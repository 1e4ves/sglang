from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Any, Sequence

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_cache_connector_mixin import UnifiedTreeConnector

logger = logging.getLogger(__name__)


class _LogicalPool:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.kv_buffer = None


class _TokenRowsPool:
    """Direct view of tensors whose first dimension stores token slots."""

    def __init__(self, buffers: Sequence[torch.Tensor], page_size: int):
        self.kv_buffer = list(buffers)
        self.page_size = page_size
        self._page_offsets = torch.arange(page_size)
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)
        self._row_sizes = tuple(
            buffer[0].numel() * buffer.element_size() * page_size
            for buffer in self.kv_buffer
        )

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def get_page_buffer_meta(self, indices: torch.Tensor):
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(
                f"Mooncake transfer has {slots.numel()} indices, expected a "
                f"multiple of page_size={self.page_size}."
            )
        if not slots.numel():
            return [], []

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + self._page_offsets
        ):
            raise ValueError(
                "Direct Mooncake requires aligned contiguous device pages."
            )
        first_row = int(starts.min())
        last_row = int(starts.max()) + self.page_size
        if first_row < 0 or last_row > self._row_count:
            bad_row = first_row if first_row < 0 else last_row - 1
            raise ValueError(
                f"Mooncake token row {bad_row} exceeds buffer shapes "
                f"{[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )

        ptrs = [
            buffer[int(start)].data_ptr()
            for start in starts.tolist()
            for buffer in self.kv_buffer
        ]
        return ptrs, list(self._row_sizes) * len(starts)


class _PageRowsPool:
    """Direct view of tensors whose first dimension stores logical pages."""

    def __init__(self, buffers: Sequence[torch.Tensor], page_size: int):
        self.kv_buffer = list(buffers)
        self.page_size = page_size
        self._page_offsets = torch.arange(page_size)
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)
        self._row_sizes = tuple(
            buffer.shape[1:].numel() * buffer.element_size()
            for buffer in self.kv_buffer
        )
        self._row_ptrs = tuple(
            tuple(
                buffer.data_ptr() + row * buffer.stride(0) * buffer.element_size()
                for buffer in self.kv_buffer
            )
            for row in range(self._row_count)
        )

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def get_page_buffer_meta(self, indices: torch.Tensor):
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(
                f"Mooncake transfer has {slots.numel()} indices, expected a "
                f"multiple of page_size={self.page_size}."
            )
        if not slots.numel():
            return [], []

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + self._page_offsets
        ):
            raise ValueError(
                "Direct Mooncake requires aligned contiguous device pages."
            )

        rows = starts.div(self.page_size, rounding_mode="floor")
        first_row = int(rows.min())
        last_row = int(rows.max())
        if first_row < 0 or last_row >= self._row_count:
            bad_row = first_row if first_row < 0 else last_row
            raise ValueError(
                f"Mooncake page row {bad_row} exceeds buffer shapes "
                f"{[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )

        row_indices = rows.tolist()
        ptrs = [ptr for row in row_indices for ptr in self._row_ptrs[row]]
        return ptrs, list(self._row_sizes) * len(row_indices)


def _state_page_views(pools: Sequence[Any], name: PoolName) -> list[torch.Tensor]:
    views = []
    for pool in pools:
        if pool is None:
            raise ValueError(f"DeepSeek V4 pool {name} has a missing state pool.")
        state = pool.kv_score_buffer.kv_score
        ring_size = int(pool.ring_size)
        usable = state.shape[0] // ring_size * ring_size
        views.append(
            state.view(torch.uint8)
            .reshape(state.shape[0], -1)[:usable]
            .reshape(usable // ring_size, -1)
        )
    return views


def _build_dsv4_pools(kvcache: Any, page_size: int):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
        HiSparseC4DevicePool,
    )

    if not isinstance(kvcache, DeepSeekV4TokenToKVPool):
        raise ValueError(
            "Direct Mooncake connector currently supports DeepSeek V4 KV pools only."
        )
    if kvcache._unified_kv:
        raise ValueError(
            "Direct Mooncake connector does not support DeepSeek V4 unified-KV layout."
        )
    if isinstance(kvcache.c4_kv_pool, HiSparseC4DevicePool):
        raise ValueError(
            "Direct Mooncake connector does not support DeepSeek V4 HiSparse."
        )
    if kvcache.swa_page_size != page_size:
        raise ValueError(
            "DeepSeek V4 SWA page size must match the tree page size: "
            f"{kvcache.swa_page_size} != {page_size}."
        )

    mapping = kvcache.layer_mapping[kvcache._stage_start : kvcache._stage_end]
    c4_layers = [
        kvcache._stage_start + i
        for i, layer in enumerate(mapping)
        if layer.compress_ratio == 4
    ]
    has_c128 = any(layer.compress_ratio == 128 for layer in mapping)

    pools = {
        PoolName.SWA: _PageRowsPool(kvcache.swa_kv_pool.kv_buffer, page_size),
    }
    sources = {PoolName.SWA: PoolName.SWA}

    def add(name: PoolName, buffers: Sequence[torch.Tensor], source: PoolName):
        if not buffers:
            raise ValueError(f"DeepSeek V4 direct Mooncake requires pool {name}.")
        pools[name] = _PageRowsPool(buffers, page_size)
        sources[name] = source

    if c4_layers:
        add(PoolName.DEEPSEEK_V4_C4, kvcache.c4_kv_pool.kv_buffer, PoolName.KV)
        add(
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
            PoolName.KV,
        )
        add(
            PoolName.DEEPSEEK_V4_C4_STATE,
            _state_page_views(
                [kvcache.compress_state_pools[i] for i in c4_layers],
                PoolName.DEEPSEEK_V4_C4_STATE,
            ),
            PoolName.SWA,
        )
        add(
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            _state_page_views(
                [kvcache.indexer_compress_state_pools[i] for i in c4_layers],
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            ),
            PoolName.SWA,
        )
    if has_c128:
        add(
            PoolName.DEEPSEEK_V4_C128,
            kvcache.c128_kv_pool.kv_buffer,
            PoolName.KV,
        )
    return _LogicalPool(page_size), pools, sources


def _build_pools(kvcache: Any, page_size: int):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
    )
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    if isinstance(kvcache, DeepSeekV4TokenToKVPool):
        return _build_dsv4_pools(kvcache, page_size)
    if not isinstance(kvcache, DSATokenToKVPool):
        raise ValueError(
            "Direct Mooncake connector currently supports standard DSA and "
            "DeepSeek V4 KV pools only."
        )
    if kvcache.page_size != page_size:
        raise ValueError(
            "DSA KV page size must match the tree page size: "
            f"{kvcache.page_size} != {page_size}."
        )

    pools = {
        PoolName.KV: _TokenRowsPool(kvcache.kv_buffer, page_size),
        PoolName.INDEXER: _PageRowsPool(kvcache.index_k_with_scale_buffer, page_size),
    }
    return (
        _LogicalPool(page_size),
        pools,
        {
            PoolName.KV: PoolName.KV,
            PoolName.INDEXER: PoolName.KV,
        },
    )


class MooncakeTreeConnector(UnifiedTreeConnector):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        _storage=None,
    ):
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        anchor, self.pools, self.sources = _build_pools(kvcache, self.page_size)

        tp_rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=params.tp_cache_group)
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=server_args.model_path,
            extra_config=extra_config,
        )
        if _storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(storage_config, mem_pool=None)
        else:
            self.storage = _storage
        self.storage.mem_pool_host = anchor
        self.storage.registered_pools = self.pools
        self.storage.mla_suffix = (
            f"tp{tp_rank}_cp{params.attn_cp_rank}_pp{params.pp_rank}"
        )
        self._stats = {"lookup": 0, "load": 0, "offload": 0}
        self._register_buffers()
        self._load_executor = ThreadPoolExecutor(
            max_workers=len(self.pools), thread_name_prefix="mooncake-get"
        )

    def _register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def _expand(
        self, transfers: list[PoolTransfer], *, allow_partial: bool = False
    ) -> list[PoolTransfer]:
        by_name = {transfer.name: transfer for transfer in transfers}
        kv = by_name.get(PoolName.KV)
        if kv is None or not kv.keys:
            return []
        if not allow_partial:
            missing = set(self.sources.values()) - set(by_name)
            if missing:
                raise ValueError(
                    "Mooncake transfer is missing required pools: "
                    + ", ".join(sorted(str(name) for name in missing))
                )

        expanded = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None:
                continue
            keys = kv.keys if source_name == PoolName.KV else source.keys
            if not keys:
                raise ValueError(f"Mooncake transfer {name} has no keys.")
            expanded.append(
                replace(
                    source,
                    name=name,
                    host_indices=source.device_indices,
                    keys=list(keys),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if source_name == PoolName.KV
                        else source.hit_policy
                    ),
                    indices_from_pool=None,
                )
            )
        return expanded

    @staticmethod
    def _all_succeeded(results: dict, transfers: list[PoolTransfer]) -> bool:
        return bool(results) and all(
            len(results.get(transfer.name, ())) == len(transfer.keys)
            and all(results[transfer.name])
            for transfer in transfers
        )

    def _page_exists(self, page_keys: list[str], transfer: PoolTransfer) -> list[bool]:
        """Per-page presence of one pool's objects over the whole candidate range."""
        component_keys, multiplier = self.storage._get_hybrid_page_component_keys(
            page_keys, transfer
        )
        if multiplier <= 0:
            return [False] * len(page_keys)
        ex = self.storage._batch_exist(self.storage._tag_keys(component_keys))
        return [
            all(r == 1 for r in ex[i * multiplier : (i + 1) * multiplier])
            for i in range(len(page_keys))
        ]

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self._expand(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        num_pages = len(page_keys)
        if num_pages == 0:
            return []

        # Contiguous-prefix bound. The DSv4 logical anchor stores no KV object,
        # so KV pages are implicitly present and the bound comes from ALL_PAGES
        # sidecars only.
        limit = (
            num_pages
            if self.storage.mem_pool_host.kv_buffer is None
            else int(self.storage.batch_exists(page_keys))
        )

        # Trailing-window pools: keep the full per-page mask as a prefix count so
        # each candidate boundary can be tested in O(1).
        windows: list[tuple[int, list[int]]] = []
        for transfer in expanded:
            page_exists = self._page_exists(page_keys, transfer)
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                limit = min(
                    limit,
                    next(
                        (i for i, ok in enumerate(page_exists) if not ok),
                        num_pages,
                    ),
                )
                continue
            counts = [0] * (num_pages + 1)
            for i, ok in enumerate(page_exists):
                counts[i + 1] = counts[i] + int(ok)
            windows.append((max(1, len(transfer.keys or ())), counts))

        valid = [
            end
            for end in range(1, limit + 1)
            if all(
                counts[end] - counts[max(0, end - window)] == end - max(0, end - window)
                for window, counts in windows
            )
        ]
        self._stats["lookup"] += 1
        if valid:
            logger.info(
                "Unified tree Mooncake lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                valid[-1],
                len(valid),
            )
        return valid

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        expanded = self._expand(transfers)
        if not expanded:
            return False
        futures = [
            self._load_executor.submit(self.storage.batch_get_v2, [transfer])
            for transfer in expanded
        ]
        wait(futures)
        results = {}
        for future in futures:
            results.update(future.result())
        if not self._all_succeeded(results, expanded):
            return False
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        loaded = len(kv.keys) * self.page_size
        self._stats["load"] += 1
        logger.info("Unified tree Mooncake load back: rid=%s tokens=%d", rid, loaded)
        return True

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self._expand(transfers, allow_partial=True)
        if not expanded:
            return False
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        self._wait_for_device()
        results = self.storage.batch_set_v2(expanded)
        if not self._all_succeeded(results, expanded):
            return False
        self._stats["offload"] += 1
        if self._stats["offload"] == 1:
            logger.info("Unified tree Mooncake offload: tokens=%d", tokens)
        return True

    def _wait_for_device(self) -> None:
        device = next(
            (
                buffer.device
                for pool in self.pools.values()
                for buffer in pool.get_hybrid_pool_buffer()
                if buffer.device.type == "cuda"
            ),
            None,
        )
        if device is not None:
            torch.cuda.synchronize(device)

    def close(self) -> None:
        logger.info("Unified tree Mooncake stats: %s", self._stats)
        self._load_executor.shutdown(wait=True)
        self.storage.close()
