from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from queue import Queue
from typing import Any

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


def _prefix_offsets(sizes: Sequence[int]) -> tuple[int, ...]:
    offsets = []
    current = 0
    for size in sizes:
        offsets.append(current)
        current += size
    return tuple(offsets)


class _LogicalPool:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.kv_buffer = None


class _LayerRowsPool:
    def __init__(
        self,
        buffers: Sequence[torch.Tensor],
        page_size: int,
        *,
        rows_are_pages: bool,
    ):
        if not buffers:
            raise ValueError("Direct Mooncake requires at least one layer buffer.")
        self.kv_buffer = list(buffers)
        self.page_size = page_size
        self._page_offsets = torch.arange(page_size)
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)
        self._row_span = 1 if rows_are_pages else page_size
        self._rows_are_pages = rows_are_pages
        self._row_sizes = tuple(
            buffer[0].numel() * buffer.element_size() * self._row_span
            for buffer in self.kv_buffer
        )
        self._layer_offsets = _prefix_offsets(self._row_sizes)

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def _rows(self, indices: torch.Tensor) -> torch.Tensor:
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(
                f"Mooncake transfer has {slots.numel()} indices, expected a "
                f"multiple of page_size={self.page_size}."
            )
        if not slots.numel():
            return torch.empty((0,), dtype=torch.int64)

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + self._page_offsets
        ):
            raise ValueError(
                "Direct Mooncake requires aligned contiguous device pages."
            )

        rows = (
            starts.div(self.page_size, rounding_mode="floor")
            if self._rows_are_pages
            else starts
        )
        first_row = int(rows.min())
        last_row = int(rows.max()) + self._row_span
        if first_row < 0 or last_row > self._row_count:
            bad_row = first_row if first_row < 0 else last_row - 1
            raise ValueError(
                f"Mooncake row {bad_row} exceeds buffer shapes "
                f"{[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )
        return rows

    def get_page_buffer_meta(self, indices: torch.Tensor):
        rows = self._rows(indices).tolist()
        ptrs = [buffer[row].data_ptr() for row in rows for buffer in self.kv_buffer]
        return ptrs, list(self._row_sizes) * len(rows)

    def prepare_locations(self, indices: torch.Tensor) -> list[int]:
        return self._rows(indices).tolist()

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        size = self._row_sizes[layer]
        offset = self._layer_offsets[layer]
        return (
            [[self.kv_buffer[layer][row].data_ptr()] for row in locations],
            [[size] for _ in locations],
            [[offset] for _ in locations],
        )

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return indices


class _TokenRowsPool(_LayerRowsPool):
    def __init__(self, buffers: Sequence[torch.Tensor], page_size: int):
        super().__init__(buffers, page_size, rows_are_pages=False)


class _PageRowsPool(_LayerRowsPool):
    def __init__(self, buffers: Sequence[torch.Tensor], page_size: int):
        super().__init__(buffers, page_size, rows_are_pages=True)


class _MappedRowsPool(_LayerRowsPool):
    def __init__(
        self,
        components: Sequence[Sequence[torch.Tensor | None]],
        page_size: int,
        *,
        rows_are_pages: bool,
        packed: bool = False,
    ):
        self.components = [list(component) for component in components]
        buffers = [
            buffer
            for component in components
            for buffer in component
            if buffer is not None
        ]
        self.num_layers = len(self.components[0])
        self.packed = packed
        super().__init__(buffers, page_size, rows_are_pages=rows_are_pages)

        self.layer_items = [[] for _ in range(self.num_layers)]
        offset = 0
        for component in self.components:
            if not packed:
                offset = 0
            for layer, buffer in enumerate(component):
                if buffer is not None:
                    size = buffer[0].nbytes * self._row_span
                    self.layer_items[layer].append((buffer, size, offset))
                    offset += size

    def get_page_buffer_meta(self, indices: torch.Tensor):
        ptrs, sizes = super().get_page_buffer_meta(indices)
        if not self.packed:
            return ptrs, sizes
        width = len(self.kv_buffer)
        return (
            [ptrs[i : i + width] for i in range(0, len(ptrs), width)],
            [sizes[i : i + width] for i in range(0, len(sizes), width)],
        )

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        items = self.layer_items[layer]
        if not items:
            return None
        ptrs, sizes, offsets = [], [], []
        for row in locations:
            row_ptrs = [buffer[row].data_ptr() for buffer, _, _ in items]
            row_sizes = [size for _, size, _ in items]
            row_offsets = [offset for _, _, offset in items]
            if self.packed:
                ptrs.append(row_ptrs)
                sizes.append(row_sizes)
                offsets.append(row_offsets)
            else:
                ptrs.extend([[value] for value in row_ptrs])
                sizes.extend([[value] for value in row_sizes])
                offsets.extend([[value] for value in row_offsets])
        return ptrs, sizes, offsets


def _mapped(
    buffers: Sequence[torch.Tensor],
    num_layers: int,
    layer_to_buffer: dict[int, int],
) -> list[torch.Tensor | None]:
    mapped: list[torch.Tensor | None] = [None] * num_layers
    for layer, buffer_index in layer_to_buffer.items():
        mapped[layer] = buffers[buffer_index]
    return mapped


def _state_view(pool: Any) -> torch.Tensor:
    state = pool.kv_score_buffer.kv_score
    ring = int(pool.ring_size)
    usable = state.shape[0] // ring * ring
    return (
        state.view(torch.uint8)
        .reshape(state.shape[0], -1)[:usable]
        .reshape(usable // ring, -1)
    )


def _build_dsv4_pools(kvcache: Any, page_size: int):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool

    if kvcache._unified_kv or isinstance(kvcache.c4_kv_pool, HiSparseC4DevicePool):
        raise ValueError("Direct Mooncake does not support unified-KV or HiSparse.")
    if kvcache.swa_page_size != page_size:
        raise ValueError(
            "DeepSeek V4 SWA page size must match the tree page size: "
            f"{kvcache.swa_page_size} != {page_size}."
        )

    stage = kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
    num_layers = len(stage)
    c4, c128 = (
        {
            layer: item.compress_layer_id
            for layer, item in enumerate(stage)
            if item.compress_ratio == ratio
        }
        for ratio in (4, 128)
    )
    kv_components = [
        _mapped(kvcache.c4_kv_pool.kv_buffer, num_layers, c4),
        _mapped(
            kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
            num_layers,
            c4,
        ),
        _mapped(kvcache.c128_kv_pool.kv_buffer, num_layers, c128),
    ]
    global_layers = [kvcache.start_layer + layer for layer in c4]
    state_map = {layer: index for index, layer in enumerate(c4)}
    swa_components = [
        list(kvcache.swa_kv_pool.kv_buffer),
        *[
            _mapped(
                [_state_view(states[layer]) for layer in global_layers],
                num_layers,
                state_map,
            )
            for states in (
                kvcache.compress_state_pools,
                kvcache.indexer_compress_state_pools,
            )
        ],
    ]

    pools = {
        PoolName.KV: _MappedRowsPool(
            kv_components,
            page_size,
            rows_are_pages=True,
            packed=True,
        ),
        PoolName.SWA: _MappedRowsPool(
            swa_components,
            page_size,
            rows_are_pages=True,
            packed=True,
        ),
    }
    return (
        _LogicalPool(page_size),
        pools,
        {PoolName.KV: PoolName.KV, PoolName.SWA: PoolName.SWA},
        num_layers,
    )


def _build_hybrid_linear_pools(kvcache: Any, page_size: int, req_to_token_pool: Any):
    if getattr(req_to_token_pool, "mamba_ckpt_pool", None) is not None:
        raise ValueError(
            "Direct Mooncake does not support int8 Mamba checkpoint storage."
        )

    layer_ids = set(kvcache.full_attention_layer_id_mapping) | set(
        req_to_token_pool.mamba_map
    )
    start_layer = min(layer_ids)
    num_layers = max(layer_ids) - start_layer + 1
    full_mapping = {
        global_layer - start_layer: local_layer
        for global_layer, local_layer in kvcache.full_attention_layer_id_mapping.items()
    }
    full_pool = kvcache.full_kv_pool
    if kvcache.use_mla or getattr(full_pool, "k_scale_buffer", None) is not None:
        raise ValueError("Direct Mooncake requires unquantized Qwen3.5 MHA KV.")
    kv_pool = _MappedRowsPool(
        [
            _mapped(full_pool.k_buffer, num_layers, full_mapping),
            _mapped(full_pool.v_buffer, num_layers, full_mapping),
        ],
        page_size,
        rows_are_pages=full_pool.k_buffer[0].shape[0] < full_pool.size + page_size,
        packed=True,
    )

    state = req_to_token_pool.mamba_pool.mamba_cache
    mamba_mapping = {
        layer - start_layer: index
        for layer, index in req_to_token_pool.mamba_map.items()
    }
    temporal_size = state.temporal[0, 0].numel() if state.temporal.numel() else 0
    state_tensors = ([state.temporal] if temporal_size else []) + list(state.conv)
    mamba_pool = _MappedRowsPool(
        [_mapped(tensor, num_layers, mamba_mapping) for tensor in state_tensors],
        page_size=1,
        rows_are_pages=True,
    )
    mamba_pool.temporal_state_elem_size = temporal_size
    mamba_pool.conv_buffer = list(state.conv)
    mamba_pool.translate_indices = req_to_token_pool.translate_mamba_indices

    return (
        _LogicalPool(page_size),
        {PoolName.KV: kv_pool, PoolName.MAMBA: mamba_pool},
        {PoolName.KV: PoolName.KV, PoolName.MAMBA: PoolName.MAMBA},
        num_layers,
    )


def _build_pools(kvcache: Any, page_size: int, req_to_token_pool: Any):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
    )
    from sglang.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        HybridLinearKVPool,
    )

    if isinstance(kvcache, DeepSeekV4TokenToKVPool):
        return _build_dsv4_pools(kvcache, page_size)
    if isinstance(kvcache, HybridLinearKVPool):
        return _build_hybrid_linear_pools(kvcache, page_size, req_to_token_pool)

    if not isinstance(kvcache, DSATokenToKVPool):
        raise TypeError(
            "Direct Mooncake supports DSA, DeepSeek V4, and hybrid linear KV pools."
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
        {PoolName.KV: PoolName.KV, PoolName.INDEXER: PoolName.KV},
        len(kvcache.kv_buffer),
    )


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._producer_index = -1
        self.consumer_index = -1
        self._events: dict[int, list[threading.Event]] = {}
        self._errors: dict[int, BaseException] = {}

    def update_producer(self) -> int:
        self._producer_index += 1
        self._events[self._producer_index] = [
            threading.Event() for _ in range(self.num_layers)
        ]
        return self._producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self._events[index][layer].set()

    def fail(self, index: int, error: BaseException) -> None:
        events = self._events.get(index)
        if events is None:
            return
        self._errors[index] = error
        for event in events:
            event.set()

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        events = self._events.get(index)
        if events is None:
            return
        events[threshold].wait()
        error = self._errors.get(index)
        if threshold == self.num_layers - 1:
            self._events.pop(index, None)
            self._errors.pop(index, None)
        if error is not None:
            raise RuntimeError("Mooncake layer-wise KV load failed.") from error

    def reset(self) -> None:
        self._producer_index = -1
        self.consumer_index = -1
        self._events.clear()
        self._errors.clear()


@dataclass
class _LayerRangePlan:
    name: PoolName
    pool: Any
    keys: list[str]
    locations: list[int]

    def get_layer_meta(self, layer: int):
        return self.pool.get_prepared_layer_range_meta(self.locations, layer)


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
        (
            anchor,
            self.pools,
            self.sources,
            self.num_layers,
        ) = _build_pools(kvcache, self.page_size, params.req_to_token_pool)

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
        rank_suffix = f"tp{tp_rank}_cp{params.attn_cp_rank}_pp{params.pp_rank}"
        self.storage.mla_suffix = rank_suffix
        self.storage.mha_suffix = rank_suffix

        self._validate_range_api()
        self._register_buffers()
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self._pending: dict[str, list[PoolTransfer]] = {}
        self._load_queue: Queue[tuple[int, list[_LayerRangePlan]] | None] = Queue()
        self._stats = {"lookup": 0, "load": 0, "offload": 0}
        self._load_thread = threading.Thread(
            target=self._load_thread_func,
            daemon=True,
            name=f"mooncake-layerwise-tp{tp_rank}",
        )
        self._load_thread.start()

    def _validate_range_api(self) -> None:
        required = (
            "batch_get_session_start",
            "batch_get_into_multi_buffer_ranges",
            "batch_get_session_end",
        )
        missing = [name for name in required if not hasattr(self.storage.store, name)]
        if missing:
            raise RuntimeError(
                "The installed Mooncake package lacks the layer-wise range API: "
                + ", ".join(missing)
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
        if not allow_partial and not set(self.sources.values()) <= set(by_name):
            return []

        expanded = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None:
                continue
            keys = kv.keys if source_name == PoolName.KV else source.keys
            indices = source.device_indices
            expanded.append(
                replace(
                    source,
                    name=name,
                    host_indices=(
                        self.pools[name].translate_indices(indices)
                        if indices is not None
                        else None
                    ),
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

        valid = list(range(1, num_pages + 1))
        for transfer in expanded:
            exists = self._page_exists(page_keys, transfer)
            counts = [0] * (num_pages + 1)
            for index, present in enumerate(exists):
                counts[index + 1] = counts[index] + int(present)
            window = (
                num_pages
                if transfer.hit_policy == PoolHitPolicy.ALL_PAGES
                else max(1, len(transfer.keys or ()))
            )
            valid = [
                end
                for end in valid
                if counts[end] - counts[max(0, end - window)]
                == end - max(0, end - window)
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
        if rid in self._pending:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self._pending[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> None:
        self._pending.pop(rid, None)

    def start_layer_wise_loading(self) -> int:
        if not self._pending:
            return -1
        pending = self._pending
        self._pending = {}

        plans = self._build_range_plans(list(pending.values()))
        counter_index = self.layer_done_counter.update_producer()
        self._load_queue.put((counter_index, plans))
        self._stats["load"] += len(pending)
        return counter_index

    def _load_thread_func(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                counter_index, plans = task
                self._run_layer_wise_batch(counter_index, plans)
            finally:
                self._load_queue.task_done()

    def _build_range_plans(
        self, request_transfers: list[list[PoolTransfer]]
    ) -> list[_LayerRangePlan]:
        grouped: dict[PoolName, list[PoolTransfer]] = {}
        for transfers in request_transfers:
            for transfer in transfers:
                grouped.setdefault(transfer.name, []).append(transfer)

        plans = []
        for name, transfers in grouped.items():
            keys = []
            locations = []
            for transfer in transfers:
                component_keys, multiplier = (
                    self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                )
                keys.extend(self.storage._tag_keys(component_keys))
                transfer_locations = self.pools[name].prepare_locations(
                    transfer.host_indices
                )
                if len(transfer_locations) * multiplier != len(component_keys):
                    raise ValueError(
                        f"Layer-wise Mooncake pool {name} has "
                        f"{len(component_keys)} component keys for "
                        f"{len(transfer_locations)} destination pages."
                    )
                locations.extend(transfer_locations)
            if not locations or not keys:
                raise ValueError(
                    f"Layer-wise Mooncake pool {name} has no destinations."
                )
            plans.append(
                _LayerRangePlan(
                    name=name,
                    pool=self.pools[name],
                    keys=keys,
                    locations=locations,
                )
            )
        return plans

    @staticmethod
    def _status_ok(result: Any, expected: int | None = None) -> bool:
        if result is None:
            return True
        if isinstance(result, int):
            return result == 0
        values = list(result)
        return (expected is None or len(values) == expected) and all(
            value == 0 for value in values
        )

    @staticmethod
    def _range_result_ok(result: Any, sizes: list[list[int]]) -> bool:
        if result is None or isinstance(result, int):
            return False
        return list(result) == [sum(key_sizes) for key_sizes in sizes]

    def _run_layer_wise_batch(
        self, counter_index: int, plans: list[_LayerRangePlan]
    ) -> None:
        started: list[_LayerRangePlan] = []
        try:
            for plan in plans:
                result = self.storage.store.batch_get_session_start(plan.keys)
                if not self._status_ok(result, len(plan.keys)):
                    raise RuntimeError(
                        f"Mooncake session start failed for pool {plan.name}: {result}"
                    )
                started.append(plan)

            if not plans:
                raise ValueError("Layer-wise Mooncake load has no page keys.")

            for layer in range(self.num_layers):
                active = False
                for plan in plans:
                    meta = plan.get_layer_meta(layer)
                    if meta is None:
                        continue
                    active = True
                    ptrs, sizes, offsets = meta
                    if len(ptrs) != len(plan.keys):
                        raise ValueError(
                            f"Mooncake pool={plan.name}, layer={layer} produced "
                            f"{len(ptrs)} ranges for {len(plan.keys)} keys."
                        )
                    result = self.storage.store.batch_get_into_multi_buffer_ranges(
                        plan.keys,
                        ptrs,
                        sizes,
                        offsets,
                    )
                    if not self._range_result_ok(result, sizes):
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={plan.name}, "
                            f"layer={layer}: transferred={result}, "
                            f"expected={[sum(item) for item in sizes]}"
                        )
                if not active:
                    raise ValueError(
                        f"Layer-wise Mooncake load has no pool for layer {layer}."
                    )
                self.layer_done_counter.complete(counter_index, layer)
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for plan in started:
                try:
                    result = self.storage.store.batch_get_session_end(plan.keys)
                    if not self._status_ok(result):
                        raise RuntimeError(
                            f"Mooncake session end failed for pool "
                            f"{plan.name}: {result}"
                        )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")

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

    def reset(self) -> None:
        self._pending.clear()
        self._load_queue.join()
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self._load_queue.put(None)
        self._load_thread.join()
        logger.info("Unified tree Mooncake stats: %s", self._stats)
        self.storage.close()
