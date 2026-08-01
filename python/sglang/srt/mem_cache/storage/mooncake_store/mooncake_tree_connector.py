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


class _TokenRowsPool(_LayerRowsPool):
    def __init__(self, buffers: Sequence[torch.Tensor], page_size: int):
        super().__init__(buffers, page_size, rows_are_pages=False)


class _PageRowsPool(_LayerRowsPool):
    def __init__(self, buffers: Sequence[torch.Tensor], page_size: int):
        super().__init__(buffers, page_size, rows_are_pages=True)


def _build_pools(kvcache: Any, page_size: int):
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    if not isinstance(kvcache, DSATokenToKVPool):
        raise TypeError(
            "Direct Mooncake connector currently supports DSA KV pools only."
        )
    if kvcache.page_size != page_size:
        raise ValueError(
            "DSA KV page size must match the tree page size: "
            f"{kvcache.page_size} != {page_size}."
        )

    return (
        _LogicalPool(page_size),
        {
            PoolName.KV: _TokenRowsPool(kvcache.kv_buffer, page_size),
            PoolName.INDEXER: _PageRowsPool(
                kvcache.index_k_with_scale_buffer, page_size
            ),
        },
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
        anchor, self.pools = _build_pools(kvcache, self.page_size)
        self.num_layers = len(self.pools[PoolName.KV].kv_buffer)

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

        self._validate_range_api()
        self._register_buffers()
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
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

    def _expand(self, transfers: list[PoolTransfer]) -> list[PoolTransfer]:
        kv = next(
            (transfer for transfer in transfers if transfer.name == PoolName.KV),
            None,
        )
        if kv is None or not kv.keys:
            return []
        return [
            replace(
                kv,
                name=name,
                host_indices=kv.device_indices,
                keys=list(kv.keys),
                indices_from_pool=None,
            )
            for name in self.pools
        ]

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
        page_keys = list(expanded[0].keys)
        present = [True] * len(page_keys)
        for transfer in expanded:
            present = [
                left and right
                for left, right in zip(present, self._page_exists(page_keys, transfer))
            ]
        hit_pages = next(
            (index for index, exists in enumerate(present) if not exists),
            len(page_keys),
        )
        valid = list(range(1, hit_pages + 1))
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
                if multiplier != 1:
                    raise ValueError(
                        "Layer-wise ranges require one object per page, got "
                        f"{multiplier} for pool {name}."
                    )
                keys.extend(self.storage._tag_keys(component_keys))
                locations.extend(
                    self.pools[name].prepare_locations(transfer.host_indices)
                )
            if len(locations) != len(keys):
                raise ValueError(
                    f"Layer-wise Mooncake pool {name} has {len(keys)} keys but "
                    f"{len(locations)} destination pages."
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

            if not plans or not plans[0].keys:
                raise ValueError("Layer-wise Mooncake load has no page keys.")
            num_keys = len(plans[0].keys)
            if any(len(plan.keys) != num_keys for plan in plans):
                raise ValueError(
                    "Layer-wise Mooncake pools must cover the same page set."
                )

            for layer in range(self.num_layers):
                for plan in plans:
                    ptrs, sizes, offsets = plan.get_layer_meta(layer)
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
        expanded = self._expand(transfers)
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
