from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from queue import Empty, Queue
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
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_mappings import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache_connector_mixin import UnifiedTreeConnector

logger = logging.getLogger(__name__)


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
        pool_group = resolve_hybrid_device_pool_group(
            kvcache, self.page_size, params.req_to_token_pool
        )
        self.pools = pool_group.entry_map
        self.sources = pool_group.sources
        self.num_layers = pool_group.num_layers

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
        self.storage.mem_pool_host = pool_group
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
        self.load_queue: Queue[tuple[int, list[_LayerRangePlan]] | None] = Queue()
        self.offload_queue: Queue[tuple[list[PoolTransfer], int] | None] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self._stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-layerwise-tp{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

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
        self.load_queue.put((counter_index, plans))
        self._stats["load"] += len(pending)
        return counter_index

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, plans = task
                self._run_layer_wise_batch(counter_index, plans)
            finally:
                self.load_queue.task_done()

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
        self.offload_queue.put((expanded, tokens))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens = task
                self._wait_for_device()
                results = self.storage.batch_set_v2(expanded)
                success = self._all_succeeded(results, expanded)
                if success:
                    self._stats["offload"] += 1
                    if self._stats["offload"] == 1:
                        logger.info("Unified tree Mooncake offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

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
        self.load_queue.join()
        self.offload_queue.join()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Unified tree Mooncake stats: %s", self._stats)
        self.storage.close()
