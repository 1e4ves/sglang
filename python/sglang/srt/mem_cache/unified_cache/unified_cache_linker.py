"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. Self-contained: this module owns both halves of the contract.

* :class:`DevicePoolEntry` / :class:`DevicePoolGroup` -- the shared zero-copy
  view over device pools used by direct-linker backends.
* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``_inc_hit_count``    -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

"""

from __future__ import annotations

import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_hash_str, hash_str_to_int64

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools."""

    layer_done_counter: object

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    def prepare_read(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Query and, when supported, pin the readable objects for ``rid``.

        Backends without a read-session primitive retain the old lookup
        semantics.  Mooncake overrides this method and holds get sessions until
        the corresponding load or cancellation releases them.
        """
        return self.lookup(rid, transfers)

    def release_read(self, rid: str) -> None:
        """Release a read prepared for ``rid`` without loading it."""

    def retain_read(self, rid: str, transfers: list[PoolTransfer]) -> None:
        """Drop prepared objects that are outside the agreed load boundary."""

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def num_completed_loads(self) -> int:
        """Return the number of completed load batches waiting to be consumed."""

    @abstractmethod
    def pop_completed_load(self) -> list[str]:
        """Consume the oldest completed load batch and return its request IDs."""

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``prefix_key`` covers the device-cached prefix plus the restorable tail, so
    it is what gets inserted once the tail lands. ``tail_hashes`` are the
    per-page storage hashes of that tail alone, starting at ``device_hit_len``.
    """

    prefix_key: RadixKey
    tail_hashes: list[str]
    device_hit_len: int


class _PrepareOperation(IntEnum):
    NONE = 0
    LOOKUP = 1
    ACQUIRE = 2


@dataclass
class _PreparedRead:
    request_id: int
    sequence: int
    signature: int
    rid: str | None
    page_hashes: tuple[str, ...] | None
    start_page: int
    num_pages: int
    local_restorable: tuple[int, ...]
    common_pages: int
    swa_window_pages: int
    acquired: bool
    read_held: bool
    anchor_node: Any = None
    anchor_lock_params: DecLockRefParams | None = None


@dataclass
class _PrepareItem:
    request_id: int
    sequence: int
    signature: int
    operation: _PrepareOperation
    rid: str | None
    page_hashes: tuple[str, ...] | None
    start_page: int
    num_pages: int
    transfers: list[PoolTransfer] | None
    swa_window_pages: int = 0
    anchor_node: Any = None
    anchor_lock_params: DecLockRefParams | None = None


@dataclass
class _PrepareBatch:
    items: list[_PrepareItem]
    completion: threading.Event | None = None
    results: list[_PreparedRead] | None = None


class DevicePoolEntry:
    """Zero-copy linker view over one physical device pool."""

    def __init__(
        self,
        *,
        name: PoolName,
        indices_from_pool: PoolName,
        device_pool: Any,
        components: Sequence[Sequence[torch.Tensor]],
        layer_mapping: dict[int, int],
        page_size: int,
        rows_are_pages: bool,
        packed: bool = True,
        index_mapper: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.name = name
        self.indices_from_pool = indices_from_pool
        self.device_pool = device_pool
        self.components = [list(component) for component in components]
        self.layer_mapping = layer_mapping
        self.page_size = page_size
        self.packed = packed
        self._index_mapper = index_mapper
        self._page_offsets = torch.arange(page_size)
        self._row_span = 1 if rows_are_pages else page_size

        if not self.components or any(not component for component in self.components):
            raise ValueError(f"Device pool {name} has no storage buffers.")
        self.kv_buffer = [buffer for group in self.components for buffer in group]
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)

        self.buffer_meta = [
            [
                (
                    buffer.data_ptr(),
                    buffer.stride(0) * buffer.element_size(),
                    buffer.nbytes // buffer.shape[0] * self._row_span,
                )
                for buffer in component
            ]
            for component in self.components
        ]

        self._component_offsets = []
        offset = 0
        for component in self.buffer_meta:
            if not packed:
                offset = 0
            offsets = []
            for _, _, size in component:
                offsets.append(offset)
                offset += size
            self._component_offsets.append(offsets)

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self._index_mapper(indices) if self._index_mapper else indices

    def _rows(self, indices: torch.Tensor) -> list[int]:
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(
                f"Pool {self.name} got {slots.numel()} indices, expected a "
                f"multiple of page_size={self.page_size}."
            )
        if not slots.numel():
            return []

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + self._page_offsets
        ):
            raise ValueError(f"Pool {self.name} requires aligned contiguous pages.")
        rows = (
            starts.div(self.page_size, rounding_mode="floor")
            if self._row_span == 1
            else starts
        )
        first_row = int(rows.min())
        last_row = int(rows.max()) + self._row_span
        if first_row < 0 or last_row > self._row_count:
            raise ValueError(
                f"Pool {self.name} row range [{first_row}, {last_row}) exceeds "
                f"buffer shapes {[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )
        return rows.tolist()

    def get_page_buffer_meta(self, indices: torch.Tensor):
        rows = self._rows(indices)
        ptrs = [
            base_ptr + row * row_stride
            for row in rows
            for component in self.buffer_meta
            for base_ptr, row_stride, _ in component
        ]
        sizes = [
            size
            for _ in rows
            for component in self.buffer_meta
            for _, _, size in component
        ]
        return ptrs, sizes

    def prepare_locations(self, indices: torch.Tensor) -> list[int]:
        return self._rows(indices)

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        buffer_index = self.layer_mapping.get(layer)
        if buffer_index is None:
            return None

        items = []
        for component, offsets in zip(self.buffer_meta, self._component_offsets):
            base_ptr, row_stride, size = component[buffer_index]
            items.append((base_ptr, row_stride, size, offsets[buffer_index]))

        ptrs, sizes, offsets = [], [], []
        for row in locations:
            row_ptrs = [
                base_ptr + row * row_stride for base_ptr, row_stride, _, _ in items
            ]
            row_sizes = [size for _, _, size, _ in items]
            row_offsets = [offset for _, _, _, offset in items]
            if self.packed:
                ptrs.append(row_ptrs)
                sizes.append(row_sizes)
                offsets.append(row_offsets)
            else:
                ptrs.extend([[value] for value in row_ptrs])
                sizes.extend([[value] for value in row_sizes])
                offsets.extend([[value] for value in row_offsets])
        return ptrs, sizes, offsets


class DevicePoolGroup:
    """Physical device pools sharing one logical linker layer range."""

    def __init__(
        self, entries: Sequence[DevicePoolEntry], num_layers: int, page_size: int
    ):
        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        if len(self.entries) != len(self.entry_map):
            raise ValueError("DevicePoolGroup contains duplicate pool names.")
        self.sources = {entry.name: entry.indices_from_pool for entry in self.entries}
        self.num_layers = num_layers
        self.page_size = page_size
        self.kv_buffer = None

    def resolve_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        allow_partial: bool = False,
        allow_missing_kv: bool = False,
    ) -> list[PoolTransfer]:
        """Expand logical component transfers into physical device pools."""
        by_name = {transfer.name: transfer for transfer in transfers}
        kv = by_name.get(PoolName.KV)
        if not any(transfer.keys for transfer in transfers):
            return []
        if not allow_missing_kv and (kv is None or not kv.keys):
            return []
        if not allow_partial and not set(self.sources.values()) <= set(by_name):
            return []

        resolved = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None or not source.keys:
                continue
            indices = source.device_indices
            resolved.append(
                replace(
                    source,
                    name=name,
                    host_indices=(
                        self.entry_map[name].translate_indices(indices)
                        if indices is not None
                        else None
                    ),
                    keys=list(source.keys),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if source_name == PoolName.KV
                        else source.hit_policy
                    ),
                    indices_from_pool=None,
                )
            )
        return resolved


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        server_args: ServerArgs,
        params: CacheInitParams,
    ):
        backend = server_args.unified_cache_external_linker_backend
        if backend == "mooncake":
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
                MooncakeDirectLinker,
            )

            linker_cls = MooncakeDirectLinker
        elif backend == "mori":
            from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
                UMBPDirectLinker,
            )

            linker_cls = UMBPDirectLinker
        else:
            raise ValueError(
                f"Unknown unified cache external linker backend: {backend!r}"
            )

        self.cache = cache
        self.cache_linker: UnifiedCacheLinker = linker_cls(
            server_args,
            params,
            components=set(cache.components),
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Loads in flight, each pinning its inserted endpoint until DMA completes.
        self.pending_loads: dict[str, tuple[NodeId, DecLockRefParams]] = {}
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[tuple[NodeId, DecLockRefParams]] = []

        # L3 reads are prepared outside the scheduler thread. For PP prefill,
        # waiting requests get a pure lookup first; only an admission-time
        # match asks the same worker to acquire a per-rank read session. Both
        # phases intersect sparse boundaries on dedicated CPU groups.
        self._pp_prefill_protocol = cache.pp_size > 1
        self._prepare_pending: dict[
            int, tuple[int, _PrepareOperation, int, tuple[str, ...] | None]
        ] = {}
        self._prepared_reads: dict[int, _PreparedRead] = {}
        self._rid_to_request_id: dict[str, int] = {}
        self._released_prepare_sequences: set[tuple[int, int]] = set()
        self._active_prepare_ids: set[int] = set()
        self._next_prepare_sequence = 1
        self._prepare_queue: Queue[_PrepareBatch | None] = Queue()
        self._prepare_results: Queue[list[_PreparedRead]] = Queue()
        self._prepare_sync_groups = self._create_prepare_sync_groups(params)
        self._prepare_thread = threading.Thread(
            target=self._prepare_thread_func,
            daemon=True,
            name=f"linker-prepare-pp{params.pp_rank}",
        )
        self._prepare_thread.start()

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    @staticmethod
    def request_id(rid: str) -> int:
        """Stable positive int64 ID used by the PP prepare manifest."""
        digest = hashlib.blake2b(rid.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)

    def _create_prepare_sync_groups(self, params) -> list[object]:
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return []

        from sglang.srt.distributed.parallel_state import (
            create_custom_parallel_group,
        )

        groups = []
        seen_rank_sets = set()
        if (
            params.attn_cp_cache_group is not None
            or params.attn_tp_cache_group is not None
        ):
            base_groups = [params.attn_cp_cache_group, params.attn_tp_cache_group]
        else:
            base_groups = [params.tp_cache_group]
        base_groups.append(params.pp_cache_group)
        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) <= 1:
                continue
            ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(ranks)
            groups.append(
                create_custom_parallel_group(group_ranks=list(ranks), backend="gloo")
            )
        return groups

    def _destroy_prepare_sync_groups(self) -> None:
        for group in self._prepare_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except RuntimeError:
                logger.debug("Failed to destroy linker prepare group", exc_info=True)
        self._prepare_sync_groups = []

    def _all_reduce_prepare_groups(self, tensor: torch.Tensor, op) -> None:
        assert tensor.device.type == "cpu"
        for group in self._prepare_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    # ---- phase 1: scheduler-window lookup, then admission-time acquire ----

    def _key_for_req(self, req: Req) -> RadixKey | None:
        if req.positional_embed_overrides is not None:
            return None
        token_ids = req.origin_input_ids + req.output_ids
        key_limit = req._compute_max_prefix_len(len(token_ids))
        reprefill_tail = self.cache.swa_reprefill_tail_tokens()
        if reprefill_tail:
            key_limit = min(key_limit, max(0, len(token_ids) - reprefill_tail))
        return RadixKey(
            token_ids=token_ids,
            extra_key=req.extra_key,
            limit=key_limit,
            cache_salt=req.cache_salt,
        )

    def _full_page_hashes(self, key: RadixKey) -> tuple[str, ...]:
        page_aligned_len = len(key) // self.cache.page_size * self.cache.page_size
        if page_aligned_len == 0:
            return ()
        return tuple(
            get_hash_str(
                key[:page_aligned_len],
                None,
                page_size=self.cache.page_size,
            )
        )

    def _match_l1(self, key: RadixKey) -> MatchResult:
        # ``req=None`` deliberately bypasses this linker in
        # UnifiedRadixCache.match_prefix, yielding a local radix snapshot only.
        return self.cache.match_prefix(MatchPrefixParams(key=key))

    def _new_prepare_sequence(self) -> int:
        sequence = self._next_prepare_sequence
        self._next_prepare_sequence += 1
        return sequence

    def build_prepare_manifest(
        self, requests: Sequence[Req], window_size: int
    ) -> list[tuple[int, int, int, int, int, int]]:
        """Build PP0's canonical window and its next background operation.

        Records are ``(request_id, signature, sequence, start_page,
        num_pages, operation)``.  PP prefill always looks up from page zero so
        its result is an absolute logical boundary.  The old single-PP path
        retains its L1-relative prepare behavior.
        """
        manifest = []
        for req in requests[:window_size]:
            request_id = self.request_id(req.rid)
            key = self._key_for_req(req)
            full_page_hashes = self._full_page_hashes(key) if key is not None else ()
            if self._pp_prefill_protocol:
                start_page = 0
            else:
                l1_match = self._match_l1(key) if key is not None else None
                start_page = min(
                    (
                        int(l1_match.device_indices.numel()) // self.cache.page_size
                        if l1_match is not None
                        else 0
                    ),
                    len(full_page_hashes),
                )
            page_hashes = full_page_hashes[start_page:]
            signature = (
                hash_str_to_int64(full_page_hashes[-1]) if full_page_hashes else 0
            )
            if not page_hashes:
                prepared = self._prepared_reads.pop(request_id, None)
                if prepared is not None:
                    self._discard_prepared(prepared)
                manifest.append((request_id, signature, 0, start_page, 0, 0))
                continue

            prepared = self._prepared_reads.get(request_id)
            if (
                prepared is not None
                and (
                    prepared.signature != signature
                    or prepared.start_page != start_page
                    or prepared.page_hashes
                    != page_hashes[: prepared.num_pages]
                )
            ):
                self._discard_prepared(prepared)
                self._prepared_reads.pop(request_id, None)
                prepared = None

            pending = self._prepare_pending.get(request_id)
            if pending is not None:
                sequence = pending[0]
                operation = _PrepareOperation.NONE
                num_pages = len(page_hashes)
            elif prepared is None:
                sequence = self._new_prepare_sequence()
                operation = (
                    _PrepareOperation.LOOKUP
                    if self._pp_prefill_protocol
                    else _PrepareOperation.ACQUIRE
                )
                num_pages = len(page_hashes)
            else:
                sequence = prepared.sequence
                operation = _PrepareOperation.NONE
                num_pages = prepared.num_pages

            manifest.append(
                (
                    request_id,
                    signature,
                    sequence,
                    start_page,
                    num_pages,
                    int(operation),
                )
            )
        return manifest

    def reconcile_prepare_window(self, active_request_ids: set[int]) -> None:
        """Drop speculative state outside PP0's bounded admission window."""
        self._active_prepare_ids = active_request_ids
        for request_id in list(self._prepared_reads):
            if request_id not in active_request_ids:
                self._discard_prepared(self._prepared_reads.pop(request_id))
        for request_id, pending in self._prepare_pending.items():
            if request_id not in active_request_ids:
                self._released_prepare_sequences.add((request_id, pending[0]))
        for rid, request_id in list(self._rid_to_request_id.items()):
            if request_id not in active_request_ids:
                self._rid_to_request_id.pop(rid, None)

    def _build_local_prepare_inputs(
        self,
        req: Req,
        signature: int,
        start_page: int,
        num_pages: int,
        operation: _PrepareOperation,
    ) -> tuple[
        tuple[str, ...] | None,
        list[PoolTransfer] | None,
        int,
        Any,
        DecLockRefParams | None,
    ]:
        page_hashes = None
        transfers = None
        swa_window_pages = 0
        anchor_node = None
        anchor_lock_params = None
        try:
            key = self._key_for_req(req)
            full_hashes = self._full_page_hashes(key) if key is not None else ()
            if not (
                full_hashes
                and hash_str_to_int64(full_hashes[-1]) == signature
                and start_page + num_pages <= len(full_hashes)
            ):
                return (None, None, 0, None, None)

            # LOOKUP describes the whole currently matchable request. ACQUIRE
            # may intentionally cover only the shorter H_lookup prefix.
            if operation == _PrepareOperation.LOOKUP and (
                start_page + num_pages != len(full_hashes)
            ):
                return (None, None, 0, None, None)

            page_hashes = full_hashes[start_page : start_page + num_pages]
            # The PP protocol deliberately holds no L1 node across the waiting
            # window. Single-PP keeps the prior anchor behavior.
            if not self._pp_prefill_protocol:
                l1_match = self._match_l1(key)
                local_device_pages = (
                    int(l1_match.device_indices.numel()) // self.cache.page_size
                )
                if local_device_pages < start_page:
                    return (None, None, 0, None, None)
                anchor_node = l1_match.last_device_node
                lock_result = self.cache.inc_lock_ref(anchor_node)
                anchor_lock_params = lock_result.to_dec_params()
            lookup_transfers = []
            for component in self.cache._components_tuple:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, page_hashes
                )
                if transfer is None:
                    lookup_transfers = []
                    break
                lookup_transfers.append(transfer)
            if lookup_transfers:
                transfers = lookup_transfers
                swa = next(
                    (
                        transfer
                        for transfer in transfers
                        if transfer.name == PoolName.SWA
                    ),
                    None,
                )
                swa_window_pages = len(swa.keys) if swa is not None else 0
            return (
                page_hashes,
                transfers,
                swa_window_pages,
                anchor_node,
                anchor_lock_params,
            )
        except BaseException:
            logger.exception("Failed to build local linker prepare: rid=%s", req.rid)
            if anchor_node is not None:
                try:
                    self.cache.dec_lock_ref(anchor_node, anchor_lock_params)
                except BaseException:
                    logger.exception(
                        "Failed to release local linker prepare anchor: rid=%s",
                        req.rid,
                    )
            return (None, None, 0, None, None)

    def enqueue_prepare_manifest(
        self,
        manifest: Sequence[tuple[int, int, int, int, int, int]],
        local_requests: Sequence[Req],
    ) -> None:
        if not manifest:
            return
        by_id = {self.request_id(req.rid): req for req in local_requests}
        items = []
        for (
            request_id,
            signature,
            sequence,
            start_page,
            num_pages,
            operation_value,
        ) in manifest:
            operation = _PrepareOperation(operation_value)
            req = by_id.get(request_id)
            rid = req.rid if req is not None else None
            page_hashes = None
            transfers = None
            swa_window_pages = 0
            anchor_node = None
            anchor_lock_params = None

            if req is not None:
                (
                    page_hashes,
                    transfers,
                    swa_window_pages,
                    anchor_node,
                    anchor_lock_params,
                ) = self._build_local_prepare_inputs(
                    req, signature, start_page, num_pages, operation
                )

            prepared = self._prepared_reads.pop(request_id, None)
            if prepared is not None:
                self._discard_prepared(prepared)

            pending = self._prepare_pending.get(request_id)
            if pending is not None:
                logger.error(
                    "Duplicate linker prepare manifest entry while pending: request_id=%d",
                    request_id,
                )
                if anchor_node is not None:
                    self.cache.dec_lock_ref(anchor_node, anchor_lock_params)
                    anchor_node = None
                    anchor_lock_params = None
                transfers = None
                page_hashes = None
            self._prepare_pending[request_id] = (
                sequence,
                operation,
                signature,
                page_hashes,
            )
            if rid is not None:
                self._rid_to_request_id[rid] = request_id
            items.append(
                _PrepareItem(
                    request_id=request_id,
                    sequence=sequence,
                    signature=signature,
                    operation=operation,
                    rid=rid,
                    page_hashes=page_hashes,
                    start_page=start_page,
                    num_pages=num_pages,
                    transfers=transfers,
                    swa_window_pages=swa_window_pages,
                    anchor_node=anchor_node,
                    anchor_lock_params=anchor_lock_params,
                )
            )
        self._prepare_queue.put(_PrepareBatch(items))

    def _prepare_thread_func(self) -> None:
        while True:
            batch = self._prepare_queue.get()
            try:
                if batch is None:
                    return
                local_restorable = []
                read_held = []
                for item in batch.items:
                    held = False
                    if item.rid is not None and item.transfers is not None:
                        try:
                            if item.operation == _PrepareOperation.LOOKUP:
                                restorable = self.cache_linker.lookup(
                                    item.rid, item.transfers
                                )
                            else:
                                restorable = self.cache_linker.prepare_read(
                                    item.rid, item.transfers
                                )
                                held = True
                        except BaseException:
                            logger.exception(
                                "External linker %s failed: rid=%s",
                                item.operation.name.lower(),
                                item.rid,
                            )
                            if item.operation == _PrepareOperation.ACQUIRE:
                                try:
                                    self.cache_linker.release_read(item.rid)
                                except BaseException:
                                    logger.exception(
                                        "External linker acquire cleanup failed: rid=%s",
                                        item.rid,
                                    )
                            restorable = []
                    else:
                        restorable = []
                    local_restorable.append(tuple(restorable))
                    read_held.append(held)

                offsets = []
                total = 0
                for item in batch.items:
                    offsets.append(total)
                    total += item.num_pages + 1
                mask = torch.zeros(total, dtype=torch.int, device="cpu")
                assert mask.device.type == "cpu"
                for item, offset, restorable in zip(
                    batch.items, offsets, local_restorable
                ):
                    for pages in restorable:
                        if 0 < pages <= item.num_pages:
                            mask[offset + pages] = 1
                self._all_reduce_prepare_groups(
                    mask, torch.distributed.ReduceOp.MIN
                )

                results = []
                for item, offset, restorable, held in zip(
                    batch.items, offsets, local_restorable, read_held
                ):
                    common = mask[offset : offset + item.num_pages + 1].nonzero()
                    common_pages = int(common[-1].item()) if common.numel() else 0
                    if common_pages == 0 and held and item.rid is not None:
                        try:
                            self.cache_linker.release_read(item.rid)
                        except BaseException:
                            logger.exception(
                                "External linker miss cleanup failed: rid=%s", item.rid
                            )
                        held = False
                    elif (
                        common_pages > 0
                        and held
                        and item.rid is not None
                        and item.page_hashes is not None
                    ):
                        retained_transfers = []
                        for component in self.cache._components_tuple:
                            transfer = component.build_external_linker_transfer(
                                LinkerTransferPhase.LOOKUP,
                                None,
                                item.page_hashes[:common_pages],
                            )
                            if transfer is None:
                                retained_transfers = []
                                break
                            retained_transfers.append(transfer)
                        if retained_transfers:
                            try:
                                self.cache_linker.retain_read(
                                    item.rid, retained_transfers
                                )
                            except BaseException:
                                # The agreed objects remain leased; failure to
                                # release the unused suffix is a bounded leak
                                # until release_read/TTL, not a read hazard.
                                logger.exception(
                                    "External linker prepare trim failed: rid=%s",
                                    item.rid,
                                )
                    results.append(
                        _PreparedRead(
                            request_id=item.request_id,
                            sequence=item.sequence,
                            signature=item.signature,
                            rid=item.rid,
                            page_hashes=item.page_hashes,
                            start_page=item.start_page,
                            num_pages=item.num_pages,
                            # retain_read has dropped objects beyond the agreed
                            # boundary, so a later local reuse must not advertise
                            # the pre-trim sparse candidates.
                            local_restorable=tuple(
                                pages
                                for pages in restorable
                                if pages <= common_pages
                            ),
                            common_pages=common_pages,
                            swa_window_pages=item.swa_window_pages,
                            acquired=item.operation == _PrepareOperation.ACQUIRE,
                            read_held=held,
                            anchor_node=item.anchor_node,
                            anchor_lock_params=item.anchor_lock_params,
                        )
                    )
                if batch.completion is None:
                    self._prepare_results.put(results)
                else:
                    batch.results = results
                    batch.completion.set()
            except BaseException:
                logger.exception("External linker prepare batch failed")
                for item in batch.items:
                    if (
                        item.rid is not None
                        and item.operation == _PrepareOperation.ACQUIRE
                    ):
                        try:
                            self.cache_linker.release_read(item.rid)
                        except BaseException:
                            logger.exception(
                                "External linker failed-batch cleanup failed: rid=%s",
                                item.rid,
                            )
                # Complete through the same channel as the successful path so
                # neither async polling nor a blocking admission can hang.
                results = [
                    _PreparedRead(
                        request_id=item.request_id,
                        sequence=item.sequence,
                        signature=item.signature,
                        rid=item.rid,
                        page_hashes=item.page_hashes,
                        start_page=item.start_page,
                        num_pages=item.num_pages,
                        local_restorable=(),
                        common_pages=0,
                        swa_window_pages=0,
                        acquired=item.operation == _PrepareOperation.ACQUIRE,
                        read_held=False,
                        anchor_node=item.anchor_node,
                        anchor_lock_params=item.anchor_lock_params,
                    )
                    for item in batch.items
                ]
                if batch.completion is None:
                    self._prepare_results.put(results)
                else:
                    batch.results = results
                    batch.completion.set()
            finally:
                self._prepare_queue.task_done()

    def num_completed_prepares(self) -> int:
        return self._prepare_results.qsize()

    def drain_prepares(self, finish_count: int) -> None:
        for _ in range(finish_count):
            for prepared in self._prepare_results.get():
                pending = self._prepare_pending.get(prepared.request_id)
                if pending is not None and pending[0] == prepared.sequence:
                    self._prepare_pending.pop(prepared.request_id, None)
                released_key = (prepared.request_id, prepared.sequence)
                if (
                    pending is None
                    or pending[0] != prepared.sequence
                    or released_key in self._released_prepare_sequences
                    or prepared.request_id not in self._active_prepare_ids
                ):
                    self._discard_prepared(prepared)
                    self._released_prepare_sequences.discard(released_key)
                    continue
                old = self._prepared_reads.pop(prepared.request_id, None)
                if old is not None and old is not prepared:
                    self._discard_prepared(old)
                if prepared.common_pages == 0:
                    self._release_anchor(prepared)
                self._prepared_reads[prepared.request_id] = prepared

    def prepare_ready(self, req: Req) -> bool:
        request_id = self.request_id(req.rid)
        if request_id not in self._active_prepare_ids:
            return False
        key = self._key_for_req(req)
        page_hashes = self._full_page_hashes(key) if key is not None else ()
        if not page_hashes:
            return True
        if not self._pp_prefill_protocol:
            l1_match = self._match_l1(key)
            if (
                int(l1_match.device_indices.numel()) // self.cache.page_size
                >= len(page_hashes)
            ):
                return True
        if request_id in self._prepare_pending:
            return False
        prepared = self._prepared_reads.get(request_id)
        if (
            prepared is None
            or prepared.signature != hash_str_to_int64(page_hashes[-1])
            or prepared.start_page + prepared.num_pages > len(page_hashes)
        ):
            return False
        # ``None`` is the safe PP fallback for a request missing locally while
        # the canonical manifest was prepared: it is ready as an L1-only miss.
        matches = (
            prepared.page_hashes is None
            or prepared.page_hashes
            == page_hashes[
                prepared.start_page : prepared.start_page + prepared.num_pages
            ]
        )
        if not matches:
            return False
        return True

    def _acquire_for_admission(
        self, req: Req, prepared: _PreparedRead
    ) -> _PreparedRead:
        """Synchronously acquire H_lookup while preserving worker FIFO order."""
        assert self._pp_prefill_protocol
        (
            page_hashes,
            transfers,
            swa_window_pages,
            anchor_node,
            anchor_lock_params,
        ) = self._build_local_prepare_inputs(
            req,
            prepared.signature,
            prepared.start_page,
            prepared.common_pages,
            _PrepareOperation.ACQUIRE,
        )
        item = _PrepareItem(
            request_id=prepared.request_id,
            sequence=prepared.sequence,
            signature=prepared.signature,
            operation=_PrepareOperation.ACQUIRE,
            rid=req.rid,
            page_hashes=page_hashes,
            start_page=prepared.start_page,
            num_pages=prepared.common_pages,
            transfers=transfers,
            swa_window_pages=swa_window_pages,
            anchor_node=anchor_node,
            anchor_lock_params=anchor_lock_params,
        )
        completion = threading.Event()
        batch = _PrepareBatch([item], completion=completion)
        self._prepare_queue.put(batch)
        completion.wait()
        assert batch.results is not None and len(batch.results) == 1
        acquired = batch.results[0]

        old = self._prepared_reads.pop(prepared.request_id, None)
        if old is not None and old is not acquired:
            self._discard_prepared(old)
        self._prepared_reads[prepared.request_id] = acquired
        return acquired

    # ---- phase 2: admission match requests/consumes an acquired boundary ----

    def match(
        self,
        key: RadixKey,
        req: Req,
        result: MatchResult,
        *,
        prefill_admission: bool = False,
    ) -> MatchResult:
        request_id = self.request_id(req.rid)
        if self._pp_prefill_protocol and not prefill_admission:
            return result
        # An admission request may be rematched after a previous attempt. Never
        # let its older marker survive a later L1-only/miss result.
        self.hit_markers.pop(req.rid, None)
        prepared = self._prepared_reads.get(request_id)
        if prepared is None or prepared.common_pages <= 0:
            return result
        full_page_hashes = self._full_page_hashes(key)
        if not full_page_hashes:
            return result
        if (
            prepared.signature != hash_str_to_int64(full_page_hashes[-1])
            or prepared.page_hashes
            != full_page_hashes[
                prepared.start_page : prepared.start_page + prepared.num_pages
            ]
        ):
            return result

        if self._pp_prefill_protocol and not prepared.acquired:
            # Admission blocks on one acquire. Running it through the existing
            # worker serializes this collective after all earlier lookups.
            prepared = self._acquire_for_admission(req, prepared)
            if prepared.common_pages <= 0:
                return result

        page = self.cache.page_size
        prepared_start = prepared.start_page * page
        common_end = (prepared.start_page + prepared.common_pages) * page
        if self._pp_prefill_protocol:
            # H_session is the logical prefix boundary shared by every PP rank.
            # Rematch all ranks at that cap: a longer incidental L1 hit must not
            # consume past H, while a shorter one will load only its local gap.
            result = self._match_l1(key[:common_end])
        device_hit_len = int(result.device_indices.numel())
        if common_end <= device_hit_len:
            if prepared.read_held and prepared.rid is not None:
                try:
                    self.cache_linker.release_read(prepared.rid)
                except BaseException:
                    logger.exception(
                        "External linker covered-read cleanup failed: rid=%s",
                        prepared.rid,
                    )
                prepared.read_held = False
            self._release_anchor(prepared)
            self._prepared_reads.pop(request_id, None)
            self._rid_to_request_id.pop(req.rid, None)
            return result
        if not prepared.read_held:
            return result
        if device_hit_len % page or device_hit_len < prepared_start:
            return result

        first_page = device_hit_len // page - prepared.start_page
        tail_hashes = list(prepared.page_hashes[first_page : prepared.common_pages])
        if not tail_hashes:
            return result
        hit_tokens = common_end - device_hit_len
        swa_host_hit_length = min(
            len(tail_hashes), prepared.swa_window_pages
        ) * page
        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            prefix_key=key[:common_end],
            tail_hashes=tail_hashes,
            device_hit_len=device_hit_len,
        )
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            full_kv_hit_length=(
                common_end
                if self._pp_prefill_protocol
                else result.full_kv_hit_length
            ),
        )

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> device, then insert ----

    def load_back(self, req: Req) -> tuple[torch.Tensor, NodeId]:
        cache = self.cache
        empty_indices = cache.tree_core.empty_match_result.device_indices
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return empty_indices, req.last_node
        request_id = self.request_id(req.rid)
        prepared = self._prepared_reads.get(request_id)

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        # Build per-component linker transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, tail_hashes
            )
            if transfer is None:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT,
                    req,
                    component_transfers,
                    prefix_len,
                )
                if prepared is not None:
                    self._prepared_reads.pop(request_id, None)
                    self._rid_to_request_id.pop(req.rid, None)
                    self._discard_prepared(prepared)
                return empty_indices, req.last_node
            component_transfers.append((component, transfer))

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        self._update_load(
            ExternalLinkerLoadPhase.PREPARE,
            req,
            component_transfers,
            prefix_len,
        )

        # Insert the newly loaded tail into the tree.
        prefix_indices = torch.cat(
            [req.prefix_indices.to(torch.int64), full_transfer.device_indices]
        )
        mamba_transfer = next(
            (
                transfer
                for _, transfer in component_transfers
                if transfer.name == PoolName.MAMBA
            ),
            None,
        )
        insert_result = cache.insert(
            InsertParams(
                key=hit.prefix_key,
                value=prefix_indices,
                mamba_value=(
                    mamba_transfer.device_indices[:1]
                    if mamba_transfer is not None
                    else None
                ),
                prev_prefix_len=device_hit_len,
                swa_evicted_seqlen=(
                    req.kv.swa_evicted_seqlen if req.kv is not None else 0
                ),
                chunked=True,
                priority=getattr(req, "priority", 0) or 0,
                track_adopted_ranges=True,
            )
        )
        if mamba_transfer is not None and insert_result.mamba_exist:
            cache.req_to_token_pool.mamba_allocator.free(
                mamba_transfer.device_indices[:1]
            )

        canonical_tail = cache.tree_core.collect_full_device_indices(
            insert_result.last_device_node, req.last_node
        )
        assert canonical_tail.numel() == len(tail_hashes) * cache.page_size
        load_transfers = self._update_load(
            ExternalLinkerLoadPhase.COMMIT,
            req,
            component_transfers,
            prefix_len,
            insert_result=insert_result,
            canonical_full=canonical_tail,
        )

        prepared = self._prepared_reads.pop(request_id, prepared)
        self._rid_to_request_id.pop(req.rid, None)
        try:
            self._queue_load(req.rid, insert_result.last_device_node, load_transfers)
        except BaseException:
            if prepared is not None:
                self._discard_prepared(prepared)
            raise
        if prepared is not None:
            # The inserted/load endpoint now has its own in-flight lock. The
            # original L1 anchor only protected the prepare-to-admission gap.
            self._release_anchor(prepared)

        node = cache.resolve_node_handle(insert_result.last_device_node)
        while node.id != req.last_node:
            node.external_cache_stored = True
            node = node.parent
        return canonical_tail, insert_result.last_device_node

    def _queue_load(
        self, rid: str, node_id: NodeId, transfers: list[PoolTransfer]
    ) -> None:
        if not transfers:
            self.cache_linker.release_read(rid)
            return
        assert rid not in self.pending_loads
        lock_params = self.cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.load(rid, transfers)
        except BaseException:
            self.cache.dec_lock_ref(node_id, lock_params)
            self.cache_linker.release_read(rid)
            raise
        if not queued:
            self.cache.dec_lock_ref(node_id, lock_params)
            self.cache_linker.release_read(rid)
            raise RuntimeError(f"Failed to queue the linker load for rid={rid!r}.")
        self.pending_loads[rid] = (node_id, lock_params)

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        insert_result=None,
        canonical_full: torch.Tensor | None = None,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            component_canonical = canonical_full
            if phase == ExternalLinkerLoadPhase.COMMIT:
                assert insert_result.adopted_ranges is not None
                coverage_start = prefix_len - len(transfer.device_indices)
                ranges = [
                    (max(start, coverage_start), min(end, prefix_len))
                    for start, end in insert_result.adopted_ranges.get(
                        component.component_type, ()
                    )
                    if max(start, coverage_start) < min(end, prefix_len)
                ]
                indices, keys = self._select_adopted_pages(
                    transfer.device_indices,
                    ranges,
                    prefix_len,
                    transfer.keys,
                )
                if not keys:
                    continue
                transfer.device_indices = indices
                transfer.keys = keys
                component_canonical, _ = self._select_adopted_pages(
                    canonical_full, ranges, prefix_len
                )
            transfer = component.update_external_linker_load(
                phase,
                req,
                full,
                transfer,
                prefix_len,
                insert_result=insert_result,
                canonical_full=component_canonical,
            )
            if transfer is not None:
                result.append(transfer)
        return result

    def _select_adopted_pages(
        self,
        indices: torch.Tensor,
        ranges: Sequence[tuple[int, int]],
        prefix_len: int,
        keys: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        page = self.cache.page_size
        coverage_start = prefix_len - len(indices)
        pages = indices.reshape(-1, page)
        if keys is not None:
            assert len(keys) == len(pages)

        chunks = []
        selected_keys = []
        for start, end in ranges:
            start = max(start, coverage_start)
            end = min(end, prefix_len)
            if start >= end:
                continue
            assert (start - coverage_start) % page == 0
            assert (end - coverage_start) % page == 0
            first = (start - coverage_start) // page
            last = (end - coverage_start) // page
            chunks.append(pages[first:last].reshape(-1))
            if keys is not None:
                selected_keys.extend(keys[first:last])

        if not chunks:
            return indices[:0], selected_keys
        selected = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return selected, selected_keys

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            if not self.cache.resolve_node_handle(node_id).external_cache_stored:
                self._offload_node(node_id)

    def _offload_node(self, node_id: NodeId) -> None:
        cache = self.cache
        node = cache.resolve_node_handle(node_id)
        transfers = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.OFFLOAD, node, None
            )
            if transfer is not None:
                transfers.append(transfer)

        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        node.external_cache_stored = True
        self.pending_offloads.append((node_id, lock_params))

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def num_completed_loads(self) -> int:
        return self.cache_linker.num_completed_loads()

    def drain_loads(self, finish_count: int) -> None:
        for _ in range(finish_count):
            for rid in self.cache_linker.pop_completed_load():
                node_id, lock_params = self.pending_loads.pop(rid)
                self.cache.dec_lock_ref(node_id, lock_params)

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            node_id, lock_params = self.pending_offloads.pop(0)
            node = self.cache.resolve_node_handle(node_id)
            node.external_cache_stored = success
            self.cache.dec_lock_ref(node_id, lock_params)

    def start_layer_wise_loading(self) -> int:
        return self.cache_linker.start_layer_wise_loading()

    # ---- lifecycle ----

    def _release_anchor(self, prepared: _PreparedRead) -> None:
        if prepared.anchor_node is None:
            return
        try:
            self.cache.dec_lock_ref(
                prepared.anchor_node, prepared.anchor_lock_params
            )
        except BaseException:
            logger.exception(
                "External linker anchor cleanup failed: rid=%s", prepared.rid
            )
        finally:
            prepared.anchor_node = None
            prepared.anchor_lock_params = None

    def _discard_prepared(self, prepared: _PreparedRead) -> None:
        if prepared.rid is not None:
            self.hit_markers.pop(prepared.rid, None)
        try:
            if prepared.read_held and prepared.rid is not None:
                try:
                    self.cache_linker.release_read(prepared.rid)
                except BaseException:
                    logger.exception(
                        "External linker prepared-read cleanup failed: rid=%s",
                        prepared.rid,
                    )
                prepared.read_held = False
        finally:
            self._release_anchor(prepared)

    def reset(self) -> None:
        for request_id, pending in self._prepare_pending.items():
            self._released_prepare_sequences.add((request_id, pending[0]))
        self._prepare_queue.join()
        while True:
            try:
                results = self._prepare_results.get_nowait()
            except Empty:
                break
            for prepared in results:
                self._discard_prepared(prepared)
        for prepared in self._prepared_reads.values():
            self._discard_prepared(prepared)
        self._prepare_pending.clear()
        self._prepared_reads.clear()
        self._rid_to_request_id.clear()
        self._released_prepare_sequences.clear()
        self._active_prepare_ids.clear()
        self.cache_linker.reset()
        self.hit_markers.clear()
        for node_id, lock_params in self.pending_loads.values():
            self.cache.dec_lock_ref(node_id, lock_params)
        self.pending_loads.clear()
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self.hit_markers.pop(rid, None)
        request_id = getattr(self, "_rid_to_request_id", {}).pop(
            rid, self.request_id(rid)
        )
        pending = getattr(self, "_prepare_pending", {}).get(request_id)
        if pending is not None:
            getattr(self, "_released_prepare_sequences", set()).add(
                (request_id, pending[0])
            )
        getattr(self, "_active_prepare_ids", set()).discard(request_id)
        prepared = getattr(self, "_prepared_reads", {}).pop(request_id, None)
        if prepared is not None:
            self._discard_prepared(prepared)
        if self.cache_linker.cancel_queued_load(rid):
            node_id, lock_params = self.pending_loads.pop(rid)
            self.cache.dec_lock_ref(node_id, lock_params)

    def release_unadmitted(self, rid: str) -> None:
        """Release an acquired ticket when admission rejects before load-back."""
        if self.hit_markers.pop(rid, None) is None:
            return
        request_id = self._rid_to_request_id.pop(rid, self.request_id(rid))
        prepared = self._prepared_reads.pop(request_id, None)
        if prepared is not None:
            self._discard_prepared(prepared)

    def close(self) -> None:
        self.reset()
        self._prepare_queue.put(None)
        self._prepare_thread.join()
        self._destroy_prepare_sync_groups()
        self.cache_linker.close()
