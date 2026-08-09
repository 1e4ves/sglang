"""External-KV connector support for :class:`UnifiedRadixCache`.

Self-contained: this module owns both halves of the contract.

* :class:`UnifiedTreeConnector` -- the transport interface a backend implements.
* :class:`UnifiedCacheConnectorMixin` -- the tree-side flow that drives it,
  keeping the whole connector path out of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheConnectorMixin.match_connector`
* ``init_load_back``    -> :meth:`UnifiedCacheConnectorMixin.load_connector`
* ``_inc_hit_count``    -> :meth:`UnifiedCacheConnectorMixin.offload_connector_node`

Division of labour: the connector owns each rank's transport, including an
optional layer-wise background load. Everything cross-rank (hit-set
intersection, IO outcome agreement) is decided on the tree side, here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import (
    ComponentType,
    ConnectorTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
    from sglang.srt.server_args import ServerArgs


class UnifiedTreeConnector(ABC):
    """Remote transport for tree-built pool transfers."""

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

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Prepare a load into device indices.

        A layer-wise implementation queues the transfer for the next
        ``start_layer_wise_loading`` call. Other implementations may complete
        the load synchronously.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    def cancel_queued_load(self, rid: str) -> None:
        """Remove a queued request before layer-wise loading starts."""

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


class ConnectorMarker(NamedTuple):
    """State carried from a connector-hit ``match_prefix`` to ``init_load_back``.

    ``keys`` are tail-relative: they start at the first device-uncached page,
    anchored at ``device_hit_len``.
    """

    key: RadixKey
    keys: list[str]
    device_hit_len: int


class UnifiedCacheConnectorMixin:
    """Connector-driven match / load / offload for the unified radix tree."""

    def init_connector(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        supported = {
            (ComponentType.FULL,),
            (ComponentType.FULL, ComponentType.SWA),
            (ComponentType.FULL, ComponentType.MAMBA),
        }
        if self.tree_components not in supported:
            raise ValueError(
                "Unified tree connector supports FULL, FULL+SWA, and FULL+MAMBA."
            )
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_tree_connector import (
            MooncakeTreeConnector,
        )

        self.connector = MooncakeTreeConnector(server_args, params)
        self.write_through_threshold = 1

    # ---- match: probe the remote store and report host_hit_length ----

    def match_connector(
        self, key: RadixKey, req: Req, result: MatchResult
    ) -> MatchResult:
        page = self.page_size
        device_hit_len = int(result.device_indices.numel())
        if device_hit_len >= len(key):
            return result

        keys = self._connector_tail_keys(key, result, device_hit_len)
        if not keys:
            return result

        transfers = []
        for component in self._components_tuple:
            transfer = component.build_connector_transfer(
                ConnectorTransferPhase.LOOKUP,
                keys=keys,
            )
            if transfer is None:
                return result
            transfers.append(transfer)
        by_pool = {transfer.name: transfer for transfer in transfers}

        # Tail-relative: page 0 of `keys` is the first uncached page.
        hit_pages = self._sync_connector_hit_pages(
            self.connector.lookup(req.rid, transfers),
            num_pages=len(keys),
            device_hit_pages=0,
        )
        if hit_pages == 0:
            return result
        hit_tokens = hit_pages * page

        swa_transfer = by_pool.get(PoolName.SWA)
        swa_host_hit_length = (
            min(len(swa_transfer.keys), hit_pages) * page
            if swa_transfer is not None
            else 0
        )
        mamba_host_hit_length = int(PoolName.MAMBA in by_pool)

        self._connector_markers[req.rid] = ConnectorMarker(
            key=key[: device_hit_len + hit_tokens],
            keys=list(keys[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _sync_connector_hit_pages(
        self, valid_pages: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix."""
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        for pages in valid_pages:
            if device_hit_pages < pages <= num_pages:
                mask[pages] = 1
        self._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        return int(common[-1].item())

    def _connector_tail_keys(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            node = result.last_device_node
            last_hash = node.get_last_hash_value() if node is not None else None
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        return get_hash_str(
            key.token_ids[device_hit_len:], last_hash, page_size=self.page_size
        )

    # ---- init_load_back: remote -> device, then insert ----

    def load_connector(self, req: Req) -> tuple[torch.Tensor, UnifiedTreeNode]:
        empty = self._empty_match_result.device_indices
        marker = self._connector_markers.pop(req.rid, None)
        if marker is None:
            return empty, req.last_node

        device_hit_len = marker.device_hit_len
        tail_keys = marker.keys
        num_tokens = len(tail_keys) * self.page_size

        # Build per-component connector transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in self._components_tuple:
            transfer = component.build_connector_transfer(
                ConnectorTransferPhase.LOAD,
                keys=tail_keys,
            )
            if transfer is None:
                break
            component_transfers.append((component, transfer))

        prefix_len = device_hit_len + num_tokens
        if len(component_transfers) != len(self._components_tuple):
            if component_transfers:
                full = component_transfers[0][1]
                for component, transfer in component_transfers:
                    component.finish_connector_load(
                        req, full, transfer, prefix_len, False
                    )
            return empty, req.last_node

        full = component_transfers[0][1]
        assert full.name == PoolName.KV

        transfers = [transfer for _, transfer in component_transfers]
        local_success = self.connector.load(req.rid, transfers)
        success = self._connector_sync_success(local_success)
        if local_success and not success:
            self.connector.cancel_queued_load(req.rid)
        for component, transfer in component_transfers:
            component.finish_connector_load(req, full, transfer, prefix_len, success)
        if not success:
            return empty, req.last_node

        # Insert the newly loaded tail into the tree.
        values = torch.cat([req.prefix_indices.to(torch.int64), full.device_indices])
        mamba_transfer = next(
            (
                transfer
                for _, transfer in component_transfers
                if transfer.name == PoolName.MAMBA
            ),
            None,
        )
        insert_result = self.insert(
            InsertParams(
                key=marker.key,
                value=values,
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
            )
        )
        if mamba_transfer is not None and insert_result.mamba_exist:
            self.req_to_token_pool.mamba_allocator.free(
                mamba_transfer.device_indices[:1]
            )

        # Rematch
        loaded = self.match_prefix(MatchPrefixParams(key=marker.key))
        canonical_full_indices = self._retarget_connector_load(
            req.rid,
            transfers,
            loaded,
            device_hit_len,
        )
        node = loaded.last_device_node
        while node is not req.last_node:
            node.connector_offloaded = True
            node = node.parent
        return canonical_full_indices, loaded.last_device_node

    def _retarget_connector_load(
        self,
        rid: str,
        transfers: list[PoolTransfer],
        loaded: MatchResult,
        device_hit_len: int,
    ) -> torch.Tensor:
        """Retarget a queued load to the slots retained by ``insert``."""
        canonical_full = loaded.device_indices[device_hit_len:]
        # Prefetch already populated L1 before insertion. Only a queued
        # layer-wise load must follow slots that insert retained from the tree.
        if getattr(self.connector, "load_strategy", "layer_wise") != "layer_wise":
            return canonical_full
        for transfer in transfers:
            if transfer.name == PoolName.KV:
                transfer.device_indices = canonical_full
            elif transfer.name == PoolName.SWA:
                swa_len = transfer.device_indices.numel()
                transfer.device_indices = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        canonical_full[-swa_len:]
                    ).to(torch.int64)
                )
            else:
                assert transfer.name == PoolName.MAMBA
                canonical_mamba = loaded.last_device_node.component_data[
                    ComponentType.MAMBA
                ].value
                assert canonical_mamba is not None
                transfer.device_indices = torch.cat(
                    [
                        canonical_mamba.to(transfer.device_indices),
                        transfer.device_indices[1:],
                    ]
                )

        self.connector.cancel_queued_load(rid)
        if not self.connector.load(rid, transfers):
            raise RuntimeError(
                f"Failed to requeue canonical connector load for {rid=}."
            )
        return canonical_full

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_connector_node(self, node: UnifiedTreeNode) -> None:
        transfers = []
        for component in self._components_tuple:
            transfer = component.build_connector_transfer(
                ConnectorTransferPhase.OFFLOAD, node=node
            )
            if transfer is not None:
                transfers.append(transfer)

        lock_params = self.inc_lock_ref(node).to_dec_params()
        try:
            queued = self.connector.offload(transfers)
        except BaseException:
            self.dec_lock_ref(node, lock_params)
            raise
        if not queued:
            self.dec_lock_ref(node, lock_params)
            return

        node.connector_offloaded = True
        self.connector_offloads.append((node, lock_params))

    def drain_connector_offloads(self) -> None:
        if self.connector is None:
            return
        count = min(
            self.connector.num_completed_offloads(), len(self.connector_offloads)
        )
        for _ in range(count):
            node, lock_params = self.connector_offloads.pop(0)
            node.connector_offloaded = self.connector.pop_completed_offload()
            self.dec_lock_ref(node, lock_params)

    def _connector_sync_success(self, success: bool) -> bool:
        """MIN-reduce a per-rank IO outcome so every rank takes the same branch."""
        synced = torch.tensor([int(success)], dtype=torch.int)
        self._all_reduce_attn_groups(synced, torch.distributed.ReduceOp.MIN)
        return bool(synced.item())

    # ---- lifecycle helpers used by the tree's own hooks ----

    def reset_connector_state(self) -> None:
        self._connector_markers: dict[str, ConnectorMarker] = {}
        self.connector_offloads = []

    def release_connector_request(self, rid: str) -> None:
        if self.connector is not None:
            self._connector_markers.pop(rid, None)

    def close_connector(self) -> None:
        if self.connector is not None:
            self.connector.close()

    if TYPE_CHECKING:
        # Provided by UnifiedRadixCache; declared for type checkers only.
        connector: Optional[UnifiedTreeConnector]
