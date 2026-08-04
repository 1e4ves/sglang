from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_tree_connector import (
    MooncakeTreeConnector,
    _build_dsv4_pools,
    _build_hybrid_linear_pools,
    _MappedRowsPool,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ConnectorTransferPhase,
)
from sglang.srt.mem_cache.unified_cache_connector_mixin import (
    UnifiedCacheConnectorMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, slots=None):
        self.slots = slots
        self.freed = []
        self.mapping = []

    def available_size(self):
        return 100

    def alloc(self, size):
        if self.slots is None:
            return torch.arange(1, size + 1, dtype=torch.int64)
        value = self.slots[:size].clone()
        self.slots = self.slots[size:]
        return value

    def free(self, value):
        self.freed.append(value.clone())

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def test_sparse_multi_component_layer_ranges():
    k0 = torch.zeros((8, 3), dtype=torch.uint8)
    k2 = torch.zeros((8, 5), dtype=torch.uint8)
    v0 = torch.zeros((8, 7), dtype=torch.uint8)
    v2 = torch.zeros((8, 11), dtype=torch.uint8)
    pool = _MappedRowsPool(
        [[k0, None, k2], [v0, None, v2]],
        page_size=2,
        rows_are_pages=False,
    )

    indices = torch.tensor([0, 1, 4, 5])
    locations = pool.prepare_locations(indices)
    assert locations == [0, 4]
    pointers, sizes = pool.get_page_buffer_meta(indices)
    assert len(pointers) == 8
    assert sizes == [6, 10, 14, 22] * 2
    assert pool.get_prepared_layer_range_meta(locations, 1) is None

    pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
    assert len(pointers) == 4
    assert sizes == [[10], [22], [10], [22]]
    assert offsets == [[6], [14], [6], [14]]


def test_lookup_returns_sparse_mamba_boundaries():
    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    connector.sources = {
        PoolName.KV: PoolName.KV,
        PoolName.MAMBA: PoolName.MAMBA,
    }
    identity_pool = SimpleNamespace(translate_indices=lambda indices: indices)
    connector.pools = {
        PoolName.KV: identity_pool,
        PoolName.MAMBA: identity_pool,
    }
    connector._stats = {"lookup": 0}
    connector._page_exists = lambda keys, transfer: (
        [True, True, True, True]
        if transfer.name == PoolName.KV
        else [False, True, False, True]
    )

    valid = connector.lookup(
        "rid",
        [
            PoolTransfer(name=PoolName.KV, keys=["a", "b", "c", "d"]),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["d"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ],
    )
    assert valid == [2, 4]


def test_deepseek_v4_builder_maps_sparse_sidecars():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4LayerItem,
        DeepSeekV4TokenToKVPool,
    )

    def state_pool():
        return SimpleNamespace(
            ring_size=2,
            kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((8, 3))),
        )

    kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    kvcache._unified_kv = False
    kvcache.start_layer = 0
    kvcache.end_layer = 3
    kvcache.swa_page_size = 2
    kvcache.swa_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 3)) for _ in range(3)]
    )
    kvcache.c4_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 5)) for _ in range(2)]
    )
    kvcache.c4_indexer_kv_pool = SimpleNamespace(
        index_k_with_scale_buffer=[torch.zeros((8, 7)) for _ in range(2)]
    )
    kvcache.c128_kv_pool = SimpleNamespace(kv_buffer=[torch.zeros((8, 11))])
    kvcache.layer_mapping = [
        DeepSeekV4LayerItem(4, 0),
        DeepSeekV4LayerItem(128, 0),
        DeepSeekV4LayerItem(4, 1),
    ]
    kvcache.compress_state_pools = [state_pool(), None, state_pool()]
    kvcache.indexer_compress_state_pools = [state_pool(), None, state_pool()]

    _, pools, sources, num_layers = _build_dsv4_pools(kvcache, 2)
    assert num_layers == 3
    assert set(pools) == {PoolName.KV, PoolName.SWA}
    assert sources == {PoolName.KV: PoolName.KV, PoolName.SWA: PoolName.SWA}
    _, sizes, _ = pools[PoolName.KV].get_prepared_layer_range_meta([0], 0)
    assert len(sizes[0]) == 2
    _, sizes, _ = pools[PoolName.KV].get_prepared_layer_range_meta([0], 1)
    assert len(sizes[0]) == 1


def test_qwen35_builder_maps_full_and_mamba_layers():
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
    kvcache.use_mla = False
    kvcache.full_attention_layer_id_mapping = {0: 0, 2: 1}
    kvcache.full_kv_pool = SimpleNamespace(
        size=6,
        k_scale_buffer=None,
        k_buffer=[torch.zeros((8, 3)), torch.zeros((8, 5))],
        v_buffer=[torch.zeros((8, 7)), torch.zeros((8, 11))],
    )
    req_pool = SimpleNamespace(
        mamba_ckpt_pool=None,
        mamba_map={1: 0, 3: 1},
        mamba_pool=SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.zeros((2, 5, 2, 3)),
                conv=[torch.zeros((2, 5, 4))],
            )
        ),
        translate_mamba_indices=lambda indices: indices,
    )

    _, pools, sources, num_layers = _build_hybrid_linear_pools(kvcache, 2, req_pool)
    assert num_layers == 4
    assert set(pools) == {PoolName.KV, PoolName.MAMBA}
    assert sources == {PoolName.KV: PoolName.KV, PoolName.MAMBA: PoolName.MAMBA}
    assert pools[PoolName.MAMBA].translate_indices(torch.tensor([1])).tolist() == [1]
    assert pools[PoolName.KV].get_prepared_layer_range_meta([0], 1) is None
    assert pools[PoolName.MAMBA].get_prepared_layer_range_meta([0], 0) is None
    pointers, sizes = pools[PoolName.KV].get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 1
    assert len(sizes[0]) == 4


def test_swa_connector_finish_maps_or_releases_slots():
    swa_allocator = _Allocator()
    allocator = SimpleNamespace(
        swa_attn_allocator=swa_allocator,
        set_full_to_swa_mapping=swa_allocator.set_full_to_swa_mapping,
    )
    component = SWAComponent.__new__(SWAComponent)
    component.cache = SimpleNamespace(
        page_size=64,
        token_to_kv_pool_allocator=allocator,
    )
    component.sliding_window_size = 128
    req = SimpleNamespace(kv=SimpleNamespace(swa_evicted_seqlen=0))
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1, 2, 3, 4]))
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([20, 21]))

    component.finish_connector_load(req, full, swa, prefix_len=256, success=True)
    mapped_full, mapped_swa = swa_allocator.mapping[0]
    assert mapped_full.tolist() == [3, 4]
    assert mapped_swa.tolist() == [20, 21]
    assert req.kv.swa_evicted_seqlen == 128

    component.finish_connector_load(req, full, swa, prefix_len=256, success=False)
    assert swa_allocator.freed[0].tolist() == [20, 21]


def test_mamba_connector_load_allocates_cache_and_request_slots():
    allocator = _Allocator(slots=torch.tensor([7, 8]))
    req_pool = SimpleNamespace(mamba_allocator=allocator, mamba_ckpt_pool=None)
    component = MambaComponent.__new__(MambaComponent)
    component.cache = SimpleNamespace(
        req_to_token_pool=req_pool,
        evict=lambda params: None,
    )

    transfer = component.build_connector_transfer(
        phase=ConnectorTransferPhase.LOAD,
        keys=["a", "b"],
    )
    assert transfer.keys == ["b", "b"]
    assert transfer.device_indices.tolist() == [7, 8]

    req = SimpleNamespace(
        mamba_pool_idx=None,
        mamba_cow_src_index=torch.tensor([99]),
        mamba_needs_clear=True,
    )
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1]))
    component.finish_connector_load(req, full, transfer, prefix_len=2, success=True)
    assert req.mamba_pool_idx.item() == 8
    assert req.mamba_cow_src_index is None
    assert not req.mamba_needs_clear

    failed = PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([9, 10]))
    component.finish_connector_load(req, full, failed, prefix_len=2, success=False)
    assert allocator.freed[-1].tolist() == [9, 10]


def test_overlapping_load_retargets_freed_slots_to_tree_values():
    mixin = UnifiedCacheConnectorMixin()
    mixin.token_to_kv_pool_allocator = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda indices: indices + 1000
    )
    queued = {"second": ["stale"]}

    def load(rid, transfers):
        queued[rid] = list(transfers)
        return True

    mixin.connector = SimpleNamespace(
        cancel_queued_load=lambda rid: queued.pop(rid),
        load=load,
    )

    full = PoolTransfer(
        name=PoolName.KV, device_indices=torch.tensor([100, 101, 102, 103])
    )
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([200, 201]))
    mamba = PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([300, 301]))
    canonical_full = torch.tensor([10, 11, 12, 13])
    loaded = SimpleNamespace(
        device_indices=torch.cat([torch.tensor([1, 2]), canonical_full]),
        last_device_node=SimpleNamespace(
            component_data={
                ComponentType.MAMBA: SimpleNamespace(value=torch.tensor([30]))
            }
        ),
    )

    returned = mixin._retarget_connector_load(
        "second",
        [full, swa, mamba],
        loaded,
        device_hit_len=2,
    )

    assert returned.tolist() == canonical_full.tolist()
    assert full.device_indices.tolist() == canonical_full.tolist()
    assert swa.device_indices.tolist() == [1012, 1013]
    assert mamba.device_indices.tolist() == [30, 301]
    assert queued["second"] == [full, swa, mamba]
