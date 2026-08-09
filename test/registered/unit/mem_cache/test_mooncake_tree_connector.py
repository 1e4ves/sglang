import threading
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_mappings import (
    DevicePoolEntry,
    DevicePoolGroup,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_tree_connector
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_tree_connector import (
    MooncakeTreeConnector,
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
    pool = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=[[k0, k2], [v0, v2]],
        layer_mapping={0: 0, 2: 1},
        page_size=2,
        rows_are_pages=False,
        packed=False,
    )

    indices = torch.tensor([0, 1, 4, 5])
    locations = pool.prepare_locations(indices)
    assert locations == [0, 4]
    pointers, sizes = pool.get_page_buffer_meta(indices)
    assert pointers == [
        buffer[row].data_ptr() for row in locations for buffer in (k0, k2, v0, v2)
    ]
    assert sizes == [6, 10, 14, 22] * 2
    assert pool.get_prepared_layer_range_meta(locations, 1) is None

    pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
    assert pointers == [
        [k2[0].data_ptr()],
        [v2[0].data_ptr()],
        [k2[4].data_ptr()],
        [v2[4].data_ptr()],
    ]
    assert sizes == [[10], [22], [10], [22]]
    assert offsets == [[6], [14], [6], [14]]


def test_lookup_returns_sparse_mamba_boundaries():
    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    pools = [
        SimpleNamespace(
            name=name,
            indices_from_pool=name,
            translate_indices=lambda indices: indices,
        )
        for name in (PoolName.KV, PoolName.MAMBA)
    ]
    connector.pool_group = DevicePoolGroup(pools, num_layers=4, page_size=1)
    connector.pools = connector.pool_group.entry_map
    connector.stats = {"lookup": 0}
    connector.storage = MooncakeStore.__new__(MooncakeStore)
    connector.storage.mem_pool_host = SimpleNamespace(kv_buffer=None)
    connector.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
        [f"{key}_{transfer.name}" for key in keys],
        1,
    )
    connector.storage._tag_keys = lambda keys: keys
    connector.storage._batch_exist = lambda keys: [
        int(key.endswith("_kv") or key[0] in ("b", "d")) for key in keys
    ]

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


def test_load_and_offload_share_gc_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(mooncake_tree_connector, "freeze_gc", calls.append)
    monkeypatch.setattr(
        mooncake_tree_connector.device_module,
        "Event",
        lambda: SimpleNamespace(record=lambda: None),
    )

    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    connector.page_size = 2
    connector.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=2)
    connector.gc_frozen = False
    connector.offload_queue = Queue()
    connector.pending_loads = {"first": [object()]}
    connector.layer_done_counter = SimpleNamespace(update_producer=lambda: 3)
    connector.load_queue = Queue()
    connector.stats = {"load": 0}

    assert connector.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert connector.start_layer_wise_loading() == 3
    connector.pending_loads = {"second": [object()]}
    assert connector.start_layer_wise_loading() == 3

    assert calls == ["Mooncake connector"]
    assert connector.stats["load"] == 2


def test_prefetch_loads_before_returning():
    worker_threads = []

    class _Storage:
        def batch_get_v2(self, transfers):
            worker_threads.append(threading.get_ident())
            return {
                transfer.name: [True] * len(transfer.keys) for transfer in transfers
            }

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    connector.page_size = 2
    connector.pool_group = DevicePoolGroup([pool], num_layers=3, page_size=2)
    connector.storage = _Storage()
    connector.load_strategy = "prefetch"
    connector.gc_frozen = True
    connector.pending_loads = {}
    connector.stats = {"load": 0}
    connector.load_queue = Queue()
    connector.load_thread = threading.Thread(
        target=connector.load_thread_func, daemon=True
    )
    connector.load_thread.start()

    assert connector.load(
        "rid",
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["a"],
                device_indices=torch.tensor([0, 1]),
            )
        ],
    )
    assert worker_threads == [connector.load_thread.ident]
    assert connector.pending_loads == {}
    assert connector.stats["load"] == 1

    connector.load_queue.put(None)
    connector.load_thread.join(timeout=5)


def test_offload_runs_on_background_thread(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()
    worker_threads = []
    event_calls = []

    monkeypatch.setattr(mooncake_tree_connector, "freeze_gc", lambda _: None)

    class _Event:
        def record(self):
            event_calls.append("record")

        def synchronize(self):
            event_calls.append("synchronize")

    monkeypatch.setattr(mooncake_tree_connector.device_module, "Event", _Event)

    class _Storage:
        def batch_set_v2(self, transfers):
            assert event_calls == ["record", "synchronize"]
            worker_threads.append(threading.get_ident())
            started.set()
            assert release.wait(timeout=5)
            return {
                transfer.name: [True] * len(transfer.keys) for transfer in transfers
            }

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
        get_hybrid_pool_buffer=lambda: [],
    )
    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    connector.page_size = 2
    connector.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=2)
    connector.pools = connector.pool_group.entry_map
    connector.storage = _Storage()
    connector.gc_frozen = False
    connector.stats = {"lookup": 0, "load": 0, "offload": 0}
    connector.offload_queue = Queue()
    connector.offload_results = Queue()
    connector.offload_thread = threading.Thread(
        target=connector.offload_thread_func, daemon=True
    )
    connector.offload_thread.start()

    assert connector.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert started.wait(timeout=5)
    assert connector.num_completed_offloads() == 0
    assert worker_threads == [connector.offload_thread.ident]
    assert worker_threads[0] != caller_thread

    release.set()
    connector.offload_queue.join()
    assert connector.num_completed_offloads() == 1
    assert connector.pop_completed_offload()
    connector.offload_queue.put(None)
    connector.offload_thread.join(timeout=5)


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_connector_transfer(self, phase, node=None):
            assert phase == ConnectorTransferPhase.OFFLOAD
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    results = []
    connector = SimpleNamespace(
        offload=lambda transfers: True,
        num_completed_offloads=lambda: len(results),
        pop_completed_offload=lambda: results.pop(0),
    )
    mixin = UnifiedCacheConnectorMixin()
    mixin.connector = connector
    mixin._components_tuple = (_Component(),)
    mixin.connector_offloads = []
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    mixin.inc_lock_ref = inc_lock_ref
    mixin.dec_lock_ref = lambda node, params: unlocks.append((node, params))
    node = SimpleNamespace(connector_offloaded=False)

    mixin.offload_connector_node(node)
    assert locks == [node]
    assert node.connector_offloaded
    assert not unlocks

    results.append(False)
    mixin.drain_connector_offloads()
    assert not node.connector_offloaded
    assert unlocks == [(node, lock_params)]


def test_deepseek_v4_device_pool_group_maps_sparse_sidecars():
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
        kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)]
    )
    kvcache.c4_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)]
    )
    kvcache.c4_indexer_kv_pool = SimpleNamespace(
        index_k_with_scale_buffer=[
            torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
        ]
    )
    kvcache.c128_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)]
    )
    kvcache.layer_mapping = [
        DeepSeekV4LayerItem(4, 0),
        DeepSeekV4LayerItem(128, 0),
        DeepSeekV4LayerItem(4, 1),
    ]
    kvcache.compress_state_pools = [state_pool(), None, state_pool()]
    kvcache.indexer_compress_state_pools = [state_pool(), None, state_pool()]

    group = resolve_hybrid_device_pool_group(kvcache, 2, None)
    assert group.num_layers == 3
    assert set(group.entry_map) == {
        PoolName.SWA,
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    }
    assert group.sources[PoolName.DEEPSEEK_V4_C4] == PoolName.KV
    assert group.sources[PoolName.DEEPSEEK_V4_C4_STATE] == PoolName.SWA
    c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
    pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 2
    assert sizes == [5, 5]
    _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
    assert sizes == [[5]]
    assert offsets == [[5]]
    assert c4_pool.get_prepared_layer_range_meta([0], 1) is None


def test_qwen35_device_pool_group_maps_full_and_mamba_layers():
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

    group = resolve_hybrid_device_pool_group(kvcache, 2, req_pool)
    pools = group.entry_map
    assert group.num_layers == 4
    assert set(pools) == {PoolName.KV, PoolName.MAMBA}
    assert group.sources == {
        PoolName.KV: PoolName.KV,
        PoolName.MAMBA: PoolName.MAMBA,
    }
    assert pools[PoolName.MAMBA].translate_indices(torch.tensor([1])).tolist() == [1]
    assert pools[PoolName.KV].get_prepared_layer_range_meta([0], 1) is None
    assert pools[PoolName.MAMBA].get_prepared_layer_range_meta([0], 0) is None
    pointers, sizes = pools[PoolName.KV].get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 4
    assert sizes == [24, 40, 56, 88]


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


def test_prefetch_load_is_not_requeued_after_insert():
    mixin = UnifiedCacheConnectorMixin()

    def fail(*args):
        raise AssertionError("prefetch must not be loaded twice")

    mixin.connector = SimpleNamespace(
        load_strategy="prefetch",
        cancel_queued_load=fail,
        load=fail,
    )
    transfer = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([100, 101]))
    loaded = SimpleNamespace(device_indices=torch.tensor([1, 2, 10, 11]))

    returned = mixin._retarget_connector_load(
        "rid", [transfer], loaded, device_hit_len=2
    )

    assert returned.tolist() == [10, 11]
    assert transfer.device_indices.tolist() == [100, 101]
