"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _split_hicache_size,
    build_hicache_draft_sidecars,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Pool:
    def __init__(self, kv_bytes):
        self._kv_bytes = kv_bytes

    def get_kv_size_bytes(self):
        return self._kv_bytes


class TestSplitHicacheSize(CustomTestCase):
    def test_splits_total_budget_by_device_bytes(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(75 * 10**9), _Pool((15 * 10**9, 10 * 10**9)))
        )
        self.assertEqual(shares, (75.0, 25.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled


class TestDraftSidecarPoolDispatch(CustomTestCase):
    def test_full_builder_unwraps_hybrid_linear_pool(self):
        full_kv_pool = SimpleNamespace(layer_num=1, size=8)
        draft_kv_pool = object.__new__(HybridLinearKVPool)
        draft_kv_pool.full_kv_pool = full_kv_pool
        host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=16),
                page_size=2,
            )
        )
        server_args = SimpleNamespace(hicache_mem_layout="layer_first")

        with (
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_build_mha_mla_host_pool",
                return_value=host_pool,
            ) as build_host_pool,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_get_allocator_type",
                return_value="allocator",
            ),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "build_pool_entry",
                return_value="entry",
            ) as build_entry,
        ):
            specs, entries = build_hicache_draft_sidecars(
                draft_device_pools=(draft_kv_pool,),
                tree_cache=tree_cache,
                server_args=server_args,
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].pool_name, PoolName.DRAFT)
        self.assertEqual(entries, ["entry"])
        build_host_pool.assert_called_once_with(
            pool=full_kv_pool,
            host_to_device_ratio=2,
            page_size=2,
            layout="layer_first",
            allocator_type="allocator",
            pool_label="draft",
        )
        build_entry.assert_called_once_with(
            name=PoolName.DRAFT,
            host_pool=host_pool,
            device_pool=full_kv_pool,
            layer_mapping={0: 0},
            transfer_layer_num=1,
        )

    @patch(
        "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
        "build_full_draft_pools"
    )
    def test_routes_non_swa_pool_to_full_builder(self, build_full_draft_pools):
        draft_kv_pool = object()
        build_full_draft_pools.return_value = ([], [])

        build_hicache_draft_sidecars(
            draft_device_pools=(draft_kv_pool,),
            tree_cache="tree-cache",
            server_args="server-args",
        )

        build_full_draft_pools.assert_called_once_with(
            draft_kv_pool=draft_kv_pool,
            tree_cache="tree-cache",
            server_args="server-args",
        )


if __name__ == "__main__":
    unittest.main()
