"""Unit test for hybrid HiCache fixed-size budget splitting."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _build_dsv4_canary_page_buffers,
    _split_hicache_size,
)
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


class TestDeepSeekV4CanaryPageBuffers(CustomTestCase):
    @staticmethod
    def _pool(*, layer_num: int, slots: int = 8, fill: int = 0):
        head = torch.full((slots, 32), fill, dtype=torch.uint8)
        tail = torch.full((slots, 32), fill + 1, dtype=torch.uint8)
        group = SimpleNamespace(
            kind=SimpleNamespace(name="SWA"),
            k_head=head,
            k_tail=tail,
        )
        return SimpleNamespace(
            swa_kv_pool=SimpleNamespace(kv_buffer=[object()] * layer_num),
            canary_buffer_groups=(group,),
        )

    def test_page_packs_target_and_draft_canaries(self):
        target = self._pool(layer_num=3, fill=10)
        draft_1 = self._pool(layer_num=2, fill=20)
        draft_2 = self._pool(layer_num=1, fill=30)

        result = _build_dsv4_canary_page_buffers(
            target_pool=target,
            draft_pools=(draft_1, draft_2),
            swa_page_size=4,
        )

        self.assertEqual(result.item_bytes, 4 * 32)
        self.assertEqual(result.layer_mapping, {0: 0, 3: 1, 5: 2})
        self.assertEqual([tuple(x.shape) for x in result.head], [(2, 128)] * 3)
        self.assertEqual([tuple(x.shape) for x in result.tail], [(2, 128)] * 3)
        self.assertEqual(
            result.head[0].data_ptr(), target.canary_buffer_groups[0].k_head.data_ptr()
        )
        self.assertTrue(torch.all(result.head[1] == 20))
        self.assertTrue(torch.all(result.tail[2] == 31))

    def test_skips_pools_without_canary_but_keeps_layer_offsets(self):
        target = SimpleNamespace(
            swa_kv_pool=SimpleNamespace(kv_buffer=[object()] * 3),
            canary_buffer_groups=(),
        )
        draft = self._pool(layer_num=2)

        result = _build_dsv4_canary_page_buffers(
            target_pool=target,
            draft_pools=(draft,),
            swa_page_size=4,
        )

        self.assertEqual(result.layer_mapping, {3: 0})

    def test_rejects_non_page_aligned_canary_slots(self):
        target = self._pool(layer_num=3, slots=7)
        with self.assertRaisesRegex(ValueError, "must be page-aligned"):
            _build_dsv4_canary_page_buffers(
                target_pool=target,
                draft_pools=(),
                swa_page_size=4,
            )

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled


if __name__ == "__main__":
    unittest.main()
