# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest

from verl.single_controller.ray.base import (
    RayResourcePool,
    split_resource_pool,
    split_resource_pool_spod_aligned,
    validate_replicas_within_superpod,
)

GPUS_PER_NODE = 8


def make_fake_pool(store: list[int], spod_ids: list[int | None]) -> RayResourcePool:
    """Build a RayResourcePool with dummy placement groups, no ray cluster needed."""
    pool = RayResourcePool(process_on_nodes=list(store), use_gpu=True, name_prefix="test_pool")
    pool.pgs = [MagicMock(name=f"pg_{i}") for i in range(len(store))]
    pool.pg_spod_ids = list(spod_ids)
    return pool


def make_spod_pool() -> RayResourcePool:
    """3 SuperPods x 4 nodes x 8 GPUs."""
    return make_fake_pool([GPUS_PER_NODE] * 12, [0] * 4 + [1] * 4 + [2] * 4)


class TestSplitResourcePoolSpodAligned:
    def test_replicas_never_cross_superpod(self):
        # 2 nodes per replica: each SuperPod block of 4 nodes holds exactly 2 replicas.
        sub_pools = split_resource_pool_spod_aligned(make_spod_pool(), split_size=2 * GPUS_PER_NODE)

        assert len(sub_pools) == 6
        assert [int(p.start_bundle_index) for p in sub_pools] == [0, 16, 32, 48, 64, 80]
        for expected_spod, sub_pool in zip([0, 0, 1, 1, 2, 2], sub_pools, strict=True):
            assert sub_pool.world_size == 2 * GPUS_PER_NODE
            assert sub_pool.pg_spod_ids == [expected_spod] * 2

    def test_block_skip_leaves_fragment_idle(self):
        # 3 nodes per replica: each 4-node SuperPod block holds 1 replica, 1 node idle.
        sub_pools = split_resource_pool_spod_aligned(make_spod_pool(), split_size=[24, 24, 24])

        assert len(sub_pools) == 3
        assert [int(p.start_bundle_index) for p in sub_pools] == [0, 32, 64]
        for expected_spod, sub_pool in zip([0, 1, 2], sub_pools, strict=True):
            assert sub_pool.pg_spod_ids == [expected_spod] * 3

    def test_replica_larger_than_any_block_raises(self):
        # 5 nodes per replica cannot fit any 4-node SuperPod block.
        with pytest.raises(ValueError, match="within a single SuperPod"):
            split_resource_pool_spod_aligned(make_spod_pool(), split_size=[40])

    def test_too_many_replicas_for_fragmented_blocks_raises(self):
        # 4 replicas of 3 nodes need 12 nodes, but block-fit wastes 1 node per block.
        with pytest.raises(ValueError, match="within a single SuperPod"):
            split_resource_pool_spod_aligned(make_spod_pool(), split_size=[24, 24, 24, 24])

    def test_split_size_exceeding_pool_raises(self):
        with pytest.raises(ValueError, match="exceeds the resource pool world size"):
            split_resource_pool_spod_aligned(make_spod_pool(), split_size=[64, 64])

    def test_non_node_aligned_replica_raises(self):
        # 12 GPUs span 1.5 nodes and cannot be aligned to whole nodes.
        with pytest.raises(ValueError, match="aligned to whole nodes"):
            split_resource_pool_spod_aligned(make_spod_pool(), split_size=12)

    def test_sub_node_replicas_stay_within_one_pg(self):
        # 4-GPU replicas pack 2 per node and never straddle a node boundary.
        sub_pools = split_resource_pool_spod_aligned(make_spod_pool(), split_size=GPUS_PER_NODE // 2)

        assert len(sub_pools) == 24
        assert [int(p.start_bundle_index) for p in sub_pools[:4]] == [0, 4, 8, 12]
        for sub_pool in sub_pools:
            start = int(sub_pool.start_bundle_index)
            assert start // GPUS_PER_NODE == (start + sub_pool.world_size - 1) // GPUS_PER_NODE

    def test_falls_back_to_flat_split_when_spod_ids_none(self):
        pool = make_fake_pool([GPUS_PER_NODE] * 12, [None] * 12)
        sub_pools = split_resource_pool_spod_aligned(pool, split_size=2 * GPUS_PER_NODE)
        flat_sub_pools = split_resource_pool(pool, split_size=2 * GPUS_PER_NODE)

        assert [int(p.start_bundle_index) for p in sub_pools] == [int(p.start_bundle_index) for p in flat_sub_pools]

    def test_flat_fallback_honors_gpus_per_replica_node(self):
        # Without SuperPod IDs the allocation is flat, but the replica node layout
        # (4 GPUs/node) is still honored via the sub-pool store.
        pool = make_fake_pool([GPUS_PER_NODE] * 4, [None] * 4)
        sub_pools = split_resource_pool_spod_aligned(pool, split_size=[16], gpus_per_replica_node=4)

        assert len(sub_pools) == 1
        assert int(sub_pools[0].start_bundle_index) == 0
        assert sub_pools[0].store[0] == 4
        assert sub_pools[0].world_size == 16

    def test_split_resource_pool_propagates_spod_ids(self):
        sub_pools = split_resource_pool(make_spod_pool(), split_size=4 * GPUS_PER_NODE)

        assert [p.pg_spod_ids for p in sub_pools] == [[0] * 4, [1] * 4, [2] * 4]

    def test_split_sub_pool_input(self):
        # Splitting a SubRayResourcePool (e.g. a per-teacher pool) stays SuperPod-aligned.
        teacher_pool = split_resource_pool(make_spod_pool(), split_size=4 * GPUS_PER_NODE)[1]
        sub_pools = split_resource_pool_spod_aligned(teacher_pool, split_size=2 * GPUS_PER_NODE)

        assert len(sub_pools) == 2
        assert [int(p.start_bundle_index) for p in sub_pools] == [32, 48]
        for sub_pool in sub_pools:
            assert sub_pool.pg_spod_ids == [1] * 2

    def test_gpus_per_replica_node_spreads_replica_over_more_nodes(self):
        # 16-GPU replicas at 4 GPUs/node occupy 4 PGs each; both 4-node blocks fill exactly.
        pool = make_fake_pool([GPUS_PER_NODE] * 8, [0] * 4 + [1] * 4)
        sub_pools = split_resource_pool_spod_aligned(pool, split_size=[16, 16], gpus_per_replica_node=4)

        assert [int(p.start_bundle_index) for p in sub_pools] == [0, 16]
        for expected_spod, sub_pool in zip([0, 1], sub_pools, strict=True):
            assert sub_pool.store[0] == 4
            assert sub_pool.world_size == 16
            assert sub_pool.pg_spod_ids == [expected_spod] * 4

    def test_gpus_per_replica_node_too_many_nodes_raises(self):
        # 16 GPUs at 2 GPUs/node needs 8 nodes, more than any 4-node SuperPod block.
        pool = make_fake_pool([GPUS_PER_NODE] * 8, [0] * 4 + [1] * 4)
        with pytest.raises(ValueError, match="within a single SuperPod"):
            split_resource_pool_spod_aligned(pool, split_size=[16], gpus_per_replica_node=2)

    def test_gpus_per_replica_node_exceeding_pool_raises(self):
        with pytest.raises(ValueError, match="per-node GPU count"):
            split_resource_pool_spod_aligned(make_spod_pool(), split_size=16, gpus_per_replica_node=16)


class TestValidateReplicasWithinSuperpod:
    def test_noop_when_ids_unavailable(self):
        validate_replicas_within_superpod(None, GPUS_PER_NODE, replica_world_size=16)
        validate_replicas_within_superpod([None] * 8, GPUS_PER_NODE, replica_world_size=16)

    def test_aligned_pool_passes(self):
        # 2 SuperPods x 4 nodes, 4-node replicas align with SuperPod boundaries.
        validate_replicas_within_superpod([0] * 4 + [1] * 4, GPUS_PER_NODE, replica_world_size=32)

    def test_cross_superpod_replica_raises(self):
        # 2-node replicas: replica 1 lands on PGs 2-3 which belong to different SuperPods.
        with pytest.raises(ValueError, match="spans multiple SuperPods"):
            validate_replicas_within_superpod([0] * 3 + [1] * 5, GPUS_PER_NODE, replica_world_size=16)

    def test_error_hint_appended(self):
        with pytest.raises(ValueError, match="trainer.nnodes"):
            validate_replicas_within_superpod(
                [0] * 3 + [1] * 5,
                GPUS_PER_NODE,
                replica_world_size=16,
                error_hint=" Align trainer.nnodes to SuperPod boundaries.",
            )

    def test_indivisible_pool_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            validate_replicas_within_superpod([0] * 8, GPUS_PER_NODE, replica_world_size=24)
