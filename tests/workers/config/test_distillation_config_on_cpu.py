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

import unittest

from verl.workers.config.distillation import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.config.rollout import RolloutConfig


def make_teacher_config(tensor_parallel_size: int, nnodes_per_replica: int = 0) -> DistillationTeacherModelConfig:
    inference = RolloutConfig(name="vllm", tensor_model_parallel_size=tensor_parallel_size)
    return DistillationTeacherModelConfig(
        key="default",
        model_path="/tmp/teacher_model",
        inference=inference,
        nnodes_per_replica=nnodes_per_replica,
    )


def make_distillation_config(teacher_model: DistillationTeacherModelConfig, n_gpus_per_node: int = 8, nnodes: int = 2):
    return DistillationConfig(
        enabled=True,
        n_gpus_per_node=n_gpus_per_node,
        nnodes=nnodes,
        teacher_models={"teacher_model": teacher_model},
        distillation_loss=DistillationLossConfig(),
    )


class TestDistillationNnodesPerReplica(unittest.TestCase):
    """Test the nnodes_per_replica validation of the distillation teacher config."""

    def test_default_derived_behavior(self):
        # nnodes_per_replica=0 keeps the derived per-node GPU count.
        config = make_distillation_config(make_teacher_config(tensor_parallel_size=4))

        teacher_model = config.teacher_models["default"]
        assert teacher_model.num_replicas == 4
        assert teacher_model.nnodes_per_replica == 0

    def test_valid_explicit_nnodes_per_replica(self):
        # per_replica_world_size=4 over 2 nodes -> 2 GPUs per node (<= 8).
        config = make_distillation_config(make_teacher_config(tensor_parallel_size=4, nnodes_per_replica=2))

        teacher_model = config.teacher_models["default"]
        assert teacher_model.per_replica_world_size == 4
        assert teacher_model.nnodes_per_replica == 2

    def test_indivisible_nnodes_per_replica_raises(self):
        # per_replica_world_size=4 is not divisible by 3 nodes.
        with self.assertRaisesRegex(ValueError, "divisible by nnodes_per_replica"):
            make_distillation_config(make_teacher_config(tensor_parallel_size=4, nnodes_per_replica=3))

    def test_gpus_per_replica_node_exceeding_pool_raises(self):
        # per_replica_world_size=16 on 1 node -> 16 GPUs per node > pool's 8 GPUs per node.
        with self.assertRaisesRegex(ValueError, "n_gpus_per_node"):
            make_distillation_config(make_teacher_config(tensor_parallel_size=16, nnodes_per_replica=1))

    def test_negative_nnodes_per_replica_raises(self):
        with self.assertRaisesRegex(ValueError, ">= 0"):
            make_distillation_config(make_teacher_config(tensor_parallel_size=4, nnodes_per_replica=-1))
