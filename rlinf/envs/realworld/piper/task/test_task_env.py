# Copyright 2026 The GIGA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Piper 测试任务环境。

继承 ``PiperEnv``，作为集成测试和新任务开发的模板。
对齐 ``rlinf.envs.realworld.franka.tasks.peg_insertion_env.PegInsertionEnv`` 的模式。
"""

from dataclasses import dataclass

from ..piper_env import PiperEnv, PiperRobotConfig


@dataclass
class PiperTestTaskConfig(PiperRobotConfig):
    """测试任务配置，继承 PiperRobotConfig 并覆写默认值。"""

    task_name: str = "piper_test_task"
    max_num_steps: int = 1000


class PiperTestTaskEnv(PiperEnv):
    """Piper 测试任务环境。

    用于验证 PiperEnv 的基本功能：使能、归零、关节控制、观测读取。
    具体任务逻辑（奖励、终止条件）可在此类中覆写。
    """

    def __init__(
        self,
        override_cfg: dict | None = None,
        worker_info=None,
        hardware_info=None,
        env_idx: int = 0,
    ) -> None:
        if override_cfg is None:
            override_cfg = {}
        config = PiperTestTaskConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self) -> str:
        return "piper test task"
