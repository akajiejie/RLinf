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
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PiperRobotState:
    qpos: np.ndarray = field(default_factory=lambda: np.zeros(14))   # puppet_arm_left.position + right.position
    qvel: np.ndarray = field(default_factory=lambda: np.zeros(14))
    effort: np.ndarray = field(default_factory=lambda: np.zeros(14))
    base_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [linear.x, angular.z]
    img_front: np.ndarray = field(default_factory=lambda: np.zeros((224,224,3), dtype=np.uint8))
    img_left: np.ndarray = field(default_factory=lambda: np.zeros((224,224,3), dtype=np.uint8))
    img_right: np.ndarray = field(default_factory=lambda: np.zeros((224,224,3), dtype=np.uint8))