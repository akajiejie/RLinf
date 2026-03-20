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

"""Pico VR 双臂 Piper 遥操作控制器（基于 ROS）。

基于 XRoboToolkit 的遥操作框架，将底层通信从 CAN 直连改为 ROS 话题，
与 piper_ros 节点配合使用。

使用方法::

    # 1. 先启动 piper_ros 节点
    roslaunch piper start_ms_piper.launch

    # 2. 运行遥操作控制器
    python -m rlinf.envs.realworld.common.pico.pico_dual_piper

架构：
    VR 头显 (XrClient) → Placo IK → PiperController → ROS 话题 → piper_ros → CAN
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import (
    HardwareTeleopController,
)
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

from rlinf.envs.realworld.piper import PiperController

# =========================================================================
# 默认配置
# =========================================================================

DEFAULT_DUAL_PIPER_URDF_PATH = os.path.join(ASSET_PATH, "agilex/piper_dual.urdf")
DEFAULT_SCALE_FACTOR = 1.5

DEFAULT_MANIPULATOR_CONFIG = {
    "right_arm": {
        "link_name": "right_link6",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["right_joint7", "right_joint8"],
            "open_pos": [0.07, -0.07],
            "close_pos": [0.0, 0.0],
        },
    },
    "left_arm": {
        "link_name": "left_link6",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": ["left_joint7", "left_joint8"],
            "open_pos": [0.07, -0.07],
            "close_pos": [0.0, 0.0],
        },
    },
}


class PicoDualPiperController(HardwareTeleopController):
    """Pico VR 双臂 Piper 遥操作控制器（ROS 版本）。

    继承 XRoboToolkit 的 HardwareTeleopController，将底层 CAN 通信
    替换为 ROS 话题通信，通过 PiperController 与 piper_ros 节点交互。

    **与 AgileXDualArmTeleopController 的区别：**

    - 原版：AgileXArmController → piper_sdk → CAN 总线
    - 本版：PiperController → ROS 话题 → piper_ros 节点 → CAN 总线

    Args:
        robot_urdf_path: 双臂 URDF 文件路径。
        manipulator_config: 机械臂配置字典。
        R_headset_world: 头显到世界坐标系的旋转矩阵。
        scale_factor: VR 手柄移动缩放因子。
        visualize_placo: 是否可视化 Placo IK。
        control_rate_hz: 控制循环频率 (Hz)。
        enable_log_data: 是否启用数据记录。
        log_dir: 日志目录。
        log_freq: 日志频率。
        enable_camera: 是否启用相机。
        camera_fps: 相机帧率。
        ns_left: 左臂 ROS 命名空间。
        ns_right: 右臂 ROS 命名空间。
        joint_speed_pct: 关节运动速度百分比 (0-100)。
    """

    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_PIPER_URDF_PATH,
        manipulator_config: dict = None,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 50,
        enable_log_data: bool = False,
        log_dir: str = "logs/pico_dual_piper",
        log_freq: float = 50,
        enable_camera: bool = False,
        camera_fps: int = 30,
        ns_left: str = "/puppet_left",
        ns_right: str = "/puppet_right",
        joint_speed_pct: int = 50,
    ):
        if manipulator_config is None:
            manipulator_config = DEFAULT_MANIPULATOR_CONFIG

        self._ns_left = ns_left
        self._ns_right = ns_right
        self._joint_speed_pct = joint_speed_pct

        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
        )

        self._previous_active_state = {
            name: False for name in manipulator_config.keys()
        }
        self._latest_joint_feedback: Dict[str, Optional[np.ndarray]] = {
            name: None for name in manipulator_config.keys()
        }
        self._last_commanded_q: Dict[str, Optional[np.ndarray]] = {
            name: None for name in manipulator_config.keys()
        }

    # ==================================================================
    # 重写：机器人初始化（ROS 替代 CAN）
    # ==================================================================

    def _robot_setup(self):
        """初始化 PiperController（ROS 通信）。

        替代原版的 AgileXArmController (CAN 通信)。
        """
        print(f"初始化 PiperController: left={self._ns_left}, right={self._ns_right}")
        self._piper = PiperController(
            ns_left=self._ns_left,
            ns_right=self._ns_right,
            use_robot_base=False,
            joint_speed_pct=self._joint_speed_pct,
        )

        print("等待 Piper 双臂机械臂就绪...")
        ready = self._piper.wait_for_robot(timeout=30.0)
        if not ready:
            raise RuntimeError("Piper 机械臂等待超时，请检查 piper_ros 节点是否启动")

        print("Piper 双臂机械臂已就绪")

    # ==================================================================
    # 重写：Placo 关节索引设置
    # ==================================================================

    def _placo_setup(self):
        """设置 Placo IK 求解器和关节索引映射。"""
        super()._placo_setup()

        self.placo_arm_joint_slice = {}
        self.placo_arm_joint_indices = {}
        all_joint_names = list(self.placo_robot.model.names)

        for arm_name in ["left_arm", "right_arm"]:
            if arm_name == "right_arm":
                prefixes_to_try = ["right_", "r_", ""]
            else:
                prefixes_to_try = ["left_", "l_", ""]

            arm_joint_names = None
            for prefix in prefixes_to_try:
                candidate_names = [f"{prefix}joint{i}" for i in range(1, 7)]
                if all(name in all_joint_names for name in candidate_names):
                    arm_joint_names = candidate_names
                    break

            if arm_joint_names is None:
                arm_joint_names = [f"joint{i}" for i in range(1, 7)]
                if not all(name in all_joint_names for name in arm_joint_names):
                    raise ValueError(
                        f"无法找到 {arm_name} 的关节名称，可用关节: {all_joint_names}"
                    )

            joint_indices = [
                self.placo_robot.get_joint_offset(joint_name)
                for joint_name in arm_joint_names
            ]
            min_idx = min(joint_indices)
            max_idx = max(joint_indices)
            self.placo_arm_joint_indices[arm_name] = joint_indices
            self.placo_arm_joint_slice[arm_name] = slice(min_idx, max_idx + 1)
            print(f"设置 {arm_name} 关节: {arm_joint_names}, 索引: {joint_indices}")

    # ==================================================================
    # 重写：从 ROS 读取机器人状态
    # ==================================================================

    def _update_robot_state(self):
        """从 ROS 话题读取关节状态并更新 Placo。

        数据流：
            PiperController.get_qpos() → 14D [left(7) + right(7)]
            → 映射到 placo_robot.state.q
        """
        qpos = self._piper.get_qpos()

        for arm_name in ["left_arm", "right_arm"]:
            if arm_name not in self.placo_arm_joint_indices:
                continue

            indices = self.placo_arm_joint_indices[arm_name]

            if arm_name == "left_arm":
                joint_positions = qpos[0:6]
            else:
                joint_positions = qpos[7:13]

            best_q = np.array(joint_positions, dtype=np.float64)
            self._latest_joint_feedback[arm_name] = best_q.copy()

            for i, joint_idx in enumerate(indices):
                self.placo_robot.state.q[joint_idx] = float(best_q[i])

    # ==================================================================
    # 重写：发送控制指令到 ROS
    # ==================================================================

    def _send_command(self):
        """将 Placo IK 结果发布到 ROS 话题。

        数据流：
            placo_robot.state.q → 提取 6 关节 + 1 夹爪
            → PiperController.move_arm() → ROS 话题 → piper_ros
        """
        left_q = None
        right_q = None
        left_gripper = 0.0
        right_gripper = 0.0

        for arm_name in ["left_arm", "right_arm"]:
            if arm_name not in self.placo_arm_joint_indices:
                continue

            is_active = self.active.get(arm_name, False)
            prev_active = self._previous_active_state.get(arm_name, False)
            self._previous_active_state[arm_name] = is_active

            joint_indices = self.placo_arm_joint_indices[arm_name]
            q_placo = np.array(
                [self.placo_robot.state.q[idx] for idx in joint_indices]
            )

            current_q = self._latest_joint_feedback.get(arm_name)
            if current_q is None or len(current_q) != len(joint_indices):
                current_q = q_placo.copy()

            if is_active:
                if not prev_active:
                    print(
                        f"[IK DEBUG] {arm_name} 激活 -> "
                        f"q_des={np.round(q_placo, 4)}, current={np.round(current_q, 4)}"
                    )
                target_q = q_placo
                self._last_commanded_q[arm_name] = q_placo.copy()
            else:
                hold_q = self._last_commanded_q.get(arm_name)
                if hold_q is None or len(hold_q) != len(joint_indices):
                    hold_q = current_q.copy()
                    self._last_commanded_q[arm_name] = hold_q.copy()
                target_q = hold_q

            gripper_pos = 0.0
            if "gripper_config" in self.manipulator_config[arm_name]:
                gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                joint_names = gripper_config["joint_names"]
                if joint_names[0] in self.gripper_pos_target.get(arm_name, {}):
                    gripper_pos = self.gripper_pos_target[arm_name][joint_names[0]]

            if arm_name == "left_arm":
                left_q = target_q
                left_gripper = gripper_pos
            else:
                right_q = target_q
                right_gripper = gripper_pos

        if left_q is not None and right_q is not None:
            left_action = np.append(left_q, left_gripper)
            right_action = np.append(right_q, right_gripper)
            self._piper.move_arm(left_action, right_action)

    # ==================================================================
    # 重写：相机初始化（暂不使用）
    # ==================================================================

    def _initialize_camera(self):
        """相机初始化（暂不使用，可通过 ROS 话题获取）。"""
        self.camera_interface = None

    # ==================================================================
    # 重写：获取末端位姿（通过 Placo FK）
    # ==================================================================

    def _get_robot_end_pose(self, src_name: str = None) -> Optional[Dict]:
        """获取机器人末端位姿。

        通过 Placo FK 计算，而非 SDK。

        Args:
            src_name: 机械臂名称 ("left_arm" 或 "right_arm")。

        Returns:
            {x, y, z, rx, ry, rz} 字典或 None。
        """
        if src_name is None:
            src_name = list(self.manipulator_config.keys())[0]

        if src_name not in self.manipulator_config:
            return None

        link_name = self.manipulator_config[src_name]["link_name"]
        ee_xyz, ee_quat = self._get_link_pose(link_name)

        import meshcat.transformations as tf

        euler = tf.euler_from_quaternion(ee_quat, axes="sxyz")

        return {
            "x": ee_xyz[0],
            "y": ee_xyz[1],
            "z": ee_xyz[2],
            "rx": euler[0],
            "ry": euler[1],
            "rz": euler[2],
        }

    # ==================================================================
    # 重写：日志数据
    # ==================================================================

    def _get_robot_state_for_logging(self) -> Dict:
        """返回用于日志的机器人状态数据。"""
        qpos = self._piper.get_qpos()
        qvel = self._piper.get_qvel()

        return {
            "qpos": {
                "left_arm": qpos[0:7].tolist(),
                "right_arm": qpos[7:14].tolist(),
            },
            "qvel": {
                "left_arm": qvel[0:7].tolist(),
                "right_arm": qvel[7:14].tolist(),
            },
            "qpos_des": {
                arm: self._last_commanded_q[arm].tolist()
                for arm in ["left_arm", "right_arm"]
                if self._last_commanded_q.get(arm) is not None
            },
        }

    def _get_camera_frame_for_logging(self) -> Dict:
        """返回用于日志的相机帧（暂不使用）。"""
        return {}

    def _should_keep_running(self) -> bool:
        """判断是否继续运行。"""
        return super()._should_keep_running()

    def _shutdown_robot(self):
        """关闭机器人。"""
        print("PicoDualPiperController 关闭")


# =========================================================================
# 主函数
# =========================================================================


def main(
    robot_urdf_path: str = DEFAULT_DUAL_PIPER_URDF_PATH,
    scale_factor: float = 1.0,
    enable_log_data: bool = False,
    visualize_placo: bool = False,
    control_rate_hz: int = 50,
    log_dir: str = "logs/pico_dual_piper",
    ns_left: str = "/puppet_left",
    ns_right: str = "/puppet_right",
    joint_speed_pct: int = 50,
):
    """运行 Pico VR 双臂 Piper 遥操作控制器。

    Args:
        robot_urdf_path: 双臂 URDF 文件路径。
        scale_factor: VR 手柄移动缩放因子。
        enable_log_data: 是否启用数据记录。
        visualize_placo: 是否可视化 Placo IK。
        control_rate_hz: 控制循环频率 (Hz)。
        log_dir: 日志目录。
        ns_left: 左臂 ROS 命名空间。
        ns_right: 右臂 ROS 命名空间。
        joint_speed_pct: 关节运动速度百分比 (0-100)。
    """
    import tyro

    controller = PicoDualPiperController(
        robot_urdf_path=robot_urdf_path,
        scale_factor=scale_factor,
        enable_log_data=enable_log_data,
        visualize_placo=visualize_placo,
        control_rate_hz=control_rate_hz,
        log_dir=log_dir,
        ns_left=ns_left,
        ns_right=ns_right,
        joint_speed_pct=joint_speed_pct,
    )
    controller.run()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
