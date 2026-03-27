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

"""Piper 双臂机械臂 Gym 环境。

作为 RLinf 的统一接口，协调 ``PiperController`` 和 ``PiperRobotState``，
提供标准的 ``gymnasium.Env`` API（step / reset / observation_space / action_space）。

架构对齐 ``rlinf.envs.realworld.franka.franka_env.FrankaEnv``。

**动作空间（Joint space）：**

- 14 维绝对关节位置：左臂 7 维 (6 关节 + 1 夹爪) + 右臂 7 维
- 对齐 ``collect_data_jeff.py`` 中 ``master_arm_left.position + master_arm_right.position``

**观测空间：**

- ``state``: qpos(14), qvel(14), effort(14), base_vel(2)
- ``frames``: cam_high(480,640,3), cam_left_wrist(480,640,3), cam_right_wrist(480,640,3)
"""

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from rlinf.utils.logging import get_logger

from .piper_controller import PiperController
from .piper_robot_state import PiperRobotState
from .utils import (
    PIPER_GRIPPER_MAX,
    PIPER_GRIPPER_MIN,
    PIPER_JOINT_LIMITS_HIGH,
    PIPER_JOINT_LIMITS_LOW,
    clip_gripper,
    clip_joint_positions,
    split_dual_arm_action,
)


# =========================================================================
# 配置
# =========================================================================


@dataclass
class PiperRobotConfig:
    """Piper 双臂机械臂环境配置。

    通过 YAML / OmegaConf 传入，对齐 ``FrankaRobotConfig`` 的风格。

    Attributes:
        ns_left: 左臂 ROS 命名空间。
        ns_right: 右臂 ROS 命名空间。
        use_robot_base: 是否使用底盘。
        robot_base_topic: 底盘里程计话题。
        camera_names: 相机名称列表，对应 ``collect_data_jeff.py`` 的 camera_names。
        img_topics: 相机 ROS 话题列表，与 camera_names 一一对应。
        img_resolution: 相机图像分辨率 (H, W)。
        obs_img_resolution: 观测中图像的分辨率 (H, W)，用于 resize。
        step_frequency: 控制频率 (Hz)，对应 ``collect_data_jeff.py`` 的 frame_rate。
        publish_rate: 发布频率 (Hz)。
        joint_speed_pct: 关节运动速度百分比 (0-100)。
        is_dummy: 是否为虚拟模式（无真实硬件）。
        max_num_steps: 每个 episode 的最大步数。
        min_qpos: 双臂关节下限 (8维: 6关节+1夹爪+1夹爪 per arm)。
        max_qpos: 双臂关节上限。
        enable_camera_player: 是否启用相机画面显示。
    """

    # ---- ROS 命名空间 ----
    ns_left: str = "/puppet_left"
    ns_right: str = "/puppet_right"

    # ---- 底盘 ----
    use_robot_base: bool = False
    robot_base_topic: str = "/odom"

    # ---- 相机配置 ----
    # 对齐 collect_data_jeff.py 的 camera_names 和 topic
    camera_names: list[str] = field(
        default_factory=lambda: ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    )
    img_topics: list[str] = field(
        default_factory=lambda: [
            "/camera_f/color/image_raw",
            "/camera_l/color/image_raw",
            "/camera_r/color/image_raw",
        ]
    )
    img_resolution: tuple[int, int] = (480, 640)  # (H, W) 原始图像分辨率
    obs_img_resolution: tuple[int, int] = (480, 640)  # (H, W) 观测空间图像分辨率

    # ---- 控制参数 ----
    step_frequency: float = 30.0  # Hz，对应 collect_data_jeff.py 的 frame_rate
    publish_rate: int = 30
    joint_speed_pct: int = 50
    pos_lookahead_step: int = 50
    chunk_size: int = 50

    # ---- 环境参数 ----
    task_name: str = "task"
    is_dummy: bool = False
    max_num_steps: int = 10000
    enable_camera_player: bool = False

    # ---- 关节限位 (per arm: 6 joints + 1 gripper) ----
    # 双臂拼接后为 16 维 (8 + 8)
    min_qpos: list[float] = field(
        default_factory=lambda: [
            -2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0, 0.0,
        ]
    )
    max_qpos: list[float] = field(
        default_factory=lambda: [
            2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 1.0, 1.0,
        ]
    )

    # ---- 人在回路配置 ----
    enable_human_intervention: bool = True
    master_left_topic: str = "/master/joint_left"
    master_right_topic: str = "/master/joint_right"
    intervention_trigger_key: str = "page_down"  # 切换遥操作介入
    policy_enable_key: str = "page_up"  # 切换策略输出

    # ---- ZMQ 推理服务（预留） ----
    inference_host: str = "127.0.0.1"
    inference_port: int = 8080


# =========================================================================
# 环境
# =========================================================================


class PiperEnv(gym.Env):
    """Piper 双臂机械臂 Gymnasium 环境。

    对齐 ``FrankaEnv`` 的接口设计，通过 ``PiperController`` 与 piper_ros 通信。
    图像通过 ROS 话题订阅获取（使用 ``cv_bridge``），
    对齐 ``collect_data_jeff.py`` 中 ``RosOperator`` 的数据流。

    **数据流：**

    1. ``__init__``: 创建 PiperController → 订阅关节/位姿/图像话题 → 等待机械臂就绪
    2. ``step(action)``: 拆分 14D 动作 → 限位裁剪 → 发布关节指令 → 频率控制 → 返回观测
    3. ``reset()``: 使能 → 归零 → 等待到位 → 返回初始观测
    4. ``_get_observation()``: 拼接 qpos/qvel/effort/base_vel + 相机图像

    Args:
        config: PiperRobotConfig 配置。
        worker_info: RLinf Worker 信息（可选）。
        hardware_info: 硬件信息（可选，预留）。
        env_idx: 环境实例索引。
    """

    def __init__(
        self,
        config: PiperRobotConfig,
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        super().__init__()
        self._logger = get_logger()
        self.config = config
        self.env_idx = env_idx
        self._num_steps = 0

        # ---- 初始化控制器 ----
        if not self.config.is_dummy:
            self._controller = PiperController(
                ns_left=config.ns_left,
                ns_right=config.ns_right,
                use_robot_base=config.use_robot_base,
                robot_base_topic=config.robot_base_topic,
                joint_speed_pct=config.joint_speed_pct,
            )
        else:
            self._controller = None

        # ---- 初始化主臂控制器（人在回路） ----
        if not self.config.is_dummy and self.config.enable_human_intervention:
            from rlinf.envs.realworld.common.controller.piper import MasterArmController

            self._master_controller = MasterArmController(
                master_left_topic=config.master_left_topic,
                master_right_topic=config.master_right_topic,
                enable_keyboard_trigger=True,
                trigger_key=config.intervention_trigger_key,
                policy_enable_key=config.policy_enable_key,
            )
        else:
            self._master_controller = None

        # ---- 图像缓存（ROS 回调线程安全写入） ----
        self._bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._latest_images: dict[str, np.ndarray] = {}
        for cam_name in config.camera_names:
            h, w = config.obs_img_resolution
            self._latest_images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)

        # ---- 订阅相机话题 ----
        if not self.config.is_dummy:
            self._img_subscribers: list[rospy.Subscriber] = []
            for cam_name, topic in zip(config.camera_names, config.img_topics):
                sub = rospy.Subscriber(
                    topic,
                    Image,
                    callback=self._make_img_callback(cam_name),
                    queue_size=1,
                    tcp_nodelay=True,
                )
                self._img_subscribers.append(sub)

        # ---- 初始化动作/观测空间 ----
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            self._logger.info("PiperEnv 以 dummy 模式初始化。")
            return

        # ---- 等待机械臂就绪 ----
        self._logger.info("等待 Piper 双臂机械臂就绪...")
        ready = self._controller.wait_for_robot(timeout=30.0)
        if not ready:
            self._logger.warning("机械臂等待超时，部分话题可能未就绪。")

        # ---- 等待主臂数据就绪（如果启用人在回路） ----
        if self._master_controller is not None:
            self._logger.info("等待主臂数据就绪...")
            master_ready = self._master_controller.wait_for_master_data(timeout=10.0)
            if not master_ready:
                self._logger.warning(
                    "主臂数据等待超时，人在回路功能可能不可用。"
                )

        self._logger.info(
            f"PiperEnv 初始化完成: env_idx={env_idx}, "
            f"cameras={config.camera_names}, freq={config.step_frequency}Hz, "
            f"intervention={config.enable_human_intervention}"
        )

    # ==================================================================
    # 图像 ROS 回调
    # ==================================================================

    def _make_img_callback(self, cam_name: str):
        """创建指定相机名称的 ROS 图像回调闭包。

        对齐 ``collect_data_jeff.py`` 中 ``img_front_callback`` 等回调的逻辑。

        Args:
            cam_name: 相机名称。

        Returns:
            ROS 回调函数。
        """

        def _callback(msg: Image) -> None:
            try:
                cv_image = self._bridge.imgmsg_to_cv2(msg, "passthrough")
                # 如果需要 resize
                h, w = self.config.obs_img_resolution
                if cv_image.shape[0] != h or cv_image.shape[1] != w:
                    cv_image = cv2.resize(cv_image, (w, h))
                with self._img_lock:
                    self._latest_images[cam_name] = cv_image
            except Exception as e:
                self._logger.warning(f"图像回调异常 ({cam_name}): {e}")

        return _callback

    # ==================================================================
    # 动作/观测空间
    # ==================================================================

    def _init_action_obs_spaces(self) -> None:
        """初始化动作空间和观测空间。

        **动作空间：**
        14 维绝对关节位置（左臂 7 + 右臂 7），每臂 6 关节 + 1 夹爪。

        **观测空间：**
        - ``state``: qpos(14), qvel(14), effort(14), base_vel(2)
        - ``frames``: 每个相机一个 (H, W, 3) 图像
        """
        # ---- 动作空间: 14D 关节空间 ----
        # 构造双臂限位: 左臂 min/max + 右臂 min/max
        min_qpos = np.array(self.config.min_qpos, dtype=np.float32)
        max_qpos = np.array(self.config.max_qpos, dtype=np.float32)
        # 拼接为 [left_min(7), right_min(7)] 和 [left_max(7), right_max(7)]
        # min_qpos/max_qpos 配置中前 7 个为单臂限位，左右臂共享
        if len(min_qpos) < 14:
            # 单臂限位，复制为双臂
            action_low = np.tile(min_qpos[:7], 2).astype(np.float32)
            action_high = np.tile(max_qpos[:7], 2).astype(np.float32)
        else:
            action_low = min_qpos[:14].astype(np.float32)
            action_high = max_qpos[:14].astype(np.float32)

        self.action_space = gym.spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )

        # ---- 观测空间 ----
        h, w = self.config.obs_img_resolution
        state_space = gym.spaces.Dict(
            {
                "qpos": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
                "qvel": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
                "effort": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
                "base_vel": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
            }
        )
        frames_space = gym.spaces.Dict(
            {
                cam_name: gym.spaces.Box(
                    0, 255, shape=(h, w, 3), dtype=np.uint8
                )
                for cam_name in self.config.camera_names
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": state_space,
                "frames": frames_space,
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # ==================================================================
    # Gym API: step
    # ==================================================================

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """执行一步环境交互。

        对齐 ``FrankaEnv.step()`` 的接口，以及 ``collect_data_jeff.py``
        中 ``RosOperator.process()`` 的控制逻辑。

        Args:
            action: 14 维关节位置数组 (左臂 7 + 右臂 7)，
                    每臂 6 关节角度 (rad) + 1 夹爪 (rad/m)。

        Returns:
            (observation, reward, terminated, truncated, info) 五元组。
        """
        start_time = time.time()

        # ---- 裁剪动作到合法范围 ----
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )

        # ---- 人在回路介入: 根据遥操作和策略状态混合动作 ----
        if self._master_controller is not None:
            # 获取当前从臂位置
            current_qpos = self._controller.get_qpos()
            # blend_action() 内部处理优先级：
            # 1. 遥操作 ON -> current_qpos + master_delta
            # 2. 策略 ON -> policy_action
            # 3. 两者 OFF -> current_qpos (保持)
            action = self._master_controller.blend_action(action, current_qpos)

        # ---- 拆分为左右臂动作 ----
        left_action, right_action = split_dual_arm_action(action)

        # ---- 发布控制指令 ----
        if not self.config.is_dummy:
            self._controller.move_arm(left_action, right_action)
        else:
            self._logger.debug(f"Dummy step: left={left_action}, right={right_action}")

        self._num_steps += 1

        # ---- 频率控制 ----
        step_time = time.time() - start_time
        sleep_time = max(0.0, (1.0 / self.config.step_frequency) - step_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        # ---- 获取观测 ----
        observation = self._get_observation()

        # ---- 计算奖励 ----
        reward = self._calc_step_reward(observation)

        # ---- 终止条件 ----
        terminated = False  # Piper 环境不自动终止，由外部任务逻辑决定
        truncated = self._num_steps >= self.config.max_num_steps

        return observation, reward, terminated, truncated, {}

    # ==================================================================
    # Gym API: reset
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """重置环境到初始状态。

        对齐 ``FrankaEnv.reset()`` 的逻辑:
        1. 使能双臂
        2. 归零
        3. 等待到位
        4. 返回初始观测

        Args:
            seed: 随机种子（预留）。
            options: 额外选项（预留）。

        Returns:
            (observation, info) 二元组。
        """
        self._num_steps = 0

        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        # ---- 使能 ----
        self._controller.enable_arm()
        time.sleep(0.5)

        # ---- 归零 ----
        self._controller.go_zero()

        # ---- 等待到位 ----
        self._controller._wait_for_joint(
            target_pos=np.zeros(6),
            side="left",
            timeout=10.0,
            atol=0.05,
        )
        self._controller._wait_for_joint(
            target_pos=np.zeros(6),
            side="right",
            timeout=10.0,
            atol=0.05,
        )

        time.sleep(0.5)

        observation = self._get_observation()
        self._logger.info("PiperEnv reset 完成。")
        return observation, {}

    # ==================================================================
    # 观测构建
    # ==================================================================

    def _get_observation(self) -> dict:
        """构建完整观测字典。

        对齐 ``collect_data_jeff.py`` 第 340-354 行的观测格式:

        - ``state.qpos``: 14D 双臂关节位置 (6关节+1夹爪) × 2
        - ``state.qvel``: 14D 双臂关节速度
        - ``state.effort``: 14D 双臂关节力矩
        - ``state.base_vel``: 2D 底盘速度 [linear.x, angular.z]
        - ``frames``: 各相机图像

        Returns:
            符合 observation_space 的字典。
        """
        if not self.config.is_dummy:
            # ---- 状态 ----
            state = {
                "qpos": self._controller.get_qpos(),
                "qvel": self._controller.get_qvel(),
                "effort": self._controller.get_effort(),
                "base_vel": self._controller.get_base_vel(),
            }

            # ---- 图像 ----
            frames = self._get_camera_frames()

            observation = {
                "state": state,
                "frames": frames,
            }
            return copy.deepcopy(observation)
        else:
            return self._base_observation_space.sample()

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """线程安全地获取所有相机的最新帧。

        Returns:
            {cam_name: np.ndarray(H, W, 3)} 字典。
        """
        frames = {}
        with self._img_lock:
            for cam_name in self.config.camera_names:
                frames[cam_name] = self._latest_images[cam_name].copy()
        return frames

    # ==================================================================
    # 奖励计算
    # ==================================================================

    def _calc_step_reward(
        self, observation: dict, **kwargs: Any
    ) -> float:
        """计算当前步的奖励。

        基础实现返回 0.0，具体任务应在子类（task env）中覆写。

        Args:
            observation: 当前观测。

        Returns:
            标量奖励值。
        """
        return 0.0

    # ==================================================================
    # 属性与工具方法
    # ==================================================================

    @property
    def num_steps(self) -> int:
        """当前 episode 已执行的步数。"""
        return self._num_steps

    @property
    def task_description(self) -> str:
        """任务描述，供 RealWorldEnv 包装器使用。"""
        return self.config.task_name

    def close(self) -> None:
        """关闭环境，释放资源。"""
        if not self.config.is_dummy and hasattr(self, "_img_subscribers"):
            for sub in self._img_subscribers:
                sub.unregister()
            self._img_subscribers = []
        
        # 清理主臂控制器资源
        if self._master_controller is not None:
            self._logger.info("关闭主臂控制器...")
            # MasterArmController 中的线程是 daemon=True，会自动退出
            self._master_controller = None
        
        self._logger.info("PiperEnv 已关闭。")
