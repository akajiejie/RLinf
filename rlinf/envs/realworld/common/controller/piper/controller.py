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

"""主臂人在回路控制器。

通过订阅主臂关节状态话题,实现人工介入功能。
当介入模式开启时,返回主臂位姿作为控制指令;否则透传RL策略输出。

架构对齐 RLinf 的 Controller-State-Env 三层架构。
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np
import rospy
from pynput import keyboard
from sensor_msgs.msg import JointState

from rlinf.envs.realworld.common.ros import ROSController
from rlinf.utils.logging import get_logger


class MasterArmController:
    """主臂人在回路控制器。

    订阅主臂关节状态,提供人工介入功能。
    当介入模式开启时,返回主臂位姿作为控制指令;
    否则透传RL策略输出。

    ROS 话题映射(对应 start_ms_piper_double_agilex_delta_qpose.launch):
    - 订阅 `/master/joint_left`:  左主臂发布的关节位置 (JointState, 7D)
    - 订阅 `/master/joint_right`: 右主臂发布的关节位置 (JointState, 7D)

    Args:
        master_left_topic: 左主臂 ROS 话题名称。
        master_right_topic: 右主臂 ROS 话题名称。
        enable_keyboard_trigger: 是否启用键盘触发介入模式切换。
        trigger_key: 触发键名称,默认 'page_down'。
    """

    def __init__(
        self,
        master_left_topic: str = "/master/joint_left",
        master_right_topic: str = "/master/joint_right",
        enable_keyboard_trigger: bool = True,
        trigger_key: str = "page_down",
    ) -> None:
        self._logger = get_logger()
        self._master_left_topic = master_left_topic
        self._master_right_topic = master_right_topic
        self._enable_keyboard = enable_keyboard_trigger
        self._trigger_key = trigger_key

        # ---- 介入状态 ----
        self._intervention_enabled: bool = False
        self._state_lock = threading.Lock()

        # ---- 主臂关节位置缓存 (线程安全) ----
        # 每条主臂 7D: 6 关节 + 1 夹爪
        self._master_left_position: Optional[np.ndarray] = None
        self._master_right_position: Optional[np.ndarray] = None
        self._master_data_lock = threading.Lock()

        # ---- ROS 控制器 ----
        self._ros = ROSController()

        # ---- 订阅主臂话题 ----
        self._init_ros_subscriptions()

        # ---- 启动键盘监听 ----
        if self._enable_keyboard:
            self._keyboard_thread = threading.Thread(
                target=self._keyboard_listener_thread, daemon=True
            )
            self._keyboard_thread.start()
            self._logger.info(
                f"MasterArmController: 键盘监听已启动, 按 '{self._trigger_key}' 切换介入模式"
            )

        self._logger.info(
            f"MasterArmController 初始化完成: "
            f"left={master_left_topic}, right={master_right_topic}"
        )

    # ==================================================================
    # ROS 订阅初始化
    # ==================================================================

    def _init_ros_subscriptions(self) -> None:
        """初始化主臂关节状态订阅。

        对齐 piper_start_ms_node_double_agilex_dyn_pos.py 中的话题结构。
        """
        self._ros.connect_ros_channel(
            self._master_left_topic,
            JointState,
            self._on_master_left_joint_state,
        )
        self._ros.connect_ros_channel(
            self._master_right_topic,
            JointState,
            self._on_master_right_joint_state,
        )
        self._logger.info(
            f"订阅主臂话题: {self._master_left_topic}, {self._master_right_topic}"
        )

    # ==================================================================
    # ROS 回调: 主臂关节状态更新
    # ==================================================================

    def _on_master_left_joint_state(self, msg: JointState) -> None:
        """左主臂关节状态回调。

        从 JointState 消息中提取 position[0:7] (6 关节 + 1 夹爪)。
        对齐 piper_start_ms_node_double_agilex_dyn_pos.py 第238-277行的数据格式。

        Args:
            msg: sensor_msgs/JointState 消息。
        """
        if len(msg.position) < 7:
            self._logger.warning(
                f"左主臂 JointState position 长度不足: {len(msg.position)} < 7"
            )
            return

        position = np.array(msg.position[:7], dtype=np.float64)
        # 应用关节限位 (对齐 piper_start_ms_node_double_agilex_dyn_pos.py 第262-273行)
        position = self._apply_joint_limits(position)

        with self._master_data_lock:
            self._master_left_position = position

    def _on_master_right_joint_state(self, msg: JointState) -> None:
        """右主臂关节状态回调。

        Args:
            msg: sensor_msgs/JointState 消息。
        """
        if len(msg.position) < 7:
            self._logger.warning(
                f"右主臂 JointState position 长度不足: {len(msg.position)} < 7"
            )
            return

        position = np.array(msg.position[:7], dtype=np.float64)
        position = self._apply_joint_limits(position)

        with self._master_data_lock:
            self._master_right_position = position

    # ==================================================================
    # 关节限位
    # ==================================================================

    def _apply_joint_limits(self, position: np.ndarray) -> np.ndarray:
        """应用 Piper 机械臂关节限位。

        对齐 piper_start_ms_node_double_agilex_dyn_pos.py 第262-273行的限位逻辑:
        - joint_0: [-2.618, 2.618]
        - joint_1: [0, 3.14]
        - joint_2: [-2.967, 0]
        - joint_3: [-1.745, 1.745]
        - joint_4: [-1.22, 1.22]
        - joint_5: [-2.0944, 2.0944]
        - joint_6: [0, 0.08] (夹爪, 单位 m)

        Args:
            position: 7 维关节位置数组 (rad/m)。

        Returns:
            限位后的关节位置。
        """
        limits_low = np.array(
            [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0],
            dtype=np.float64,
        )
        limits_high = np.array(
            [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 0.08],
            dtype=np.float64,
        )
        return np.clip(position, limits_low, limits_high)

    # ==================================================================
    # 键盘监听
    # ==================================================================

    def _keyboard_listener_thread(self) -> None:
        """键盘监听线程。

        对齐 piper_start_ms_node_double_agilex_dyn_pos.py 第146-185行的键盘监听实现。
        使用 pynput 库监听键盘事件,按 page_down 切换介入模式。
        """

        def on_press(key):
            """键盘按键回调。"""
            try:
                # 提取按键字符串
                try:
                    key_str = key.char
                except AttributeError:
                    key_str = str(key).replace("Key.", "")

                if key_str == self._trigger_key:
                    # 切换介入状态
                    with self._state_lock:
                        self._intervention_enabled = not self._intervention_enabled
                        state_str = "ENABLED" if self._intervention_enabled else "DISABLED"
                        self._logger.info(f"人工介入模式: {state_str}")

            except Exception as e:
                self._logger.error(f"键盘按键处理异常: {e}")

        # 创建监听器
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        # 保持线程运行
        while not rospy.is_shutdown():
            time.sleep(1)

        listener.stop()

    # ==================================================================
    # 公共接口
    # ==================================================================

    def get_intervention_state(self) -> bool:
        """获取当前是否处于人工介入模式。

        Returns:
            True 表示介入模式已启用,False 表示使用 RL 策略。
        """
        with self._state_lock:
            return self._intervention_enabled

    def set_intervention_state(self, enabled: bool) -> None:
        """设置人工介入模式状态。

        Args:
            enabled: True 启用介入模式,False 禁用。
        """
        with self._state_lock:
            self._intervention_enabled = enabled
            state_str = "ENABLED" if enabled else "DISABLED"
            self._logger.info(f"人工介入模式: {state_str}")

    def get_master_action(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取主臂的最新关节位置。

        Returns:
            (left_action, right_action) 元组,每个为 7D numpy 数组。
            如果主臂数据尚未接收,返回 (None, None)。
        """
        with self._master_data_lock:
            left = self._master_left_position.copy() if self._master_left_position is not None else None
            right = self._master_right_position.copy() if self._master_right_position is not None else None
        return left, right

    def is_master_data_ready(self) -> bool:
        """检查主臂数据是否已就绪。

        Returns:
            True 表示左右主臂数据均已接收。
        """
        with self._master_data_lock:
            return (
                self._master_left_position is not None
                and self._master_right_position is not None
            )

    def blend_action(self, policy_action: np.ndarray) -> np.ndarray:
        """根据介入状态返回最终 action。

        - 介入模式: 返回主臂位姿 (如果数据就绪)
        - RL 模式: 返回策略输出

        Args:
            policy_action: RL 策略输出的 14D 动作 (左臂 7D + 右臂 7D)。

        Returns:
            最终的 14D 动作数组。
        """
        policy_action = np.asarray(policy_action, dtype=np.float64)

        # 检查是否处于介入模式
        if not self.get_intervention_state():
            return policy_action

        # 介入模式: 使用主臂数据
        left_master, right_master = self.get_master_action()

        if left_master is None or right_master is None:
            self._logger.warning(
                "介入模式已启用,但主臂数据尚未就绪,使用策略输出。"
            )
            return policy_action

        # 拼接主臂动作 (左 7D + 右 7D = 14D)
        master_action = np.concatenate([left_master, right_master])
        return master_action

    # ==================================================================
    # 上层接口
    # ==================================================================

    def wait_for_master_data(self, timeout: float = 10.0, poll_interval: float = 0.1) -> bool:
        """阻塞等待主臂数据就绪。

        Args:
            timeout: 最大等待时间 (秒)。
            poll_interval: 轮询间隔 (秒)。

        Returns:
            True 表示数据就绪,False 表示超时。
        """
        start_time = time.time()
        while not self.is_master_data_ready():
            if time.time() - start_time > timeout:
                self._logger.warning(
                    f"等待主臂数据超时 ({timeout}s)。"
                    f" 左主臂={self._master_left_position is not None},"
                    f" 右主臂={self._master_right_position is not None}"
                )
                return False
            time.sleep(poll_interval)
        self._logger.info("主臂数据已就绪。")
        return True
