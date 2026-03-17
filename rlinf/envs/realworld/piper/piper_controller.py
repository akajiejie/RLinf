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

"""Piper 双臂机械臂控制器。

通过 ROSController 与 piper_ros 节点通信，实现：
- 订阅双臂关节状态 (``/puppet/joint_left``, ``/puppet/joint_right``)
- 订阅末端位姿 (``end_pose``, ``end_pose_euler``)
- 发布关节控制指令 (``/puppet/joint_left``, ``/puppet/joint_right``)
- 调用使能/夹爪/归零等 ROS 服务

架构对齐 ``rlinf.envs.realworld.franka.franka_controller.FrankaController``。
"""

import time
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from rlinf.envs.realworld.common.ros import ROSController
from rlinf.utils.logging import get_logger

from .piper_robot_state import PiperRobotState
from .utils import clip_gripper, clip_joint_positions


class PiperController:
    """Piper 双臂机械臂控制器。

    负责通过 ROS 话题和服务与 piper_ros 节点通信。
    对齐 ``collect_data_jeff.py`` 中 ``RosOperator`` 的数据流，
    并遵循 ``FrankaController`` 的架构模式。

    ROS 话题映射（双臂场景，每条臂各自运行一个 piper_ctrl_single_node）：

    **订阅（状态读取）：**

    - ``{ns_left}/joint_states_single``  → 左臂 JointState (7: 6关节+1夹爪)
    - ``{ns_right}/joint_states_single`` → 右臂 JointState
    - ``{ns_left}/end_pose``             → 左臂末端位姿 PoseStamped (四元数)
    - ``{ns_right}/end_pose``            → 右臂末端位姿 PoseStamped (四元数)
    - ``{ns_left}/end_pose_euler``       → 左臂末端欧拉角位姿
    - ``{ns_right}/end_pose_euler``      → 右臂末端欧拉角位姿
    - ``/odom`` (可选)                   → 底盘里程计

    **发布（指令下发）：**

    - ``{ns_left}/joint_ctrl_single``    → 左臂关节控制 JointState
    - ``{ns_right}/joint_ctrl_single``   → 右臂关节控制 JointState

    **服务调用：**

    - ``{ns}/enable_srv``   → 使能/去使能
    - ``{ns}/gripper_srv``  → 夹爪控制
    - ``{ns}/go_zero_srv``  → 归零
    - ``{ns}/stop_srv``     → 急停
    - ``{ns}/reset_srv``    → 复位

    Args:
        ns_left: 左臂 ROS 命名空间，默认 ``/puppet_left``。
        ns_right: 右臂 ROS 命名空间，默认 ``/puppet_right``。
        use_robot_base: 是否订阅底盘里程计话题。
        robot_base_topic: 底盘里程计话题名称。
        joint_speed_pct: 关节运动速度百分比 (0-100)，对应 MotionCtrl_2 的 speed 参数。
    """

    def __init__(
        self,
        ns_left: str = "/puppet_left",
        ns_right: str = "/puppet_right",
        use_robot_base: bool = False,
        robot_base_topic: str = "/odom",
        joint_speed_pct: int = 50,
    ) -> None:
        self._logger = get_logger()
        self._ns_left = ns_left
        self._ns_right = ns_right
        self._use_robot_base = use_robot_base
        self._robot_base_topic = robot_base_topic
        self._joint_speed_pct = joint_speed_pct

        # ---- 双臂状态 ----
        self._state_left = PiperRobotState()
        self._state_right = PiperRobotState()

        # ---- 底盘速度 ----
        self._base_vel: np.ndarray = np.zeros(2, dtype=np.float64)  # [linear.x, angular.z]

        # ---- ROS 控制器 ----
        self._ros = ROSController()

        # ---- 初始化所有 ROS 通道 ----
        self._init_ros_channels()

        self._logger.info(
            f"PiperController initialized: left={ns_left}, right={ns_right}, "
            f"base={use_robot_base}"
        )

    # ==================================================================
    # ROS 通道初始化
    # ==================================================================

    def _init_ros_channels(self) -> None:
        """初始化所有 ROS 订阅和发布通道。

        对齐 ``collect_data_jeff.py`` 中 ``RosOperator.init_ros()`` 的话题列表，
        以及 ``piper_ctrl_single_node.py`` 中的发布/订阅话题。
        """
        # ---- 话题名称定义 ----
        # 订阅：状态读取（piper_ctrl_single_node 发布的话题）
        # 适配 start_ms_piper.launch 的 topic 结构
        if self._ns_left == "/puppet_left" and self._ns_right == "/puppet_right":
            # 主从模式 (start_ms_piper.launch)
            self._left_joint_state_topic = "/puppet/joint_left"
            self._right_joint_state_topic = "/puppet/joint_right"
            self._left_end_pose_topic = "/puppet/end_pose_left"
            self._right_end_pose_topic = "/puppet/end_pose_right"
            self._left_joint_ctrl_topic = "/master/joint_left"
            self._right_joint_ctrl_topic = "/master/joint_right"
        else:
            # 单臂节点模式
            self._left_joint_state_topic = f"{self._ns_left}/joint_states_single"
            self._right_joint_state_topic = f"{self._ns_right}/joint_states_single"
            self._left_end_pose_topic = f"{self._ns_left}/end_pose"
            self._right_end_pose_topic = f"{self._ns_right}/end_pose"
            self._left_joint_ctrl_topic = f"{self._ns_left}/joint_ctrl_single"
            self._right_joint_ctrl_topic = f"{self._ns_right}/joint_ctrl_single"

        # ---- 订阅：左臂状态 ----
        self._ros.connect_ros_channel(
            self._left_joint_state_topic,
            JointState,
            self._on_left_joint_state,
        )
        self._ros.connect_ros_channel(
            self._left_end_pose_topic,
            PoseStamped,
            self._on_left_end_pose,
        )

        # ---- 订阅：右臂状态 ----
        self._ros.connect_ros_channel(
            self._right_joint_state_topic,
            JointState,
            self._on_right_joint_state,
        )
        self._ros.connect_ros_channel(
            self._right_end_pose_topic,
            PoseStamped,
            self._on_right_end_pose,
        )

        # ---- 订阅：底盘里程计（可选） ----
        if self._use_robot_base:
            self._ros.connect_ros_channel(
                self._robot_base_topic,
                Odometry,
                self._on_robot_base,
            )

        # ---- 发布：关节控制指令 ----
        self._ros.create_ros_channel(
            self._left_joint_ctrl_topic, JointState, queue_size=1
        )
        self._ros.create_ros_channel(
            self._right_joint_ctrl_topic, JointState, queue_size=1
        )

    # ==================================================================
    # ROS 回调：状态更新（线程安全，由 ROS Subscriber 线程调用）
    # ==================================================================

    def _on_left_joint_state(self, msg: JointState) -> None:
        """左臂关节状态回调。

        piper_ctrl_single_node 以 200Hz 发布 joint_states_single，
        包含 7 个值：6 关节 + 1 夹爪 (position/velocity/effort)。

        Args:
            msg: sensor_msgs/JointState 消息。
        """
        self._state_left.update_joint_state(
            position=np.array(msg.position, dtype=np.float64),
            velocity=np.array(msg.velocity, dtype=np.float64),
            effort=np.array(msg.effort, dtype=np.float64),
        )

    def _on_right_joint_state(self, msg: JointState) -> None:
        """右臂关节状态回调。

        Args:
            msg: sensor_msgs/JointState 消息。
        """
        self._state_right.update_joint_state(
            position=np.array(msg.position, dtype=np.float64),
            velocity=np.array(msg.velocity, dtype=np.float64),
            effort=np.array(msg.effort, dtype=np.float64),
        )

    def _on_left_end_pose(self, msg: PoseStamped) -> None:
        """左臂末端位姿回调（四元数）。

        piper_ctrl_single_node 以 200Hz 发布 end_pose (PoseStamped)，
        其中位置单位为米，姿态为四元数 (qx, qy, qz, qw)。

        Args:
            msg: geometry_msgs/PoseStamped 消息。
        """
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64)
        self._state_left.update_tcp_pose(pos, quat)

    def _on_right_end_pose(self, msg: PoseStamped) -> None:
        """右臂末端位姿回调（四元数）。

        Args:
            msg: geometry_msgs/PoseStamped 消息。
        """
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64)
        self._state_right.update_tcp_pose(pos, quat)

    def _on_robot_base(self, msg: Odometry) -> None:
        """底盘里程计回调。

        对应 ``collect_data_jeff.py`` 第 352 行:
        ``[robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]``

        Args:
            msg: nav_msgs/Odometry 消息。
        """
        self._base_vel = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.angular.z],
            dtype=np.float64,
        )

    # ==================================================================
    # 状态获取接口
    # ==================================================================

    def get_left_state(self) -> PiperRobotState:
        """获取左臂当前状态。

        Returns:
            左臂 PiperRobotState 实例。
        """
        return self._state_left

    def get_right_state(self) -> PiperRobotState:
        """获取右臂当前状态。

        Returns:
            右臂 PiperRobotState 实例。
        """
        return self._state_right

    def get_base_vel(self) -> np.ndarray:
        """获取底盘速度。

        Returns:
            [linear.x, angular.z] 数组。
        """
        return self._base_vel.copy()

    def get_qpos(self) -> np.ndarray:
        """获取双臂拼接的 qpos (14维)。

        对应 ``collect_data_jeff.py`` 第 348 行:
        ``np.concatenate((puppet_arm_left.position, puppet_arm_right.position))``

        每条臂 7 维: 6 关节位置 + 1 夹爪位置。

        Returns:
            长度为 14 的 numpy 数组。
        """
        left_snap = self._state_left.snapshot()
        right_snap = self._state_right.snapshot()
        left_qpos = np.append(left_snap["arm_joint_position"], left_snap["gripper_position"])
        right_qpos = np.append(right_snap["arm_joint_position"], right_snap["gripper_position"])
        return np.concatenate([left_qpos, right_qpos])

    def get_qvel(self) -> np.ndarray:
        """获取双臂拼接的 qvel (14维)。

        对应 ``collect_data_jeff.py`` 第 349 行。

        Returns:
            长度为 14 的 numpy 数组。
        """
        left_snap = self._state_left.snapshot()
        right_snap = self._state_right.snapshot()
        # 夹爪速度在 piper_ros 中为 0.0
        left_qvel = np.append(left_snap["arm_joint_velocity"], 0.0)
        right_qvel = np.append(right_snap["arm_joint_velocity"], 0.0)
        return np.concatenate([left_qvel, right_qvel])

    def get_effort(self) -> np.ndarray:
        """获取双臂拼接的 effort (14维)。

        对应 ``collect_data_jeff.py`` 第 350 行。

        Returns:
            长度为 14 的 numpy 数组。
        """
        left_snap = self._state_left.snapshot()
        right_snap = self._state_right.snapshot()
        left_effort = np.append(left_snap["arm_joint_effort"], left_snap["gripper_effort"])
        right_effort = np.append(right_snap["arm_joint_effort"], right_snap["gripper_effort"])
        return np.concatenate([left_effort, right_effort])

    # ==================================================================
    # 控制指令发布
    # ==================================================================

    def move_arm(
        self,
        left_action: np.ndarray,
        right_action: np.ndarray,
        left_speed_pct: Optional[int] = None,
        right_speed_pct: Optional[int] = None,
    ) -> None:
        """发布双臂关节控制指令。

        对应 ``piper_ctrl_single_node.py`` 中 ``joint_callback`` 的接收逻辑:
        - ``position[0:6]``: 6 个关节角度 (rad)
        - ``position[6]``: 夹爪角度 (rad)
        - ``velocity[6]``: 全局速度百分比 (0-100)，可选
        - ``effort[6]``: 夹爪力矩 (N·m)，可选

        Args:
            left_action: 左臂动作，长度为 7 (6 关节 + 1 夹爪)，单位 rad。
            right_action: 右臂动作，长度为 7 (6 关节 + 1 夹爪)，单位 rad。
            left_speed_pct: 左臂速度百分比 (0-100)，None 使用默认值。
            right_speed_pct: 右臂速度百分比 (0-100)，None 使用默认值。
        """
        left_action = np.asarray(left_action, dtype=np.float64)
        right_action = np.asarray(right_action, dtype=np.float64)

        assert len(left_action) == 7, (
            f"左臂动作维度应为 7，实际为 {len(left_action)}"
        )
        assert len(right_action) == 7, (
            f"右臂动作维度应为 7，实际为 {len(right_action)}"
        )

        # 关节限位裁剪
        left_action[:6] = clip_joint_positions(left_action[:6])
        right_action[:6] = clip_joint_positions(right_action[:6])
        left_action[6] = clip_gripper(left_action[6])
        right_action[6] = clip_gripper(right_action[6])

        self._publish_joint_ctrl(
            self._left_joint_ctrl_topic,
            left_action,
            left_speed_pct or self._joint_speed_pct,
        )
        self._publish_joint_ctrl(
            self._right_joint_ctrl_topic,
            right_action,
            right_speed_pct or self._joint_speed_pct,
        )

    def move_left_arm(
        self, action: np.ndarray, speed_pct: Optional[int] = None
    ) -> None:
        """发布左臂关节控制指令。

        Args:
            action: 长度为 7 的动作数组 (6 关节 + 1 夹爪)，单位 rad。
            speed_pct: 速度百分比 (0-100)，None 使用默认值。
        """
        action = np.asarray(action, dtype=np.float64)
        assert len(action) == 7, f"动作维度应为 7，实际为 {len(action)}"
        action[:6] = clip_joint_positions(action[:6])
        action[6] = clip_gripper(action[6])
        self._publish_joint_ctrl(
            self._left_joint_ctrl_topic,
            action,
            speed_pct or self._joint_speed_pct,
        )

    def move_right_arm(
        self, action: np.ndarray, speed_pct: Optional[int] = None
    ) -> None:
        """发布右臂关节控制指令。

        Args:
            action: 长度为 7 的动作数组 (6 关节 + 1 夹爪)，单位 rad。
            speed_pct: 速度百分比 (0-100)，None 使用默认值。
        """
        action = np.asarray(action, dtype=np.float64)
        assert len(action) == 7, f"动作维度应为 7，实际为 {len(action)}"
        action[:6] = clip_joint_positions(action[:6])
        action[6] = clip_gripper(action[6])
        self._publish_joint_ctrl(
            self._right_joint_ctrl_topic,
            action,
            speed_pct or self._joint_speed_pct,
        )

    def _publish_joint_ctrl(
        self, topic: str, action: np.ndarray, speed_pct: int
    ) -> None:
        """构造 JointState 消息并发布到指定话题。

        消息格式对齐 ``piper_ctrl_single_node.py`` 中 ``joint_callback`` 的期望:
        - ``position``: [j1, j2, j3, j4, j5, j6, gripper] (rad)
        - ``velocity``: [0, 0, 0, 0, 0, 0, speed_pct] — 第 7 个元素为全局速度百分比
        - ``effort``:   [0, 0, 0, 0, 0, 0, gripper_effort] — 夹爪力矩 (N·m)

        Args:
            topic: 发布的 ROS 话题名称。
            action: 长度为 7 的动作数组。
            speed_pct: 速度百分比 (0-100)。
        """
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6",
            "gripper",
        ]
        msg.position = action.tolist()
        # velocity[6] = 全局速度百分比，piper_ctrl_single_node 会读取此值
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(speed_pct)]
        # effort[6] = 夹爪力矩，默认 1.0 N·m
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        self._ros.put_channel(topic, msg)

    # ==================================================================
    # ROS 服务调用
    # ==================================================================

    def enable_arm(self, ns: Optional[str] = None) -> bool:
        """使能机械臂。

        调用 piper_ros 的 ``enable_srv`` 服务。

        Args:
            ns: 指定命名空间，None 则同时使能左右臂。

        Returns:
            是否使能成功。
        """
        from piper_msgs.srv import Enable

        namespaces = [ns] if ns else [self._ns_left, self._ns_right]
        all_ok = True
        for arm_ns in namespaces:
            service_name = f"{arm_ns}/enable_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                enable_proxy = rospy.ServiceProxy(service_name, Enable)
                resp = enable_proxy(enable_request=True)
                if not resp.enable_response:
                    self._logger.warning(f"使能失败: {service_name}")
                    all_ok = False
                else:
                    self._logger.info(f"使能成功: {service_name}")
            except (rospy.ServiceException, rospy.ROSException) as e:
                self._logger.error(f"使能服务调用异常 {service_name}: {e}")
                all_ok = False
        return all_ok

    def disable_arm(self, ns: Optional[str] = None) -> bool:
        """去使能机械臂。

        Args:
            ns: 指定命名空间，None 则同时去使能左右臂。

        Returns:
            是否去使能成功。
        """
        from piper_msgs.srv import Enable

        namespaces = [ns] if ns else [self._ns_left, self._ns_right]
        all_ok = True
        for arm_ns in namespaces:
            service_name = f"{arm_ns}/enable_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                enable_proxy = rospy.ServiceProxy(service_name, Enable)
                resp = enable_proxy(enable_request=False)
                if not resp.enable_response:
                    self._logger.warning(f"去使能失败: {service_name}")
                    all_ok = False
                else:
                    self._logger.info(f"去使能成功: {service_name}")
            except (rospy.ServiceException, rospy.ROSException) as e:
                self._logger.error(f"去使能服务调用异常 {service_name}: {e}")
                all_ok = False
        return all_ok

    def gripper_ctrl(
        self,
        gripper_angle: float = 0.0,
        gripper_effort: float = 1.0,
        gripper_code: int = 0x01,
        set_zero: int = 0x00,
        ns: Optional[str] = None,
    ) -> bool:
        """控制夹爪。

        调用 piper_ros 的 ``gripper_srv`` 服务。

        Args:
            gripper_angle: 夹爪开合角度 (m)，范围 [0, 0.07]。
            gripper_effort: 夹爪力矩 (N·m)，范围 [0.5, 2.0]。
            gripper_code: 夹爪控制码 (0x00=Disable, 0x01=Enable, 0x02/0x03=清错)。
            set_zero: 归零设置 (0x00=无效, 0xAE=设置零点)。
            ns: 指定命名空间，None 则同时控制左右臂夹爪。

        Returns:
            是否控制成功。
        """
        from piper_msgs.srv import Gripper

        namespaces = [ns] if ns else [self._ns_left, self._ns_right]
        all_ok = True
        for arm_ns in namespaces:
            service_name = f"{arm_ns}/gripper_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                gripper_proxy = rospy.ServiceProxy(service_name, Gripper)
                resp = gripper_proxy(
                    gripper_angle=gripper_angle,
                    gripper_effort=gripper_effort,
                    gripper_code=gripper_code,
                    set_zero=set_zero,
                )
                if not resp.status:
                    self._logger.warning(
                        f"夹爪控制失败: {service_name}, code={resp.code}"
                    )
                    all_ok = False
            except (rospy.ServiceException, rospy.ROSException) as e:
                self._logger.error(f"夹爪服务调用异常 {service_name}: {e}")
                all_ok = False
        return all_ok

    def go_zero(self, is_mit_mode: bool = False, ns: Optional[str] = None) -> bool:
        """机械臂归零。

        调用 piper_ros 的 ``go_zero_srv`` 服务。

        Args:
            is_mit_mode: 是否使用 MIT 模式归零。
            ns: 指定命名空间，None 则同时归零左右臂。

        Returns:
            是否归零成功。
        """
        from piper_msgs.srv import GoZero

        namespaces = [ns] if ns else [self._ns_left, self._ns_right]
        all_ok = True
        for arm_ns in namespaces:
            service_name = f"{arm_ns}/go_zero_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                go_zero_proxy = rospy.ServiceProxy(service_name, GoZero)
                resp = go_zero_proxy(is_mit_mode=is_mit_mode)
                if not resp.status:
                    self._logger.warning(
                        f"归零失败: {service_name}, code={resp.code}"
                    )
                    all_ok = False
                else:
                    self._logger.info(f"归零成功: {service_name}")
            except (rospy.ServiceException, rospy.ROSException) as e:
                self._logger.error(f"归零服务调用异常 {service_name}: {e}")
                all_ok = False
        return all_ok

    def stop_arm(self, ns: Optional[str] = None) -> bool:
        """急停机械臂。

        调用 piper_ros 的 ``stop_srv`` 服务 (std_srvs/Trigger)。

        Args:
            ns: 指定命名空间，None 则同时急停左右臂。

        Returns:
            是否急停成功。
        """
        from std_srvs.srv import Trigger

        namespaces = [ns] if ns else [self._ns_left, self._ns_right]
        all_ok = True
        for arm_ns in namespaces:
            service_name = f"{arm_ns}/stop_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                stop_proxy = rospy.ServiceProxy(service_name, Trigger)
                resp = stop_proxy()
                if not resp.success:
                    self._logger.warning(
                        f"急停失败: {service_name}, msg={resp.message}"
                    )
                    all_ok = False
                else:
                    self._logger.info(f"急停成功: {service_name}")
            except (rospy.ServiceException, rospy.ROSException) as e:
                self._logger.error(f"急停服务调用异常 {service_name}: {e}")
                all_ok = False
        return all_ok

    def reset_arm(self, ns: Optional[str] = None) -> bool:
        """复位机械臂（清除错误状态）。

        调用 piper_ros 的 ``reset_srv`` 服务 (std_srvs/Trigger)。

        Args:
            ns: 指定命名空间，None 则同时复位左右臂。

        Returns:
            是否复位成功。
        """
        from std_srvs.srv import Trigger

        namespaces = [ns] if ns else [self._ns_left, self._ns_right]
        all_ok = True
        for arm_ns in namespaces:
            service_name = f"{arm_ns}/reset_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                reset_proxy = rospy.ServiceProxy(service_name, Trigger)
                resp = reset_proxy()
                if not resp.success:
                    self._logger.warning(
                        f"复位失败: {service_name}, msg={resp.message}"
                    )
                    all_ok = False
                else:
                    self._logger.info(f"复位成功: {service_name}")
            except (rospy.ServiceException, rospy.ROSException) as e:
                self._logger.error(f"复位服务调用异常 {service_name}: {e}")
                all_ok = False
        return all_ok

    # ==================================================================
    # 上层接口
    # ==================================================================

    def is_robot_up(self) -> bool:
        """检查双臂关键 ROS 话题是否已有数据到达。

        对齐 ``FrankaController.is_robot_up()`` 的语义。

        Returns:
            True 表示左右臂的关节状态和末端位姿话题均已收到数据。
        """
        left_joint_ok = self._ros.get_input_channel_status(
            self._left_joint_state_topic
        )
        right_joint_ok = self._ros.get_input_channel_status(
            self._right_joint_state_topic
        )
        left_pose_ok = self._ros.get_input_channel_status(
            self._left_end_pose_topic
        )
        right_pose_ok = self._ros.get_input_channel_status(
            self._right_end_pose_topic
        )
        return left_joint_ok and right_joint_ok and left_pose_ok and right_pose_ok

    def wait_for_robot(self, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
        """阻塞等待机械臂就绪。

        Args:
            timeout: 最大等待时间 (秒)。
            poll_interval: 轮询间隔 (秒)。

        Returns:
            True 表示机械臂就绪，False 表示超时。
        """
        start_time = time.time()
        while not self.is_robot_up():
            if time.time() - start_time > timeout:
                self._logger.warning(
                    f"等待机械臂就绪超时 ({timeout}s)。"
                    f" 左关节={self._ros.get_input_channel_status(self._left_joint_state_topic)},"
                    f" 右关节={self._ros.get_input_channel_status(self._right_joint_state_topic)},"
                    f" 左位姿={self._ros.get_input_channel_status(self._left_end_pose_topic)},"
                    f" 右位姿={self._ros.get_input_channel_status(self._right_end_pose_topic)}"
                )
                return False
            time.sleep(poll_interval)
        self._logger.info("双臂机械臂就绪。")
        return True

    def _wait_for_joint(
        self,
        target_pos: np.ndarray,
        side: str = "left",
        timeout: float = 30.0,
        atol: float = 1e-2,
    ) -> bool:
        """等待指定臂到达目标关节位置。

        Args:
            target_pos: 目标关节位置，长度为 6 (rad)。
            side: ``"left"`` 或 ``"right"``。
            timeout: 最大等待时间 (秒)。
            atol: 位置容差 (rad)。

        Returns:
            True 表示到达目标，False 表示超时。
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        state = self._state_left if side == "left" else self._state_right
        wait_step = 0.01
        waited = 0.0
        while waited < timeout:
            current = state.snapshot()["arm_joint_position"]
            if np.allclose(current, target_pos, atol=atol):
                self._logger.debug(
                    f"{side} 臂到达目标位置: {current}"
                )
                return True
            time.sleep(wait_step)
            waited += wait_step
        self._logger.warning(f"{side} 臂等待关节位置超时 ({timeout}s)")
        return False
