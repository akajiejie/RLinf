from rlinf.envs.realworld.common.ros import ROSController
from .piper_robot_state import PiperRobotState
# ... 把 RosOperator 的订阅/发布逻辑迁移进来

class PiperController:
    def __init__(self, use_robot_base=False):
        self._state = PiperRobotState()
        self._ros = ROSController()   # 自动管理 roscore
        self._init_ros_channels()     # 替代 RosOperator.init_ros()

    def get_state(self) -> PiperRobotState:
        return self._state

    def move_arm(self, left_action, right_action):
        # 对应 ros_operator.puppet_arm_publish()
        ...

    def is_robot_up(self) -> bool:
        # 检查关键 topic 是否有数据
        ...