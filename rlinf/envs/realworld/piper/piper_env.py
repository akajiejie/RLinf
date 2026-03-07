@dataclass
class PiperRobotConfig:
    use_robot_base: bool = False
    publish_rate: int = 30
    pos_lookahead_step: int = 50
    chunk_size: int = 50
    task_name: str = "task"
    is_dummy: bool = False
    max_num_steps: int = 10000
    # ZMQ 推理服务地址（对应你现有的 RobotInferenceClient）
    inference_host: str = "127.0.0.1"
    inference_port: int = 8080
    min_qpos: list = field(default_factory=lambda: [-2.618,0.0,-2.967,-1.745,-1.22,-2.0944,0,0])
    max_qpos: list = field(default_factory=lambda: [2.618,3.14,0.0,1.745,1.22,2.0944,1,1])

class PiperEnv(gym.Env):
    def __init__(self, config, worker_info, hardware_info, env_idx):
        self.config = config
        self._state = PiperRobotState()
        self._controller = PiperController(use_robot_base=config.use_robot_base)
        self._init_action_obs_spaces()  # 14/16维关节空间

    def step(self, action):
        # action: [14] joint delta 或绝对位置
        left_action, right_action = action[:7], action[7:14].
        # clip 关节限位（你现有代码的 min_qpos/max_qpos 逻辑）
        self._controller.move_arm(left_action, right_action)
        self._state = self._controller.get_state()
        obs = self._get_observation()
        reward = self._calc_step_reward()
        ...

    def _get_observation(self):
        return {
            "state": {"qpos": self._state.qpos, "qvel": self._state.qvel, ...},
            "frames": {"cam_high": ..., "cam_left_wrist": ..., "cam_right_wrist": ...}
        }