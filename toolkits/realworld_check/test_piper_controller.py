# Copyright 2025 The RLinf Authors.
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

"""Test script for Piper controller.

This script tests the Piper dual-arm controller functionality:
- Connection to ROS topics
- Reading joint states and end-effector poses
- Human teleoperation (PiperEnv + MasterArmController), command ``teleop``

Prerequisites (basic commands: getpos, getjoint, home):
    1. ROS master should be running: roscore
    2. Piper ROS nodes should be launched:
       roslaunch piper start_ms_piper.launch

Prerequisites (teleop):
    1. roscore
    2. Piper ROS master-slave nodes (delta qpos mode), e.g.:
       roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch

Usage:
    python test_piper_controller.py

Commands:
    q              - Quit
    getpos         - Get current TCP pose (quaternion) for both arms
    getpos_euler   - Get current TCP pose (euler angles) for both arms
    getjoint       - Get current joint positions for both arms
    home           - Move both arms to home position (all zeros)
    teleop         - Test human teleoperation 
"""

import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation as R

# Add RLinf to path when running as a script from toolkits/realworld_check
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_piper_ros_python_path = "/workspace/code/piper_ros/devel/lib/python3/dist-packages"
if os.path.exists(_piper_ros_python_path) and _piper_ros_python_path not in sys.path:
    sys.path.insert(0, _piper_ros_python_path)

from rlinf.envs.realworld.piper import PiperEnv, PiperRobotConfig  # noqa: E402
from rlinf.envs.realworld.piper.piper_controller import PiperController  # noqa: E402

# Used only during ``teleop`` run for graceful Ctrl+C
_teleop_shutdown = False


def _teleop_signal_handler(signum, frame):
    global _teleop_shutdown
    print("\n\n[SIGNAL] Ctrl+C detected, exiting teleop...")
    _teleop_shutdown = True


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_status(step, intervention_enabled, master_ready, qpos, master_delta=None):
    """Print current teleop status."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    print(f"\n[{timestamp}] Step {step:04d}")
    print(f"  Teleoperation: {'ON ' if intervention_enabled else 'OFF'} | Master Data Ready: {master_ready}")

    print(f"  Left  Arm qpos: {qpos[:7]}")
    print(f"  Right Arm qpos: {qpos[7:14]}")

    if master_delta is not None and intervention_enabled:
        print(f"  Master Delta (L): {master_delta[:7]}")
        print(f"  Master Delta (R): {master_delta[7:14]}")


def run_teleop(ns_left: str, ns_right: str) -> None:
    """Human teleoperation test using PiperEnv (no policy/RL action)."""
    global _teleop_shutdown

    def print_header(title: str) -> None:
        print_separator()
        print(f" {title}")
        print_separator()

    print_header("Piper Teleoperation Test")
    print("\nChecking ROS environment...")
    if "ROS_MASTER_URI" not in os.environ:
        print("WARNING: ROS_MASTER_URI not set. Using default: http://localhost:11311")
        os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
    print(f"  ROS_MASTER_URI: {os.environ.get('ROS_MASTER_URI')}")

    config = PiperRobotConfig(
        ns_left=ns_left,
        ns_right=ns_right,
        enable_human_intervention=True,
        master_left_topic="/master/joint_left",
        master_right_topic="/master/joint_right",
        intervention_trigger_key="page_down",
        camera_names=["cam_high"],
        img_topics=["/camera_f/color/image_raw"],
        step_frequency=30.0,
        joint_speed_pct=50,
        is_dummy=False,
        max_num_steps=10000,
        min_qpos=[-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0],
        max_qpos=[2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 0.08],
    )

    print("\nInitializing PiperEnv for teleop...")
    print(f"  Control frequency: {config.step_frequency} Hz")
    print(f"  Intervention enabled: {config.enable_human_intervention}")
    print(f"  Trigger key: {config.intervention_trigger_key}")

    try:
        env = PiperEnv(
            config=config,
            worker_info=None,
            hardware_info=None,
            env_idx=0,
        )
    except Exception as e:
        print(f"\nERROR: Failed to initialize environment: {e}")
        print("\nPlease check:")
        print("  1. roscore is running")
        print("  2. Piper master-slave launch is running, e.g.:")
        print("     roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch")
        return

    if env._master_controller is None:
        print("\nERROR: MasterArmController not initialized.")
        print("Set enable_human_intervention=True in PiperRobotConfig.")
        try:
            env.close()
        except Exception:
            pass
        return

    print("\n✓ PiperEnv ready (MasterArmController active)")
    print(f"  Subscribed to: {config.master_left_topic}, {config.master_right_topic}")

    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"\nERROR: Failed to reset environment: {e}")
        try:
            env.close()
        except Exception:
            pass
        return

    # Disable policy action (teleop-only mode)
    env._master_controller.set_policy_enabled(False)

    print_header("Teleop Controls")
    print("  This mode tests teleoperation only (no policy/RL action).")
    print("  Initial: Teleoperation=OFF — robot holds current position.")
    print()
    print("  [page_down] - Toggle teleoperation ON/OFF")
    print("                ON:  robot follows master arm delta")
    print("                OFF: robot holds current position")
    print("  [Ctrl+C]    - Exit teleop and return to piper> prompt")
    print_separator()

    _teleop_shutdown = False
    old_sigint = signal.signal(signal.SIGINT, _teleop_signal_handler)

    step = 0
    try:
        print("\nStarting teleop in 3 seconds...")
        time.sleep(3)
        print("\n" + "=" * 80)
        print("Teleop loop running... (Ctrl+C to stop)")
        print("=" * 80)

        while not _teleop_shutdown:
            loop_start = time.time()

            # Placeholder action: current joint targets. blend_action ignores it while policy is off.
            action_in = np.asarray(obs["state"]["qpos"], dtype=np.float64).copy()
            obs, reward, terminated, truncated, info = env.step(action_in)

            intervention_enabled = env._master_controller.get_intervention_state()
            master_ready = env._master_controller.is_master_data_ready()

            if step % 10 == 0:
                qpos = obs["state"]["qpos"]
                master_delta = None
                if intervention_enabled and master_ready:
                    left_delta, right_delta = env._master_controller.get_master_action()
                    master_delta = np.concatenate([left_delta, right_delta])
                print_status(step, intervention_enabled, master_ready, qpos, master_delta)

            if terminated or truncated:
                print(f"\nEpisode ended at step {step} (terminated={terminated}, truncated={truncated})")
                obs, info = env.reset()
                env._master_controller.set_policy_enabled(False)
                step = 0
                continue

            step += 1
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / config.step_frequency) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[EXCEPTION] KeyboardInterrupt in teleop")
    except Exception as e:
        print(f"\nERROR during teleop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        _teleop_shutdown = False
        print("\nClosing teleop environment...")
        try:
            env.close()
            print("✓ Environment closed")
        except Exception as e:
            print(f"Warning during env.close(): {e}")

    print(f"Teleop finished after {step} steps.\n")


def print_help():
    """Print available commands."""
    print("\n" + "=" * 60)
    print("Available Commands:")
    print("=" * 60)
    print("  q              - Quit")
    print("  getpos         - Get current TCP pose (quaternion)")
    print("  getpos_euler   - Get current TCP pose (euler angles)")
    print("  getjoint       - Get current joint positions")
    print("  home           - Move both arms to home position (all zeros)")
    print("  teleop         - Test human teleoperation (needs delta-qpos master-slave launch)")
    print("  help           - Show this help message")
    print("=" * 60 + "\n")


def main():
    """Main test function."""
    ns_left = os.environ.get("PIPER_NS_LEFT", "/puppet_left")
    ns_right = os.environ.get("PIPER_NS_RIGHT", "/puppet_right")

    print("\n" + "=" * 60)
    print("Piper Controller Test")
    print("=" * 60)
    print(f"Left arm namespace:  {ns_left}")
    print(f"Right arm namespace: {ns_right}")
    print()
    print("Initializing controller...")

    controller = PiperController(
        ns_left=ns_left,
        ns_right=ns_right,
        use_robot_base=False,
        joint_speed_pct=50,
    )

    print("Waiting for Piper robot to be ready...")
    start_time = time.time()

    while not controller.is_robot_up():
        elapsed = time.time() - start_time
        if elapsed > 30:
            print(f"\n✗ ERROR: Robot not ready after {elapsed:.1f} seconds")
            print("Please check:")
            print("  1. Is roscore running?")
            print("  2. Is 'roslaunch piper start_ms_piper.launch' running?")
            print("  3. Are the ROS topics publishing?")
            print(f"     rostopic list | grep {ns_left}")
            print(f"     rostopic list | grep {ns_right}")
            return 1

        if int(elapsed) % 5 == 0 and elapsed > 0:
            print(f"  Still waiting... ({elapsed:.0f}s)")

        time.sleep(0.5)

    print(f"✓ Robot ready! (took {time.time() - start_time:.1f}s)\n")

    print_help()

    while True:
        try:
            cmd_str = input("piper> ").strip().lower()

            if cmd_str == "q" or cmd_str == "quit":
                print("Exiting...")
                break

            elif cmd_str == "help" or cmd_str == "h":
                print_help()

            elif cmd_str == "getpos":
                state_left = controller.get_left_state()
                state_right = controller.get_right_state()

                print("\nLeft arm TCP pose (x, y, z, qx, qy, qz, qw):")
                print(f"  {state_left.tcp_pose}")
                print("\nRight arm TCP pose (x, y, z, qx, qy, qz, qw):")
                print(f"  {state_right.tcp_pose}")

            elif cmd_str == "getpos_euler":
                state_left = controller.get_left_state()
                state_right = controller.get_right_state()

                tcp_pose_left = state_left.tcp_pose
                r_left = R.from_quat(tcp_pose_left[3:].copy())
                euler_left = r_left.as_euler("xyz")

                tcp_pose_right = state_right.tcp_pose
                r_right = R.from_quat(tcp_pose_right[3:].copy())
                euler_right = r_right.as_euler("xyz")

                print("\nLeft arm TCP pose (x, y, z, roll, pitch, yaw):")
                print(f"  {np.concatenate([tcp_pose_left[:3], euler_left])}")
                print("\nRight arm TCP pose (x, y, z, roll, pitch, yaw):")
                print(f"  {np.concatenate([tcp_pose_right[:3], euler_right])}")

            elif cmd_str == "getjoint":
                state_left = controller.get_left_state().snapshot()
                state_right = controller.get_right_state().snapshot()

                print("\nLeft arm joint positions (6 joints):")
                print(f"  Joints: {state_left['arm_joint_position']}")
                print(f"  Gripper: {state_left['gripper_position']}")
                print("\nRight arm joint positions (6 joints):")
                print(f"  Joints: {state_right['arm_joint_position']}")
                print(f"  Gripper: {state_right['gripper_position']}")

            elif cmd_str == "home":
                print("Moving both arms to home position (all zeros)...")
                home_joints = np.zeros(6)

                state_left = controller.get_left_state().snapshot()
                state_right = controller.get_right_state().snapshot()

                left_action = np.append(home_joints, state_left["gripper_position"])
                right_action = np.append(home_joints, state_right["gripper_position"])

                controller.move_arm(left_action, right_action)
                time.sleep(2.0)

                new_left = controller.get_left_state().snapshot()
                new_right = controller.get_right_state().snapshot()
                print(f"✓ Left arm:  {new_left['arm_joint_position']}")
                print(f"✓ Right arm: {new_right['arm_joint_position']}")

            elif cmd_str == "teleop":
                run_teleop(ns_left, ns_right)

            elif cmd_str == "":
                continue

            else:
                print(f"Unknown command: '{cmd_str}'")
                print("Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n✗ Error executing command: {e}")
            import traceback

            traceback.print_exc()

    print("\nTest completed.")
    return 0


if __name__ == "__main__":
    exit(main())
