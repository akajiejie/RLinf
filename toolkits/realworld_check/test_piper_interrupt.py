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

"""Test script for Piper human-in-the-loop intervention.

This script tests the master-slave arm intervention functionality:
- PiperEnv initialization with MasterArmController
- Reading master arm joint states (delta qpos mode)
- Dual keyboard control: policy enable + teleoperation intervention
- Action replacement logic with priority system
- Data flow validation: qpos/qvel/effort/frames

Prerequisites:
    1. ROS master should be running: roscore
    2. Piper ROS master-slave nodes should be launched:
       roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch

Usage:
    python test_piper_interrupt.py

Controls:
    page_down  - Toggle teleoperation (遥操作 on/off)
    page_up    - Toggle policy output (策略输出 on/off)
    Ctrl+C     - Quit

Control Priority (highest to lowest):
    1. Teleoperation ON + master data ready: action = current_qpos + master_delta
    2. Policy ON: action = policy_action (sinusoidal motion)
    3. Both OFF: action = current_qpos (hold position, no movement)

Test Flow:
    1. Initialize PiperEnv with intervention enabled
    2. Reset environment
    3. Loop:
       - Generate policy action (small sinusoidal motion at 1Hz)
       - Check intervention & policy state
       - Blend action based on priority
       - Step environment
       - Display status (qpos, mode states, action source)

State Matrix:
    | Policy | Teleoperation | Result                                    |
    |--------|---------------|-------------------------------------------|
    | ON     | OFF           | Sinusoidal motion (policy)                |
    | OFF    | OFF           | Hold position (no movement)               |
    | ON     | ON            | Human control (policy blocked)            |
    | OFF    | ON            | Human control                             |
"""

import os
import signal
import sys
import time
from datetime import datetime

import numpy as np

# Add RLinf to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Add piper_ros devel Python packages to path
piper_ros_python_path = "/workspace/code/piper_ros/devel/lib/python3/dist-packages"
if os.path.exists(piper_ros_python_path) and piper_ros_python_path not in sys.path:
    sys.path.insert(0, piper_ros_python_path)

from rlinf.envs.realworld.piper import PiperEnv, PiperRobotConfig

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for graceful shutdown."""
    global shutdown_requested
    print("\n\n[SIGNAL] Ctrl+C detected, initiating graceful shutdown...")
    shutdown_requested = True


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_header(title):
    """Print a formatted header."""
    print_separator()
    print(f" {title}")
    print_separator()


def print_status(step, intervention_enabled, policy_enabled, master_ready, qpos, action_source, master_delta=None):
    """Print current status information.
    
    Args:
        step: Current step number.
        intervention_enabled: Whether teleoperation intervention is enabled.
        policy_enabled: Whether policy output is enabled.
        master_ready: Whether master arm data is ready.
        qpos: Current joint positions (14D).
        action_source: Source of action.
        master_delta: Master arm delta values (14D), optional.
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print(f"\n[{timestamp}] Step {step:04d}")
    print(f"  Teleoperation: {'ON ' if intervention_enabled else 'OFF'} | "
          f"Policy: {'ON ' if policy_enabled else 'OFF'} | "
          f"Action: {action_source}")
    print(f"  Master Data Ready: {master_ready}")
    
    # Print joint positions (left arm)
    print(f"  Left  Arm qpos: {qpos[:7]}")
    # Print joint positions (right arm)
    print(f"  Right Arm qpos: {qpos[7:14]}")
    
    # Print master delta if in intervention mode
    if master_delta is not None and intervention_enabled:
        print(f"  Master Delta (L): {master_delta[:7]}")
        print(f"  Master Delta (R): {master_delta[7:14]}")


def generate_policy_action(step, base_qpos, amplitude=0.05, frequency=1.0):
    """Generate a sinusoidal policy action for testing.
    
    Creates small periodic motions at 1Hz to simulate policy output.
    
    Args:
        step: Current step number.
        base_qpos: Base joint positions (14D).
        amplitude: Motion amplitude (rad).
        frequency: Motion frequency (Hz).
    
    Returns:
        14D action array with sinusoidal motion.
    """
    # Time in seconds (assuming 30Hz control frequency)
    t = step / 30.0
    
    # Sinusoidal pattern at specified frequency
    delta = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Create action: base_qpos + small sinusoidal delta
    action = base_qpos.copy()
    
    # Apply delta to first 3 joints of each arm (for smooth motion)
    action[0] += delta  # Left arm joint 0
    action[1] += delta * 0.5  # Left arm joint 1
    action[7] += delta  # Right arm joint 0
    action[8] += delta * 0.5  # Right arm joint 1
    
    return action


def main():
    """Main test function."""
    print_header("Piper Human-in-the-Loop Intervention Test")
    
    # Check environment variables
    print("\nChecking ROS environment...")
    if "ROS_MASTER_URI" not in os.environ:
        print("WARNING: ROS_MASTER_URI not set. Using default: http://localhost:11311")
        os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
    
    print(f"  ROS_MASTER_URI: {os.environ.get('ROS_MASTER_URI')}")
    
    # Configuration
    config = PiperRobotConfig(
        # ROS namespaces (matching launch file)
        ns_left="/puppet_left",
        ns_right="/puppet_right",
        
        # Human intervention settings
        enable_human_intervention=True,
        master_left_topic="/master/joint_left",
        master_right_topic="/master/joint_right",
        intervention_trigger_key="page_down",
        
        # Camera settings (optional for this test)
        camera_names=["cam_high"],
        img_topics=["/camera_f/color/image_raw"],
        
        # Control settings
        step_frequency=30.0,  # 30Hz control loop
        joint_speed_pct=50,
        
        # Environment settings
        is_dummy=False,  # Real robot mode
        max_num_steps=10000,
        
        # Joint limits (per arm: 6 joints + 1 gripper)
        min_qpos=[-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0],
        max_qpos=[2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 0.08],
    )
    
    print("\nInitializing PiperEnv...")
    print(f"  Control frequency: {config.step_frequency} Hz")
    print(f"  Intervention enabled: {config.enable_human_intervention}")
    print(f"  Trigger key: {config.intervention_trigger_key}")
    
    # Initialize environment
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
        print("  2. Piper launch file is running:")
        print("     roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch")
        return
    
    print("\n✓ Environment initialized successfully!")
    
    # Check master controller
    if env._master_controller is None:
        print("\nERROR: MasterArmController not initialized!")
        print("Please check enable_human_intervention=True in config.")
        return
    
    print("\n✓ MasterArmController initialized")
    print(f"  Subscribed to: {config.master_left_topic}, {config.master_right_topic}")
    
    # Reset environment
    print("\nResetting environment...")
    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"\nERROR: Failed to reset environment: {e}")
        return
    
    print("✓ Environment reset successful")
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  State keys: {list(obs['state'].keys())}")
    print(f"  qpos shape: {obs['state']['qpos'].shape}")
    print(f"  qvel shape: {obs['state']['qvel'].shape}")
    print(f"  effort shape: {obs['state']['effort'].shape}")
    
    # Get initial qpos for policy action generation
    initial_qpos = obs['state']['qpos'].copy()
    print(f"\nInitial qpos: {initial_qpos}")
    
    # Instructions
    print_header("Test Instructions")
    print("  Initial State: Policy=ON, Teleoperation=OFF")
    print("  Robot executes sinusoidal motions (1Hz) from policy")
    print()
    print("  Keyboard Controls:")
    print("    [page_up]   - Toggle policy output (ON/OFF)")
    print("                  OFF: robot holds position, policy blocked")
    print("    [page_down] - Toggle teleoperation (ON/OFF)")
    print("                  ON: human controls via master arms (overrides policy)")
    print("    [Ctrl+C]    - Quit program")
    print()
    print("  Control Matrix:")
    print("    Policy=ON,  Teleoperation=OFF  -> Sinusoidal motion (policy)")
    print("    Policy=OFF, Teleoperation=OFF  -> Hold position (no movement)")
    print("    Policy=ON,  Teleoperation=ON   -> Human control (policy blocked)")
    print("    Policy=OFF, Teleoperation=ON   -> Human control")
    print_separator()
    
    print("\nStarting test in 3 seconds...")
    time.sleep(3)
    
    # Test loop
    step = 0
    paused = False
    
    print("\n" + "=" * 80)
    print("Starting test loop... (Press Ctrl+C to stop)")
    print("=" * 80)
    
    try:
        while not shutdown_requested:
            loop_start_time = time.time()
            
            # Generate policy action (small sinusoidal motion at 1Hz)
            policy_action = generate_policy_action(
                step=step,
                base_qpos=initial_qpos,
                amplitude=0.05,  # 0.05 rad amplitude (~2.9 degrees)
                frequency=1.0,   # 1 Hz
            )
            
            # Step environment
            # The env.step() will internally check intervention & policy state
            # and blend action based on priority system
            obs, reward, terminated, truncated, info = env.step(policy_action)
            
            # Get intervention and policy state
            intervention_enabled = env._master_controller.get_intervention_state()
            policy_enabled = env._master_controller.get_policy_enabled()
            master_ready = env._master_controller.is_master_data_ready()
            
            # Determine action source based on priority
            if intervention_enabled and master_ready:
                action_source = "TELEOPERATION (master delta)"
            elif policy_enabled:
                action_source = "POLICY (sinusoidal)"
            else:
                action_source = "HOLD (current_qpos)"
            
            # Print status every 10 steps (to avoid flooding console)
            if step % 10 == 0:
                qpos = obs['state']['qpos']
                
                # Get master delta if in intervention mode
                master_delta = None
                if intervention_enabled and master_ready:
                    left_delta, right_delta = env._master_controller.get_master_action()
                    master_delta = np.concatenate([left_delta, right_delta])
                
                print_status(step, intervention_enabled, policy_enabled, master_ready, qpos, action_source, master_delta)
            
            # Check for termination
            if terminated or truncated:
                print(f"\n{'!' * 80}")
                print(f"Episode terminated at step {step}")
                print(f"  Terminated: {terminated}, Truncated: {truncated}")
                print(f"{'!' * 80}\n")
                
                print("Resetting environment...")
                obs, info = env.reset()
                initial_qpos = obs['state']['qpos'].copy()
                step = 0
                continue
            
            step += 1
            
            # Maintain control frequency (30Hz -> ~33ms per loop)
            loop_elapsed = time.time() - loop_start_time
            sleep_time = max(0, (1.0 / config.step_frequency) - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\n[EXCEPTION] Test interrupted by KeyboardInterrupt")
    except Exception as e:
        print(f"\n\nERROR during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            env.close()
            print("✓ Environment closed successfully")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        
        print_header("Test Summary")
        print(f"  Total steps executed: {step}")
        if 'intervention_enabled' in locals() and 'policy_enabled' in locals():
            print(f"  Final teleoperation state: {intervention_enabled}")
            print(f"  Final policy state: {policy_enabled}")
            print(f"  Master data ready: {master_ready if 'master_ready' in locals() else 'Unknown'}")
        print_separator()
        print("\nTest completed.")


if __name__ == "__main__":
    main()
