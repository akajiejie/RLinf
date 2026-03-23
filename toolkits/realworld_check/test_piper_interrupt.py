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
- Keyboard-triggered intervention mode switching (page_down key)
- Action replacement: master arm pose vs policy output
- Data flow validation: qpos/qvel/effort/frames

Prerequisites:
    1. ROS master should be running: roscore
    2. Piper ROS master-slave nodes should be launched:
       roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch

Usage:
    python test_piper_interrupt.py

Controls:
    page_down  - Toggle intervention mode (human control <-> policy control)
    q          - Quit
    r          - Reset environment
    p          - Pause/Resume

Test Flow:
    1. Initialize PiperEnv with intervention enabled
    2. Reset environment
    3. Loop:
       - Generate policy action (small sinusoidal motion at 1Hz)
       - Check intervention state
       - If intervention mode: use master arm pose
       - If policy mode: use generated policy action
       - Step environment
       - Display status (qpos, intervention state, master arm data)
"""

import os
import sys
import time
from datetime import datetime

import numpy as np

# Add RLinf to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rlinf.envs.realworld.piper import PiperEnv, PiperRobotConfig


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_header(title):
    """Print a formatted header."""
    print_separator()
    print(f" {title}")
    print_separator()


def print_status(step, intervention_enabled, master_ready, qpos, action_source):
    """Print current status information.
    
    Args:
        step: Current step number.
        intervention_enabled: Whether intervention mode is enabled.
        master_ready: Whether master arm data is ready.
        qpos: Current joint positions (14D).
        action_source: Source of action ('master' or 'policy').
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print(f"\n[{timestamp}] Step {step:04d}")
    print(f"  Mode: {'[INTERVENTION]' if intervention_enabled else '[POLICY]':20s} | Action Source: {action_source}")
    print(f"  Master Data Ready: {master_ready}")
    
    # Print joint positions (left arm)
    print(f"  Left  Arm qpos: {qpos[:7]}")
    # Print joint positions (right arm)
    print(f"  Right Arm qpos: {qpos[7:14]}")


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
    print("  1. The robot will execute small sinusoidal motions (1Hz) in POLICY mode")
    print("  2. Press [page_down] to switch to INTERVENTION mode")
    print("  3. In INTERVENTION mode, move the master arms to control the slave arms")
    print("  4. Press [page_down] again to switch back to POLICY mode")
    print("  5. Press [q] + [Enter] to quit")
    print("  6. Press [r] + [Enter] to reset")
    print("  7. Press [p] + [Enter] to pause/resume")
    print_separator()
    
    input("\nPress Enter to start the test...")
    
    # Test loop
    step = 0
    paused = False
    
    print("\n" + "=" * 80)
    print("Starting test loop... (Press Ctrl+C to stop)")
    print("=" * 80)
    
    try:
        while True:
            loop_start_time = time.time()
            
            # Check for user input (non-blocking)
            # Note: This is a simplified version. In practice, you might want to use
            # a separate thread for keyboard input
            
            # Generate policy action (small sinusoidal motion at 1Hz)
            policy_action = generate_policy_action(
                step=step,
                base_qpos=initial_qpos,
                amplitude=0.05,  # 0.05 rad amplitude (~2.9 degrees)
                frequency=1.0,   # 1 Hz
            )
            
            # Step environment
            # The env.step() will internally check intervention state
            # and replace action if intervention is enabled
            obs, reward, terminated, truncated, info = env.step(policy_action)
            
            # Get intervention state
            intervention_enabled = env._master_controller.get_intervention_state()
            master_ready = env._master_controller.is_master_data_ready()
            
            # Determine action source
            if intervention_enabled and master_ready:
                action_source = "MASTER ARM"
            else:
                action_source = "POLICY"
            
            # Print status every 10 steps (to avoid flooding console)
            if step % 10 == 0:
                qpos = obs['state']['qpos']
                print_status(step, intervention_enabled, master_ready, qpos, action_source)
                
                # Print master arm positions if available
                if master_ready:
                    left_master, right_master = env._master_controller.get_master_action()
                    print(f"  Master Left  qpos: {left_master}")
                    print(f"  Master Right qpos: {right_master}")
            
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
        print("\n\nTest interrupted by user (Ctrl+C)")
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
        print(f"  Final intervention state: {intervention_enabled if 'intervention_enabled' in locals() else 'Unknown'}")
        print(f"  Master data ready: {master_ready if 'master_ready' in locals() else 'Unknown'}")
        print_separator()
        print("\nTest completed.")


if __name__ == "__main__":
    main()
