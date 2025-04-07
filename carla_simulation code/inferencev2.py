"""
Improved Inference Script for Trained PPO Model in CARLA Environment with Lane Detection

- Shows a single window with the lane detection overlay.
- Implements smooth logging and environment reset on episode end.
- Uses a try-finally block to guarantee proper cleanup.
- Reduces penalties for minor lane invasions via a modified lane_data handler.

Make sure the CARLA simulator is running before executing this script.
Ensure the Ultra-Fast-Lane-Detection project (and its configs) is in your PYTHONPATH.
"""

import time
import cv2
import numpy as np
import carla
from stable_baselines3 import PPO
from environment import CarEnv
from lane_detection import LaneDetector

def run_inference(model_path, duration=600, render=True):
    # Initialize RL environment (CarEnv) and modify lane invasion behavior.
    env = CarEnv()
    
    # Modify lane invasion handling to allow minor violations without resetting.
    env.lane_invade_hist = []  # Clear any existing history
    
    original_lane_data = env.lane_data
    def modified_lane_data(event):
        # If the event has crossed_lane_markings, only treat solid lines as critical.
        if hasattr(event, 'crossed_lane_markings'):
            for lane_marking in event.crossed_lane_markings:
                if lane_marking.type == carla.LaneMarkingType.Solid:
                    original_lane_data(event)
                    return
        # For other lane markings, log and ignore.
        print("[INFO] Minor lane marking crossed – self-correction allowed")
    env.lane_data = modified_lane_data

    # Reset the environment to spawn vehicle and attach sensors.
    obs, _ = env.reset()
    
    # Load the trained PPO model.
    model = PPO.load(model_path, env=env, verbose=1)
    
    # Initialize the lane detector with the absolute weights path.
    lane_detector = LaneDetector(
        weights_path=r"C:\Umich\Winter 2025\ECE 579\Final Project\Ultra-Fast-Lane-Detection\culane_18.pth",
        use_gpu=True
    )
    
    print("[INFO] Starting inference episode with lane detection...")
    start_time = time.time()
    step_count = 0
    total_reward = 0
    
    try:
        # Run until the specified duration is reached.
        while time.time() - start_time < duration:
            # Get an action from the model deterministically.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Reset environment if episode is done.
            if done:
                print(f"[INFO] Episode ended at step {step_count} with reward {total_reward:.2f}. Resetting environment...")
                obs, _ = env.reset()
                # Reset timer and counters if desired (or break out if only one episode is needed).
                start_time = time.time()
                step_count = 0
                total_reward = 0
                continue

            # Process lane detection overlay if a valid front camera frame is available.
            if env.front_camera is not None and render:
                try:
                    lane_overlay = lane_detector.detect_lanes(env.front_camera)
                    cv2.imshow("Self-Driving with Lane Detection", lane_overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"[ERROR] Lane detection failed: {e}")
            
            # Small delay to allow sensor updates.
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[INFO] Inference interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected error during inference: {e}")
    finally:
        print(f"[INFO] Episode finished after {step_count} steps with total reward: {total_reward:.2f}")
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "models/20250406-004342/final_model.zip"  # Update with your model path if needed
    run_inference(model_path, duration=600, render=True)
