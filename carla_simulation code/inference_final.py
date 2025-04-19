import time
import cv2
import numpy as np
import carla
import torch
import traceback
from stable_baselines3 import PPO
from environment_final import CarEnv
from lane_detection import LaneDetector
from gymnasium import spaces
from pathlib import Path

class YOLODetector:
    """YOLOv8 detector for object detection in CARLA"""
    def __init__(self, model_path='yolov8n.pt', confidence=0.5):
        try:
            # Check if the model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                print(f"[ERROR] YOLO model file not found at: {model_path}")
                self.initialized = False
                return
            
            # Load YOLOv8 model using ultralytics
            try:
                # First attempt with ultralytics import
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[INFO] YOLO detector initialized using ultralytics package: {model_path}")
            except ImportError:
                # Fallback to torch hub if ultralytics package not installed
                print("[INFO] Ultralytics package not found, trying torch hub...")
                self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, trust_repo=True)
                print(f"[INFO] YOLO detector initialized using torch hub: {model_path}")
            
            # Set confidence threshold
            self.conf = confidence
            
            # Use GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[INFO] YOLO detector initialized using {self.device}")
            self.initialized = True
        except Exception as e:
            print(f"[ERROR] Failed to initialize YOLO detector: {e}")
            traceback.print_exc()
            self.initialized = False

    def detect(self, image):
        """Run inference on the provided image"""
        if not self.initialized:
            return None
        
        try:
            # Convert to RGB if needed (YOLO expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already 3 channels, make sure it's RGB and not BGR
                if isinstance(image, np.ndarray):
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image
            else:
                print("[WARNING] Image format incorrect for YOLO detection")
                return None
            
            # Run inference with confidence threshold
            results = self.model(rgb_image, conf=self.conf)
            return results
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            traceback.print_exc()
            return None

    def process_detections(self, results, image):
        """Process detection results and return annotated image"""
        if results is None or not self.initialized:
            return image
        
        try:
            # Get annotated image with bounding boxes
            annotated_img = results[0].plot()
            
            # Process detection results for ACC functionality
            try:
                # Extract detection information
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get class name (COCO dataset)
                        cls_name = result.names[cls] if hasattr(result, 'names') else f"class_{cls}"
                        
                        # Add to detections list
                        detections.append({
                            'class': cls,
                            'name': cls_name,
                            'confidence': conf,
                            'xmin': x1,
                            'ymin': y1,
                            'xmax': x2,
                            'ymax': y2,
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        })
                
                # Count detections by class
                class_counts = {}
                for d in detections:
                    cls_name = d['name']
                    if cls_name in class_counts:
                        class_counts[cls_name] += 1
                    else:
                        class_counts[cls_name] = 1
                        
                if class_counts:
                    print(f"[DETECTED] {class_counts}")
                
                # Find vehicles that could be in our lane (near center of image)
                vehicles = [d for d in detections if d['name'] in ['car', 'truck', 'bus']]
                if vehicles:
                    # Sort by width (larger width = closer vehicle)
                    vehicles.sort(key=lambda x: x['height'], reverse=True)
                    
                    # Get center of image
                    center_x = image.shape[1] / 2
                    
                    # Find vehicles near center (potentially in our lane)
                    center_vehicles = [v for v in vehicles 
                                    if abs(v['center_x'] - center_x) < image.shape[1] * 0.2]
                    
                    if center_vehicles:
                        # Get the closest vehicle (largest in frame)
                        closest_vehicle = center_vehicles[0]
                        
                        # Calculate distance based on bounding box height
                        # This is a simple heuristic - in real applications, calibrate this
                        relative_size = closest_vehicle['height'] / image.shape[0]
                        est_distance = max(5, 30 * (1 - relative_size))
                        
                        print(f"[ACC] Estimated distance to vehicle ahead: {est_distance:.2f}m")
                        
                        # Annotate on image
                        cv2.putText(
                            annotated_img,
                            f"Distance: {est_distance:.1f}m",
                            (int(closest_vehicle['xmin']), int(closest_vehicle['ymin'] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                        )
            except Exception as e:
                print(f"[ERROR] Failed to process detection details: {e}")
            
            return annotated_img
            
        except Exception as e:
            print(f"[ERROR] Failed to process detections: {e}")
            traceback.print_exc()
            return image

def run_inference(model_path, yolo_weights_path, duration=600, render=True):
    try:
        # Initialize RL environment
        env = CarEnv()
        
        # Patch action space to match trained model
        env.action_space = spaces.MultiDiscrete([9, 3])

        # Modify lane invasion handling
        env.lane_invade_hist = []
        original_lane_data = env.lane_data
        
        def modified_lane_data(event):
            if hasattr(event, 'crossed_lane_markings'):
                for lane_marking in event.crossed_lane_markings:
                    if lane_marking.type == carla.LaneMarkingType.Solid:
                        return original_lane_data(event)
                print("[INFO] Minor lane marking crossed - self-correction allowed")
        env.lane_data = modified_lane_data

        print("[INFO] Initializing environment...")
        obs, _ = env.reset()

        # Load RL model
        model = PPO.load(model_path, env=env, verbose=1)

        # Initialize lane detector
        lane_detector = None
        try:
            lane_detector = LaneDetector(
                weights_path=r"C:\Users\mynam\Downloads\Um_dearborn\Intelligent_Systems_ECE_579\Project\CARLA\Carla-Autonomous-Vehicle\Ultra-Fast-Lane-Detection\culane_18.pth",
                use_gpu=False
            )
            print("[INFO] Lane detector initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Lane detector initialization failed: {e}")
            traceback.print_exc()

        # Initialize YOLO object detector with proper path handling
        full_yolo_path = str(Path(yolo_weights_path).resolve())
        print(f"[INFO] Loading YOLO model from: {full_yolo_path}")
        
        yolo_detector = YOLODetector(model_path=full_yolo_path, confidence=0.45)
        if not yolo_detector.initialized:
            print("[WARNING] YOLO detector initialization failed. Continuing without object detection.")

        print("[INFO] Starting inference episode with lane and object detection...")
        start_time = time.time()
        step_count = 0
        total_reward = 0

        # For display
        window_name = "Self-Driving with Lane and Object Detection"
        if render:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)

        while time.time() - start_time < duration:
            try:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1

                # Log distance to lead vehicle from environment
                if hasattr(env, 'lead_vehicle') and env.lead_vehicle:
                    try:
                        ego_loc = env.vehicle.get_location()
                        lead_loc = env.lead_vehicle.get_location()
                        dist = ego_loc.distance(lead_loc)
                        print(f"[DISTANCE] Ego-to-Lead: {dist:.2f} meters")
                    except Exception as e:
                        print(f"[ERROR] Distance check failed: {e}")

                # Render visualizations if enabled
                if render:
                    # Use RGB camera image for visualization and detection
                    if hasattr(env, 'rgb_camera_image') and env.rgb_camera_image is not None:
                        # Create a copy to avoid modifying the original
                        display_img = env.rgb_camera_image.copy()
                        
                        # Process with YOLO object detection
                        if yolo_detector.initialized:
                            try:
                                # Run detection
                                detection_results = yolo_detector.detect(display_img)
                                
                                # Process and annotate results
                                if detection_results is not None:
                                    display_img = yolo_detector.process_detections(detection_results, display_img)
                            except Exception as e:
                                print(f"[ERROR] YOLO detection failed: {e}")
                                traceback.print_exc()
                        
                        # Add lane detection overlay if available
                        if lane_detector:
                            try:
                                lane_overlay = lane_detector.detect_lanes(display_img)
                                if lane_overlay is not None:
                                    display_img = lane_overlay
                            except Exception as e:
                                print(f"[ERROR] Lane detection failed: {e}")
                        
                        # Add ACC information overlay
                        if hasattr(env, 'ACC_ACTIVE') and env.ACC_ACTIVE:
                            try:
                                # Get current speed
                                v = env.vehicle.get_velocity()
                                kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))
                                
                                # Get filtered distance from environment if available
                                distance_info = ""
                                if hasattr(env, 'FILTERED_DISTANCE'):
                                    distance_info = f"Distance: {env.FILTERED_DISTANCE:.1f}m"
                                
                                # Create ACC info text
                                acc_info = f"ACC: {kmh}km/h Target: {env.PREFERRED_SPEED}km/h {distance_info}"
                                
                                # Add text overlay on image
                                cv2.putText(
                                    display_img,
                                    acc_info,
                                    (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, 
                                    (0, 255, 0), 
                                    2
                                )
                            except Exception as e:
                                print(f"[ERROR] ACC overlay failed: {e}")
                        
                        # Show the image with all overlays
                        cv2.imshow(window_name, display_img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("[WARNING] No RGB camera image available for display")

                if done:
                    print(f"[INFO] Episode ended. Total reward: {total_reward:.2f}")
                    obs, _ = env.reset()
                    start_time = time.time()
                    step_count = 0
                    total_reward = 0

                # Small delay to keep the simulation running smoothly
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("[INFO] Inference interrupted by user.")
                break
            except Exception as e:
                print(f"[ERROR] Step failed: {e}")
                traceback.print_exc()
                break

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        traceback.print_exc()
    finally:
        print(f"[INFO] Finished after {step_count} steps. Total reward: {total_reward:.2f}")
        try:
            env.cleanup()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "models/20250415-233928"  
    
    # YOLOv8 weights file path - update with the correct path to your yolov8 weights
    yolo_weights_path = r"C:\Users\mynam\Downloads\Um_dearborn\Intelligent_Systems_ECE_579\Project\CARLA\Carla-Autonomous-Vehicle\Yolov8\yolov8n.pt"  # Change to your actual path if needed
    
    # Get absolute path for YOLO weights
    yolo_weights_path = str(Path(yolo_weights_path).resolve())
    
    print(f"Using YOLO weights from: {yolo_weights_path}")
    print(f"Using PPO model from: {model_path}")
    
    # Run inference with object detection
    run_inference(model_path, yolo_weights_path, duration=600, render=True)