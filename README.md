
# AI-Powered Adaptive Cruise Control 

This project implements an end-to-end Adaptive Cruise Control (ACC) system in the CARLA simulator, combining Reinforcement Learning (PPO), semantic perception (CNN), real-time lane detection (UFLD), and object detection (YOLOv8).
> Note:  **This project was built on the foundation of [vadim7s/SelfDrive](https://github.com/vadim7s/SelfDrive) and extended to include Adaptive Cruise Control (ACC) logic using reinforcement learning and perception modules.**


---

## Project Structure

- `train_safe_final.py`: Trains the PPO agent with CNN-encoded semantic features and custom reward shaping.
- `evaluate_RL.py`: Evaluates the trained PPO model using custom metrics like steering jerk, throttle jerk, and following-distance error. Also generates plots.
- `inference_final.py`: Runs real-time inference with object and lane detection overlays.
- `environment_final.py`: Custom CARLA Gym-compatible environment.
- `find_best_model_final.py`: Searches training logs to locate the best-performing saved PPO model.
- `model_saved_from_CNN.h5`: Pretrained CNN model for extracting semantic features.
- `requirements.txt`: List of required Python packages and dependencies.

---

##  Setup Instructions

1. **Install CARLA 0.9.15** and ensure it runs properly.
2. **Install required Python packages:**

```bash
pip install -r requirements.txt
```

3. **Ensure pretrained models exist:**
   - `model_saved_from_CNN.h5` – semantic feature extractor
   - `yolov8n.pt` – object detector
   - `culane_18.pth` – UFLD lane detection weights

---

## Run Training

```bash
python train_safe_final.py
```

Models and TensorBoard logs will be saved to `models/<timestamp>` and `logs/<timestamp>`.

---

## Evaluate Trained RL Agent

```bash
python evaluate_RL.py
```

Generates:
- Following-distance error
- Steering & throttle jerk
- Steering variability per episode
- Matplotlib plots

---

## Run Full Inference with Perception Modules

```bash
python inference_final.py
```

This launches:
- RL agent control loop
- YOLOv8 vehicle/pedestrian detection
- UFLD ego-lane boundary visualization

Use `q` to quit the inference window.

---

##  Find Best Model from Logs

```bash
python find_best_model_final.py
```

Scans TensorBoard logs and locates the model checkpoint closest to the best reward.

---
## Results

Below are qualitative outputs from various components of the system:

### YOLOv8 – Object Detection

<img src="carla_simulation code/output_yolo.jpg" alt="YOLOv8 object detection example" width="600"/>

> Vehicles and pedestrians detected using pretrained YOLOv8n (COCO) weights.

---

###  UFLD – Lane Detection

<img src="carla_simulation code/output_UFLD.png" alt="UFLD lane detection output" width="600"/>

> Ego-lane boundaries predicted using Ultra-Fast Lane Detection (ResNet-18 pretrained on CULane).

---

### Full System Demo

[[Watch the demo]](https://www.youtube.com/watch?v=lP6mK7VfUtQ)

---

Screenshots are located in the `screenshots/` directory. Replace these placeholders with your actual images

## Folder Structure

```
project/
├── models/                # Saved PPO model checkpoints
├── logs/                  # TensorBoard logs for training
├── yolov8n.pt             # YOLO object detection weights (place here)
├── culane_18.pth          # UFLD pretrained weights (optional)
├── model_saved_from_CNN.h5
├── environment_final.py
├── evaluate_RL.py
├── inference_final.py
├── train_safe_final.py
├── find_best_model_final.py
├── requirements.txt
```

---

##  Acknowledgements

- **CARLA Simulator** – https://carla.org/
- **YOLOv8** – https://github.com/ultralytics/ultralytics
- **Ultra-Fast Lane Detection** – https://github.com/cfzd/Ultra-Fast-Lane-Detection
- **CNN Feature Extractor** – https://github.com/vadim7s/SelfDrive

---

## 💡 Authors

- **Sankalp Rajeev** – Lane detection + RL steering integration
- **Shlok Sharma** – PPO + throttle control + reward tuning
- **Kavan Kumareshan** – Object detection + braking logic

---

Enjoy your ride! 
