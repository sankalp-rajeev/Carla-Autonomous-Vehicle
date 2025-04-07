import sys
import os
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Add the Ultra-Fast-Lane-Detection folders to the Python path
sys.path.append(os.path.abspath("../Ultra-Fast-Lane-Detection"))
sys.path.append(os.path.abspath("../Ultra-Fast-Lane-Detection/configs"))

from vehicle_setup import VehicleSetup  # ✅ Vehicle management
from model.model import parsingNet      # ✅ Ultra-Fast-Lane-Detection model
import culane as cfg                    # ✅ Dataset config (CULane)
from utils.visualization import visualize  # ✅ Lane visualization

# Show OpenCV windows?
SHOW_SENSOR_OUTPUT = True

lane_detector = None

class LanePostProcessor:
    def __init__(self, griding_num, num_lanes, row_anchor, img_w, img_h):
        self.griding_num = griding_num
        self.num_lanes = num_lanes
        # Scale row anchors to match image height
        self.row_anchor = [int(anchor / max(row_anchor) * img_h) for anchor in row_anchor]
        self.img_w = img_w
        self.img_h = img_h

    def decode(self, output):
        lanes = []
        for lane_idx in range(self.num_lanes):
            lane_points = []
            for r, grid_pos in enumerate(output[lane_idx]):
                if grid_pos <= 0:
                    lane_points.append(None)
                    continue
                grid_pos = min(grid_pos, self.griding_num)
                x = int(grid_pos * self.img_w / self.griding_num)
                y = int(self.row_anchor[r])
                lane_points.append((x, y))
            lanes.append(lane_points)
        return lanes

class LaneDetector:
    def __init__(self, weights_path='../Ultra-Fast-Lane-Detection/culane_18.pth', use_gpu=True):
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        print(f"[INFO] Using device: {self.device}")

        griding_num = cfg.griding_num
        num_lanes = cfg.num_lanes
        row_anchor = cfg.row_anchor
        cls_dim = (griding_num + 1, len(row_anchor), num_lanes)
        
        self.model = parsingNet(
            size=(288, 800),
            pretrained=False,
            backbone='18',
            cls_dim=cls_dim,
            use_aux=False
        )

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])

        self.model.to(self.device)
        self.model.eval()

        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.lane_postprocessor = LanePostProcessor(
            griding_num=griding_num,
            num_lanes=num_lanes,
            row_anchor=row_anchor,
            img_w=800,
            img_h=288
        )

        print("[INFO] Lane detector initialized successfully.")

    def detect_lanes(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        input_tensor = self.img_transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        cls_out = output.cpu().numpy()  

        # Check if there is a batch dimension (expected shape: (batch, 201, 18, 4))
        if cls_out.ndim == 4:
            # Remove the batch dimension
            cls_out = cls_out[0]  # Now shape should be (201, 18, 4)
            
        # Verify the shape
        # print("Shape of cls_out before transpose:", cls_out.shape)

        # Transpose to get (num_lanes, num_rows, griding_num)
        # Your desired shape is (4, 18, 201) since:
        #   - axis 0: num_lanes (4)
        #   - axis 1: num_row_anchors (18)
        #   - axis 2: griding_num+1 (201)
        if cls_out.ndim == 3 and cls_out.shape[0] == 201:
            # If cls_out is (201, 18, 4), transpose to (4, 18, 201)
            cls_out = np.transpose(cls_out, (2, 1, 0))
            # print("Shape of cls_out after transpose:", cls_out.shape)
        else:
            # Otherwise, adjust according to the actual shape or skip transposition
            print("Unexpected shape, adjust the axes for transpose accordingly.")

        # Now perform argmax on the griding dimension (last axis)
        out_loc = np.argmax(cls_out, axis=2)  # Expected shape: (4, 18)

        decoded_lanes = self.lane_postprocessor.decode(out_loc)

        # print(f"[DEBUG] Decoded lanes (first lane): {decoded_lanes[0]}")

        lanes_overlay = visualize(image_bgr, decoded_lanes, num_lanes=cfg.num_lanes)

        return lanes_overlay


def lane_detection_callback(image, name):
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img_array.reshape((image.height, image.width, 4))[:, :, :3]

    lanes_img = lane_detector.detect_lanes(img)

    if SHOW_SENSOR_OUTPUT:
        cv2.imshow(f'{name}_lanes', lanes_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exit requested by user (q key).")
            setup.shutdown()
            exit(0)

if __name__ == "__main__":
    try:
        print("[INFO] Initializing Lane Detection Pipeline...")

        setup = VehicleSetup()
        lane_detector = LaneDetector()

        setup.spawn_vehicle()
        setup.attach_sensors(camera_callback=lane_detection_callback)
        setup.set_spectator_camera()

        print("[INFO] Lane detection running. Press Ctrl+C to stop...")

        while True:
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt received. Shutting down...")

    finally:
        setup.shutdown()
        print("[INFO] Lane detection pipeline stopped.")