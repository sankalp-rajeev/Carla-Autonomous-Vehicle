import glob
import os
import sys
import time
import numpy as np
import cv2
import math
import random
import argparse
from ultralytics import YOLO

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


# Configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CAMERA_FOV = 90

class CarlaSensor:
    def __init__(self, world, blueprint_library, vehicle):
        self.world = world
        self.vehicle = vehicle
        
        # Set up RGB camera
        self.camera_rgb = None
        self.camera_rgb_image = None
        self.setup_camera(blueprint_library)

    def setup_camera(self, blueprint_library):
        # RGB Camera setup
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))
        
        # Camera spawn location - above the vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_rgb = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        
        # Set up the callback
        self.camera_rgb.listen(lambda image: self.process_rgb_image(image))

    def process_rgb_image(self, image):
        # Convert carla.Image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        self.camera_rgb_image = array

    def get_camera_image(self):
        return self.camera_rgb_image
    
    def destroy(self):
        if self.camera_rgb:
            self.camera_rgb.destroy()


class YoloDetector:
    def __init__(self, model_path=None):
        # Load YOLO model
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Use pretrained YOLOv8 model
            self.model = YOLO("yolov8n.pt")  # Load the small model variant
    
    def detect(self, image):
        if image is None:
            return None
        
        # Run YOLOv8 inference on the image
        results = self.model(image)
        
        # Get detection results
        return results[0]
    
    def draw_detections(self, image, results):
        if image is None or results is None:
            return image
        
        # Make a copy of the image
        annotated_image = image.copy()
        
        # Get boxes, confidence scores, and class IDs
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Draw bounding boxes and labels
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            class_name = results.names[class_id]
            
            # Set color based on class
            if class_name == 'person':
                color = (0, 255, 0)  # Green for pedestrians
            elif class_name == 'car' or class_name == 'truck' or class_name == 'bus':
                color = (0, 0, 255)  # Blue for vehicles
            else:
                color = (255, 0, 0)  # Red for other objects
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image


def spawn_npc_vehicles(world, traffic_manager, num_vehicles=30):
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    
    # Limit number of vehicles to number of spawn points
    num_vehicles = min(num_vehicles, len(spawn_points))
    
    # Spawn vehicles
    vehicle_actors = []
    for i in range(num_vehicles):
        blueprint = random.choice(vehicle_blueprints)
        
        # Set autopilot
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        
        vehicle = world.try_spawn_actor(blueprint, spawn_points[i])
        if vehicle:
            vehicle_actors.append(vehicle)
            vehicle.set_autopilot(True, traffic_manager.get_port())
            
    print(f'Spawned {len(vehicle_actors)} vehicles')
    return vehicle_actors


def spawn_npc_pedestrians(world, client, num_pedestrians=15):
    # Get pedestrian blueprints
    pedestrian_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    
    # Get spawn points
    spawn_points = []
    for i in range(num_pedestrians):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    
    # Spawn pedestrian actors
    pedestrians = []
    pedestrian_controllers = []
    
    # Create pedestrians and their controllers
    for i, spawn_point in enumerate(spawn_points):
        ped_bp = random.choice(pedestrian_blueprints)
        
        # Set pedestrian attributes
        if ped_bp.has_attribute('is_invincible'):
            ped_bp.set_attribute('is_invincible', 'false')
        
        pedestrian = world.try_spawn_actor(ped_bp, spawn_point)
        
        if pedestrian:
            controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(float(1 + random.random()))
            
            pedestrians.append(pedestrian)
            pedestrian_controllers.append(controller)
    
    print(f'Spawned {len(pedestrians)} pedestrians')
    return pedestrians, pedestrian_controllers


def main():
    argparser = argparse.ArgumentParser(description='CARLA Object Detection using YOLOv8')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the CARLA server')
    argparser.add_argument('--port', default=2000, type=int, help='Port of the CARLA server')
    argparser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    argparser.add_argument('--async', dest='sync', action='store_false', help='Enable asynchronous mode')
    argparser.add_argument('--model', default=None, type=str, help='Path to YOLO model')
    # Fix: Use the parser instance to set defaults, not the module
    argparser.set_defaults(sync=True)
    args = argparser.parse_args()

    # Connect to CARLA server
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Traffic Manager setup for controlling NPC vehicles
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(args.sync)
    
    # Set synchronous mode
    if args.sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
    
    try:
        # Get blueprint library and spawn points
        blueprint_library = world.get_blueprint_library()
        
        # Spawn ego vehicle (the one with camera)
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'ego')
        spawn_points = world.get_map().get_spawn_points()
        ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
        
        # Setup autopilot for ego vehicle
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        
        # Spawn NPC vehicles and pedestrians
        npc_vehicles = spawn_npc_vehicles(world, traffic_manager, num_vehicles=30)
        npc_pedestrians, pedestrian_controllers = spawn_npc_pedestrians(world, client, num_pedestrians=15)
        
        # Setup camera sensor
        sensor = CarlaSensor(world, blueprint_library, ego_vehicle)
        
        # Setup YOLO detector
        detector = YoloDetector(args.model)
        
        # Main loop
        cv2.namedWindow('CARLA YOLOv8 Object Detection', cv2.WINDOW_AUTOSIZE)
        
        while True:
            # Tick world if in synchronous mode
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()
            
            # Get camera image
            camera_image = sensor.get_camera_image()
            
            if camera_image is not None:
                # Perform object detection
                results = detector.detect(camera_image)
                
                # Draw detections
                if results is not None:
                    annotated_image = detector.draw_detections(camera_image, results)
                    cv2.imshow('CARLA YOLOv8 Object Detection', annotated_image)
                else:
                    cv2.imshow('CARLA YOLOv8 Object Detection', camera_image)
            
            # Break loop on 'ESC' key press
            if cv2.waitKey(1) == 27:
                break
    
    finally:
        # Clean up
        cv2.destroyAllWindows()
        
        if 'sensor' in locals():
            sensor.destroy()
        
        if 'npc_vehicles' in locals():
            print('Destroying NPC vehicles')
            client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicles])
        
        if 'npc_pedestrians' in locals() and 'pedestrian_controllers' in locals():
            print('Destroying NPC pedestrians and controllers')
            for controller in pedestrian_controllers:
                controller.stop()
            client.apply_batch([carla.command.DestroyActor(x) for x in pedestrian_controllers])
            client.apply_batch([carla.command.DestroyActor(x) for x in npc_pedestrians])
        
        if 'ego_vehicle' in locals():
            ego_vehicle.destroy()
        
        # Reset world settings
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')