import carla
import numpy as np
import cv2
import time
from vehicle_setup import VehicleSetup  

MAP_SIZE_PIXELS = 5000
MAP_SIZE_METERS = 200  
PIXELS_PER_METER = MAP_SIZE_PIXELS / MAP_SIZE_METERS

SHOW_MAP = True

class SimpleLiDARSLAM:
    def __init__(self):
        self.map = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=np.uint8)
        self.origin = MAP_SIZE_PIXELS // 2, MAP_SIZE_PIXELS // 2

    def world_to_map(self, x, y):
        """Converts world coordinates to map pixel coordinates."""
        mx = int(self.origin[0] + x * PIXELS_PER_METER)
        my = int(self.origin[1] - y * PIXELS_PER_METER)
        return mx, my

    def add_scan_to_map(self, pose, scan_points):
        """Adds a LiDAR scan to the occupancy map."""
        x, y, yaw = pose
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        for point in scan_points:
            # Transform point from sensor to world coordinates
            px, py = point[0], point[1]
            wx = x + (cos_yaw * px - sin_yaw * py)
            wy = y + (sin_yaw * px + cos_yaw * py)

            mx, my = self.world_to_map(wx, wy)

            if 0 <= mx < MAP_SIZE_PIXELS and 0 <= my < MAP_SIZE_PIXELS:
                self.map[my, mx] = 255  # Mark as occupied

    def show_map(self):
        """Displays the map in OpenCV."""
        cv2.imshow('SLAM Map', self.map)
        cv2.waitKey(1)
    
    def save_map_image(self, filename='slam_map2.png'):
        """Saves the occupancy grid map as an image (PNG)."""
        map_to_save = 255 - self.map  # Optional inversion: occupied (black), free (white)
        cv2.imwrite(filename, map_to_save)
        print(f"[INFO] Map saved as {filename}")

    def save_map_numpy(self, filename='slam_map.npy'):
        """Saves the map as a NumPy array."""
        np.save(filename, self.map)
        print(f"[INFO] Map saved as {filename}")


class CarlaSLAMSystem:
    def __init__(self):
        self.vehicle_setup = VehicleSetup() 
        self.slam = SimpleLiDARSLAM()

    def run(self):
        # Spawn vehicle and attach sensors using your class
        self.vehicle_setup.spawn_vehicle()

        try:
            while True:
                # Get vehicle pose
                pose = self.get_vehicle_pose()

                # Get LiDAR points
                lidar_points = self.vehicle_setup.lidar_points

                if lidar_points is not None:
                    # Use only X and Y for 2D SLAM
                    scan_points = lidar_points[:, :2]

                    # Add scan to SLAM map
                    self.slam.add_scan_to_map(pose, scan_points)

                    if SHOW_MAP:
                        self.slam.show_map()

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("[INFO] Shutting down.")

        finally:
            print("[INFO] Saving SLAM map before shutdown...")

            self.slam.save_map_image('my_slam_map.png')
            self.slam.save_map_numpy('my_slam_map.npy')

            print("[INFO] Shutting down simulation actors...")

            # Optional flag to stop callbacks
            self.vehicle_setup.is_running = False

            if self.vehicle_setup.vehicle:
                self.vehicle_setup.vehicle.destroy()

            for sensor in self.vehicle_setup.sensors.values():
                sensor.stop()
                sensor.destroy()

            cv2.destroyAllWindows()

            print("[INFO] All actors destroyed. Simulation shutdown complete.")


    def get_vehicle_pose(self):
        """Gets vehicle position and orientation from VehicleSetup."""
        transform = self.vehicle_setup.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        return location.x, location.y, np.deg2rad(rotation.yaw)

if __name__ == "__main__":
    slam_system = CarlaSLAMSystem()
    slam_system.run()
