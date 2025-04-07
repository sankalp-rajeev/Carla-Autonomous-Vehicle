import numpy as np
import cv2
import math
import time
from vehicle_setup import VehicleSetup

def euler_to_rotation_matrix(yaw, pitch, roll):
    # Create rotation matrices around Z (yaw), Y (pitch) and X (roll)
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw),  math.cos(yaw), 0],
                    [0,              0,             1]])
    R_y = np.array([[ math.cos(pitch), 0, math.sin(pitch)],
                    [0,                1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    R_x = np.array([[1, 0,              0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll),  math.cos(roll)]])
    return R_z @ R_y @ R_x

class Simple3DSLAM:
    def __init__(self):
        # Define the resolution and overall map size in meters
        self.resolution = 0.2  # meters per voxel
        self.map_size = np.array([200, 200, 20])  # [x, y, z] in meters
        # Determine voxel grid dimensions
        self.voxel_dims = (self.map_size / self.resolution).astype(int)
        # Create the 3D occupancy grid (initialized to 0)
        self.map = np.zeros(self.voxel_dims, dtype=np.uint8)
        # Set the origin at the center of x,y and bottom for z (or choose as needed)
        self.origin = np.array([self.voxel_dims[0] // 2, self.voxel_dims[1] // 2, 0])
    
    def world_to_voxel(self, x, y, z):
        # Convert world coordinates (in meters) to voxel indices
        vx = int(self.origin[0] + x / self.resolution)
        vy = int(self.origin[1] - y / self.resolution)
        vz = int(z / self.resolution)
        return vx, vy, vz

    def add_scan_to_map(self, pose, scan_points):
        """
        Update the voxel grid using a new LiDAR scan.
        pose: (x, y, z, yaw, pitch, roll) in world frame, with angles in radians.
        scan_points: LiDAR points as an array of shape (N, 3)
        """
        x, y, z, yaw, pitch, roll = pose
        R = euler_to_rotation_matrix(yaw, pitch, roll)
        for point in scan_points:
            # Transform the point from sensor to world coordinates
            world_point = np.dot(R, point) + np.array([x, y, z])
            vx, vy, vz = self.world_to_voxel(world_point[0], world_point[1], world_point[2])
            if (0 <= vx < self.map.shape[0] and 
                0 <= vy < self.map.shape[1] and 
                0 <= vz < self.map.shape[2]):
                self.map[vx, vy, vz] = 255  # Mark as occupied
    def update_origin_if_needed(self, vehicle_world_pos):
        """
        Check if the vehicle is near the boundaries of the occupancy grid.
        If so, shift the grid to re-center the vehicle.
        vehicle_world_pos: (x, y, z) in world coordinates.
        """
        vx, vy, vz = self.world_to_voxel(*vehicle_world_pos)
        margin_x = int(self.voxel_dims[0] * 0.2)  # 20% margin
        margin_y = int(self.voxel_dims[1] * 0.2)
        
        if (vx < margin_x or vx >= self.voxel_dims[0] - margin_x or 
            vy < margin_y or vy >= self.voxel_dims[1] - margin_y):
            center_x = self.voxel_dims[0] // 2
            center_y = self.voxel_dims[1] // 2
            shift_x = vx - center_x
            shift_y = vy - center_y
            
            # Create a new empty map of the same size
            new_map = np.zeros_like(self.map)
            
            # Calculate source indices from the old map (clipped to valid range)
            src_x_start = max(0, shift_x)
            src_x_end = self.map.shape[0] + min(0, shift_x)
            src_y_start = max(0, shift_y)
            src_y_end = self.map.shape[1] + min(0, shift_y)
            
            # Calculate destination indices in the new map
            dest_x_start = max(0, -shift_x)
            dest_x_end = dest_x_start + (src_x_end - src_x_start)
            dest_y_start = max(0, -shift_y)
            dest_y_end = dest_y_start + (src_y_end - src_y_start)
            
            # Copy overlapping region from old map to new map
            new_map[dest_x_start:dest_x_end, dest_y_start:dest_y_end, :] = \
                self.map[src_x_start:src_x_end, src_y_start:src_y_end, :]
            
            self.map = new_map
            # Update origin to remain at the center of the grid
            self.origin = np.array([center_x, center_y, self.origin[2]])
            print("[INFO] Map recentered. New origin:", self.origin)

    def show_map_projection(self, vehicle_pose=None):
        """
        Display a top-down projection (max over z) of the 3D occupancy grid.
        Optionally overlay the vehicle's position if vehicle_pose is provided.
        
        vehicle_pose: (x, y, z, yaw, pitch, roll)
        """
        projection = np.max(self.map, axis=2).copy()  # Copy so we don't modify the original map
        
        if vehicle_pose is not None:
            # Convert vehicle's world (x, y, z) to voxel indices
            vx, vy, _ = self.world_to_voxel(vehicle_pose[0], vehicle_pose[1], vehicle_pose[2])
            # Draw a small circle on the projection to mark the vehicle's position.
            # Adjust the radius and color as needed (here using 128 as a grayscale value)
            cv2.circle(projection, (vx, vy), radius=5, color=128, thickness=-1)
        
        cv2.imshow('3D SLAM Map Projection', projection)
        cv2.waitKey(1)

    
    def save_map_numpy(self, filename='slam_map_3d.npy'):
        np.save(filename, self.map)
        print(f"[INFO] 3D SLAM map saved as {filename}")
    def save_map_png(self, filename='slam_map_3d.png'):
        projection = np.max(self.map, axis=2)
        cv2.imwrite(filename, projection)
        print(f"[INFO] 3D SLAM map projection saved as {filename}")


class Carla3DSLAMSystem:
    def __init__(self):
        # Use your existing VehicleSetup class for spawning the vehicle and attaching sensors.
        self.vehicle_setup = VehicleSetup() 
        # Use the new 3D SLAM mapping class
        self.slam = Simple3DSLAM()

    def run(self):
        # Spawn vehicle and sensors as before
        self.vehicle_setup.spawn_vehicle()
        self.vehicle_setup.attach_sensors()
        try:
            while True:
                # Get the full 6DoF pose (x, y, z, yaw, pitch, roll)
                pose = self.get_full_vehicle_pose()
                self.slam.update_origin_if_needed(pose[:3])
                # Get LiDAR points (all 3 dimensions: x, y, z)
                lidar_points = self.vehicle_setup.lidar_points

                if lidar_points is not None:
                    # LiDAR data shape assumed (N, 4); use first three columns (x, y, z)
                    scan_points = lidar_points[:, :3]
                    self.slam.add_scan_to_map(pose, scan_points)
                    self.slam.show_map_projection(vehicle_pose=pose)

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("[INFO] Shutting down.")

        finally:
            self.slam.save_map_numpy('my_slam_map_3d.npy')
            print("[INFO] Shutting down simulation actors...")
            # Stop sensors and destroy actors as appropriate
            self.vehicle_setup.running = False
            if self.vehicle_setup.vehicle:
                self.vehicle_setup.vehicle.destroy()
            for sensor in self.vehicle_setup.sensors.values():
                sensor.stop()
                sensor.destroy()
            cv2.destroyAllWindows()
            print("[INFO] Simulation shutdown complete.")

    def get_full_vehicle_pose(self):
        """
        Retrieve the full vehicle pose (position and orientation).
        """
        transform = self.vehicle_setup.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        # Convert angles from degrees to radians
        return (location.x, location.y, location.z, 
                np.deg2rad(rotation.yaw), 
                np.deg2rad(rotation.pitch), 
                np.deg2rad(rotation.roll))

if __name__ == "__main__":
    slam_system = Carla3DSLAMSystem()
    slam_system.run()
