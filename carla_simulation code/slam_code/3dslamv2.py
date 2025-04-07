import numpy as np
import math
import time
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vehicle_setup import VehicleSetup

def euler_to_rotation_matrix(yaw, pitch, roll):
    """Return a 3x3 rotation matrix given Euler angles (in radians)."""
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

# Semantic palette (BGR values) for semantic LiDAR
# Colors will be converted to normalized RGB (0-1) for plotting
# Carla’s common defaults (tag -> BGR)
semantic_palette = {
    0:  (  0,   0,   0),  # None
    1:  ( 70,  70,  70),  # Buildings
    2:  (100,  40,  40),  # Fences
    3:  ( 55,  90,  80),  # Other
    4:  (220,  20,  60),  # Pedestrians
    5:  (153, 153, 153),  # Poles
    6:  (157, 234,  50),  # RoadLines
    7:  (128,  64, 128),  # Roads
    8:  (244,  35, 232),  # Sidewalks
    9:  (107, 142,  35),  # Vegetation
    10: (  0,   0, 142),  # Vehicles
    11: (102, 102, 156),  # Walls
    12: (220, 220,   0),  # TrafficSigns
    13: ( 70, 130, 180),  # Sky
    14: ( 81,   0,  81),  # Ground
    15: (150, 100, 100),  # Bridge
    16: (230, 150, 140),  # RailTrack
    17: (180, 165, 180),  # GuardRail
    18: (250, 170,  30),  # TrafficLight
    19: (110, 190, 160),  # Static
    20: (170, 120,  50),  # Dynamic
    21: ( 45,  60, 150),  # Water
    22: (145, 170, 100)   # Terrain
}

# Tags for dynamic objects (skip these for static mapping)
dynamic_tags = {12, 13, 14, 15, 16, 17, 18, 19, 21}  # Various vehicles, pedestrians, etc.

class Optimized3DPointCloudMap:
    def __init__(self, max_points=2000000):
        """
        Initialize a 3D point cloud map optimized for performance.
        
        Args:
            max_points: Maximum number of points to store in the map
        """
        self.points = []   # Each element is a [x, y, z] list
        self.colors = []   # Each element is an (r, g, b) tuple
        self.max_points = max_points
        self.current_vehicle_pose = None
        
        # Performance tracking
        self.last_status_time = time.time()
        self.scans_processed = 0
        
    def add_scan(self, pose, scan_points, downsample_factor=4):
        """
        Transforms standard LiDAR scan points into world coordinates and adds them to the map.
        
        Args:
            pose: (x, y, z, yaw, pitch, roll)
            scan_points: NumPy array of shape (N, 3)
            downsample_factor: Only keep 1/downsample_factor points to improve performance
        """
        self.current_vehicle_pose = pose
        x, y, z, yaw, pitch, roll = pose
        
        # Downsample to improve performance
        if downsample_factor > 1:
            indices = np.random.choice(scan_points.shape[0], 
                                      size=scan_points.shape[0]//downsample_factor, 
                                      replace=False)
            scan_points = scan_points[indices]
        
        # Transform points to world coordinates
        R = euler_to_rotation_matrix(yaw, pitch, roll)
        translation = np.array([x, y, z])
        transformed_points = (R @ scan_points.T).T + translation
        
        # Filter out points that are too far away (optional)
        #distances = np.linalg.norm(transformed_points - translation, axis=1)
        #transformed_points = transformed_points[distances < 80]  # Keep points within 80m
        
        # Add new points and colors
        self.points.extend(transformed_points.tolist())
        # Use gray for standard LiDAR points
        self.colors.extend([(0.7, 0.7, 0.7)] * transformed_points.shape[0])
        
        # Limit the number of points to prevent memory issues
        if len(self.points) > self.max_points:
            excess = len(self.points) - self.max_points
            self.points = self.points[excess:]
            self.colors = self.colors[excess:]
        
        # Track processing stats
        self.scans_processed += 1
        current_time = time.time()
        if current_time - self.last_status_time > 5.0:
            print(f"[INFO] Processed {self.scans_processed} scans in last {current_time - self.last_status_time:.1f}s")
            print(f"[INFO] Current map size: {len(self.points)} points")
            self.last_status_time = current_time
            self.scans_processed = 0
    
    def add_semantic_scan(self, pose, semantic_detections, downsample_factor=4):
        """
        Processes semantic LiDAR detections with colors based on semantic classes.
        
        Args:
            pose: (x, y, z, yaw, pitch, roll)
            semantic_detections: List of detections with point and semantic_tag
            downsample_factor: Only keep 1/downsample_factor points
        """
        self.current_vehicle_pose = pose
        x, y, z, yaw, pitch, roll = pose
        
        # Transform matrix
        R = euler_to_rotation_matrix(yaw, pitch, roll)
        translation = np.array([x, y, z])
        
        # Process by semantic class for cleaner downsampling
        points_by_class = {}
        for detection in semantic_detections:
            # if detection.semantic_tag in dynamic_tags:
            #     continue  # Skip dynamic objects
            
            if detection.semantic_tag not in points_by_class:
                points_by_class[detection.semantic_tag] = []
            
            point = np.array([detection.point.x, detection.point.y, detection.point.z])
            points_by_class[detection.semantic_tag].append(point)
        
        # Process each semantic class
        for semantic_tag, points in points_by_class.items():
            if len(points) == 0:
                continue
            
            points_array = np.array(points)
            # Downsample within class
            if downsample_factor > 1 and len(points) > downsample_factor:
                indices = np.random.choice(len(points), 
                                          size=len(points)//downsample_factor, 
                                          replace=False)
                points_array = points_array[indices]
            
            # Transform to world coordinates
            transformed_points = (R @ points_array.T).T + translation
            
            # Add points to map
            self.points.extend(transformed_points.tolist())
            
            # Convert BGR to RGB and normalize
            color = semantic_palette.get(semantic_tag, (255, 255, 255))
            r, g, b = color[2]/255.0, color[1]/255.0, color[0]/255.0
            self.colors.extend([(r, g, b)] * transformed_points.shape[0])
        
        # Limit total points
        if len(self.points) > self.max_points:
            excess = len(self.points) - self.max_points
            self.points = self.points[excess:]
            self.colors = self.colors[excess:]
        
        # Track processing stats
        self.scans_processed += 1
        current_time = time.time()
        if current_time - self.last_status_time > 5.0:
            print(f"[INFO] Processed {self.scans_processed} scans in last {current_time - self.last_status_time:.1f}s")
            print(f"[INFO] Current map size: {len(self.points)} points")
            self.last_status_time = current_time
            self.scans_processed = 0
    
    

    def visualize(self, point_size=1.0, dpi=150, show_axes=True):
        """
        Visualize the accumulated point cloud map using Open3D.
        
        Args:
            point_size: Not used in Open3D (adjust via render settings if needed)
            dpi: Ignored in Open3D
            show_axes: Whether to display coordinate axes (Open3D always shows them)
        """
        if not self.points:
            print("[WARNING] No points to display in the point cloud.")
            return

        print("[INFO] Preparing Open3D visualization...")

        # Convert to numpy arrays
        pts = np.array(self.points)
        cols = np.array(self.colors)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Point Cloud Map", width=1280, height=720)
        vis.add_geometry(pcd)

        # Optionally add vehicle marker
        if self.current_vehicle_pose:
            x, y, z = self.current_vehicle_pose[:3]
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            mesh.translate((x, y, z))
            vis.add_geometry(mesh)

        # Customize render options
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.asarray([0, 0, 0])

        print("[INFO] Displaying point cloud in Open3D...")
        vis.run()
        vis.destroy_window()

    
    def save_map(self, filename="3d_slam_map.npz"):
        """Save the current map to a file."""
        print(f"[INFO] Saving map with {len(self.points)} points to {filename}...")
        np.savez(filename, 
                points=np.array(self.points), 
                colors=np.array(self.colors))
        print(f"[INFO] Map saved successfully to {filename}")
    
    def load_map(self, filename="3d_slam_map.npz"):
        """Load a previously saved map."""
        print(f"[INFO] Loading map from {filename}...")
        data = np.load(filename)
        self.points = data['points'].tolist()
        self.colors = data['colors'].tolist()
        print(f"[INFO] Map loaded with {len(self.points)} points")


class Carla3DPointCloudMappingSystem:
    def __init__(self, max_points=2000000):
        """
        Initialize the CARLA 3D SLAM system.
        
        Args:
            max_points: Maximum number of points to keep in the map
        """
        self.vehicle_setup = VehicleSetup()
        self.point_cloud_map = Optimized3DPointCloudMap(max_points=max_points)
        self.running = True
    
    def run(self):
        """Run the SLAM system."""
        # Spawn vehicle and attach sensors
        self.vehicle_setup.spawn_vehicle()
        self.vehicle_setup.attach_sensors()
        
        print("[INFO] 3D SLAM system started. Press Ctrl+C to stop and visualize the map.")
        start_time = time.time()
        
        try:
            # Main loop for map building
            while self.running:
                # Get current vehicle pose
                pose = self.get_vehicle_pose()
                
                # Process sensor data
                if self.vehicle_setup.semantic_lidar_points is not None:
                    self.point_cloud_map.add_semantic_scan(
                        pose, 
                        self.vehicle_setup.semantic_lidar_points,
                        downsample_factor=4  # Adjust as needed for performance
                    )
                elif self.vehicle_setup.lidar_points is not None:
                    scan_points = self.vehicle_setup.lidar_points[:, :3]
                    self.point_cloud_map.add_scan(
                        pose, 
                        scan_points,
                        downsample_factor=4
                    )
                
                # Allow world to tick (no need to wait too long)
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            elapsed_time = time.time() - start_time
            print(f"[INFO] SLAM process interrupted after {elapsed_time:.1f} seconds.")
        finally:
            # Save map
            self.point_cloud_map.save_map()
            
            # Visualize final map
            print("[INFO] Visualizing final map...")
            self.point_cloud_map.visualize(point_size=0.5)
            
            # Clean shutdown
            self.shutdown()
    
    def get_vehicle_pose(self):
        """
        Get the current vehicle pose.
        Returns: (x, y, z, yaw, pitch, roll)
        """
        transform = self.vehicle_setup.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        return (location.x, location.y, location.z,
                np.deg2rad(rotation.yaw),
                np.deg2rad(rotation.pitch),
                np.deg2rad(rotation.roll))
    
    def shutdown(self):
        """Clean shutdown of the SLAM system."""
        print("[INFO] Shutting down SLAM system...")
        
        # Stop threads
        self.running = False
        
        # Shutdown vehicle and sensors
        self.vehicle_setup.shutdown()
        
        print("[INFO] SLAM system shutdown complete.")


if __name__ == "__main__":
    try:
        # Set higher max_points for higher quality maps
        mapping_system = Carla3DPointCloudMappingSystem(max_points=5000000)
        mapping_system.run()
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
        import traceback
        traceback.print_exc()