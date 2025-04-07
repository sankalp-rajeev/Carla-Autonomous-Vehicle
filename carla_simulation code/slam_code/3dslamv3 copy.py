import numpy as np
import math
import time
import open3d as o3d
from vehicle_setup import VehicleSetup

def euler_to_rotation_matrix(yaw, pitch, roll):
    """
    Return a 3x3 rotation matrix given Euler angles (in radians).
    """
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

# Carla’s default palette (semantic_tag -> BGR)
semantic_palette = {
    0:  (  0,   0,   0),
    1:  ( 70,  70,  70),
    2:  (100,  40,  40),
    3:  ( 55,  90,  80),
    4:  (220,  20,  60),
    5:  (153, 153, 153),
    6:  (157, 234,  50),
    7:  (128,  64, 128),
    8:  (244,  35, 232),
    9:  (107, 142,  35),
    10: (  0,   0, 142),
    11: (102, 102, 156),
    12: (220, 220,   0),
    13: ( 70, 130, 180),
    14: ( 81,   0,  81),
    15: (150, 100, 100),
    16: (230, 150, 140),
    17: (180, 165, 180),
    18: (250, 170,  30),
    19: (110, 190, 160),
    20: (170, 120,  50),
    21: ( 45,  60, 150),
    22: (145, 170, 100)
}

# Example dynamic tags to skip if you want only static objects
dynamic_tags = {12, 13, 14, 15, 16, 17, 18, 19, 21}

class Optimized3DPointCloudMap:
    def __init__(self, max_points=2000000):
        """
        Maintains a point cloud map (points + colors) plus a vehicle trajectory.
        """
        self.points = []   # [ [x, y, z], ...]
        self.colors = []   # [ (r, g, b), ...], each in [0..1]
        self.path = []     # For storing vehicle positions (x, y, z)
        self.current_vehicle_pose = None
        
        self.max_points = max_points
        self.last_status_time = time.time()
        self.scans_processed = 0

    def add_scan(self, pose, scan_points, downsample_factor=4):
        """
        Add standard (non-semantic) LiDAR scan points to the map.
        """
        self.current_vehicle_pose = pose
        self.path.append(np.array(pose[:3]))  # store (x, y, z)

        x, y, z, yaw, pitch, roll = pose

        # Optional downsampling
        if downsample_factor > 1 and len(scan_points) > downsample_factor:
            indices = np.random.choice(
                scan_points.shape[0],
                size=scan_points.shape[0] // downsample_factor,
                replace=False
            )
            scan_points = scan_points[indices]

        # Transform local (vehicle) points to world frame
        R = euler_to_rotation_matrix(yaw, pitch, roll)
        translation = np.array([x, y, z])
        transformed_points = (R @ scan_points.T).T + translation
        
        # Append to map
        self.points.extend(transformed_points.tolist())
        # Gray color for standard LiDAR points
        self.colors.extend([(0.7, 0.7, 0.7)] * transformed_points.shape[0])

        # Limit map size
        if len(self.points) > self.max_points:
            excess = len(self.points) - self.max_points
            self.points = self.points[excess:]
            self.colors = self.colors[excess:]
        
        # Print occasionally
        self.scans_processed += 1
        now = time.time()
        if now - self.last_status_time > 5.0:
            print(f"[INFO] Processed {self.scans_processed} scans in last {now - self.last_status_time:.1f}s")
            print(f"[INFO] Current map size: {len(self.points)} points")
            self.last_status_time = now
            self.scans_processed = 0

    def add_semantic_scan(self, pose, semantic_detections, downsample_factor=4):
        """
        Add a semantic LiDAR scan, coloring points by semantic class.
        """
        self.current_vehicle_pose = pose
        self.path.append(np.array(pose[:3]))  # track (x, y, z)
        
        x, y, z, yaw, pitch, roll = pose
        R = euler_to_rotation_matrix(yaw, pitch, roll)
        translation = np.array([x, y, z])

        # Group points by class for separate downsampling
        points_by_class = {}
        for det in semantic_detections:
            # Example: if det.semantic_tag in dynamic_tags: continue
            if det.semantic_tag not in points_by_class:
                points_by_class[det.semantic_tag] = []
            points_by_class[det.semantic_tag].append([det.point.x, det.point.y, det.point.z])

        for semantic_tag, pts in points_by_class.items():
            if not pts:
                continue
            pts_array = np.array(pts)

            # Downsample
            if downsample_factor > 1 and len(pts_array) > downsample_factor:
                indices = np.random.choice(
                    len(pts_array),
                    size=len(pts_array) // downsample_factor,
                    replace=False
                )
                pts_array = pts_array[indices]
            
            # Transform to world coords
            transformed_points = (R @ pts_array.T).T + translation
            self.points.extend(transformed_points.tolist())

            # Convert BGR -> normalized RGB
            bgr = semantic_palette.get(semantic_tag, (255, 255, 255))
            r, g, b = bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0
            self.colors.extend([(r, g, b)] * transformed_points.shape[0])

        if len(self.points) > self.max_points:
            excess = len(self.points) - self.max_points
            self.points = self.points[excess:]
            self.colors = self.colors[excess:]

        self.scans_processed += 1
        now = time.time()
        if now - self.last_status_time > 5.0:
            print(f"[INFO] Processed {self.scans_processed} scans in last {now - self.last_status_time:.1f}s")
            print(f"[INFO] Current map size: {len(self.points)} points")
            self.last_status_time = now
            self.scans_processed = 0

    def compute_bounding_boxes(self):
        """
        Compute axis-aligned bounding boxes (AABB) per semantic class (by color).
        """
        print("[INFO] Computing bounding boxes for semantic classes...")

        # Build a reverse palette: bgr -> semantic_tag
        reverse_palette = {v: k for k, v in semantic_palette.items()}

        tag_points = {}
        for pt, color in zip(self.points, self.colors):
            # color is (r,g,b) in [0..1], convert to BGR in 0..255
            bgr = tuple(int(255 * c) for c in color[::-1])
            tag = reverse_palette.get(bgr, None)
            if tag is None:
                continue
            tag_points.setdefault(tag, []).append(pt)

        boxes = []
        for tag, pts in tag_points.items():
            if len(pts) < 10:
                continue  # skip noise
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(np.array(pts))
            aabb = cloud.get_axis_aligned_bounding_box()
            # Yellow color for bounding boxes
            aabb.color = np.array([1, 1, 0])
            boxes.append((tag, aabb))
        return boxes

    def create_voxel_grid(self, voxel_size=0.2):
        """
        Create a 3D voxel grid (surface-based) from the current point cloud
        using Open3D. This is not a 'true occupancy grid' with free/occupied,
        just a surface voxel representation.
        """
        if not self.points:
            print("[WARNING] No points to create a voxel grid.")
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(self.colors))

        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        
        # Build the voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=voxel_size,
            min_bound=min_bound,
            max_bound=max_bound
        )
        return voxel_grid

    def visualize(self, point_size=1.0, voxel_size=0.2):
        """
        Visualize the final map with:
          - Points
          - Voxel grid (optional)
          - Bounding boxes
          - Vehicle trajectory
        """
        if not self.points:
            print("[WARNING] No points to display in the point cloud.")
            return

        print("[INFO] Preparing Open3D visualization...")

        # Prepare the point cloud
        pts = np.array(self.points)
        cols = np.array(self.colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        # Optionally create a voxel grid
        voxel_grid = self.create_voxel_grid(voxel_size=voxel_size)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Point Cloud Map", width=1280, height=720)

        # Add the raw point cloud
        vis.add_geometry(pcd)

        # Add voxel grid
        if voxel_grid is not None:
            vis.add_geometry(voxel_grid)

        # Add bounding boxes
        boxes = self.compute_bounding_boxes()
        for tag, box in boxes:
            vis.add_geometry(box)
            print(f"[INFO] Added bounding box for semantic tag {tag}")

        # Optional vehicle marker
        if self.current_vehicle_pose:
            vx, vy, vz = self.current_vehicle_pose[:3]
            vehicle_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            vehicle_marker.translate((vx, vy, vz))
            vis.add_geometry(vehicle_marker)
        
        # Plot trajectory
        if len(self.path) > 1:
            path_np = np.vstack(self.path)
            lines = [[i, i+1] for i in range(len(self.path)-1)]
            line_colors = [[1, 0, 0] for _ in lines]  # red
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(path_np),
                lines=o3d.utility.Vector2iVector(lines)
            )
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            vis.add_geometry(line_set)

        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.asarray([0, 0, 0])

        print("[INFO] Displaying point cloud, voxel grid, bounding boxes, and trajectory in Open3D...")
        vis.run()
        vis.destroy_window()

    def save_map(self, filename="3d_slam_map.npz"):
        print(f"[INFO] Saving map with {len(self.points)} points to {filename}...")
        np.savez(
            filename,
            points=np.array(self.points),
            colors=np.array(self.colors),
            path=np.array(self.path)
        )
        print(f"[INFO] Map saved successfully to {filename}")
    
    def load_map(self, filename="3d_slam_map.npz"):
        print(f"[INFO] Loading map from {filename}...")
        data = np.load(filename)
        self.points = data["points"].tolist()
        self.colors = data["colors"].tolist()
        if "path" in data:
            self.path = data["path"].tolist()
            print(f"[INFO] Trajectory loaded with {len(self.path)} points")
        print(f"[INFO] Map loaded with {len(self.points)} points")


class Carla3DPointCloudMappingSystem:
    def __init__(self, max_points=2000000):
        self.vehicle_setup = VehicleSetup()
        self.point_cloud_map = Optimized3DPointCloudMap(max_points=max_points)
        self.running = True
    
    def run(self):
        """
        Main loop for building the point cloud map.
        Press Ctrl+C to stop and visualize the final map.
        """
        self.vehicle_setup.spawn_vehicle()
        self.vehicle_setup.attach_sensors()
        
        print("[INFO] 3D SLAM system started. Press Ctrl+C to stop and visualize the map.")
        start_time = time.time()
        
        try:
            while self.running:
                pose = self.get_vehicle_pose()
                
                # Priority: semantic LiDAR if present, else standard LiDAR
                if self.vehicle_setup.semantic_lidar_points is not None:
                    self.point_cloud_map.add_semantic_scan(
                        pose,
                        self.vehicle_setup.semantic_lidar_points,
                        downsample_factor=4
                    )
                elif self.vehicle_setup.lidar_points is not None:
                    scan_points = self.vehicle_setup.lidar_points[:, :3]
                    self.point_cloud_map.add_scan(
                        pose,
                        scan_points,
                        downsample_factor=4
                    )
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            elapsed_time = time.time() - start_time
            print(f"[INFO] SLAM process interrupted after {elapsed_time:.1f} seconds.")
        finally:
            self.point_cloud_map.save_map()
            print("[INFO] Visualizing final map with voxel grid and bounding boxes...")
            # Visualize with, e.g. point_size=1.0, voxel_size=0.2
            self.point_cloud_map.visualize(point_size=1.0, voxel_size=0.2)
            self.shutdown()
    
    def get_vehicle_pose(self):
        """
        Returns (x, y, z, yaw_rad, pitch_rad, roll_rad)
        from the CARLA vehicle transform.
        """
        transform = self.vehicle_setup.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        return (
            location.x,
            location.y,
            location.z,
            np.deg2rad(rotation.yaw),
            np.deg2rad(rotation.pitch),
            np.deg2rad(rotation.roll)
        )

    def shutdown(self):
        """
        Clean shutdown of vehicle & sensors.
        """
        print("[INFO] Shutting down SLAM system...")
        self.running = False
        self.vehicle_setup.shutdown()
        print("[INFO] SLAM system shutdown complete.")


if __name__ == "__main__":
    try:
        # For example, 5 million max points
        mapping_system = Carla3DPointCloudMappingSystem(max_points=5000000)
        mapping_system.run()
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
        import traceback
        traceback.print_exc()
