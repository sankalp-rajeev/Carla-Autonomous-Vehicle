import carla
import numpy as np
import cv2
import time
import signal
import open3d as o3d

from vehicle_setup import VehicleSetup

stop_flag = False

# Function to handle CTRL + C
def signal_handler(sig, frame):
    global stop_flag
    print("\n[INFO] CTRL + C detected. Stopping the program...")
    stop_flag = True

signal.signal(signal.SIGINT, signal_handler)

class CarlaVisualSLAM:
    """
    This class handles the camera-based SLAM logic (ORB feature tracking, essential matrix,
    2D occupancy mapping, etc.), and uses the camera data from vehicle_setup.latest_camera_frame.
    """
    def __init__(self, vehicle_setup: VehicleSetup):
        self.vehicle_setup = vehicle_setup

        # Confirm we have a front camera sensor:
        self.camera = self.vehicle_setup.get_sensor("front_camera")
        if not self.camera:
            print("[ERROR] Front Camera not found! Exiting...")
            exit(1)

        # Enable Autopilot
        self.vehicle_setup.vehicle.set_autopilot(True)
        print("[INFO] Autopilot enabled. Vehicle will drive autonomously.")

        # ORB + BFMatcher for feature detection
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Bookkeeping for frame-to-frame
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None

        # 2D path and “map”
        self.trajectory = []
        self.map_size = 1000
        self.map_scale = 2
        self.occupancy_grid = np.ones((self.map_size, self.map_size, 3), dtype=np.uint8) * 255  # white

        # Start the overhead spectator in the Carla engine window
        self.vehicle_setup.set_spectator_camera()

        # We will store the latest “Feature Matches” image here for main loop to display:
        self.latest_slam_view = None
        self.latest_map_view = self.occupancy_grid.copy()

    def process_camera_frame(self, frame):
        """
        Given a new camera frame (BGR, shape [H,W,3]), run ORB, match features,
        estimate relative motion, and update occupancy grid + self.latest_slam_view.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # If we have a previous frame, try to match
        if (
            self.prev_frame is not None
            and descriptors is not None
            and self.prev_descriptors is not None
        ):
            matches = self.bf.match(self.prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)[:50]  # best 50

            if len(matches) > 10:  # ensure sufficient matches
                src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)

                    if len(self.trajectory) == 0:
                        # Start at center, assign initial position to new_pos
                        initial_pos = np.array([self.map_size // 2, self.map_size // 2])
                        self.trajectory.append(initial_pos)
                        new_pos = initial_pos
                    else:
                        new_pos = self.trajectory[-1] + (t[:2].flatten() * self.map_scale)
                        new_pos = np.clip(new_pos, 0, self.map_size - 1)
                        self.trajectory.append(new_pos)

                    # Mark keypoints in our occupancy_grid using new_pos which is now defined
                    for pt in dst_pts:
                        mapped_x = int(new_pos[0] + pt[0][0] * 0.1)
                        mapped_y = int(new_pos[1] + pt[0][1] * 0.1)
                        if 0 <= mapped_x < self.map_size and 0 <= mapped_y < self.map_size:
                            self.occupancy_grid[mapped_y, mapped_x] = (0, 0, 255)  # red

            # For visualization, draw feature matches
            slam_view = cv2.drawMatches(
                self.prev_frame, self.prev_keypoints,
                frame, keypoints,
                matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            self.latest_slam_view = slam_view
            self.latest_map_view = self.occupancy_grid.copy()

        # Update “prev” for next iteration
        self.prev_frame = frame
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def destroy(self):
        """Clean up any leftover references, though real cleanup is in vehicle_setup.destroy()."""
        print("[INFO] CarlaVisualSLAM cleanup called (nothing special here).")


def main():
    global stop_flag

    # 1) Create the VehicleSetup (spawns & attaches sensors).
    vehicle_setup = VehicleSetup()
    vehicle_setup.spawn_vehicle()
    vehicle_setup.attach_sensors()

    # 2) Create our SLAM instance
    slam = CarlaVisualSLAM(vehicle_setup)

    # 3) Create a single persistent Open3D Visualizer for LiDAR:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Point Cloud", width=800, height=600)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    try:
        while not stop_flag:
            if vehicle_setup.latest_camera_frame is not None:
                slam.process_camera_frame(vehicle_setup.latest_camera_frame)

            if slam.latest_slam_view is not None:
                cv2.imshow("Feature Matches", slam.latest_slam_view)
            if slam.latest_map_view is not None:
                cv2.imshow("SLAM Map", slam.latest_map_view)

            if vehicle_setup.latest_lidar_points is not None:
                pcd.points = o3d.utility.Vector3dVector(vehicle_setup.latest_lidar_points)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                stop_flag = True

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        slam.destroy()
        vehicle_setup.destroy()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("[INFO] All cleaned up. Exiting.")

if __name__ == "__main__":
    main()
