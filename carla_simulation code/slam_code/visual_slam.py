import numpy as np
import cv2
import time
import open3d as o3d
from vehicle_setup import VehicleSetup  # Make sure this is in the same directory

class VisualSLAM:
    def __init__(self, focal_length=500, principal_point=(320, 240)):
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.focal_length = focal_length
        self.pp = principal_point

        self.cur_pose = np.eye(4)
        self.poses = []
        self.map_points = []

        self.prev_img = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_pose = np.eye(4)

        print("[INFO] VisualSLAM system initialized.")

    def process_frame(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_kp is not None and self.prev_des is not None:
            matches = self.bf.match(des, self.prev_des)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 10:
                src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.prev_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                E, mask = cv2.findEssentialMat(
                    src_pts, dst_pts,
                    focal=self.focal_length,
                    pp=self.pp,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )

                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, focal=self.focal_length, pp=self.pp)

                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.flatten()

                    self.cur_pose = self.cur_pose @ np.linalg.inv(T)
                    self.poses.append(self.cur_pose.copy())

                    pts1 = dst_pts[mask_pose.ravel() == 1]
                    pts2 = src_pts[mask_pose.ravel() == 1]

                    if pts1.shape[0] >= 8:
                        points_3d = self.triangulate_points(
                            np.eye(3), np.zeros((3, 1)),
                            R, t, pts1, pts2
                        )

                        global_points = (self.prev_pose[:3, :3] @ points_3d.T + self.prev_pose[:3, 3:4]).T
                        self.map_points.extend(global_points.tolist())

                    self.prev_pose = self.cur_pose.copy()

        self.prev_img = gray
        self.prev_kp = kp
        self.prev_des = des

    def triangulate_points(self, R1, t1, R2, t2, pts1, pts2):
        K = np.array([
            [self.focal_length, 0, self.pp[0]],
            [0, self.focal_length, self.pp[1]],
            [0, 0, 1]
        ])

        proj_matrix1 = K @ np.hstack((R1, t1))
        proj_matrix2 = K @ np.hstack((R2, t2))

        pts4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1.T, pts2.T)
        pts3d = pts4d[:3, :] / pts4d[3, :]

        return pts3d.T

    def save_point_cloud(self, filename="visual_slam_map.ply"):
        if not self.map_points:
            print("[WARNING] No map points to save.")
            return

        points_np = np.array(self.map_points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        o3d.io.write_point_cloud(filename, pcd)
        print(f"[INFO] 3D map point cloud saved to {filename}")

    def save_trajectory(self, filename="visual_slam_trajectory.csv"):
        if not self.poses:
            print("[WARNING] No trajectory poses to save.")
            return

        poses_array = np.array([pose[:3, 3] for pose in self.poses])

        np.savetxt(filename, poses_array, delimiter=",", header="x,y,z", comments='')
        print(f"[INFO] Trajectory saved to {filename}")

# =====================================================================================
# MAIN - Run Visual SLAM & Vehicle Setup
# =====================================================================================
if __name__ == "__main__":
    setup = VehicleSetup()
    slam_system = VisualSLAM()
    setup.slam_system = slam_system
    setup.spawn_vehicle()

    try:
        print("[INFO] Running real-time Visual SLAM... Close CARLA to save the map automatically.")

        while True:
            time.sleep(0.05)

            # Check if CARLA is still running
            if not setup.vehicle.is_alive:
                print("[INFO] Vehicle no longer exists. CARLA might be closed.")
                break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected. Stopping simulation...")

    finally:
        # Clean shutdown
        setup.shutdown()

        # Save outputs
        slam_system.save_point_cloud("visual_slam_map.ply")
        slam_system.save_trajectory("visual_slam_trajectory.csv")

        print("[INFO] Visual SLAM outputs saved. Exiting...")

        # Close OpenCV windows
        cv2.destroyAllWindows()
