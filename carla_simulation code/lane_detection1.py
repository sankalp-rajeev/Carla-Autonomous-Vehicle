import cv2
import numpy as np
import time
from vehicle_setup import VehicleSetup  # Import the VehicleSetup class

class LaneDetector:
    def __init__(self):
        """
        Initialize the LaneDetector with default parameters for edge detection and Hough Transform.
        """
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 20
        self.hough_min_line_length = 20
        self.hough_max_line_gap = 300

    def detect_lanes(self, frame):
        """
        Detect lanes in the given frame using edge detection and Hough Transform.
        :param frame: Input image frame from the camera.
        :return: Frame with detected lanes drawn.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)

        # Define a region of interest (ROI) to focus on the road area
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),  # Bottom-left corner
            (width // 2 - 100, height // 2 + 50),  # Top-left corner
            (width // 2 + 100, height // 2 + 50),  # Top-right corner
            (width, height),  # Bottom-right corner
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Use Hough Transform to detect lines
        lines = cv2.HoughLinesP(
            masked_edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            np.array([]),
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        # Draw detected lines on the original frame
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Combine the line image with the original frame
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

        return result

class LaneDetectionApp:
    def __init__(self):
        """
        Initialize the LaneDetectionApp with a VehicleSetup instance and LaneDetector.
        """
        self.setup = VehicleSetup()
        self.lane_detector = LaneDetector()
        self.front_camera = None

    def process_image(self, image):
        """
        Process the camera image to detect lanes and display the result.
        :param image: Image from the camera sensor.
        """
        # Convert the CARLA image to an OpenCV-compatible format
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img_array.reshape((image.height, image.width, 4))[:, :, :3]

        # Detect lanes in the image
        lane_image = self.lane_detector.detect_lanes(img)

        # Display the result
        cv2.imshow("Lane Detection", lane_image)
        cv2.waitKey(1)

    def run(self):
        """
        Main function to initialize the vehicle, attach sensors, and run lane detection.
        """
        # Spawn the vehicle and attach sensors
        self.setup.spawn_vehicle()
        self.setup.attach_sensors()

        # Allow camera to initialize
        time.sleep(2)

        # Get the front camera sensor from the sensors dictionary
        if 'front_camera' not in self.setup.sensors:
            print("[ERROR] Front camera not found! Exiting...")
            self.setup.destroy()
            exit()

        # Store the front camera as an instance variable to retain it in memory
        self.front_camera = self.setup.sensors['front_camera']

        # Start processing the camera feed for lane detection
        self.front_camera.listen(lambda image: self.process_image(image))

        try:
            # Keep the script running
            print("[INFO] Lane detection is running. Press Ctrl+C to stop.")
            while True:
                # No need to call world.tick() in asynchronous mode
                time.sleep(0.1)  # Avoid busy looping
        except KeyboardInterrupt:
            print("[INFO] Simulation stopped by user.")
        finally:
            # Clean up
            if self.front_camera:
                self.front_camera.stop()  # Stop the camera callback
            self.setup.destroy()
            cv2.destroyAllWindows()
            print("[INFO] Cleanup complete. Exiting.")

if __name__ == "__main__":
    app = LaneDetectionApp()
    app.run()