import cv2
import numpy as np

def visualize(image, lanes, num_lanes=4):
    """
    Draw decoded lanes (pixel points) on the image, 
    after scaling them from model's (288x800) space 
    to the actual 'image' resolution.
    """

    # 1) Figure out your final display size
    height_display, width_display = image.shape[:2]

    # 2) The UFLD model uses 288x800 internally
    height_inference = 288
    width_inference = 800

    # 3) Compute scaling factors
    y_scale = height_display / float(height_inference)
    x_scale = width_display / float(width_inference)

    # 4) Scale the lane points
    scaled_lanes = []
    for lane in lanes:
        scaled_lane = []
        for point in lane:
            if point is None:
                scaled_lane.append(None)
            else:
                pt_x, pt_y = point
                # Scale from model space (288x800) to the actual image size
                scaled_lane.append((int(pt_x * x_scale), int(pt_y * y_scale)))
        scaled_lanes.append(scaled_lane)

    # 5) Now draw the scaled lanes on a copy of 'image'
    img = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for lane_idx, lane in enumerate(scaled_lanes):
        prev_point = None
        for point in lane:
            if point is None:
                continue
            pt_x, pt_y = point

            # Draw a circle for each anchor
            cv2.circle(img, (pt_x, pt_y), 5, colors[lane_idx % len(colors)], -1)

            # Optionally draw a line from the previous anchor to this one
            if prev_point is not None:
                cv2.line(img, prev_point, (pt_x, pt_y), colors[lane_idx % len(colors)], 2)

            prev_point = (pt_x, pt_y)

    return img
