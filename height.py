# -*- coding: utf-8 -*-

"""
A script for REAL-TIME LIVE FEED from Intel RealSense camera for
3D top position and vertical height detection.

(Version 32: Introduced Y_PERCENTILE_FILTER to ignore the bottom 5% of points,
specifically addressing the noise and Y_min corruption caused by the metal nut holder.)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.stats import scoreatpercentile # Import for percentile calculation

# -- IMPORTS for Tip Calculation --
import open3d as o3d
from skimage.morphology import skeletonize as skeletonize_3d
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import networkx as nx
# ------------------------------------------

# ---------------------------------
# -- CONFIGURATION --
# ---------------------------------

# -- Stream settings
DEPTH_WIDTH = 1024
DEPTH_HEIGHT = 768
COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
FRAME_RATE = 30

# -- Cropping settings
CROP_PERCENT_X = 10
CROP_PERCENT_Y = 10

# -- Color Segmentation (HSV Range for Green) --
LOWER_GREEN_1 = np.array([35, 20, 20])
UPPER_GREEN_1 = np.array([85, 255, 255])
LOWER_GREEN_2 = np.array([20, 20, 20])
UPPER_GREEN_2 = np.array([34, 255, 255])

# -- Point Cloud settings
DEPTH_SCALE = 0.001  # Convert mm to meters (L515 default)
MIN_DEPTH = 0.1
MAX_DEPTH = 3.5
POINT_CLOUD_DOWNSAMPLE = 1

# --- CRITICAL CONSTANTS FOR ERROR COMPENSATION ---
# Based on your physical measurement (28 inches / 0.71 meters)
KNOWN_DEPTH_Z = 0.71
# Loosening the error threshold to capture more points
Z_ERROR_THRESHOLD = 0.7

# --- EMPIRICAL CORRECTION FACTORS ---
# (Actual Height / Measured Height) -> 10 cm / 3.0 cm = ~3.33
# This factor corrects the vertical compression error caused by the camera angle/faulty depth.
Y_CORRECTION_FACTOR = 3.33

# --- NEW: FILTER TO REMOVE HOLDER/NOISE AT BASE ---
# Ignore the bottom 5% of points in the Y-dimension to exclude the holder/floor noise.
Y_PERCENTILE_FILTER = 5
# ----------------------------------------------------

# -- Display settings
DISPLAY_SCALE = 0.6
WINDOW_NAMES = ['Live Feed', 'HSV Space', 'Color Mask', 'Segmented']

# ---------------------------------
# -- GLOBAL VARIABLES --
# ---------------------------------
pipeline = None
align = None
depth_intrinsics = None
fps_counter = deque(maxlen=30)
point_cloud_stats = {}

# Matplotlib figure for 3D point cloud
fig_3d = None
ax_3d = None
canvas_3d = None

# ---------------------------------
# -- CAMERA INITIALIZATION --
# ---------------------------------

def initialize_camera():
    """Initialize RealSense camera pipeline"""
    global pipeline, align, depth_intrinsics

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FRAME_RATE)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FRAME_RATE)

    print("🎥 Starting RealSense pipeline...")
    profile = pipeline.start(config)

    depth_profile = profile.get_stream(rs.stream.depth)
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    align = rs.align(rs.stream.color)

    print("⏳ Warming up camera...")
    for _ in range(30):
        pipeline.wait_for_frames()

    print("✅ Camera ready for live feed!")

# ---------------------------------
# -- TOP DETECTION & HEIGHT FUNCTION (MODIFIED FOR SCALING) --
# ---------------------------------

def find_asparagus_top_and_height(points_3d,
                         voxel_size=0.002,
                         nn_radius_mult=1.75):
    """
    Finds the 3D coordinates of the top (highest Y) of the asparagus
    and calculates its vertical height in cm, applying an empirical correction factor.

    Returns (top_point, centerline_points, height_cm)
    """
    MIN_POINTS_THRESHOLD = 5
    if points_3d.shape[0] < MIN_POINTS_THRESHOLD:
        return None, np.array([]), 0.0

    # 1. Calculate the robust minimum Y boundary, ignoring the bottom X% of points.
    y_coords = points_3d[:, 1]

    # Calculate the Y value corresponding to the Y_PERCENTILE_FILTER (e.g., 5th percentile)
    robust_min_y = scoreatpercentile(y_coords, Y_PERCENTILE_FILTER)

    # The true minimum Y is now the robust minimum, while max_y remains the peak.
    max_y = y_coords.max()

    # Calculate Raw Vertical Height (Max Y - Robust Min Y)
    vertical_height_raw_m = max_y - robust_min_y

    # --- CRITICAL: Apply the empirical correction factor ---
    vertical_height_m = vertical_height_raw_m * Y_CORRECTION_FACTOR
    vertical_height_cm = vertical_height_m * 100.0

    # 2. Find the 3D position of the Top point (Highest Y)
    top_point_index = np.argmax(y_coords)
    top_point = points_3d[top_point_index]

    # 3. Generate centerline for visualization (optional)
    centerline = np.array([])
    if points_3d.shape[0] >= 50:
         try:
            mn = points_3d.min(axis=0)
            grid_idx = np.floor((points_3d - mn) / voxel_size).astype(int)
            grid_max = grid_idx.max(axis=0) + 1
            vol = np.zeros(grid_max, dtype=np.uint8)
            vol[grid_idx[:,0], grid_idx[:,1], grid_idx[:,2]] = 1

            sk = skeletonize_3d(vol).astype(bool)
            sk_vox = np.argwhere(sk)

            if sk_vox.shape[0] >= 5:
                sk_pts = mn + (sk_vox + 0.5) * voxel_size
                centerline = sk_pts
         except Exception as e:
             pass

    return top_point, centerline, vertical_height_cm

# ---------------------------------
# -- POINT CLOUD GENERATION (Z COMPENSATION UNCHANGED) --
# ---------------------------------

def generate_point_cloud_fast(depth_image, mask, color_image, crop_offset_x, crop_offset_y):
    """
    Fast point cloud generation. If depth is too far (likely background),
    it is compensated using KNOWN_DEPTH_Z to fix Y-scaling.
    """
    global depth_intrinsics, point_cloud_stats

    h, w = depth_image.shape
    depth_ds = depth_image[::POINT_CLOUD_DOWNSAMPLE, ::POINT_CLOUD_DOWNSAMPLE]
    mask_ds = mask[::POINT_CLOUD_DOWNSAMPLE, ::POINT_CLOUD_DOWNSAMPLE]
    color_ds = color_image[::POINT_CLOUD_DOWNSAMPLE, ::POINT_CLOUD_DOWNSAMPLE]

    depth_meters = depth_ds * DEPTH_SCALE
    # Filter points by mask and general depth boundaries
    valid_mask = (depth_meters > MIN_DEPTH) & (depth_meters < MAX_DEPTH) & (depth_meters > 0) & (mask_ds > 0)

    if not np.any(valid_mask):
        point_cloud_stats = {'count': 0, 'bounds': None}
        return np.array([]), np.array([])

    valid_points = np.where(valid_mask)
    num_points = len(valid_points[0])

    points_3d = []
    colors_3d = []

    for i in range(num_points):
        y_idx, x_idx = valid_points[0][i], valid_points[1][i]
        pixel_x = x_idx * POINT_CLOUD_DOWNSAMPLE + crop_offset_x
        pixel_y = y_idx * POINT_CLOUD_DOWNSAMPLE + crop_offset_y
        depth_val = depth_meters[y_idx, x_idx]

        compensated = False
        # --- Z-COMPENSATION LOGIC ---
        # If the measured depth is NOT close to the known depth, and it IS close to the known faulty depth (~3m),
        # then we compensate it.
        if abs(depth_val - KNOWN_DEPTH_Z) > Z_ERROR_THRESHOLD and depth_val > 2.5:
             depth_val = KNOWN_DEPTH_Z # Overwrite with known correct depth
             compensated = True

        # NOTE: We keep the compensated Z-depth for the point cloud, so the output Z coordinate is
        # KNOWN_DEPTH_Z, and the X/Y scaling is correct for measurement.
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixel_x, pixel_y], depth_val)

        # If the Z coordinate was compensated, ensure the Z component of the point_3d uses the compensated value
        if compensated:
             point_3d[2] = KNOWN_DEPTH_Z

        points_3d.append(point_3d)
        color_bgr = color_ds[y_idx, x_idx]
        color_rgb = [color_bgr[2]/255.0, color_bgr[1]/255.0, color_bgr[0]/255.0]
        colors_3d.append(color_rgb)

    points_3d = np.array(points_3d)
    colors_3d = np.array(colors_3d)

    if len(points_3d) > 0:
        point_cloud_stats = {
            'count': len(points_3d),
            'bounds': {
                'x': (points_3d[:, 0].min(), points_3d[:, 0].max()),
                'y': (points_3d[:, 1].min(), points_3d[:, 1].max()),
                'z': (points_3d[:, 2].min(), points_3d[:, 2].max())
            }
        }
    return points_3d, colors_3d

# ---------------------------------
# -- FRAME PROCESSING (UNCHANGED) --
# ---------------------------------

def process_frame_live():
    """Process single frame for live display, calculating top position and height."""
    global pipeline, align, fps_counter

    try:
        frame_start = time.time()
        frames = pipeline.wait_for_frames(timeout_ms=100)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        color_image_full = np.asanyarray(color_frame.get_data())
        depth_image_full = np.asanyarray(depth_frame.get_data())

        crop_x = int(color_image_full.shape[1] * (CROP_PERCENT_X / 100))
        crop_y = int(color_image_full.shape[0] * (CROP_PERCENT_Y / 100))

        color_image = color_image_full[crop_y:-crop_y, crop_x:-crop_x]
        depth_image = depth_image_full[crop_y:-crop_y, crop_x:-crop_x]

        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, LOWER_GREEN_1, UPPER_GREEN_1)
        mask2 = cv2.inRange(hsv_image, LOWER_GREEN_2, UPPER_GREEN_2)
        color_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((11, 11), np.uint8)
        cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(cleaned_mask)
        color_with_bbox = color_image.copy()

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                cv2.drawContours(filled_mask, [largest_contour], 0, 255, -1)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(color_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 3)
                area = cv2.contourArea(largest_contour)
                cv2.putText(color_with_bbox, f'Area: {int(area)}px', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        segmented_image = cv2.bitwise_and(color_image, color_image, mask=filled_mask)
        points_3d, colors_3d = generate_point_cloud_fast(depth_image, filled_mask, color_image, crop_x, crop_y)

        # --- Calculate Top Position and Height ---
        top_point_3d, centerline_pts, vertical_height_cm = find_asparagus_top_and_height(points_3d)

        frame_time = time.time() - frame_start
        fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_counter.append(fps)
        avg_fps = np.mean(fps_counter)

        cv2.putText(color_with_bbox, f'FPS: {avg_fps:.1f}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(color_with_bbox, f'Points: {point_cloud_stats.get("count", 0)}', (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return {
            'color': color_with_bbox,
            'hsv': hsv_image,
            'mask': cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR),
            'segmented': segmented_image,
            'fps': avg_fps,
            'points_3d': points_3d,
            'colors_3d': colors_3d,
            'centerline_pts': centerline_pts,
            'top_point_3d': top_point_3d,
            'vertical_height_cm': vertical_height_cm
        }

    except Exception as e:
        print(f"⚠️  Frame processing error: {e}")
        return None

# ---------------------------------
# -- VISUALIZATION FUNCTIONS (MODIFIED) --
# ---------------------------------

def setup_3d_matplotlib():
    global fig_3d, ax_3d, canvas_3d
    plt.ioff()
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    canvas_3d = FigureCanvasAgg(fig_3d)

def create_3d_pointcloud_image(points_3d, colors_3d, centerline_pts=None, top_point=None, vertical_height_cm=0.0):
    """ Creates 3D point cloud visualization, including the detected top and height """
    global fig_3d, ax_3d, canvas_3d
    if fig_3d is None:
        setup_3d_matplotlib()
    ax_3d.clear()
    ax_3d.set_xlabel('X (m)'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title(f'Asparagus Point Cloud\nVertical Height: {vertical_height_cm:.1f} cm (Corrected)')

    if len(points_3d) > 0:
        ax_3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                     c=colors_3d, s=2, alpha=0.6)

        if centerline_pts is not None and len(centerline_pts) > 0:
            ax_3d.plot(centerline_pts[:, 0], centerline_pts[:, 1], centerline_pts[:, 2],
                       color='r', linewidth=2, label='Centerline')

        if top_point is not None:
            ax_3d.scatter(top_point[0], top_point[1], top_point[2],
                          c='cyan', marker='o', s=100, label='TOP', depthshade=True)
            ax_3d.legend()

        # Dynamic axis scaling based on object bounds
        max_range = np.array([points_3d[:,0].max()-points_3d[:,0].min(),
                             points_3d[:,1].max()-points_3d[:,1].min(),
                             points_3d[:,2].max()-points_3d[:,2].min()]).max() / 2.0
        mid_x = (points_3d[:,0].max()+points_3d[:,0].min()) * 0.5
        mid_y = (points_3d[:,1].max()+points_3d[:,1].min()) * 0.5
        mid_z = (points_3d[:,2].max()+points_3d[:,2].min()) * 0.5
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

        top_text = (f'TOP (X, Y, Z):\n'
                    f'{top_point[0]:.3f}, {top_point[1]:.3f}, {top_point[2]:.3f} m') if top_point is not None else 'TOP: Not Found'

        stats_text = (f'Point Count: {len(points_3d)}\n'
                      f'Z Range: {points_3d[:, 2].min():.3f} to {points_3d[:, 2].max():.3f} m\n'
                      f'Y Range (Uncorrected): {points_3d[:, 1].min():.3f} to {points_3d[:, 1].max():.3f} m')

        ax_3d.text2D(0.02, 0.98, stats_text, transform=ax_3d.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_3d.text2D(0.02, 0.75, top_text, transform=ax_3d.transAxes,
                    verticalalignment='top', fontsize=10, color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    else:
        ax_3d.text(0.5, 0.5, 0.5, 'No valid point cloud data (Check Depth Filter)',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax_3d.transAxes, fontsize=12)

    ax_3d.xaxis.pane.fill = False; ax_3d.yaxis.pane.fill = False; ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('gray'); ax_3d.yaxis.pane.set_edgecolor('gray'); ax_3d.zaxis.pane.set_edgecolor('gray')
    ax_3d.grid(True, alpha=0.3)
    canvas_3d.draw()
    buf = np.frombuffer(canvas_3d.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas_3d.get_width_height()[::-1] + (3,))
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def setup_display_windows():
    for name in WINDOW_NAMES:
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('3D Point Cloud', cv2.WINDOW_AUTOSIZE)
    window_positions = [(0, 0), (700, 0), (0, 500), (700, 500), (1400, 0)]
    for name, pos in zip(WINDOW_NAMES + ['3D Point Cloud'], window_positions):
        cv2.moveWindow(name, pos[0], pos[1])

# ---------------------------------
# -- MAIN LIVE FEED LOOP (MODIFIED) --
# ---------------------------------

def main():
    global pipeline
    try:
        initialize_camera()
        setup_display_windows()
        print(f"🚀 Starting LIVE FEED (Known Z Compensation: {KNOWN_DEPTH_Z:.2f}m, Y Factor: {Y_CORRECTION_FACTOR:.2f})...")
        print("📺 Multiple windows will show different processing stages")
        print("⏹️  Press 'q' in any window or ESC to quit")

        frame_count = 0
        while True:
            result = process_frame_live()

            if result is not None:
                frame_count += 1
                scale = DISPLAY_SCALE

                vertical_height_cm = result['vertical_height_cm']
                top_point_3d = result['top_point_3d']
                centerline_pts = result['centerline_pts']

                # Overlay Height and Top Position on Live Feed
                top_text_coord = "TOP: Not Found"
                if top_point_3d is not None:
                    # Note: We display the Z as compensated (0.71m) for clarity
                    top_text_coord = (f'TOP: X={top_point_3d[0]:.3f} Y={top_point_3d[1]:.3f} Z={top_point_3d[2]:.3f} m (Corrected)')

                cv2.putText(result['color'], f'HEIGHT: {vertical_height_cm:.1f} cm', (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.putText(result['color'], top_text_coord, (20, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


                cv2.imshow('Live Feed', cv2.resize(result['color'], None, fx=scale, fy=scale))
                cv2.imshow('HSV Space', cv2.resize(result['hsv'], None, fx=scale, fy=scale))
                cv2.imshow('Color Mask', cv2.resize(result['mask'], None, fx=scale, fy=scale))
                cv2.imshow('Segmented', cv2.resize(result['segmented'], None, fx=scale, fy=scale))

                pointcloud_img = create_3d_pointcloud_image(
                    result['points_3d'],
                    result['colors_3d'],
                    centerline_pts,
                    top_point_3d,
                    vertical_height_cm
                )
                cv2.imshow('3D Point Cloud', pointcloud_img)

                if frame_count % 100 == 0:
                    print(f"📊 Frame {frame_count}: {result['fps']:.1f} FPS, "
                          f"Height: {vertical_height_cm:.1f} cm, {top_text_coord}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("🛑 Quitting live feed...")
                break

    except Exception as e:
        print(f"❌ Error in live feed: {e}")
    finally:
        if pipeline:
            print("🔌 Stopping camera...")
            pipeline.stop()
        cv2.destroyAllWindows()
        print("✅ Live feed stopped successfully")

if __name__ == "__main__":
    main()
