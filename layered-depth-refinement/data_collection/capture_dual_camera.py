"""
Dual Camera Capture Script

Captures synchronized RGB + depth frames from two RealSense D405 cameras.

Usage:
    python data_collection/capture_dual_camera.py --cam1 <serial1> --cam2 <serial2> --output data/raw
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    print("Warning: pyrealsense2 not available. Camera capture disabled.")


def capture_synchronized_frames(cam1_serial, cam2_serial, output_dir,
                                 duration_seconds=60, capture_fps=10):
    """
    Capture synchronized frames from both cameras.
    
    Args:
        cam1_serial: serial number of camera 1 (top)
        cam2_serial: serial number of camera 2 (angled)
        output_dir: output directory
        duration_seconds: capture duration
        capture_fps: frames per second to save
    """
    if not HAS_REALSENSE:
        print("Error: pyrealsense2 required for camera capture")
        return
    
    # Create output directories
    for subdir in ['rgb_top', 'rgb_angled', 'depth_top', 'depth_angled']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Configure cameras
    pipeline1 = rs.pipeline()
    pipeline2 = rs.pipeline()
    
    config1 = rs.config()
    config1.enable_device(cam1_serial)
    config1.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config1.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    
    config2 = rs.config()
    config2.enable_device(cam2_serial)
    config2.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config2.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    
    # Start pipelines
    profile1 = pipeline1.start(config1)
    profile2 = pipeline2.start(config2)
    
    # Align depth to color
    align1 = rs.align(rs.stream.color)
    align2 = rs.align(rs.stream.color)
    
    print(f"Capturing for {duration_seconds}s at {capture_fps} fps...")
    print("Press Ctrl+C to stop early")
    
    frame_count = 0
    saved_count = 0
    save_interval = 30 // capture_fps  # At 30fps, save every N frames
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            # Get frames
            frames1 = pipeline1.wait_for_frames()
            frames2 = pipeline2.wait_for_frames()
            
            # Align
            frames1 = align1.process(frames1)
            frames2 = align2.process(frames2)
            
            frame_count += 1
            
            if frame_count % save_interval != 0:
                continue
            
            # Extract data
            color1 = np.asanyarray(frames1.get_color_frame().get_data())
            depth1 = np.asanyarray(frames1.get_depth_frame().get_data()).astype(np.float32) * 0.001
            
            color2 = np.asanyarray(frames2.get_color_frame().get_data())
            depth2 = np.asanyarray(frames2.get_depth_frame().get_data()).astype(np.float32) * 0.001
            
            # Save
            frame_id = f"{saved_count:06d}"
            
            cv2.imwrite(os.path.join(output_dir, 'rgb_top', f'{frame_id}.jpg'), color1)
            cv2.imwrite(os.path.join(output_dir, 'rgb_angled', f'{frame_id}.jpg'), color2)
            np.save(os.path.join(output_dir, 'depth_top', f'{frame_id}.npy'), depth1)
            np.save(os.path.join(output_dir, 'depth_angled', f'{frame_id}.npy'), depth2)
            
            saved_count += 1
            
            if saved_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Saved {saved_count} frames ({elapsed:.0f}s elapsed)")
    
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    finally:
        pipeline1.stop()
        pipeline2.stop()
    
    print(f"\nCapture complete: {saved_count} frame pairs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Dual camera capture')
    parser.add_argument('--cam1', type=str, required=True, help='Camera 1 (top) serial')
    parser.add_argument('--cam2', type=str, required=True, help='Camera 2 (angled) serial')
    parser.add_argument('--output', type=str, default='data/raw')
    parser.add_argument('--duration', type=int, default=60, help='Capture duration (seconds)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second to save')
    args = parser.parse_args()
    
    capture_synchronized_frames(
        args.cam1, args.cam2, args.output,
        duration_seconds=args.duration,
        capture_fps=args.fps,
    )


if __name__ == '__main__':
    main()

