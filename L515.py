"""
L515 Simple Data Extractor - Works around API limitations
Uses known L515 specifications
"""

import pyrealsense2 as rs
import numpy as np
import cv2


def extract_l515_data_simple():
    """
    Simple L515 data extraction using known specifications
    """
    print("=== L515 DATA EXTRACTOR (SIMPLIFIED) ===")

    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()

    # L515 native resolutions
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    print("Starting L515...")
    profile = pipeline.start(config)

    try:
        # Get device info
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)

        print(f"Device: {device_name}")
        print(f"Serial: {serial}")

        # L515 known depth scale (from Intel specifications)
        # L515 uses 0.25mm per unit typically
        DEPTH_SCALE = 0.00025  # meters per unit
        print(f"Using L515 standard depth scale: {DEPTH_SCALE} meters per unit")

        # Let camera stabilize
        print("\nStabilizing camera...")
        for i in range(30):
            pipeline.wait_for_frames()

        # Capture frame
        print("Capturing frame...")
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Failed to get frames!")
            return

        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        print(f"\n=== FRAME INFO ===")
        print(f"Depth: {depth_image.shape}, dtype: {depth_image.dtype}")
        print(f"Color: {color_image.shape}, dtype: {color_image.dtype}")

        # Raw data analysis
        print(f"\n=== RAW DEPTH ANALYSIS ===")
        print(f"Raw depth range: {depth_image.min()} to {depth_image.max()}")

        # Convert to meters
        depth_meters = depth_image.astype(np.float32) * DEPTH_SCALE
        valid_mask = depth_image > 0
        valid_depths = depth_meters[valid_mask]

        print(
            f"Valid pixels: {np.sum(valid_mask)} / {depth_image.size} ({100 * np.sum(valid_mask) / depth_image.size:.1f}%)")
        print(f"Distance range: {valid_depths.min():.3f}m to {valid_depths.max():.3f}m")
        print(f"Average distance: {valid_depths.mean():.3f}m")

        # Your setup analysis
        print(f"\n=== YOUR SETUP MEASUREMENTS ===")
        h, w = depth_image.shape

        # Sample key points
        points_to_check = [
            (w // 2, h // 2, "Center (Wooden Block)"),
            (w // 4, h // 2, "Left (Aluminum Rail)"),
            (3 * w // 4, h // 2, "Right Side"),
            (w // 2, h // 4, "Upper Center"),
            (w // 2, 3 * h // 4, "Lower Center")
        ]

        measurements = {}

        for x, y, label in points_to_check:
            raw_value = depth_image[y, x]
            distance_m = raw_value * DEPTH_SCALE
            distance_cm = distance_m * 100

            measurements[label] = {
                'raw': raw_value,
                'meters': distance_m,
                'cm': distance_cm
            }

            print(f"{label}: {raw_value} → {distance_cm:.1f}cm")

        # Show 10x10 raw data sample from center
        print(f"\n=== RAW DATA SAMPLE (Center 10x10) ===")
        cx, cy = w // 2, h // 2
        sample = depth_image[cy - 5:cy + 5, cx - 5:cx + 5]
        print("Raw 16-bit values:")
        print(sample)
        print("\nConverted to centimeters:")
        sample_cm = sample.astype(np.float32) * DEPTH_SCALE * 100
        print(sample_cm.astype(np.int32))

        # Ruler validation
        print(f"\n=== RULER VALIDATION ===")
        center_dist = measurements["Center (Wooden Block)"]['cm']
        rail_dist = measurements["Left (Aluminum Rail)"]['cm']

        print(f"Camera to wooden block: {center_dist:.1f}cm")
        print(f"Camera to aluminum rail: {rail_dist:.1f}cm")
        print(f"Difference: {abs(center_dist - rail_dist):.1f}cm")
        print("\nCompare these with your ruler measurements!")

        # Create simple visualization
        print(f"\n=== CREATING VISUALIZATION ===")

        # Colorize depth for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Resize color to match depth
        color_resized = cv2.resize(color_image, (w, h))

        # Combine side by side
        combined = np.hstack((color_resized, depth_colormap))

        # Add text overlay with measurements
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f"Wooden Block: {center_dist:.1f}cm", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"Rail: {rail_dist:.1f}cm", (10, 60), font, 0.7, (255, 255, 255), 2)

        # Save results
        cv2.imwrite('l515_analysis.png', combined)
        np.save('raw_depth_data.npy', depth_image)

        print("Saved: l515_analysis.png, raw_depth_data.npy")

        # Show for 5 seconds
        cv2.imshow('L515 Analysis - Your Setup', combined)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        # Precision analysis
        print(f"\n=== PRECISION ANALYSIS ===")
        center_region = depth_image[cy - 10:cy + 10, cx - 10:cx + 10]
        center_valid = center_region[center_region > 0]
        if len(center_valid) > 0:
            center_std = np.std(center_valid) * DEPTH_SCALE * 100  # in cm
            print(f"Center region precision: ±{center_std:.2f}cm")
            print("(This shows the L515's measurement consistency)")

        return measurements

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        pipeline.stop()
        print("\nDone!")


if __name__ == "__main__":
    results = extract_l515_data_simple()
