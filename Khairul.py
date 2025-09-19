import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d


class L515SimpleMeasurement:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None

        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def start(self):
        """Start the camera pipeline"""
        profile = self.pipeline.start(self.config)

        # Create align object
        self.align = rs.align(rs.stream.color)

        print("Camera started. Resolution: 640x480")

    def get_frames(self):
        """Get aligned frames"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image, depth_frame, color_frame

    def create_roi_point_cloud(self, depth_frame, color_frame, roi):
        """Create point cloud for ROI"""
        x, y, w, h = roi

        # Get intrinsics
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # L515 depth scale (usually 0.00025m = 0.25mm per unit)
        depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

        points = []
        colors = []

        # Sample points from ROI - use step for faster processing
        step = 2  # Sample every 2nd pixel for speed
        for py in range(y, min(y + h, depth_image.shape[0]), step):
            for px in range(x, min(x + w, depth_image.shape[1]), step):
                depth_value = depth_image[py, px]
                if depth_value == 0:
                    continue

                depth_in_meters = depth_value * depth_scale

                if 0.2 < depth_in_meters < 1.5:  # Valid range for L515
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_in_meters)
                    points.append(point)

                    if py < color_image.shape[0] and px < color_image.shape[1]:
                        colors.append(color_image[py, px] / 255.0)

        if len(points) < 10:
            return None

        print(f"Created point cloud with {len(points)} points")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        return pcd

    def simple_measure(self, pcd):
        """Simple measurement without ground plane removal"""
        if pcd is None or len(pcd.points) < 10:
            return None

        # Remove outliers first
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(pcd.points) < 10:
            return None

        # Get bounding box
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()

        # Get points array for more analysis
        points = np.asarray(pcd.points)

        # Calculate dimensions
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])

        # Convert to cm
        results = {
            'width_cm': x_range * 100,
            'height_cm': y_range * 100,
            'depth_cm': z_range * 100,
            'diagonal_cm': np.sqrt(x_range ** 2 + y_range ** 2 + z_range ** 2) * 100,
            'max_dimension_cm': max(x_range, y_range, z_range) * 100,
            'point_count': len(pcd.points)
        }

        # Visualize
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        o3d.visualization.draw_geometries([pcd, bbox, coord_frame],
                                          window_name="Simple Measurement",
                                          width=800, height=600)

        return results

    def stop(self):
        self.pipeline.stop()


class ROISelector:
    def __init__(self):
        self.roi_selecting = False
        self.roi_selected = False
        self.roi_start = None
        self.roi_end = None
        self.current_roi = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_selecting = True
            self.roi_selected = False
            self.roi_start = (x, y)
            self.roi_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.roi_selecting:
                self.roi_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_selecting = False
            self.roi_selected = True
            self.roi_end = (x, y)
            x1 = min(self.roi_start[0], self.roi_end[0])
            y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0])
            y2 = max(self.roi_start[1], self.roi_end[1])
            self.current_roi = (x1, y1, x2 - x1, y2 - y1)

    def draw_roi(self, image):
        if self.roi_selecting and self.roi_start and self.roi_end:
            cv2.rectangle(image, self.roi_start, self.roi_end, (0, 255, 0), 2)
        elif self.roi_selected and self.roi_start and self.roi_end:
            cv2.rectangle(image, self.roi_start, self.roi_end, (0, 255, 255), 2)
            cv2.putText(image, "Press 'm' to measure",
                        (self.roi_start[0], self.roi_start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def reset(self):
        self.roi_selected = False
        self.current_roi = None


if __name__ == "__main__":
    measurer = L515SimpleMeasurement()
    measurer.start()

    print("\n=== L515 Simple Length Measurement ===")
    print("\nTest objects to try:")
    print("- A pen or pencil")
    print("- A ruler (to verify accuracy)")
    print("- Your finger (hold still)")
    print("- A book spine")
    print("\nControls:")
    print("- Click and drag to select object")
    print("- 'm' = measure")
    print("- 'r' = reset")
    print("- 'q' = quit")

    roi_selector = ROISelector()

    cv2.namedWindow('Color')
    cv2.setMouseCallback('Color', roi_selector.mouse_callback)

    try:
        while True:
            color_image, depth_image, depth_frame, color_frame = measurer.get_frames()

            if color_image is None:
                continue

            display_image = color_image.copy()
            roi_selector.draw_roi(display_image)

            cv2.putText(display_image, "L515 Simple Measurement", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Color', display_image)

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('Depth', depth_colormap)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('m') and roi_selector.roi_selected:
                print("\n--- Measuring ---")
                pcd = measurer.create_roi_point_cloud(depth_frame, color_frame,
                                                      roi_selector.current_roi)
                if pcd:
                    results = measurer.simple_measure(pcd)
                    if results:
                        print(f"\n📏 MEASUREMENTS:")
                        print(f"  Width:  {results['width_cm']:.1f} cm")
                        print(f"  Height: {results['height_cm']:.1f} cm")
                        print(f"  Depth:  {results['depth_cm']:.1f} cm")
                        print(f"  Max dimension: {results['max_dimension_cm']:.1f} cm")
                        print(f"  Points used: {results['point_count']}")
                else:
                    print("Insufficient points captured")
            elif key == ord('r'):
                roi_selector.reset()
                print("Selection reset")

    finally:
        measurer.stop()
        cv2.destroyAllWindows()
