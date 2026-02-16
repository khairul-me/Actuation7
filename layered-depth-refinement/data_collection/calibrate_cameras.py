"""
Stereo Camera Calibration

Calibrate intrinsics and extrinsics for dual RealSense D405 cameras.

Usage:
    python data_collection/calibrate_cameras.py --cam1_serial <serial1> --cam2_serial <serial2>
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import cv2


def calibrate_intrinsics(image_dir, checkerboard_size=(9, 6), square_size=0.025):
    """
    Calibrate camera intrinsics from checkerboard images.
    
    Args:
        image_dir: directory containing checkerboard images
        checkerboard_size: (cols, rows) inner corners
        square_size: size of each square in meters
    
    Returns:
        K: 3x3 camera matrix
        dist: distortion coefficients
        error: mean reprojection error
    """
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    obj_points = []
    img_points = []
    
    images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) + 
                    glob.glob(os.path.join(image_dir, '*.png')))
    
    img_size = None
    
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_size is None:
            img_size = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
    
    if len(obj_points) < 5:
        print(f"Warning: Only {len(obj_points)} valid images found. Need at least 5.")
        return None, None, None
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    
    print(f"Intrinsic calibration: {len(obj_points)} images, error = {ret:.4f} px")
    
    return K, dist, ret


def calibrate_stereo(images_cam1, images_cam2, K1, dist1, K2, dist2,
                     checkerboard_size=(9, 6), square_size=0.025):
    """
    Calibrate stereo extrinsics between two cameras.
    
    Returns:
        R: 3x3 rotation matrix
        T: 3x1 translation vector
        E: essential matrix
        F: fundamental matrix
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    obj_points = []
    img_points1 = []
    img_points2 = []
    img_size = None
    
    for img1_path, img2_path in zip(sorted(images_cam1), sorted(images_cam2)):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        if img_size is None:
            img_size = gray1.shape[::-1]
        
        ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_size, None)
        
        if ret1 and ret2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points1.append(corners1)
            img_points2.append(corners2)
    
    if len(obj_points) < 5:
        print(f"Warning: Only {len(obj_points)} valid stereo pairs. Need at least 5.")
        return None, None, None, None
    
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2,
        K1, dist1, K2, dist2,
        img_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    baseline = np.linalg.norm(T)
    print(f"Stereo calibration: error = {ret:.4f} px, baseline = {baseline*100:.1f} cm")
    
    return R, T, E, F


def main():
    parser = argparse.ArgumentParser(description='Stereo camera calibration')
    parser.add_argument('--cam1_dir', type=str, default='calib/top',
                       help='Directory with camera 1 checkerboard images')
    parser.add_argument('--cam2_dir', type=str, default='calib/angled',
                       help='Directory with camera 2 checkerboard images')
    parser.add_argument('--checkerboard', type=str, default='9x6',
                       help='Checkerboard size (cols x rows)')
    parser.add_argument('--square_size', type=float, default=0.025,
                       help='Square size in meters')
    parser.add_argument('--output', type=str, default='calibration.json')
    args = parser.parse_args()
    
    cb_size = tuple(map(int, args.checkerboard.split('x')))
    
    print("=" * 50)
    print("Stereo Camera Calibration")
    print("=" * 50)
    
    # Calibrate intrinsics
    print("\n--- Camera 1 (Top) ---")
    K1, dist1, err1 = calibrate_intrinsics(args.cam1_dir, cb_size, args.square_size)
    
    print("\n--- Camera 2 (Angled) ---")
    K2, dist2, err2 = calibrate_intrinsics(args.cam2_dir, cb_size, args.square_size)
    
    if K1 is None or K2 is None:
        print("\nCalibration failed. Not enough valid images.")
        return
    
    # Calibrate stereo
    print("\n--- Stereo Calibration ---")
    images1 = sorted(glob.glob(os.path.join(args.cam1_dir, '*.jpg')) + 
                     glob.glob(os.path.join(args.cam1_dir, '*.png')))
    images2 = sorted(glob.glob(os.path.join(args.cam2_dir, '*.jpg')) + 
                     glob.glob(os.path.join(args.cam2_dir, '*.png')))
    
    R, T, E, F = calibrate_stereo(images1, images2, K1, dist1, K2, dist2, cb_size, args.square_size)
    
    if R is None:
        print("\nStereo calibration failed.")
        return
    
    # Save
    calibration = {
        'intrinsics_top': {
            'K': K1.tolist(),
            'dist': dist1.tolist(),
            'fx': float(K1[0, 0]),
            'fy': float(K1[1, 1]),
            'cx': float(K1[0, 2]),
            'cy': float(K1[1, 2]),
            'reprojection_error': float(err1),
        },
        'intrinsics_angled': {
            'K': K2.tolist(),
            'dist': dist2.tolist(),
            'fx': float(K2[0, 0]),
            'fy': float(K2[1, 1]),
            'cx': float(K2[0, 2]),
            'cy': float(K2[1, 2]),
            'reprojection_error': float(err2),
        },
        'extrinsics': {
            'R': R.tolist(),
            'T': T.tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'baseline': float(np.linalg.norm(T)),
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"\nCalibration saved to {args.output}")


if __name__ == '__main__':
    main()

