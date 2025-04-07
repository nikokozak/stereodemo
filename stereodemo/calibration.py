import cv2
import numpy as np
from pathlib import Path
import time
import json
from typing import List, Tuple, Optional
from .methods import Calibration
from .visualizer import InputPair

class StereoCalibrator:
    def __init__(self, 
                 checkerboard_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0):
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints_left = []  # 2D points in left image plane
        self.imgpoints_right = []  # 2D points in right image plane
        
        self.image_size = None
        
    def detect_corners(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Returns (success, corners_left, corners_right)"""
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left, gray_right = img_left, img_right
            
        if self.image_size is None:
            self.image_size = gray_left.shape[::-1]
        
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.checkerboard_size, flags)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.checkerboard_size, flags)
        
        if ret_left and ret_right:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            return True, corners_left, corners_right
        return False, None, None
    
    def add_calibration_pair(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns (success, visualization_image)"""
        success, corners_left, corners_right = self.detect_corners(img_left, img_right)
        if success:
            self.objpoints.append(self.objp)
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            
            # Create visualization
            vis_left = cv2.drawChessboardCorners(img_left.copy(), self.checkerboard_size, corners_left, True)
            vis_right = cv2.drawChessboardCorners(img_right.copy(), self.checkerboard_size, corners_right, True)
            vis = np.hstack((vis_left, vis_right))
            return True, vis
        return False, None
    
    def calibrate(self) -> Optional[Calibration]:
        """Perform stereo calibration and return Calibration object"""
        if len(self.objpoints) < 5:
            print("Need at least 5 valid image pairs for calibration")
            return None
            
        # Calibrate each camera individually
        ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, self.image_size, None, None)
        ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, self.image_size, None, None)
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right, self.image_size,
            flags=flags)
            
        # Get rectification transforms
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right, self.image_size, R, T)
            
        # Create Calibration object
        baseline_meters = float(abs(T[0]) / 1000.0)  # Convert from mm to meters and ensure float type
        calib = Calibration(
            width=int(self.image_size[0]),
            height=int(self.image_size[1]),
            fx=float(P1[0,0]),  # Focal length from rectified projection matrix
            fy=float(P1[1,1]),
            cx0=float(P1[0,2]),  # Principal point from left camera
            cx1=float(P2[0,2]),  # Principal point from right camera
            cy=float(P1[1,2]),
            baseline_meters=baseline_meters,
            depth_range=(0.3, 20.0),  # Default depth range is already Python native types
            left_image_rect_normalized=np.array([
                float(roi_left[0])/float(self.image_size[0]),
                float(roi_left[1])/float(self.image_size[1]),
                float(roi_left[0] + roi_left[2])/float(self.image_size[0]),
                float(roi_left[1] + roi_left[3])/float(self.image_size[1])
            ])
        )
        return calib

def calibrate_from_source(source, num_pairs: int = 15, output_path: Path = None) -> Optional[Calibration]:
    """Calibrate from a stereo source (like SyphonSource)"""
    calibrator = StereoCalibrator()
    collected = 0
    last_success_time = 0
    delay_between_captures = 1.0  # seconds
    
    print(f"Starting calibration. Need {num_pairs} valid image pairs.")
    print("Press 'q' to quit, 'c' to capture when checkerboard is detected")
    
    while collected < num_pairs:
        pair = source.get_next_pair()
        if not pair.has_data():
            continue
            
        success, corners_left, corners_right = calibrator.detect_corners(pair.left_image, pair.right_image)
        vis = pair.left_image.copy()
        
        if success:
            cv2.putText(vis, "Checkerboard detected! Press 'c' to capture", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No checkerboard detected", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.putText(vis, f"Collected: {collected}/{num_pairs}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow("Calibration", vis)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c') and success and (time.time() - last_success_time) > delay_between_captures:
            success, vis = calibrator.add_calibration_pair(pair.left_image, pair.right_image)
            if success:
                collected += 1
                last_success_time = time.time()
                if vis is not None:
                    cv2.imshow("Last captured pair", vis)
                print(f"Captured pair {collected}/{num_pairs}")
    
    cv2.destroyAllWindows()
    
    if collected < 5:
        print("Not enough image pairs collected for calibration")
        return None
        
    print("Calibrating...")
    calib = calibrator.calibrate()
    
    if calib and output_path:
        with open(output_path, 'w') as f:
            f.write(calib.to_json())
        print(f"Calibration saved to {output_path}")
    
    return calib 