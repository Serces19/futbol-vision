"""
PnLCalib integration for football field calibration
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import torch
import yaml

# Add PnLCalib to path
PNLCALIB_PATH = Path(__file__).parent.parent.parent / "PnLCalib"
if PNLCALIB_PATH.exists():
    sys.path.insert(0, str(PNLCALIB_PATH))

try:
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (
        get_keypoints_from_heatmap_batch_maxpool, 
        get_keypoints_from_heatmap_batch_maxpool_l,
        complete_keypoints, 
        coords_to_dict
    )
    import torchvision.transforms as T
    import torchvision.transforms.functional as f
    from PIL import Image
    PNLCALIB_AVAILABLE = True
except ImportError as e:
    PNLCALIB_AVAILABLE = False
    IMPORT_ERROR = str(e)

from ..core.interfaces import BaseCalibrator
from ..core.models import FieldLine, KeyPoint
from ..core.config import FieldDimensions
from ..core.exceptions import CalibrationError


class PnLCalibrator(BaseCalibrator):
    """
    PnLCalib-based field calibrator for football analytics
    """
    
    def __init__(
        self,
        field_dimensions: FieldDimensions,
        weights_kp: str,
        weights_line: str,
        device: str = "cuda",
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        pnl_refine: bool = False
    ):
        """
        Initialize PnLCalibrator
        
        Args:
            field_dimensions: Field dimensions (not used by PnLCalib but kept for compatibility)
            weights_kp: Path to keypoint detection model
            weights_line: Path to line detection model
            device: Device to run inference on
            kp_threshold: Threshold for keypoint detection
            line_threshold: Threshold for line detection
            pnl_refine: Enable PnL refinement module
        """
        
        if not PNLCALIB_AVAILABLE:
            raise CalibrationError(f"PnLCalib not available: {IMPORT_ERROR}")
        
        self.field_dimensions = field_dimensions
        self.weights_kp = weights_kp
        self.weights_line = weights_line
        self.device = device
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        self.pnl_refine = pnl_refine
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model state
        self.model_kp = None
        self.model_line = None
        self.transform = None
        self.cam_calibrator = None
        
        # Calibration state
        self.is_calibrated_flag = False
        self.last_params = None
        self.projection_matrix = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load PnLCalib models"""
        try:
            # Check if model files exist
            if not Path(self.weights_kp).exists():
                raise CalibrationError(f"Keypoint model not found: {self.weights_kp}")
            if not Path(self.weights_line).exists():
                raise CalibrationError(f"Line model not found: {self.weights_line}")
            
            # Load configurations
            config_path = PNLCALIB_PATH / "config"
            cfg_kp = yaml.safe_load(open(config_path / "hrnetv2_w48.yaml", 'r'))
            cfg_line = yaml.safe_load(open(config_path / "hrnetv2_w48_l.yaml", 'r'))
            
            # Load keypoint model
            self.logger.info(f"Loading keypoint model: {self.weights_kp}")
            loaded_state_kp = torch.load(self.weights_kp, map_location=self.device)
            self.model_kp = get_cls_net(cfg_kp)
            self.model_kp.load_state_dict(loaded_state_kp)
            self.model_kp.to(self.device)
            self.model_kp.eval()
            
            # Load line model
            self.logger.info(f"Loading line model: {self.weights_line}")
            loaded_state_line = torch.load(self.weights_line, map_location=self.device)
            self.model_line = get_cls_net_l(cfg_line)
            self.model_line.load_state_dict(loaded_state_line)
            self.model_line.to(self.device)
            self.model_line.eval()
            
            # Setup transform
            self.transform = T.Resize((540, 960))
            
            self.logger.info("PnLCalib models loaded successfully")
            
        except Exception as e:
            raise CalibrationError(f"Failed to load PnLCalib models: {str(e)}")
    
    def calibrate(self, field_lines: List[FieldLine], key_points: List[KeyPoint]) -> Optional[np.ndarray]:
        """
        Calibrate using PnLCalib (ignores input field_lines and key_points)
        
        Note: This method requires a frame to be set first using set_frame()
        """
        raise NotImplementedError(
            "PnLCalibrator requires frame-based calibration. Use calibrate_frame() instead."
        )
    
    def calibrate_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Calibrate field using a frame with PnLCalib
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Projection matrix if calibration successful, None otherwise
        """
        try:
            # Initialize camera calibrator if needed
            if self.cam_calibrator is None:
                h, w = frame.shape[:2]
                self.cam_calibrator = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
            
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = f.to_tensor(frame_pil).float().unsqueeze(0)
            
            # Resize if needed
            if frame_tensor.size()[-1] != 960:
                frame_tensor = self.transform(frame_tensor)
            
            frame_tensor = frame_tensor.to(self.device)
            b, c, h, w = frame_tensor.size()
            
            # Run inference
            with torch.no_grad():
                heatmaps_kp = self.model_kp(frame_tensor)
                heatmaps_line = self.model_line(frame_tensor)
            
            # Extract keypoints and lines
            kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps_kp[:,:-1,:,:])
            line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_line[:,:-1,:,:])
            
            # Convert to dictionaries
            kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
            lines_dict = coords_to_dict(line_coords, threshold=self.line_threshold)
            
            # Complete keypoints
            kp_dict, lines_dict = complete_keypoints(
                kp_dict[0], lines_dict[0], w=w, h=h, normalize=True
            )
            
            # Update camera calibrator
            self.cam_calibrator.update(kp_dict, lines_dict)
            
            # Get calibration parameters
            final_params_dict = self.cam_calibrator.heuristic_voting(refine_lines=self.pnl_refine)
            
            if final_params_dict is not None:
                self.last_params = final_params_dict
                self.projection_matrix = self._projection_from_cam_params(final_params_dict)
                self.is_calibrated_flag = True
                
                self.logger.info("PnLCalib calibration successful")
                return self.projection_matrix
            else:
                self.logger.warning("PnLCalib calibration failed")
                self.is_calibrated_flag = False
                return None
                
        except Exception as e:
            self.logger.error(f"PnLCalib calibration error: {str(e)}")
            self.is_calibrated_flag = False
            return None
    
    def _projection_from_cam_params(self, params_dict: Dict[str, Any]) -> np.ndarray:
        """Convert camera parameters to projection matrix"""
        cam_params = params_dict["cam_params"]
        x_focal_length = cam_params['x_focal_length']
        y_focal_length = cam_params['y_focal_length']
        principal_point = np.array(cam_params['principal_point'])
        position_meters = np.array(cam_params['position_meters'])
        rotation = np.array(cam_params['rotation_matrix'])

        It = np.eye(4)[:-1]
        It[:, -1] = -position_meters
        Q = np.array([[x_focal_length, 0, principal_point[0]],
                      [0, y_focal_length, principal_point[1]],
                      [0, 0, 1]])
        P = Q @ (rotation @ It)

        return P    

    def transform_to_field_coordinates(self, pixel_coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Transform pixel coordinates to field coordinates
        
        Note: PnLCalib uses a different coordinate system. This is an approximation.
        """
        if not self.is_calibrated():
            raise CalibrationError("Not calibrated. Call calibrate_frame() first.")
        
        # This is a simplified transformation - PnLCalib uses 3D world coordinates
        # For a proper implementation, you'd need to implement inverse projection
        field_coords = []
        
        for px, py in pixel_coords:
            # Simplified transformation (placeholder)
            # In a real implementation, you'd use the inverse of the projection matrix
            # and assume a ground plane (z=0)
            
            # For now, return normalized coordinates as an approximation
            if self.cam_calibrator:
                # Use frame dimensions for normalization
                w, h = self.cam_calibrator.iwidth, self.cam_calibrator.iheight
                norm_x = (px - w/2) / (w/2)  # -1 to 1
                norm_y = (py - h/2) / (h/2)  # -1 to 1
                
                # Map to approximate field coordinates (very rough approximation)
                field_x = norm_x * 52.5  # Half field length
                field_y = norm_y * 34.0  # Half field width
                
                field_coords.append((field_x, field_y))
            else:
                field_coords.append((0.0, 0.0))
        
        return field_coords
    
    def is_calibrated(self) -> bool:
        """Check if calibrated"""
        return self.is_calibrated_flag and self.projection_matrix is not None
    
    def get_projection_matrix(self) -> Optional[np.ndarray]:
        """Get the projection matrix"""
        return self.projection_matrix
    
    def get_camera_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the camera parameters"""
        return self.last_params
    
    def project_field_lines_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Project field lines onto the frame using calibration
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with projected field lines
        """
        if not self.is_calibrated():
            return frame
        
        try:
            # Field line coordinates (from PnLCalib)
            lines_coords = [
                [[0., 54.16, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
                [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                [[52.5, 0., 0.], [52.5, 68, 0.]],
                [[0., 68., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [0., 68., 0.]],
                [[105., 0., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [105., 0., 0.]],
                [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
                [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                [[99.5, 24.84, 0.], [105., 24.84, 0.]]
            ]
            
            projected_frame = frame.copy()
            P = self.projection_matrix
            
            # Project lines
            for line in lines_coords:
                w1 = line[0]
                w2 = line[1]
                i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
                i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
                
                if i1[-1] != 0 and i2[-1] != 0:
                    i1 /= i1[-1]
                    i2 /= i2[-1]
                    
                    # Check if points are within frame
                    if (0 <= i1[0] < frame.shape[1] and 0 <= i1[1] < frame.shape[0] and
                        0 <= i2[0] < frame.shape[1] and 0 <= i2[1] < frame.shape[0]):
                        
                        projected_frame = cv2.line(
                            projected_frame, 
                            (int(i1[0]), int(i1[1])), 
                            (int(i2[0]), int(i2[1])), 
                            (255, 0, 0), 3
                        )
            
            # Project center circle
            r = 9.15
            base_pos = np.array([0, 0, 0., 0.])
            circle_points = []
            
            for ang in np.linspace(0, 360, 100):
                ang = np.deg2rad(ang)
                pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
                ipos = P @ pos
                if ipos[-1] != 0:
                    ipos /= ipos[-1]
                    if (0 <= ipos[0] < frame.shape[1] and 0 <= ipos[1] < frame.shape[0]):
                        circle_points.append([int(ipos[0]), int(ipos[1])])
            
            if len(circle_points) > 2:
                circle_points = np.array(circle_points, np.int32)
                projected_frame = cv2.polylines(projected_frame, [circle_points], True, (255, 0, 0), 3)
            
            return projected_frame
            
        except Exception as e:
            self.logger.error(f"Error projecting field lines: {str(e)}")
            return frame
    
    def reset_calibration(self) -> None:
        """Reset calibration state"""
        self.is_calibrated_flag = False
        self.last_params = None
        self.projection_matrix = None
        # Don't reset cam_calibrator as it doesn't have iheight/iwidth attributes
        # The calibrator will be reinitialized on next frame if needed
    
    def __repr__(self) -> str:
        """String representation"""
        status = "calibrated" if self.is_calibrated() else "not calibrated"
        return f"PnLCalibrator({status}, device={self.device})"


def create_pnl_calibrator_from_config(
    field_dimensions: FieldDimensions,
    model_paths: Dict[str, str],
    device: str = "cuda"
) -> PnLCalibrator:
    """
    Factory function to create PnLCalibrator from configuration
    
    Args:
        field_dimensions: Field dimensions
        model_paths: Dictionary with 'keypoints' and 'lines' model paths
        device: Device to run on
        
    Returns:
        Configured PnLCalibrator instance
    """
    return PnLCalibrator(
        field_dimensions=field_dimensions,
        weights_kp=model_paths['keypoints'],
        weights_line=model_paths['lines'],
        device=device
    )