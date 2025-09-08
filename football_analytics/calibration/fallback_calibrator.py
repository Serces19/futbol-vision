"""
Fallback calibration system for when field calibration fails
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ..core.interfaces import BaseCalibrator
from ..core.models import FieldLine, KeyPoint
from ..core.config import FieldDimensions
from ..core.exceptions import CalibrationError


class FallbackCalibrator(BaseCalibrator):
    """
    Fallback calibration system that provides default field dimensions
    and identity transformation when proper calibration fails.
    
    This calibrator is used as a backup when the main FieldCalibrator
    cannot establish a proper homography transformation.
    """
    
    def __init__(
        self,
        field_dimensions: FieldDimensions,
        default_frame_size: Tuple[int, int] = (1920, 1080),
        pixels_per_meter: float = 10.0
    ):
        """
        Initialize the FallbackCalibrator.
        
        Args:
            field_dimensions: Standard field dimensions in meters
            default_frame_size: Default frame size (width, height) in pixels
            pixels_per_meter: Default scaling factor for pixel to meter conversion
        
        Raises:
            ValueError: If parameters are invalid
        """
        self.field_dimensions = field_dimensions
        self.default_frame_size = self._validate_frame_size(default_frame_size)
        self.pixels_per_meter = self._validate_pixels_per_meter(pixels_per_meter)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Fallback calibration state
        self.is_fallback_active = False
        self.fallback_homography: Optional[np.ndarray] = None
        self.fallback_inverse_homography: Optional[np.ndarray] = None
        self.calibration_warnings: List[str] = []
        
        # Setup default transformation
        self._setup_default_transformation()
    
    def _validate_frame_size(self, frame_size: Tuple[int, int]) -> Tuple[int, int]:
        """Validate frame size parameters"""
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Frame size must be positive, got {frame_size}")
        return frame_size
    
    def _validate_pixels_per_meter(self, pixels_per_meter: float) -> float:
        """Validate pixels per meter parameter"""
        if pixels_per_meter <= 0:
            raise ValueError(f"Pixels per meter must be positive, got {pixels_per_meter}")
        return pixels_per_meter
    
    def _setup_default_transformation(self) -> None:
        """Setup default transformation matrix for fallback mode"""
        # Create a simple scaling transformation that maps the field
        # to the center of the default frame size
        
        frame_width, frame_height = self.default_frame_size
        field_length = self.field_dimensions.length
        field_width = self.field_dimensions.width
        
        # Calculate scaling to fit field in frame with some margin
        margin_factor = 0.8  # Use 80% of frame to leave margins
        scale_x = (frame_width * margin_factor) / field_length
        scale_y = (frame_height * margin_factor) / field_width
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Center the field in the frame
        offset_x = frame_width / 2
        offset_y = frame_height / 2
        
        # Create homography matrix for field-to-pixel transformation
        # This maps field coordinates (meters) to pixel coordinates
        self.fallback_homography = np.array([
            [scale, 0.0, offset_x],
            [0.0, scale, offset_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Compute inverse for pixel-to-field transformation
        self.fallback_inverse_homography = np.linalg.inv(self.fallback_homography)
        
        self.logger.info(
            f"Setup fallback transformation: scale={scale:.2f}, "
            f"offset=({offset_x:.1f}, {offset_y:.1f})"
        )
    
    def calibrate(self, field_lines: List[FieldLine], key_points: List[KeyPoint]) -> Optional[np.ndarray]:
        """
        Activate fallback calibration mode.
        
        Args:
            field_lines: Detected field lines (ignored in fallback mode)
            key_points: Detected field keypoints (ignored in fallback mode)
            
        Returns:
            Fallback homography matrix
        """
        self.logger.warning(
            "Activating fallback calibration mode. "
            "Using default field dimensions and identity transformation."
        )
        
        self.is_fallback_active = True
        self.calibration_warnings = [
            "Fallback calibration active - using default field mapping",
            "Coordinate transformations may not be accurate",
            "Consider improving field detection for better calibration"
        ]
        
        return self.fallback_homography.copy()
    
    def transform_to_field_coordinates(self, pixel_coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Transform pixel coordinates to field coordinates using fallback transformation.
        
        Args:
            pixel_coords: List of (x, y) pixel coordinates
            
        Returns:
            List of (x, y) field coordinates in meters
        """
        if not self.is_fallback_active:
            raise CalibrationError("Fallback calibration not active. Call calibrate() first.")
        
        try:
            # Convert to numpy array and add homogeneous coordinate
            pixel_array = np.array(pixel_coords, dtype=np.float32)
            ones = np.ones((len(pixel_coords), 1), dtype=np.float32)
            homogeneous_pixels = np.hstack([pixel_array, ones])
            
            # Transform using inverse homography
            field_homogeneous = homogeneous_pixels @ self.fallback_inverse_homography.T
            
            # Convert back to Cartesian coordinates
            field_coords = field_homogeneous[:, :2] / field_homogeneous[:, 2:3]
            
            return [(float(x), float(y)) for x, y in field_coords]
            
        except Exception as e:
            raise CalibrationError(f"Failed to transform coordinates in fallback mode: {str(e)}")
    
    def transform_to_pixel_coordinates(self, field_coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Transform field coordinates to pixel coordinates using fallback transformation.
        
        Args:
            field_coords: List of (x, y) field coordinates in meters
            
        Returns:
            List of (x, y) pixel coordinates
        """
        if not self.is_fallback_active:
            raise CalibrationError("Fallback calibration not active. Call calibrate() first.")
        
        try:
            # Convert to numpy array and add homogeneous coordinate
            field_array = np.array(field_coords, dtype=np.float32)
            ones = np.ones((len(field_coords), 1), dtype=np.float32)
            homogeneous_field = np.hstack([field_array, ones])
            
            # Transform using homography
            pixel_homogeneous = homogeneous_field @ self.fallback_homography.T
            
            # Convert back to Cartesian coordinates
            pixel_coords = pixel_homogeneous[:, :2] / pixel_homogeneous[:, 2:3]
            
            return [(int(x), int(y)) for x, y in pixel_coords]
            
        except Exception as e:
            raise CalibrationError(f"Failed to transform coordinates in fallback mode: {str(e)}")
    
    def is_calibrated(self) -> bool:
        """
        Check if fallback calibration is active.
        
        Returns:
            True if fallback calibration is active, False otherwise
        """
        return self.is_fallback_active
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Get fallback calibration information.
        
        Returns:
            Dictionary with calibration details
        """
        return {
            "is_calibrated": self.is_calibrated(),
            "calibration_type": "fallback",
            "calibration_quality": 0.5,  # Medium quality for fallback
            "homography_matrix": self.fallback_homography.tolist() if self.fallback_homography is not None else None,
            "field_dimensions": {
                "length": self.field_dimensions.length,
                "width": self.field_dimensions.width,
                "goal_width": self.field_dimensions.goal_width,
                "goal_height": self.field_dimensions.goal_height
            },
            "calibration_warnings": self.calibration_warnings,
            "default_frame_size": self.default_frame_size,
            "pixels_per_meter": self.pixels_per_meter,
            "fallback_active": self.is_fallback_active
        }
    
    def get_field_boundaries_pixels(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get field boundary coordinates in pixel space using fallback transformation.
        
        Returns:
            List of field boundary points in pixels if calibrated, None otherwise
        """
        if not self.is_calibrated():
            return None
        
        # Define field boundary in field coordinates
        half_length = self.field_dimensions.length / 2
        half_width = self.field_dimensions.width / 2
        
        field_boundary = [
            (-half_length, -half_width),  # Top-left
            (half_length, -half_width),   # Top-right
            (half_length, half_width),    # Bottom-right
            (-half_length, half_width)    # Bottom-left
        ]
        
        try:
            pixel_boundary = self.transform_to_pixel_coordinates(field_boundary)
            return pixel_boundary
        except CalibrationError:
            return None
    
    def validate_point_in_field(self, field_coord: Tuple[float, float]) -> bool:
        """
        Check if a field coordinate is within the field boundaries.
        
        Args:
            field_coord: Field coordinate (x, y) in meters
            
        Returns:
            True if point is within field, False otherwise
        """
        x, y = field_coord
        half_length = self.field_dimensions.length / 2
        half_width = self.field_dimensions.width / 2
        
        return (-half_length <= x <= half_length and 
                -half_width <= y <= half_width)
    
    def reset_calibration(self) -> None:
        """Reset fallback calibration state"""
        self.is_fallback_active = False
        self.calibration_warnings.clear()
        self.logger.info("Fallback calibration reset")
    
    def update_frame_size(self, frame_size: Tuple[int, int]) -> None:
        """
        Update the default frame size and recalculate transformation.
        
        Args:
            frame_size: New frame size (width, height) in pixels
        """
        self.default_frame_size = self._validate_frame_size(frame_size)
        self._setup_default_transformation()
        
        if self.is_fallback_active:
            self.logger.info(f"Updated fallback transformation for frame size: {frame_size}")
    
    def __repr__(self) -> str:
        """String representation of the fallback calibrator"""
        status = "active" if self.is_fallback_active else "inactive"
        return f"FallbackCalibrator(status={status}, frame_size={self.default_frame_size})"


class CalibrationQualityAssessment:
    """
    Utility class for assessing calibration quality and determining
    when to use fallback calibration.
    """
    
    def __init__(
        self,
        min_quality_threshold: float = 0.6,
        min_lines_threshold: int = 4,
        min_keypoints_threshold: int = 2
    ):
        """
        Initialize quality assessment parameters.
        
        Args:
            min_quality_threshold: Minimum quality score to accept calibration
            min_lines_threshold: Minimum number of lines required
            min_keypoints_threshold: Minimum number of keypoints required
        """
        self.min_quality_threshold = min_quality_threshold
        self.min_lines_threshold = min_lines_threshold
        self.min_keypoints_threshold = min_keypoints_threshold
        
        self.logger = logging.getLogger(__name__)
    
    def assess_field_elements(
        self, 
        field_lines: List[FieldLine], 
        key_points: List[KeyPoint]
    ) -> Dict[str, Any]:
        """
        Assess the quality of detected field elements.
        
        Args:
            field_lines: Detected field lines
            key_points: Detected field keypoints
            
        Returns:
            Assessment results dictionary
        """
        assessment = {
            "sufficient_elements": True,
            "quality_score": 1.0,
            "warnings": [],
            "recommendations": [],
            "use_fallback": False
        }
        
        # Check quantity of elements
        if len(field_lines) < self.min_lines_threshold:
            assessment["sufficient_elements"] = False
            assessment["warnings"].append(
                f"Insufficient field lines: {len(field_lines)} < {self.min_lines_threshold}"
            )
        
        if len(key_points) < self.min_keypoints_threshold:
            assessment["sufficient_elements"] = False
            assessment["warnings"].append(
                f"Insufficient keypoints: {len(key_points)} < {self.min_keypoints_threshold}"
            )
        
        # Assess quality of elements
        if field_lines:
            line_confidences = [line.confidence for line in field_lines]
            avg_line_confidence = np.mean(line_confidences)
            assessment["quality_score"] *= avg_line_confidence
        else:
            assessment["quality_score"] = 0.0
        
        if key_points:
            keypoint_confidences = [kp.confidence for kp in key_points]
            avg_keypoint_confidence = np.mean(keypoint_confidences)
            assessment["quality_score"] *= avg_keypoint_confidence
        else:
            assessment["quality_score"] *= 0.5  # Penalty for no keypoints
        
        # Check for line type diversity
        if field_lines:
            line_types = set(line.line_type for line in field_lines)
            required_types = {"sideline", "goal_line"}
            missing_types = required_types - line_types
            
            if missing_types:
                assessment["warnings"].append(f"Missing line types: {missing_types}")
                assessment["quality_score"] *= 0.8
        
        # Determine if fallback should be used
        if (not assessment["sufficient_elements"] or 
            assessment["quality_score"] < self.min_quality_threshold):
            assessment["use_fallback"] = True
            assessment["recommendations"].append("Use fallback calibration")
        
        return assessment
    
    def assess_calibration_result(
        self, 
        homography_matrix: Optional[np.ndarray],
        calibration_quality: float,
        calibration_warnings: List[str]
    ) -> Dict[str, Any]:
        """
        Assess the result of a calibration attempt.
        
        Args:
            homography_matrix: Computed homography matrix
            calibration_quality: Quality score from calibration
            calibration_warnings: Warnings from calibration process
            
        Returns:
            Assessment results dictionary
        """
        assessment = {
            "calibration_successful": False,
            "quality_acceptable": False,
            "use_fallback": True,
            "warnings": calibration_warnings.copy(),
            "recommendations": []
        }
        
        if homography_matrix is not None:
            assessment["calibration_successful"] = True
            
            if calibration_quality >= self.min_quality_threshold:
                assessment["quality_acceptable"] = True
                assessment["use_fallback"] = False
                assessment["recommendations"].append("Use computed calibration")
            else:
                assessment["warnings"].append(
                    f"Low calibration quality: {calibration_quality:.3f} < {self.min_quality_threshold}"
                )
                assessment["recommendations"].append("Consider using fallback calibration")
        else:
            assessment["warnings"].append("Homography computation failed")
            assessment["recommendations"].append("Use fallback calibration")
        
        return assessment


def create_fallback_calibrator_from_config(
    field_dimensions: FieldDimensions,
    frame_size: Optional[Tuple[int, int]] = None
) -> FallbackCalibrator:
    """
    Factory function to create FallbackCalibrator from configuration.
    
    Args:
        field_dimensions: Field dimensions configuration
        frame_size: Optional frame size, uses default if not provided
    
    Returns:
        Configured FallbackCalibrator instance
    """
    kwargs = {"field_dimensions": field_dimensions}
    if frame_size is not None:
        kwargs["default_frame_size"] = frame_size
    
    return FallbackCalibrator(**kwargs)