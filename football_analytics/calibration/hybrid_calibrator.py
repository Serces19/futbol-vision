"""
Hybrid calibration system that combines main calibration with fallback mechanisms
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ..core.interfaces import BaseCalibrator
from ..core.models import FieldLine, KeyPoint
from ..core.config import FieldDimensions
from ..core.exceptions import CalibrationError

from .field_calibrator import FieldCalibrator
from .fallback_calibrator import FallbackCalibrator, CalibrationQualityAssessment


class HybridCalibrator(BaseCalibrator):
    """
    Hybrid calibration system that attempts proper field calibration
    and falls back to default transformation when calibration fails.
    
    This calibrator provides robust field calibration by combining
    the accuracy of homography-based calibration with the reliability
    of fallback mechanisms.
    """
    
    def __init__(
        self,
        field_dimensions: FieldDimensions,
        min_quality_threshold: float = 0.6,
        enable_fallback: bool = True,
        fallback_frame_size: Tuple[int, int] = (1920, 1080),
        **calibrator_kwargs
    ):
        """
        Initialize the HybridCalibrator.
        
        Args:
            field_dimensions: Standard field dimensions in meters
            min_quality_threshold: Minimum quality to accept main calibration
            enable_fallback: Whether to enable fallback calibration
            fallback_frame_size: Default frame size for fallback mode
            **calibrator_kwargs: Additional arguments for FieldCalibrator
        
        Raises:
            ValueError: If parameters are invalid
        """
        self.field_dimensions = field_dimensions
        self.min_quality_threshold = self._validate_threshold(min_quality_threshold)
        self.enable_fallback = enable_fallback
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize main and fallback calibrators
        self.main_calibrator = FieldCalibrator(
            field_dimensions=field_dimensions,
            homography_confidence_threshold=min_quality_threshold,
            **calibrator_kwargs
        )
        
        self.fallback_calibrator = FallbackCalibrator(
            field_dimensions=field_dimensions,
            default_frame_size=fallback_frame_size
        )
        
        # Quality assessment
        self.quality_assessor = CalibrationQualityAssessment(
            min_quality_threshold=min_quality_threshold
        )
        
        # Calibration state
        self.active_calibrator: Optional[BaseCalibrator] = None
        self.calibration_mode: str = "none"  # "main", "fallback", or "none"
        self.last_assessment: Optional[Dict[str, Any]] = None
    
    def _validate_threshold(self, threshold: float) -> float:
        """Validate threshold values"""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        return threshold
    
    def calibrate(self, field_lines: List[FieldLine], key_points: List[KeyPoint]) -> Optional[np.ndarray]:
        """
        Attempt calibration using main calibrator, fall back if necessary.
        
        Args:
            field_lines: Detected field lines
            key_points: Detected field keypoints
            
        Returns:
            Homography matrix if calibration successful, None otherwise
        """
        self.logger.info("Starting hybrid calibration process")
        
        # Reset previous calibration
        self._reset_calibration()
        
        # Assess field elements quality
        element_assessment = self.quality_assessor.assess_field_elements(
            field_lines, key_points
        )
        
        self.logger.debug(f"Field elements assessment: {element_assessment}")
        
        # Try main calibration if elements are sufficient
        if not element_assessment["use_fallback"]:
            try:
                self.logger.info("Attempting main field calibration")
                homography = self.main_calibrator.calibrate(field_lines, key_points)
                
                # Assess calibration result
                calibration_assessment = self.quality_assessor.assess_calibration_result(
                    homography,
                    self.main_calibrator.calibration_quality,
                    self.main_calibrator.calibration_warnings
                )
                
                self.logger.debug(f"Main calibration assessment: {calibration_assessment}")
                
                # Use main calibration if successful and quality is acceptable
                if not calibration_assessment["use_fallback"]:
                    self.active_calibrator = self.main_calibrator
                    self.calibration_mode = "main"
                    self.last_assessment = calibration_assessment
                    
                    self.logger.info(
                        f"Main calibration successful. Quality: {self.main_calibrator.calibration_quality:.3f}"
                    )
                    return homography
                
            except CalibrationError as e:
                self.logger.warning(f"Main calibration failed: {str(e)}")
        
        # Fall back to default calibration if enabled
        if self.enable_fallback:
            self.logger.info("Falling back to default calibration")
            
            try:
                fallback_homography = self.fallback_calibrator.calibrate(field_lines, key_points)
                self.active_calibrator = self.fallback_calibrator
                self.calibration_mode = "fallback"
                
                # Create assessment for fallback
                self.last_assessment = {
                    "calibration_successful": True,
                    "quality_acceptable": True,
                    "use_fallback": True,
                    "warnings": self.fallback_calibrator.calibration_warnings,
                    "recommendations": ["Using fallback calibration due to main calibration failure"]
                }
                
                self.logger.info("Fallback calibration activated")
                return fallback_homography
                
            except Exception as e:
                self.logger.error(f"Fallback calibration failed: {str(e)}")
        
        # Both calibrations failed
        self.logger.error("All calibration methods failed")
        raise CalibrationError("Both main and fallback calibration failed")
    
    def _reset_calibration(self) -> None:
        """Reset calibration state"""
        self.active_calibrator = None
        self.calibration_mode = "none"
        self.last_assessment = None
        self.main_calibrator.reset_calibration()
        self.fallback_calibrator.reset_calibration()
    
    def transform_to_field_coordinates(self, pixel_coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Transform pixel coordinates to field coordinates using active calibrator.
        
        Args:
            pixel_coords: List of (x, y) pixel coordinates
            
        Returns:
            List of (x, y) field coordinates in meters
        
        Raises:
            CalibrationError: If no calibrator is active
        """
        if self.active_calibrator is None:
            raise CalibrationError("No active calibrator. Call calibrate() first.")
        
        return self.active_calibrator.transform_to_field_coordinates(pixel_coords)
    
    def transform_to_pixel_coordinates(self, field_coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Transform field coordinates to pixel coordinates using active calibrator.
        
        Args:
            field_coords: List of (x, y) field coordinates in meters
            
        Returns:
            List of (x, y) pixel coordinates
        
        Raises:
            CalibrationError: If no calibrator is active
        """
        if self.active_calibrator is None:
            raise CalibrationError("No active calibrator. Call calibrate() first.")
        
        # Handle different method names between calibrators
        if hasattr(self.active_calibrator, 'transform_to_pixel_coordinates'):
            return self.active_calibrator.transform_to_pixel_coordinates(field_coords)
        else:
            # Fallback for calibrators that don't have this method
            raise CalibrationError("Active calibrator does not support pixel coordinate transformation")
    
    def is_calibrated(self) -> bool:
        """
        Check if any calibrator is active.
        
        Returns:
            True if calibrated, False otherwise
        """
        return (self.active_calibrator is not None and 
                self.active_calibrator.is_calibrated())
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Get comprehensive calibration information.
        
        Returns:
            Dictionary with calibration details
        """
        base_info = {
            "is_calibrated": self.is_calibrated(),
            "calibration_mode": self.calibration_mode,
            "enable_fallback": self.enable_fallback,
            "min_quality_threshold": self.min_quality_threshold,
            "last_assessment": self.last_assessment
        }
        
        if self.active_calibrator is not None:
            active_info = self.active_calibrator.get_calibration_info()
            base_info.update({
                "active_calibrator_info": active_info,
                "calibration_quality": active_info.get("calibration_quality", 0.0),
                "calibration_warnings": active_info.get("calibration_warnings", [])
            })
        
        # Add information about both calibrators
        base_info.update({
            "main_calibrator_info": self.main_calibrator.get_calibration_info(),
            "fallback_calibrator_info": self.fallback_calibrator.get_calibration_info()
        })
        
        return base_info
    
    def get_field_boundaries_pixels(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get field boundary coordinates in pixel space using active calibrator.
        
        Returns:
            List of field boundary points in pixels if calibrated, None otherwise
        """
        if self.active_calibrator is None:
            return None
        
        return self.active_calibrator.get_field_boundaries_pixels()
    
    def validate_point_in_field(self, field_coord: Tuple[float, float]) -> bool:
        """
        Check if a field coordinate is within the field boundaries.
        
        Args:
            field_coord: Field coordinate (x, y) in meters
            
        Returns:
            True if point is within field, False otherwise
        """
        # This validation is the same regardless of calibrator
        x, y = field_coord
        half_length = self.field_dimensions.length / 2
        half_width = self.field_dimensions.width / 2
        
        return (-half_length <= x <= half_length and 
                -half_width <= y <= half_width)
    
    def reset_calibration(self) -> None:
        """Reset all calibration state"""
        self._reset_calibration()
        self.logger.info("Hybrid calibration reset")
    
    def update_frame_size(self, frame_size: Tuple[int, int]) -> None:
        """
        Update frame size for fallback calibrator.
        
        Args:
            frame_size: New frame size (width, height) in pixels
        """
        self.fallback_calibrator.update_frame_size(frame_size)
        
        if self.calibration_mode == "fallback":
            self.logger.info(f"Updated fallback calibrator frame size: {frame_size}")
    
    def force_fallback_mode(self) -> Optional[np.ndarray]:
        """
        Force activation of fallback calibration mode.
        
        Returns:
            Fallback homography matrix
        """
        self.logger.info("Forcing fallback calibration mode")
        
        try:
            fallback_homography = self.fallback_calibrator.calibrate([], [])
            self.active_calibrator = self.fallback_calibrator
            self.calibration_mode = "fallback"
            
            self.last_assessment = {
                "calibration_successful": True,
                "quality_acceptable": True,
                "use_fallback": True,
                "warnings": ["Forced fallback mode"],
                "recommendations": ["Using fallback calibration by user request"]
            }
            
            return fallback_homography
            
        except Exception as e:
            raise CalibrationError(f"Failed to force fallback mode: {str(e)}")
    
    def get_calibration_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about calibration attempts.
        
        Returns:
            Dictionary with calibration statistics
        """
        stats = {
            "calibration_mode": self.calibration_mode,
            "is_calibrated": self.is_calibrated(),
            "fallback_enabled": self.enable_fallback
        }
        
        if self.calibration_mode == "main":
            stats.update({
                "main_calibration_quality": self.main_calibrator.calibration_quality,
                "main_calibration_warnings": len(self.main_calibrator.calibration_warnings),
                "homography_available": self.main_calibrator.homography_matrix is not None
            })
        elif self.calibration_mode == "fallback":
            stats.update({
                "fallback_active": True,
                "fallback_frame_size": self.fallback_calibrator.default_frame_size,
                "fallback_warnings": len(self.fallback_calibrator.calibration_warnings)
            })
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of the hybrid calibrator"""
        status = f"mode={self.calibration_mode}" if self.is_calibrated() else "not calibrated"
        return f"HybridCalibrator({status}, fallback_enabled={self.enable_fallback})"


def create_hybrid_calibrator_from_config(
    field_dimensions: FieldDimensions,
    processing_config: Optional[Dict[str, Any]] = None
) -> HybridCalibrator:
    """
    Factory function to create HybridCalibrator from configuration.
    
    Args:
        field_dimensions: Field dimensions configuration
        processing_config: Optional processing configuration dictionary
    
    Returns:
        Configured HybridCalibrator instance
    """
    kwargs = {"field_dimensions": field_dimensions}
    
    if processing_config:
        if "calibration_confidence_threshold" in processing_config:
            kwargs["min_quality_threshold"] = processing_config["calibration_confidence_threshold"]
        if "enable_field_calibration" in processing_config:
            kwargs["enable_fallback"] = processing_config["enable_field_calibration"]
    
    return HybridCalibrator(**kwargs)