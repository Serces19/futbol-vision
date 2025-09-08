"""
Field calibration system for football analytics using homography transformation
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from ..core.interfaces import BaseCalibrator
from ..core.models import FieldLine, KeyPoint
from ..core.config import FieldDimensions
from ..core.exceptions import CalibrationError


class FieldCalibrator(BaseCalibrator):
    """
    Field calibration system for computing homography transformation between
    pixel coordinates and real-world field coordinates.
    
    This class analyzes detected field lines and keypoints to establish a mapping
    between the camera view and the actual football field dimensions.
    """
    
    def __init__(
        self,
        field_dimensions: FieldDimensions,
        min_lines_required: int = 4,
        min_keypoints_required: int = 2,
        homography_confidence_threshold: float = 0.8
    ):
        """
        Initialize the FieldCalibrator.
        
        Args:
            field_dimensions: Standard field dimensions in meters
            min_lines_required: Minimum number of lines needed for calibration
            min_keypoints_required: Minimum number of keypoints needed for calibration
            homography_confidence_threshold: Minimum confidence for homography validation
        
        Raises:
            ValueError: If parameters are invalid
        """
        self.field_dimensions = field_dimensions
        self.min_lines_required = self._validate_min_required(min_lines_required, "lines")
        self.min_keypoints_required = self._validate_min_required(min_keypoints_required, "keypoints")
        self.homography_confidence_threshold = self._validate_threshold(homography_confidence_threshold)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Calibration state
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.calibration_quality: float = 0.0
        self.calibration_warnings: List[str] = []
        
        # Field coordinate system (origin at center of field)
        self._setup_field_coordinates()
    
    def _validate_min_required(self, value: int, param_name: str) -> int:
        """Validate minimum required parameters"""
        if value < 1:
            raise ValueError(f"Minimum {param_name} required must be at least 1, got {value}")
        return value
    
    def _validate_threshold(self, threshold: float) -> float:
        """Validate threshold values"""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        return threshold
    
    def _setup_field_coordinates(self) -> None:
        """Setup standard field coordinate system with origin at center"""
        # Field boundaries (origin at center)
        half_length = self.field_dimensions.length / 2
        half_width = self.field_dimensions.width / 2
        
        # Key field points in real-world coordinates (meters)
        self.field_reference_points = {
            # Corner points
            "corner_top_left": (-half_length, -half_width),
            "corner_top_right": (half_length, -half_width),
            "corner_bottom_left": (-half_length, half_width),
            "corner_bottom_right": (half_length, half_width),
            
            # Center points
            "center_spot": (0.0, 0.0),
            "center_left": (-half_length, 0.0),
            "center_right": (half_length, 0.0),
            "center_top": (0.0, -half_width),
            "center_bottom": (0.0, half_width),
            
            # Penalty spots
            "penalty_spot_left": (-self.field_dimensions.penalty_spot_distance, 0.0),
            "penalty_spot_right": (self.field_dimensions.penalty_spot_distance, 0.0),
            
            # Goal posts
            "goal_post_left_top": (-half_length, -self.field_dimensions.goal_width/2),
            "goal_post_left_bottom": (-half_length, self.field_dimensions.goal_width/2),
            "goal_post_right_top": (half_length, -self.field_dimensions.goal_width/2),
            "goal_post_right_bottom": (half_length, self.field_dimensions.goal_width/2),
        }
        
        # Field lines in real-world coordinates
        self.field_reference_lines = {
            "sideline_top": [(-half_length, -half_width), (half_length, -half_width)],
            "sideline_bottom": [(-half_length, half_width), (half_length, half_width)],
            "goal_line_left": [(-half_length, -half_width), (-half_length, half_width)],
            "goal_line_right": [(half_length, -half_width), (half_length, half_width)],
            "center_line": [(0.0, -half_width), (0.0, half_width)],
        }
    
    def calibrate(self, field_lines: List[FieldLine], key_points: List[KeyPoint]) -> Optional[np.ndarray]:
        """
        Calibrate field using detected lines and keypoints.
        
        Args:
            field_lines: Detected field lines
            key_points: Detected field keypoints
            
        Returns:
            Homography matrix if calibration successful, None otherwise
        
        Raises:
            CalibrationError: If calibration fails due to invalid input
        """
        try:
            # Reset previous calibration
            self._reset_calibration()
            
            # Validate input
            if len(field_lines) < self.min_lines_required:
                raise CalibrationError(
                    f"Insufficient field lines for calibration. "
                    f"Required: {self.min_lines_required}, Got: {len(field_lines)}"
                )
            
            if len(key_points) < self.min_keypoints_required:
                raise CalibrationError(
                    f"Insufficient keypoints for calibration. "
                    f"Required: {self.min_keypoints_required}, Got: {len(key_points)}"
                )
            
            # Extract correspondence points
            pixel_points, field_points = self._extract_correspondences(field_lines, key_points)
            
            if len(pixel_points) < 4:
                raise CalibrationError(
                    f"Need at least 4 correspondence points for homography. Got: {len(pixel_points)}"
                )
            
            # Compute homography matrix
            self.homography_matrix = self._compute_homography(pixel_points, field_points)
            
            if self.homography_matrix is None:
                raise CalibrationError("Failed to compute homography matrix")
            
            # Compute inverse homography
            self.inverse_homography = np.linalg.inv(self.homography_matrix)
            
            # Validate calibration quality
            self.calibration_quality = self._validate_calibration_quality(
                pixel_points, field_points
            )
            
            if self.calibration_quality < self.homography_confidence_threshold:
                self.calibration_warnings.append(
                    f"Low calibration quality: {self.calibration_quality:.3f} "
                    f"< {self.homography_confidence_threshold}"
                )
            
            self.logger.info(
                f"Field calibration successful. Quality: {self.calibration_quality:.3f}, "
                f"Points used: {len(pixel_points)}"
            )
            
            return self.homography_matrix
            
        except Exception as e:
            self.logger.error(f"Field calibration failed: {str(e)}")
            self._reset_calibration()
            raise CalibrationError(f"Calibration failed: {str(e)}")
    
    def _reset_calibration(self) -> None:
        """Reset calibration state"""
        self.homography_matrix = None
        self.inverse_homography = None
        self.calibration_quality = 0.0
        self.calibration_warnings.clear()
    
    def _extract_correspondences(
        self, 
        field_lines: List[FieldLine], 
        key_points: List[KeyPoint]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
        """
        Extract correspondence points between pixel and field coordinates.
        
        Args:
            field_lines: Detected field lines
            key_points: Detected field keypoints
            
        Returns:
            Tuple of (pixel_points, field_points)
        """
        pixel_points = []
        field_points = []
        
        # Process keypoints first (more reliable)
        for keypoint in key_points:
            field_coord = self._map_keypoint_to_field(keypoint)
            if field_coord is not None:
                pixel_points.append(keypoint.position)
                field_points.append(field_coord)
        
        # Process line intersections
        line_intersections = self._find_line_intersections(field_lines)
        for pixel_point, field_point in line_intersections:
            pixel_points.append(pixel_point)
            field_points.append(field_point)
        
        # Process line endpoints if needed
        if len(pixel_points) < 6:  # Need more points
            line_endpoints = self._extract_line_endpoints(field_lines)
            for pixel_point, field_point in line_endpoints:
                pixel_points.append(pixel_point)
                field_points.append(field_point)
        
        return pixel_points, field_points
    
    def _map_keypoint_to_field(self, keypoint: KeyPoint) -> Optional[Tuple[float, float]]:
        """
        Map detected keypoint to field coordinates.
        
        Args:
            keypoint: Detected keypoint
            
        Returns:
            Field coordinates if mapping successful, None otherwise
        """
        keypoint_mapping = {
            "corner": self._find_closest_corner(keypoint.position),
            "penalty_spot": self._find_closest_penalty_spot(keypoint.position),
            "center_spot": (0.0, 0.0),
            "goal_post": self._find_closest_goal_post(keypoint.position),
        }
        
        # Try exact match first
        if keypoint.keypoint_type in keypoint_mapping:
            return keypoint_mapping[keypoint.keypoint_type]
        
        # Try partial matches
        for key_type, field_coord in keypoint_mapping.items():
            if key_type in keypoint.keypoint_type.lower():
                return field_coord
        
        return None
    
    def _find_closest_corner(self, pixel_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Find the closest field corner to a pixel position"""
        # Simple heuristic: use pixel position to determine which corner
        x, y = pixel_pos
        
        # Assume image coordinates: (0,0) at top-left
        # Map to field corners based on position
        corners = [
            self.field_reference_points["corner_top_left"],
            self.field_reference_points["corner_top_right"],
            self.field_reference_points["corner_bottom_left"],
            self.field_reference_points["corner_bottom_right"]
        ]
        
        # Return first corner for now (could be improved with more sophisticated logic)
        return corners[0]
    
    def _find_closest_penalty_spot(self, pixel_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Find the closest penalty spot to a pixel position"""
        x, y = pixel_pos
        
        # Simple heuristic: left side vs right side
        # This would need frame width to be more accurate
        penalty_spots = [
            self.field_reference_points["penalty_spot_left"],
            self.field_reference_points["penalty_spot_right"]
        ]
        
        return penalty_spots[0]  # Return left penalty spot for now
    
    def _find_closest_goal_post(self, pixel_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Find the closest goal post to a pixel position"""
        goal_posts = [
            self.field_reference_points["goal_post_left_top"],
            self.field_reference_points["goal_post_left_bottom"],
            self.field_reference_points["goal_post_right_top"],
            self.field_reference_points["goal_post_right_bottom"]
        ]
        
        return goal_posts[0]  # Return first goal post for now
    
    def _find_line_intersections(
        self, 
        field_lines: List[FieldLine]
    ) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Find intersections between field lines and map to field coordinates.
        
        Args:
            field_lines: List of detected field lines
            
        Returns:
            List of (pixel_point, field_point) tuples
        """
        intersections = []
        
        # Find intersections between different line types
        for i, line1 in enumerate(field_lines):
            for j, line2 in enumerate(field_lines[i+1:], i+1):
                if line1.line_type != line2.line_type:
                    intersection = self._compute_line_intersection(line1, line2)
                    if intersection is not None:
                        field_coord = self._map_intersection_to_field(
                            intersection, line1.line_type, line2.line_type
                        )
                        if field_coord is not None:
                            intersections.append((intersection, field_coord))
        
        return intersections
    
    def _compute_line_intersection(
        self, 
        line1: FieldLine, 
        line2: FieldLine
    ) -> Optional[Tuple[int, int]]:
        """
        Compute intersection point between two lines.
        
        Args:
            line1: First line
            line2: Second line
            
        Returns:
            Intersection point if lines intersect, None otherwise
        """
        x1, y1 = line1.start_point
        x2, y2 = line1.end_point
        x3, y3 = line2.start_point
        x4, y4 = line2.end_point
        
        # Calculate line intersection using determinants
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            return (x, y)
        
        return None
    
    def _map_intersection_to_field(
        self, 
        pixel_point: Tuple[int, int], 
        line_type1: str, 
        line_type2: str
    ) -> Optional[Tuple[float, float]]:
        """
        Map line intersection to field coordinates based on line types.
        
        Args:
            pixel_point: Pixel coordinates of intersection
            line_type1: Type of first line
            line_type2: Type of second line
            
        Returns:
            Field coordinates if mapping successful, None otherwise
        """
        # Define intersection mappings
        intersection_mappings = {
            ("sideline", "goal_line"): [
                self.field_reference_points["corner_top_left"],
                self.field_reference_points["corner_top_right"],
                self.field_reference_points["corner_bottom_left"],
                self.field_reference_points["corner_bottom_right"]
            ],
            ("sideline", "center_line"): [
                self.field_reference_points["center_top"],
                self.field_reference_points["center_bottom"]
            ],
            ("goal_line", "center_line"): [
                self.field_reference_points["center_left"],
                self.field_reference_points["center_right"]
            ]
        }
        
        # Normalize line type order
        line_types = tuple(sorted([line_type1, line_type2]))
        
        if line_types in intersection_mappings:
            # Return first available mapping (could be improved with position analysis)
            return intersection_mappings[line_types][0]
        
        return None 
   
    def _extract_line_endpoints(
        self, 
        field_lines: List[FieldLine]
    ) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Extract line endpoints and map to field coordinates.
        
        Args:
            field_lines: List of detected field lines
            
        Returns:
            List of (pixel_point, field_point) tuples
        """
        endpoints = []
        
        for line in field_lines:
            # Map line endpoints based on line type
            field_coords = self._get_line_field_coordinates(line.line_type)
            if field_coords is not None:
                start_field, end_field = field_coords
                endpoints.append((line.start_point, start_field))
                endpoints.append((line.end_point, end_field))
        
        return endpoints
    
    def _get_line_field_coordinates(self, line_type: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get field coordinates for a line type.
        
        Args:
            line_type: Type of the line
            
        Returns:
            Tuple of (start_field_coord, end_field_coord) if available, None otherwise
        """
        if line_type in self.field_reference_lines:
            coords = self.field_reference_lines[line_type]
            return (coords[0], coords[1])
        
        return None
    
    def _compute_homography(
        self, 
        pixel_points: List[Tuple[int, int]], 
        field_points: List[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix from correspondence points.
        
        Args:
            pixel_points: List of pixel coordinates
            field_points: List of corresponding field coordinates
            
        Returns:
            Homography matrix if successful, None otherwise
        """
        try:
            # Convert to numpy arrays
            src_points = np.array(pixel_points, dtype=np.float32)
            dst_points = np.array(field_points, dtype=np.float32)
            
            # Compute homography using RANSAC for robustness
            homography, mask = cv2.findHomography(
                src_points, 
                dst_points, 
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.99,
                maxIters=2000
            )
            
            if homography is None:
                self.logger.warning("Failed to compute homography matrix")
                return None
            
            # Check if homography is reasonable (not degenerate)
            if not self._is_valid_homography(homography):
                self.logger.warning("Computed homography appears to be degenerate")
                return None
            
            return homography
            
        except Exception as e:
            self.logger.error(f"Error computing homography: {str(e)}")
            return None
    
    def _is_valid_homography(self, homography: np.ndarray) -> bool:
        """
        Check if homography matrix is valid (not degenerate).
        
        Args:
            homography: Homography matrix to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check matrix shape
            if homography.shape != (3, 3):
                return False
            
            # Check determinant (should not be zero or very close to zero)
            det = np.linalg.det(homography)
            if abs(det) < 1e-6:
                return False
            
            # Check condition number (should not be too large)
            cond = np.linalg.cond(homography)
            if cond > 1e6:
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(homography)) or np.any(np.isinf(homography)):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_calibration_quality(
        self, 
        pixel_points: List[Tuple[int, int]], 
        field_points: List[Tuple[float, float]]
    ) -> float:
        """
        Validate calibration quality by computing reprojection error.
        
        Args:
            pixel_points: Original pixel coordinates
            field_points: Corresponding field coordinates
            
        Returns:
            Quality score between 0 and 1 (higher is better)
        """
        try:
            if self.homography_matrix is None:
                return 0.0
            
            # Transform field points back to pixel coordinates
            field_array = np.array(field_points, dtype=np.float32).reshape(-1, 1, 2)
            reprojected_pixels = cv2.perspectiveTransform(field_array, self.inverse_homography)
            reprojected_pixels = reprojected_pixels.reshape(-1, 2)
            
            # Calculate reprojection errors
            pixel_array = np.array(pixel_points, dtype=np.float32)
            errors = np.linalg.norm(pixel_array - reprojected_pixels, axis=1)
            
            # Convert to quality score (lower error = higher quality)
            mean_error = np.mean(errors)
            max_acceptable_error = 10.0  # pixels
            
            quality = max(0.0, 1.0 - (mean_error / max_acceptable_error))
            
            self.logger.debug(f"Calibration quality: {quality:.3f}, Mean error: {mean_error:.2f} pixels")
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Error validating calibration quality: {str(e)}")
            return 0.0
    
    def transform_to_field_coordinates(self, pixel_coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Transform pixel coordinates to field coordinates.
        
        Args:
            pixel_coords: List of (x, y) pixel coordinates
            
        Returns:
            List of (x, y) field coordinates in meters
        
        Raises:
            CalibrationError: If field is not calibrated
        """
        if not self.is_calibrated():
            raise CalibrationError("Field is not calibrated. Call calibrate() first.")
        
        try:
            # Convert to numpy array and reshape for cv2.perspectiveTransform
            pixel_array = np.array(pixel_coords, dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform using homography
            field_array = cv2.perspectiveTransform(pixel_array, self.homography_matrix)
            
            # Reshape back to list of tuples
            field_coords = field_array.reshape(-1, 2)
            return [(float(x), float(y)) for x, y in field_coords]
            
        except Exception as e:
            raise CalibrationError(f"Failed to transform coordinates: {str(e)}")
    
    def transform_to_pixel_coordinates(self, field_coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Transform field coordinates to pixel coordinates.
        
        Args:
            field_coords: List of (x, y) field coordinates in meters
            
        Returns:
            List of (x, y) pixel coordinates
        
        Raises:
            CalibrationError: If field is not calibrated
        """
        if not self.is_calibrated():
            raise CalibrationError("Field is not calibrated. Call calibrate() first.")
        
        try:
            # Convert to numpy array and reshape for cv2.perspectiveTransform
            field_array = np.array(field_coords, dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform using inverse homography
            pixel_array = cv2.perspectiveTransform(field_array, self.inverse_homography)
            
            # Reshape back to list of tuples
            pixel_coords = pixel_array.reshape(-1, 2)
            return [(int(x), int(y)) for x, y in pixel_coords]
            
        except Exception as e:
            raise CalibrationError(f"Failed to transform coordinates: {str(e)}")
    
    def is_calibrated(self) -> bool:
        """
        Check if field is calibrated.
        
        Returns:
            True if calibrated, False otherwise
        """
        return (self.homography_matrix is not None and 
                self.inverse_homography is not None and
                self.calibration_quality > 0.0)
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Get detailed calibration information.
        
        Returns:
            Dictionary with calibration details
        """
        return {
            "is_calibrated": self.is_calibrated(),
            "calibration_quality": self.calibration_quality,
            "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            "field_dimensions": {
                "length": self.field_dimensions.length,
                "width": self.field_dimensions.width,
                "goal_width": self.field_dimensions.goal_width,
                "goal_height": self.field_dimensions.goal_height
            },
            "calibration_warnings": self.calibration_warnings,
            "min_lines_required": self.min_lines_required,
            "min_keypoints_required": self.min_keypoints_required,
            "confidence_threshold": self.homography_confidence_threshold
        }
    
    def get_field_boundaries_pixels(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get field boundary coordinates in pixel space.
        
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
        """Reset calibration state"""
        self._reset_calibration()
        self.logger.info("Field calibration reset")
    
    def __repr__(self) -> str:
        """String representation of the calibrator"""
        status = "calibrated" if self.is_calibrated() else "not calibrated"
        quality = f"quality={self.calibration_quality:.3f}" if self.is_calibrated() else ""
        return f"FieldCalibrator({status}, {quality})"


def create_field_calibrator_from_config(field_dimensions: FieldDimensions) -> FieldCalibrator:
    """
    Factory function to create FieldCalibrator from configuration.
    
    Args:
        field_dimensions: Field dimensions configuration
    
    Returns:
        Configured FieldCalibrator instance
    """
    return FieldCalibrator(field_dimensions=field_dimensions)