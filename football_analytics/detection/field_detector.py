"""
Field detection system for football analytics using specialized models
"""

import os
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn

from ..core.models import FieldLine, KeyPoint
from ..core.config import ModelPaths


class FieldDetectionError(Exception):
    """Custom exception for field detection errors"""
    pass


class FieldModelLoadError(FieldDetectionError):
    """Raised when field detection model fails to load"""
    pass


class FieldInferenceError(FieldDetectionError):
    """Raised during field detection inference failures"""
    pass


class FieldDetector:
    """
    Field detection system for detecting lines and keypoints in football fields.
    
    This class wraps specialized field detection models (SV_lines and SV_kp) and provides
    a clean interface for detecting field elements with validation and error handling.
    """
    
    def __init__(
        self,
        lines_model_path: str,
        keypoints_model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the FieldDetector.
        
        Args:
            lines_model_path: Path to the field lines detection model
            keypoints_model_path: Path to the field keypoints detection model
            device: Device to run inference on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
        
        Raises:
            FieldModelLoadError: If models fail to load
            ValueError: If parameters are invalid
        """
        self.lines_model_path = lines_model_path
        self.keypoints_model_path = keypoints_model_path
        self.device = self._validate_device(device)
        self.confidence_threshold = self._validate_threshold(confidence_threshold)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.lines_model = None
        self.keypoints_model = None
        self._load_models()
        
        # Field line types mapping
        self.line_types = {
            0: "sideline",
            1: "goal_line", 
            2: "center_line",
            3: "penalty_area",
            4: "goal_area",
            5: "center_circle"
        }
        
        # Keypoint types mapping
        self.keypoint_types = {
            0: "corner",
            1: "penalty_spot",
            2: "center_spot",
            3: "goal_post",
            4: "penalty_area_corner",
            5: "goal_area_corner"
        }
    
    def _validate_device(self, device: str) -> str:
        """Validate and setup device"""
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        elif device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cuda' or 'cpu'")
        return device
    
    def _validate_threshold(self, threshold: float) -> float:
        """Validate threshold values"""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {threshold}")
        return threshold
    
    def _load_models(self) -> None:
        """Load field detection models with error handling"""
        try:
            # Load lines model
            if not os.path.exists(self.lines_model_path):
                raise FieldModelLoadError(f"Lines model file not found: {self.lines_model_path}")
            
            self.logger.info(f"Loading field lines model from: {self.lines_model_path}")
            self.lines_model = torch.load(self.lines_model_path, map_location=self.device)
            
            # Ensure model is in evaluation mode
            if hasattr(self.lines_model, 'eval'):
                self.lines_model.eval()
            
            # Load keypoints model
            if not os.path.exists(self.keypoints_model_path):
                raise FieldModelLoadError(f"Keypoints model file not found: {self.keypoints_model_path}")
            
            self.logger.info(f"Loading field keypoints model from: {self.keypoints_model_path}")
            self.keypoints_model = torch.load(self.keypoints_model_path, map_location=self.device)
            
            # Ensure model is in evaluation mode
            if hasattr(self.keypoints_model, 'eval'):
                self.keypoints_model.eval()
                
        except Exception as e:
            raise FieldModelLoadError(f"Failed to load field detection models: {str(e)}")
    
    def _preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """
        Preprocess frame for field detection models.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            target_size: Target size for model input (width, height)
        
        Returns:
            Preprocessed tensor ready for model inference
        """
        # Resize frame
        resized = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _postprocess_lines(
        self, 
        model_output: torch.Tensor, 
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int] = (512, 512)
    ) -> List[FieldLine]:
        """
        Postprocess lines model output to FieldLine objects.
        
        Args:
            model_output: Raw model output tensor
            original_shape: Original frame shape (height, width)
            input_shape: Model input shape (height, width)
        
        Returns:
            List of FieldLine objects
        """
        lines = []
        
        try:
            # Convert tensor to numpy
            if isinstance(model_output, torch.Tensor):
                output = model_output.detach().cpu().numpy()
            else:
                output = model_output
            
            # Handle different output formats
            if len(output.shape) == 4:  # Batch dimension
                output = output[0]
            
            # Scale factors for coordinate conversion
            scale_x = original_shape[1] / input_shape[1]
            scale_y = original_shape[0] / input_shape[0]
            
            # Extract line segments (assuming output format: [N, 5] where 5 = [x1, y1, x2, y2, confidence])
            if len(output.shape) == 2 and output.shape[1] >= 5:
                for detection in output:
                    confidence = detection[4]
                    
                    if confidence >= self.confidence_threshold:
                        # Scale coordinates back to original frame size
                        x1 = int(detection[0] * scale_x)
                        y1 = int(detection[1] * scale_y)
                        x2 = int(detection[2] * scale_x)
                        y2 = int(detection[3] * scale_y)
                        
                        # Determine line type (simplified classification)
                        line_type = self._classify_line_type((x1, y1), (x2, y2), original_shape)
                        
                        line = FieldLine(
                            start_point=(x1, y1),
                            end_point=(x2, y2),
                            line_type=line_type,
                            confidence=float(confidence)
                        )
                        lines.append(line)
            
            # Alternative: Handle heatmap-style output
            elif len(output.shape) == 3:  # [C, H, W] format
                lines = self._extract_lines_from_heatmap(output, original_shape, input_shape)
            
        except Exception as e:
            self.logger.error(f"Error postprocessing lines: {e}")
        
        return lines
    
    def _postprocess_keypoints(
        self, 
        model_output: torch.Tensor, 
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int] = (512, 512)
    ) -> List[KeyPoint]:
        """
        Postprocess keypoints model output to KeyPoint objects.
        
        Args:
            model_output: Raw model output tensor
            original_shape: Original frame shape (height, width)
            input_shape: Model input shape (height, width)
        
        Returns:
            List of KeyPoint objects
        """
        keypoints = []
        
        try:
            # Convert tensor to numpy
            if isinstance(model_output, torch.Tensor):
                output = model_output.detach().cpu().numpy()
            else:
                output = model_output
            
            # Handle different output formats
            if len(output.shape) == 4:  # Batch dimension
                output = output[0]
            
            # Scale factors for coordinate conversion
            scale_x = original_shape[1] / input_shape[1]
            scale_y = original_shape[0] / input_shape[0]
            
            # Extract keypoints (assuming output format: [N, 4] where 4 = [x, y, class, confidence])
            if len(output.shape) == 2 and output.shape[1] >= 3:
                for detection in output:
                    if len(detection) >= 4:
                        confidence = detection[3]
                    else:
                        confidence = 1.0  # Default confidence if not provided
                    
                    if confidence >= self.confidence_threshold:
                        # Scale coordinates back to original frame size
                        x = int(detection[0] * scale_x)
                        y = int(detection[1] * scale_y)
                        
                        # Get keypoint type
                        if len(detection) >= 3:
                            class_id = int(detection[2])
                            keypoint_type = self.keypoint_types.get(class_id, f"keypoint_{class_id}")
                        else:
                            keypoint_type = "unknown"
                        
                        keypoint = KeyPoint(
                            position=(x, y),
                            keypoint_type=keypoint_type,
                            confidence=float(confidence)
                        )
                        keypoints.append(keypoint)
            
            # Alternative: Handle heatmap-style output
            elif len(output.shape) == 3:  # [C, H, W] format
                keypoints = self._extract_keypoints_from_heatmap(output, original_shape, input_shape)
            
        except Exception as e:
            self.logger.error(f"Error postprocessing keypoints: {e}")
        
        return keypoints
    
    def _classify_line_type(
        self, 
        start_point: Tuple[int, int], 
        end_point: Tuple[int, int], 
        frame_shape: Tuple[int, int]
    ) -> str:
        """
        Classify line type based on position and orientation.
        
        Args:
            start_point: Line start coordinates
            end_point: Line end coordinates
            frame_shape: Frame dimensions (height, width)
        
        Returns:
            Line type string
        """
        x1, y1 = start_point
        x2, y2 = end_point
        height, width = frame_shape
        
        # Calculate line properties
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine if line is horizontal or vertical
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
        is_horizontal = angle < 45
        is_vertical = angle > 45
        
        # Classify based on position and orientation
        if is_horizontal:
            if center_y < height * 0.2 or center_y > height * 0.8:
                return "goal_line"
            elif abs(center_y - height/2) < height * 0.1:
                return "center_line"
            else:
                return "penalty_area"
        elif is_vertical:
            if center_x < width * 0.2 or center_x > width * 0.8:
                return "sideline"
            else:
                return "penalty_area"
        else:
            # Diagonal or curved lines
            if length < min(width, height) * 0.3:
                return "goal_area"
            else:
                return "center_circle"
    
    def _extract_lines_from_heatmap(
        self, 
        heatmap: np.ndarray, 
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int]
    ) -> List[FieldLine]:
        """Extract line segments from heatmap output using image processing"""
        lines = []
        
        try:
            # Process each channel (assuming different channels for different line types)
            for channel_idx, channel in enumerate(heatmap):
                # Threshold the heatmap
                binary = (channel > self.confidence_threshold).astype(np.uint8) * 255
                
                # Apply morphological operations to clean up
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # Detect lines using HoughLinesP
                detected_lines = cv2.HoughLinesP(
                    binary, 
                    rho=1, 
                    theta=np.pi/180, 
                    threshold=50,
                    minLineLength=30,
                    maxLineGap=10
                )
                
                if detected_lines is not None:
                    scale_x = original_shape[1] / input_shape[1]
                    scale_y = original_shape[0] / input_shape[0]
                    
                    for line_coords in detected_lines:
                        x1, y1, x2, y2 = line_coords[0]
                        
                        # Scale coordinates
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        line_type = self.line_types.get(channel_idx, f"line_type_{channel_idx}")
                        
                        line = FieldLine(
                            start_point=(x1, y1),
                            end_point=(x2, y2),
                            line_type=line_type,
                            confidence=self.confidence_threshold
                        )
                        lines.append(line)
        
        except Exception as e:
            self.logger.error(f"Error extracting lines from heatmap: {e}")
        
        return lines
    
    def _extract_keypoints_from_heatmap(
        self, 
        heatmap: np.ndarray, 
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int]
    ) -> List[KeyPoint]:
        """Extract keypoints from heatmap output using peak detection"""
        keypoints = []
        
        try:
            # Process each channel (assuming different channels for different keypoint types)
            for channel_idx, channel in enumerate(heatmap):
                # Find local maxima
                from scipy.ndimage import maximum_filter
                
                # Apply maximum filter to find peaks
                local_maxima = maximum_filter(channel, size=5) == channel
                
                # Threshold and find coordinates
                peaks = np.where((local_maxima) & (channel > self.confidence_threshold))
                
                if len(peaks[0]) > 0:
                    scale_x = original_shape[1] / input_shape[1]
                    scale_y = original_shape[0] / input_shape[0]
                    
                    for y, x in zip(peaks[0], peaks[1]):
                        # Scale coordinates
                        scaled_x = int(x * scale_x)
                        scaled_y = int(y * scale_y)
                        
                        confidence = channel[y, x]
                        keypoint_type = self.keypoint_types.get(channel_idx, f"keypoint_type_{channel_idx}")
                        
                        keypoint = KeyPoint(
                            position=(scaled_x, scaled_y),
                            keypoint_type=keypoint_type,
                            confidence=float(confidence)
                        )
                        keypoints.append(keypoint)
        
        except Exception as e:
            self.logger.error(f"Error extracting keypoints from heatmap: {e}")
        
        return keypoints
    
    def detect_lines(self, frame: np.ndarray) -> List[FieldLine]:
        """
        Detect field lines in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
        
        Returns:
            List of FieldLine objects
        
        Raises:
            FieldInferenceError: If inference fails
        """
        if self.lines_model is None:
            raise FieldInferenceError("Lines model not loaded")
        
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                output = self.lines_model(input_tensor)
            
            # Postprocess results
            lines = self._postprocess_lines(output, frame.shape[:2])
            
            self.logger.debug(f"Detected {len(lines)} field lines")
            return lines
            
        except Exception as e:
            raise FieldInferenceError(f"Line detection failed: {str(e)}")
    
    def detect_keypoints(self, frame: np.ndarray) -> List[KeyPoint]:
        """
        Detect field keypoints in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
        
        Returns:
            List of KeyPoint objects
        
        Raises:
            FieldInferenceError: If inference fails
        """
        if self.keypoints_model is None:
            raise FieldInferenceError("Keypoints model not loaded")
        
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                output = self.keypoints_model(input_tensor)
            
            # Postprocess results
            keypoints = self._postprocess_keypoints(output, frame.shape[:2])
            
            self.logger.debug(f"Detected {len(keypoints)} field keypoints")
            return keypoints
            
        except Exception as e:
            raise FieldInferenceError(f"Keypoint detection failed: {str(e)}")
    
    def detect_field_elements(self, frame: np.ndarray) -> Tuple[List[FieldLine], List[KeyPoint]]:
        """
        Detect both lines and keypoints in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
        
        Returns:
            Tuple of (lines, keypoints)
        """
        lines = self.detect_lines(frame)
        keypoints = self.detect_keypoints(frame)
        return lines, keypoints
    
    def validate_field_elements(
        self, 
        lines: List[FieldLine], 
        keypoints: List[KeyPoint]
    ) -> Dict[str, Any]:
        """
        Validate detected field elements for consistency and completeness.
        
        Args:
            lines: List of detected field lines
            keypoints: List of detected keypoints
        
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "completeness_score": 0.0,
            "consistency_score": 0.0
        }
        
        try:
            # Check for minimum required elements
            required_line_types = ["sideline", "goal_line"]
            detected_line_types = set(line.line_type for line in lines)
            
            missing_lines = set(required_line_types) - detected_line_types
            if missing_lines:
                validation_results["warnings"].append(f"Missing required line types: {missing_lines}")
            
            # Check for reasonable number of elements
            if len(lines) < 4:
                validation_results["warnings"].append("Too few lines detected for a complete field")
            elif len(lines) > 20:
                validation_results["warnings"].append("Too many lines detected, may include noise")
            
            if len(keypoints) < 2:
                validation_results["warnings"].append("Too few keypoints detected")
            
            # Calculate completeness score
            expected_elements = 10  # Approximate number of major field elements
            detected_elements = len(lines) + len(keypoints)
            validation_results["completeness_score"] = min(1.0, detected_elements / expected_elements)
            
            # Calculate consistency score based on confidence
            if lines or keypoints:
                all_confidences = [line.confidence for line in lines] + [kp.confidence for kp in keypoints]
                validation_results["consistency_score"] = np.mean(all_confidences)
            
            # Overall validation
            if validation_results["errors"]:
                validation_results["is_valid"] = False
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["is_valid"] = False
        
        return validation_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            Dictionary with model information
        """
        return {
            "lines_model_path": self.lines_model_path,
            "keypoints_model_path": self.keypoints_model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "lines_model_loaded": self.lines_model is not None,
            "keypoints_model_loaded": self.keypoints_model is not None,
            "line_types": self.line_types,
            "keypoint_types": self.keypoint_types
        }
    
    def update_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold for detections.
        
        Args:
            threshold: New confidence threshold
        """
        self.confidence_threshold = self._validate_threshold(threshold)
        self.logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def __repr__(self) -> str:
        """String representation of the detector"""
        return (f"FieldDetector(lines_model='{self.lines_model_path}', "
                f"keypoints_model='{self.keypoints_model_path}', "
                f"device='{self.device}', "
                f"confidence_threshold={self.confidence_threshold})")


def create_field_detector_from_config(model_paths: ModelPaths, device: str = "cuda") -> FieldDetector:
    """
    Factory function to create FieldDetector from configuration.
    
    Args:
        model_paths: Model paths configuration
        device: Device to run inference on
    
    Returns:
        Configured FieldDetector instance
    """
    return FieldDetector(
        lines_model_path=model_paths.field_lines_model,
        keypoints_model_path=model_paths.field_keypoints_model,
        device=device
    )