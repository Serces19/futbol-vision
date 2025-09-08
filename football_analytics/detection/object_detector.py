"""
Object detection system for football analytics using YOLO models
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
from ultralytics import YOLO

from ..core.models import Detection
from ..core.config import ProcessingConfig, ModelPaths


class ObjectDetectionError(Exception):
    """Custom exception for object detection errors"""
    pass


class ModelLoadError(ObjectDetectionError):
    """Raised when model fails to load"""
    pass


class InferenceError(ObjectDetectionError):
    """Raised during inference failures"""
    pass


class ObjectDetector:
    """
    YOLO-based object detector for football players and ball detection.
    
    This class wraps YOLO models and provides a clean interface for detecting
    objects in football video frames with confidence filtering and error handling.
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        target_classes: Optional[List[int]] = None
    ):
        """
        Initialize the ObjectDetector.
        
        Args:
            model_path: Path to the YOLO model file (.pt)
            device: Device to run inference on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            target_classes: List of class IDs to detect (None for all classes)
        
        Raises:
            ModelLoadError: If model fails to load
            ValueError: If parameters are invalid
        """
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        self.model_path = model_path
        self.device = self._validate_device(device)
        self.confidence_threshold = self._validate_threshold(confidence_threshold, "confidence")
        self.nms_threshold = self._validate_threshold(nms_threshold, "NMS")
        self.target_classes = target_classes
        
        # Initialize model
        self.model = None
        self.class_names = {}
        self._load_model()
    
    def _validate_device(self, device: str) -> str:
        """Validate and setup device"""
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        elif device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cuda' or 'cpu'")
        return device
    
    def _validate_threshold(self, threshold: float, threshold_type: str) -> float:
        """Validate threshold values"""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"{threshold_type} threshold must be between 0 and 1, got {threshold}")
        return threshold
    
    def _load_model(self) -> None:
        """Load YOLO model with error handling"""
        try:
            if not os.path.exists(self.model_path):
                raise ModelLoadError(f"Model file not found: {self.model_path}")
            
            self.logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to specified device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            # Store class names for reference
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                self.logger.info(f"Model loaded with {len(self.class_names)} classes:")
                for idx, name in self.class_names.items():
                    self.logger.info(f"  {idx}: {name}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {self.model_path}: {str(e)}")
    
    def detect(
        self, 
        frame: np.ndarray, 
        imgsz: int = 1280,
        verbose: bool = False
    ) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            imgsz: Input image size for model
            verbose: Whether to print verbose output
        
        Returns:
            List of Detection objects
        
        Raises:
            InferenceError: If inference fails
        """
        if self.model is None:
            raise InferenceError("Model not loaded. Call _load_model() first.")
        
        try:
            # Run inference
            results = self.model(
                frame,
                imgsz=imgsz,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                device=self.device,
                classes=self.target_classes,
                verbose=verbose
            )
            
            # Convert results to Detection objects
            detections = []
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        # Apply confidence filtering
                        if conf >= self.confidence_threshold:
                            detection = Detection(
                                bbox=tuple(map(int, box)),  # Convert to int tuple
                                confidence=float(conf),
                                class_id=int(cls_id),
                                class_name=self.class_names.get(cls_id, f"class_{cls_id}")
                            )
                            detections.append(detection)
            
            self.logger.debug(f"Detected {len(detections)} objects in frame")
            return detections
            
        except Exception as e:
            raise InferenceError(f"Inference failed: {str(e)}")
    
    def filter_detections(
        self, 
        detections: List[Detection], 
        conf_threshold: Optional[float] = None,
        class_filter: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        Filter detections by confidence and class.
        
        Args:
            detections: List of Detection objects
            conf_threshold: Minimum confidence (uses instance threshold if None)
            class_filter: List of class IDs to keep (uses target_classes if None)
        
        Returns:
            Filtered list of Detection objects
        """
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        
        if class_filter is None:
            class_filter = self.target_classes
        
        filtered = []
        for detection in detections:
            # Check confidence
            if detection.confidence < conf_threshold:
                continue
            
            # Check class filter
            if class_filter is not None and detection.class_id not in class_filter:
                continue
            
            filtered.append(detection)
        
        return filtered
    
    def detect_players(self, frame: np.ndarray, **kwargs) -> List[Detection]:
        """
        Convenience method to detect only players.
        
        Args:
            frame: Input frame
            **kwargs: Additional arguments for detect()
        
        Returns:
            List of player detections
        """
        # Common player class IDs (adjust based on your model)
        player_classes = [0, 1, 2]  # Adjust these based on your model's class mapping
        
        # Temporarily set target classes
        original_classes = self.target_classes
        self.target_classes = player_classes
        
        try:
            detections = self.detect(frame, **kwargs)
            return detections
        finally:
            # Restore original target classes
            self.target_classes = original_classes
    
    def detect_ball(self, frame: np.ndarray, **kwargs) -> Optional[Detection]:
        """
        Convenience method to detect the ball.
        
        Args:
            frame: Input frame
            **kwargs: Additional arguments for detect()
        
        Returns:
            Ball detection or None if not found
        """
        # Common ball class ID (adjust based on your model)
        ball_class = [0]  # Adjust this based on your model's class mapping
        
        # Temporarily set target classes
        original_classes = self.target_classes
        self.target_classes = ball_class
        
        try:
            detections = self.detect(frame, **kwargs)
            # Return the detection with highest confidence
            if detections:
                return max(detections, key=lambda d: d.confidence)
            return None
        finally:
            # Restore original target classes
            self.target_classes = original_classes
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "target_classes": self.target_classes,
            "class_names": self.class_names,
            "model_loaded": self.model is not None
        }
    
    def update_thresholds(
        self, 
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None
    ) -> None:
        """
        Update detection thresholds.
        
        Args:
            confidence_threshold: New confidence threshold
            nms_threshold: New NMS threshold
        """
        if confidence_threshold is not None:
            self.confidence_threshold = self._validate_threshold(confidence_threshold, "confidence")
        
        if nms_threshold is not None:
            self.nms_threshold = self._validate_threshold(nms_threshold, "NMS")
        
        self.logger.info(f"Updated thresholds - Confidence: {self.confidence_threshold}, NMS: {self.nms_threshold}")
    
    def __repr__(self) -> str:
        """String representation of the detector"""
        return (f"ObjectDetector(model_path='{self.model_path}', "
                f"device='{self.device}', "
                f"confidence_threshold={self.confidence_threshold})")


def create_object_detector_from_config(
    config: ProcessingConfig, 
    model_paths: ModelPaths,
    detector_type: str = "player"
) -> ObjectDetector:
    """
    Factory function to create ObjectDetector from configuration.
    
    Args:
        config: Processing configuration
        model_paths: Model paths configuration
        detector_type: Type of detector ("player" or "ball")
    
    Returns:
        Configured ObjectDetector instance
    
    Raises:
        ValueError: If detector_type is invalid
    """
    if detector_type == "player":
        model_path = model_paths.yolo_player_model
        target_classes = [1, 2]  # Player classes
    elif detector_type == "ball":
        model_path = model_paths.yolo_ball_model
        target_classes = [0]  # Ball class
    else:
        raise ValueError(f"Invalid detector_type: {detector_type}. Must be 'player' or 'ball'")
    
    return ObjectDetector(
        model_path=model_path,
        device=config.device,
        confidence_threshold=config.confidence_threshold,
        nms_threshold=config.nms_threshold,
        target_classes=target_classes
    )