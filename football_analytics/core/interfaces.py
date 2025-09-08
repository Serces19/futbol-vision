"""
Base interfaces and abstract classes for the football analytics system
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .models import (
    Detection, TrackedObject, FieldLine, KeyPoint, 
    FrameResults, ProcessingResults, Position
)


class BaseDetector(ABC):
    """Base interface for object detection components"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load detection model from path
        
        Args:
            model_path: Path to the model file
        """
        pass
    
    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set confidence threshold for detections
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        pass


class BaseTracker(ABC):
    """Base interface for object tracking components"""
    
    @abstractmethod
    def update(self, detections: List[Detection], frame_shape: Tuple[int, int]) -> List[TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections from current frame
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            List of tracked objects with persistent IDs
        """
        pass
    
    @abstractmethod
    def get_trajectories(self) -> Dict[int, List[Position]]:
        """
        Get trajectories for all tracked objects
        
        Returns:
            Dictionary mapping track_id to list of positions
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state"""
        pass


class BaseClassifier(ABC):
    """Base interface for team classification components"""
    
    @abstractmethod
    def classify_teams(self, player_crops: List[np.ndarray]) -> List[int]:
        """
        Classify players into teams
        
        Args:
            player_crops: List of cropped player images
            
        Returns:
            List of team IDs for each player
        """
        pass
    
    @abstractmethod
    def update_model(self, new_embeddings: np.ndarray, team_labels: List[int]) -> None:
        """
        Update classification model with new data
        
        Args:
            new_embeddings: New embedding vectors
            team_labels: Corresponding team labels
        """
        pass
    
    @abstractmethod
    def get_team_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Get color mapping for teams
        
        Returns:
            Dictionary mapping team_id to RGB color tuple
        """
        pass


class BaseCalibrator(ABC):
    """Base interface for field calibration components"""
    
    @abstractmethod
    def calibrate(self, field_lines: List[FieldLine], key_points: List[KeyPoint]) -> Optional[np.ndarray]:
        """
        Calibrate field using detected lines and keypoints
        
        Args:
            field_lines: Detected field lines
            key_points: Detected field keypoints
            
        Returns:
            Homography matrix if calibration successful, None otherwise
        """
        pass
    
    @abstractmethod
    def transform_to_field_coordinates(self, pixel_coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Transform pixel coordinates to field coordinates
        
        Args:
            pixel_coords: List of (x, y) pixel coordinates
            
        Returns:
            List of (x, y) field coordinates in meters
        """
        pass
    
    @abstractmethod
    def is_calibrated(self) -> bool:
        """
        Check if field is calibrated
        
        Returns:
            True if calibrated, False otherwise
        """
        pass


class BaseAnalyticsEngine(ABC):
    """Base interface for analytics processing components"""
    
    @abstractmethod
    def update_frame_data(self, tracked_objects: List[TrackedObject], timestamp: float) -> None:
        """
        Update analytics with new frame data
        
        Args:
            tracked_objects: List of tracked objects in current frame
            timestamp: Current frame timestamp
        """
        pass
    
    @abstractmethod
    def calculate_velocities(self) -> Dict[int, float]:
        """
        Calculate current velocities for all tracked objects
        
        Returns:
            Dictionary mapping track_id to velocity in m/s
        """
        pass
    
    @abstractmethod
    def generate_heatmaps(self) -> Dict[int, np.ndarray]:
        """
        Generate position heatmaps for all tracked objects
        
        Returns:
            Dictionary mapping track_id to heatmap array
        """
        pass
    
    @abstractmethod
    def export_data(self) -> Dict[str, Any]:
        """
        Export all analytics data
        
        Returns:
            Dictionary containing all analytics data
        """
        pass


class BaseVisualizer(ABC):
    """Base interface for visualization components"""
    
    @abstractmethod
    def draw_detections(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """
        Draw detection overlays on frame
        
        Args:
            frame: Input frame
            tracked_objects: List of tracked objects to draw
            
        Returns:
            Frame with overlays drawn
        """
        pass
    
    @abstractmethod
    def draw_field_overlay(self, frame: np.ndarray, field_lines: List[FieldLine]) -> np.ndarray:
        """
        Draw field line overlays on frame
        
        Args:
            frame: Input frame
            field_lines: List of field lines to draw
            
        Returns:
            Frame with field overlays drawn
        """
        pass
    
    @abstractmethod
    def create_2d_map(self, field_positions: Dict[int, Tuple[float, float]]) -> np.ndarray:
        """
        Create 2D field map with player positions
        
        Args:
            field_positions: Dictionary mapping track_id to field coordinates
            
        Returns:
            2D map image as numpy array
        """
        pass


class BaseProcessor(ABC):
    """Base interface for video processing components"""
    
    @abstractmethod
    def process_video(self, video_source: str) -> ProcessingResults:
        """
        Process entire video
        
        Args:
            video_source: Path to video file or stream URL
            
        Returns:
            Complete processing results
        """
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> FrameResults:
        """
        Process single frame
        
        Args:
            frame: Input frame
            frame_id: Frame number
            timestamp: Frame timestamp
            
        Returns:
            Frame processing results
        """
        pass
    
    @abstractmethod
    def set_callbacks(self, callbacks: Dict[str, Any]) -> None:
        """
        Set callback functions for real-time updates
        
        Args:
            callbacks: Dictionary of callback functions
        """
        pass


class BaseExporter(ABC):
    """Base interface for data export components"""
    
    @abstractmethod
    def export_json(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Export data to JSON format
        
        Args:
            data: Data to export
            filepath: Output file path
        """
        pass
    
    @abstractmethod
    def export_csv(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Export data to CSV format
        
        Args:
            data: Data to export
            filepath: Output file path
        """
        pass
    
    @abstractmethod
    def export_video(self, frames: List[np.ndarray], filepath: str, fps: int) -> None:
        """
        Export processed frames as video
        
        Args:
            frames: List of processed frames
            filepath: Output video path
            fps: Frames per second
        """
        pass


class ComponentFactory(ABC):
    """Factory interface for creating system components"""
    
    @abstractmethod
    def create_detector(self, detector_type: str, **kwargs) -> BaseDetector:
        """Create detector instance"""
        pass
    
    @abstractmethod
    def create_tracker(self, tracker_type: str, **kwargs) -> BaseTracker:
        """Create tracker instance"""
        pass
    
    @abstractmethod
    def create_classifier(self, classifier_type: str, **kwargs) -> BaseClassifier:
        """Create classifier instance"""
        pass
    
    @abstractmethod
    def create_calibrator(self, calibrator_type: str, **kwargs) -> BaseCalibrator:
        """Create calibrator instance"""
        pass
    
    @abstractmethod
    def create_analytics_engine(self, engine_type: str, **kwargs) -> BaseAnalyticsEngine:
        """Create analytics engine instance"""
        pass
    
    @abstractmethod
    def create_visualizer(self, visualizer_type: str, **kwargs) -> BaseVisualizer:
        """Create visualizer instance"""
        pass