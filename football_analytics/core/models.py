"""
Core data models for the football analytics system
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Dict
import numpy as np


@dataclass
class Detection:
    """Represents a detected object in a frame"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    
    def __post_init__(self):
        """Validate detection data"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.class_id < 0:
            raise ValueError(f"Class ID must be non-negative, got {self.class_id}")


@dataclass
class TrackedObject:
    """Represents a tracked object with persistent ID"""
    track_id: int
    detection: Detection
    team_id: Optional[int] = None
    field_position: Optional[Tuple[float, float]] = None
    velocity: Optional[float] = None
    trajectory: Optional[List[Tuple[float, float]]] = None
    
    def __post_init__(self):
        """Initialize trajectory if not provided"""
        if self.trajectory is None:
            self.trajectory = []


@dataclass
class FieldLine:
    """Represents a detected field line"""
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    line_type: str  # "sideline", "goal_line", "center_line", etc.
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate line data"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class KeyPoint:
    """Represents a detected field keypoint"""
    position: Tuple[int, int]
    keypoint_type: str  # "corner", "penalty_spot", "center_circle", etc.
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate keypoint data"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class FrameResults:
    """Results from processing a single frame"""
    frame_id: int
    timestamp: float
    tracked_objects: List[TrackedObject]
    field_lines: List[FieldLine]
    key_points: List[KeyPoint]
    ball_position: Optional[Tuple[int, int]] = None
    is_calibrated: bool = False
    homography_matrix: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate frame results"""
        if self.frame_id < 0:
            raise ValueError(f"Frame ID must be non-negative, got {self.frame_id}")
        if self.timestamp < 0:
            raise ValueError(f"Timestamp must be non-negative, got {self.timestamp}")


@dataclass
class ProcessingResults:
    """Complete results from video processing"""
    total_frames: int
    processing_time: float
    frame_results: List[FrameResults]
    analytics_data: Dict[str, Any]
    export_paths: List[str]
    
    def __post_init__(self):
        """Validate processing results"""
        if self.total_frames < 0:
            raise ValueError(f"Total frames must be non-negative, got {self.total_frames}")
        if self.processing_time < 0:
            raise ValueError(f"Processing time must be non-negative, got {self.processing_time}")


@dataclass
class Position:
    """Represents a position in 2D space"""
    x: float
    y: float
    timestamp: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class PlayerStats:
    """Statistics for a single player"""
    player_id: int
    team_id: int
    total_distance: float
    average_speed: float
    max_speed: float
    positions: List[Position]
    heatmap: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate player stats"""
        if self.player_id < 0:
            raise ValueError(f"Player ID must be non-negative, got {self.player_id}")
        if self.total_distance < 0:
            raise ValueError(f"Total distance must be non-negative, got {self.total_distance}")


@dataclass
class TeamStats:
    """Statistics for a team"""
    team_id: int
    player_stats: List[PlayerStats]
    formation_analysis: Dict[str, Any]
    possession_time: float
    average_position: Tuple[float, float]
    
    def __post_init__(self):
        """Validate team stats"""
        if self.team_id < 0:
            raise ValueError(f"Team ID must be non-negative, got {self.team_id}")
        if self.possession_time < 0:
            raise ValueError(f"Possession time must be non-negative, got {self.possession_time}")