"""
Configuration management system for the football analytics MVP
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
import json
import yaml
from pathlib import Path


@dataclass
class ProcessingConfig:
    """Main processing configuration"""
    # Detection parameters
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    device: str = "cuda"
    
    # Team classification
    n_teams: int = 2
    embedding_model: str = "osnet"
    
    # Field calibration
    enable_field_calibration: bool = True
    calibration_confidence_threshold: float = 0.7
    
    # Visualization
    enable_2d_visualization: bool = True
    enable_trajectory_visualization: bool = True
    output_fps: int = 30
    
    # Performance
    max_tracking_objects: int = 50
    trajectory_history_length: int = 100
    
    # Performance optimization
    enable_multithreading: bool = True
    max_worker_threads: int = 4
    frame_buffer_size: int = 10
    memory_cleanup_interval: int = 100
    gc_collection_interval: int = 50
    enable_frame_skipping: bool = True
    memory_limit_mb: float = 2000.0
    enable_resource_monitoring: bool = True
    max_trajectory_length: int = 1000
    embedding_cache_size: int = 500
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {self.confidence_threshold}")
        if not (0.0 <= self.nms_threshold <= 1.0):
            raise ValueError(f"NMS threshold must be between 0 and 1, got {self.nms_threshold}")
        if self.n_teams < 2:
            raise ValueError(f"Number of teams must be at least 2, got {self.n_teams}")
        if self.output_fps <= 0:
            raise ValueError(f"Output FPS must be positive, got {self.output_fps}")
        if self.max_tracking_objects <= 0:
            raise ValueError(f"Max tracking objects must be positive, got {self.max_tracking_objects}")
        if self.trajectory_history_length <= 0:
            raise ValueError(f"Trajectory history length must be positive, got {self.trajectory_history_length}")
        if self.max_worker_threads <= 0:
            raise ValueError(f"Max worker threads must be positive, got {self.max_worker_threads}")
        if self.frame_buffer_size <= 0:
            raise ValueError(f"Frame buffer size must be positive, got {self.frame_buffer_size}")
        if self.memory_limit_mb <= 0:
            raise ValueError(f"Memory limit must be positive, got {self.memory_limit_mb}")
        if self.max_trajectory_length <= 0:
            raise ValueError(f"Max trajectory length must be positive, got {self.max_trajectory_length}")
        if self.embedding_cache_size <= 0:
            raise ValueError(f"Embedding cache size must be positive, got {self.embedding_cache_size}")


@dataclass
class FieldDimensions:
    """Standard football field dimensions in meters"""
    length: float = 105.0  # FIFA standard
    width: float = 68.0    # FIFA standard
    goal_width: float = 7.32
    goal_height: float = 2.44
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.32
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    corner_arc_radius: float = 1.0
    
    def __post_init__(self):
        """Validate field dimensions"""
        if self.length <= 0 or self.width <= 0:
            raise ValueError("Field dimensions must be positive")
        if self.goal_width <= 0 or self.goal_height <= 0:
            raise ValueError("Goal dimensions must be positive")


@dataclass
class ModelPaths:
    """Paths to ML models"""
    yolo_player_model: str = "models/yolov8m-football.pt"
    yolo_ball_model: str = "models/yolov8m-football.pt"
    field_lines_model: str = "models/SV_FT_WC14_lines"
    field_keypoints_model: str = "models/SV_FT_WC14_kp"
    embedding_model_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate model paths exist"""
        for model_name, model_path in self.__dict__.items():
            if model_path and not os.path.exists(model_path):
                print(f"Warning: Model path does not exist: {model_path}")


@dataclass
class TrackerConfig:
    """ByteTrack tracker configuration"""
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30
    
    def __post_init__(self):
        """Validate tracker configuration"""
        if not (0.0 <= self.track_thresh <= 1.0):
            raise ValueError(f"Track threshold must be between 0 and 1, got {self.track_thresh}")
        if not (0.0 <= self.match_thresh <= 1.0):
            raise ValueError(f"Match threshold must be between 0 and 1, got {self.match_thresh}")
        if self.track_buffer <= 0:
            raise ValueError(f"Track buffer must be positive, got {self.track_buffer}")
        if self.frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {self.frame_rate}")


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    # Colors (BGR format for OpenCV)
    team_colors: Dict[int, tuple] = field(default_factory=lambda: {
        0: (255, 0, 0),    # Blue
        1: (0, 0, 255),    # Red
        2: (0, 255, 0),    # Green
        3: (255, 255, 0),  # Cyan
    })
    ball_color: tuple = (0, 255, 255)  # Yellow
    line_color: tuple = (255, 255, 255)  # White
    
    # Drawing parameters
    bbox_thickness: int = 2
    trajectory_thickness: int = 2
    text_font_scale: float = 0.6
    text_thickness: int = 2
    
    # 2D map settings
    map_width: int = 800
    map_height: int = 600
    map_background_color: tuple = (34, 139, 34)  # Forest Green
    
    def __post_init__(self):
        """Validate visualization configuration"""
        if self.bbox_thickness <= 0:
            raise ValueError(f"Bbox thickness must be positive, got {self.bbox_thickness}")
        if self.map_width <= 0 or self.map_height <= 0:
            raise ValueError("Map dimensions must be positive")


@dataclass
class ExportConfig:
    """Data export configuration"""
    output_directory: str = "output"
    export_json: bool = True
    export_csv: bool = True
    export_video: bool = True
    export_heatmaps: bool = True
    
    # File naming
    json_filename: str = "analytics_data.json"
    csv_filename: str = "player_stats.csv"
    video_filename: str = "processed_video.mp4"
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_directory, exist_ok=True)


class ConfigManager:
    """Manages loading and saving of configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.processing_config = ProcessingConfig()
        self.field_dimensions = FieldDimensions()
        self.model_paths = ModelPaths()
        self.tracker_config = TrackerConfig()
        self.visualization_config = VisualizationConfig()
        self.export_config = ExportConfig()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from file"""
        path = config_path or self.config_path
        
        if not os.path.exists(path):
            print(f"Config file not found: {path}. Using default configuration.")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.json'):
                    config_data = json.load(f)
                elif path.endswith(('.yaml', '.yml')):
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {path}")
            
            # Update configurations with loaded data
            if 'processing' in config_data:
                self._update_dataclass(self.processing_config, config_data['processing'])
            if 'field_dimensions' in config_data:
                self._update_dataclass(self.field_dimensions, config_data['field_dimensions'])
            if 'model_paths' in config_data:
                self._update_dataclass(self.model_paths, config_data['model_paths'])
            if 'tracker' in config_data:
                self._update_dataclass(self.tracker_config, config_data['tracker'])
            if 'visualization' in config_data:
                self._update_dataclass(self.visualization_config, config_data['visualization'])
            if 'export' in config_data:
                self._update_dataclass(self.export_config, config_data['export'])
                
        except Exception as e:
            print(f"Error loading config file {path}: {e}")
            print("Using default configuration.")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        path = config_path or self.config_path
        
        config_data = {
            'processing': self._dataclass_to_dict(self.processing_config),
            'field_dimensions': self._dataclass_to_dict(self.field_dimensions),
            'model_paths': self._dataclass_to_dict(self.model_paths),
            'tracker': self._dataclass_to_dict(self.tracker_config),
            'visualization': self._dataclass_to_dict(self.visualization_config),
            'export': self._dataclass_to_dict(self.export_config),
        }
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                if path.endswith('.json'):
                    json.dump(config_data, f, indent=2)
                elif path.endswith(('.yaml', '.yml')):
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {path}")
                    
        except Exception as e:
            print(f"Error saving config file {path}: {e}")
    
    def _update_dataclass(self, dataclass_instance, update_dict: Dict[str, Any]) -> None:
        """Update dataclass instance with dictionary values"""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _dataclass_to_dict(self, dataclass_instance) -> Dict[str, Any]:
        """Convert dataclass instance to dictionary"""
        result = {}
        for key, value in dataclass_instance.__dict__.items():
            if isinstance(value, dict):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary"""
        return {
            'processing': self.processing_config,
            'field_dimensions': self.field_dimensions,
            'model_paths': self.model_paths,
            'tracker': self.tracker_config,
            'visualization': self.visualization_config,
            'export': self.export_config,
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ConfigManager':
        """Create ConfigManager instance from configuration file"""
        config_manager = cls(config_path)
        config_manager.load_config()
        return config_manager
    
    def update_processing_config(self, **kwargs) -> None:
        """Update processing configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self.processing_config, key):
                setattr(self.processing_config, key, value)
            else:
                raise ValueError(f"Unknown processing config parameter: {key}")
    
    def update_field_dimensions(self, **kwargs) -> None:
        """Update field dimensions with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self.field_dimensions, key):
                setattr(self.field_dimensions, key, value)
            else:
                raise ValueError(f"Unknown field dimension parameter: {key}")
    
    def update_model_paths(self, **kwargs) -> None:
        """Update model paths with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self.model_paths, key):
                setattr(self.model_paths, key, value)
            else:
                raise ValueError(f"Unknown model path parameter: {key}")
    
    def update_tracker_config(self, **kwargs) -> None:
        """Update tracker configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self.tracker_config, key):
                setattr(self.tracker_config, key, value)
            else:
                raise ValueError(f"Unknown tracker config parameter: {key}")
    
    def update_visualization_config(self, **kwargs) -> None:
        """Update visualization configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self.visualization_config, key):
                setattr(self.visualization_config, key, value)
            else:
                raise ValueError(f"Unknown visualization config parameter: {key}")
    
    def update_export_config(self, **kwargs) -> None:
        """Update export configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self.export_config, key):
                setattr(self.export_config, key, value)
            else:
                raise ValueError(f"Unknown export config parameter: {key}")


def load_config_from_env() -> ConfigManager:
    """Load configuration from environment variables"""
    config_manager = ConfigManager()
    
    # Override with environment variables if they exist
    if os.getenv('CONFIDENCE_THRESHOLD'):
        config_manager.processing_config.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))
    if os.getenv('DEVICE'):
        config_manager.processing_config.device = os.getenv('DEVICE')
    if os.getenv('N_TEAMS'):
        config_manager.processing_config.n_teams = int(os.getenv('N_TEAMS'))
    
    return config_manager