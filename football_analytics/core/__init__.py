"""
Core components and interfaces for the football analytics system
"""

from .models import (
    Detection, TrackedObject, FieldLine, KeyPoint, FrameResults, 
    ProcessingResults, Position, PlayerStats, TeamStats
)
from .config import (
    ProcessingConfig, FieldDimensions, ModelPaths, TrackerConfig,
    VisualizationConfig, ExportConfig, ConfigManager, load_config_from_env
)
from .interfaces import (
    BaseDetector, BaseTracker, BaseClassifier, BaseCalibrator,
    BaseAnalyticsEngine, BaseVisualizer, BaseProcessor, BaseExporter,
    ComponentFactory
)
from .exceptions import (
    FootballAnalyticsError, ModelLoadError, CalibrationError,
    ProcessingError, DetectionError, TrackingError, ClassificationError,
    VideoError, ConfigurationError, ExportError, VisualizationError,
    AnalyticsError, ErrorSeverity, RecoveryAction
)
from .error_recovery import ErrorRecoveryManager, with_error_recovery, create_error_context
from .error_handler import ErrorHandler, ErrorAggregator, safe_execute, get_global_error_handler, set_global_error_handler
from .logging_system import (
    FootballAnalyticsLogger, PerformanceMonitor, AnalyticsLogger, LogLevel,
    PerformanceMetric, AnalyticsMetric, get_global_logger, set_global_logger, setup_logging
)
from .monitoring import (
    SystemMonitor, ResourceMonitor, ProcessingMonitor, QualityMonitor,
    DiagnosticCollector, SystemMetrics, ProcessingMetrics, QualityMetrics
)
from .debug_tools import (
    DebugManager, DebugProfiler, FrameDebugger, ComponentDebugger, MemoryDebugger,
    get_global_debug_manager, set_global_debug_manager, setup_debugging
)
from .factory import DefaultComponentFactory
from .video_processor import VideoProcessor
from .video_io import VideoSource, VideoWriter, VideoStreamManager, FrameRateController

__all__ = [
    # Models
    'Detection', 'TrackedObject', 'FieldLine', 'KeyPoint', 'FrameResults',
    'ProcessingResults', 'Position', 'PlayerStats', 'TeamStats',
    
    # Configuration
    'ProcessingConfig', 'FieldDimensions', 'ModelPaths', 'TrackerConfig',
    'VisualizationConfig', 'ExportConfig', 'ConfigManager', 'load_config_from_env',
    
    # Interfaces
    'BaseDetector', 'BaseTracker', 'BaseClassifier', 'BaseCalibrator',
    'BaseAnalyticsEngine', 'BaseVisualizer', 'BaseProcessor', 'BaseExporter',
    'ComponentFactory',
    
    # Exceptions
    'FootballAnalyticsError', 'ModelLoadError', 'CalibrationError',
    'ProcessingError', 'DetectionError', 'TrackingError', 'ClassificationError',
    'VideoError', 'ConfigurationError', 'ExportError', 'VisualizationError',
    'AnalyticsError', 'ErrorSeverity', 'RecoveryAction',
    
    # Error Handling
    'ErrorRecoveryManager', 'ErrorHandler', 'ErrorAggregator', 'with_error_recovery',
    'create_error_context', 'safe_execute', 'get_global_error_handler', 'set_global_error_handler',
    
    # Logging and Monitoring
    'FootballAnalyticsLogger', 'PerformanceMonitor', 'AnalyticsLogger', 'LogLevel',
    'PerformanceMetric', 'AnalyticsMetric', 'get_global_logger', 'set_global_logger', 'setup_logging',
    'SystemMonitor', 'ResourceMonitor', 'ProcessingMonitor', 'QualityMonitor',
    'DiagnosticCollector', 'SystemMetrics', 'ProcessingMetrics', 'QualityMetrics',
    
    # Debug Tools
    'DebugManager', 'DebugProfiler', 'FrameDebugger', 'ComponentDebugger', 'MemoryDebugger',
    'get_global_debug_manager', 'set_global_debug_manager', 'setup_debugging',
    
    # Factory
    'DefaultComponentFactory',
    
    # Video Processing
    'VideoProcessor', 'VideoSource', 'VideoWriter', 'VideoStreamManager', 'FrameRateController',
]