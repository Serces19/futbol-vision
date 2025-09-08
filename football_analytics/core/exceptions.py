"""
Custom exceptions for the football analytics system with enhanced error handling capabilities
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import traceback
import time


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Available recovery actions for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    RESET = "reset"


class FootballAnalyticsError(Exception):
    """
    Base exception for football analytics system with enhanced error handling
    
    Attributes:
        message: Error message
        severity: Error severity level
        recovery_action: Suggested recovery action
        context: Additional context information
        timestamp: When the error occurred
        component: Which component raised the error
        recoverable: Whether the error is recoverable
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_action: RecoveryAction = RecoveryAction.ABORT,
        context: Optional[Dict[str, Any]] = None,
        component: Optional[str] = None,
        recoverable: bool = True,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.recovery_action = recovery_action
        self.context = context or {}
        self.component = component
        self.recoverable = recoverable
        self.original_exception = original_exception
        self.timestamp = time.time()
        self.traceback_str = traceback.format_exc() if original_exception else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "recovery_action": self.recovery_action.value,
            "context": self.context,
            "component": self.component,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "original_exception": str(self.original_exception) if self.original_exception else None,
            "traceback": self.traceback_str
        }
    
    def __str__(self) -> str:
        base_msg = f"[{self.severity.value.upper()}] {self.message}"
        if self.component:
            base_msg = f"[{self.component}] {base_msg}"
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            base_msg += f" (Context: {context_str})"
        return base_msg


class ModelLoadError(FootballAnalyticsError):
    """Raised when models fail to load"""
    
    def __init__(
        self, 
        message: str, 
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if model_path:
            context['model_path'] = model_path
        if model_type:
            context['model_type'] = model_type
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.FALLBACK,
            context=context,
            component="ModelLoader",
            **kwargs
        )


class CalibrationError(FootballAnalyticsError):
    """Raised when field calibration fails"""
    
    def __init__(
        self, 
        message: str,
        calibration_method: Optional[str] = None,
        field_lines_detected: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if calibration_method:
            context['calibration_method'] = calibration_method
        if field_lines_detected is not None:
            context['field_lines_detected'] = field_lines_detected
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.FALLBACK,
            context=context,
            component="FieldCalibrator",
            **kwargs
        )


class ProcessingError(FootballAnalyticsError):
    """Raised during frame processing"""
    
    def __init__(
        self, 
        message: str,
        frame_id: Optional[int] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if frame_id is not None:
            context['frame_id'] = frame_id
        if processing_stage:
            context['processing_stage'] = processing_stage
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.SKIP,
            context=context,
            component="FrameProcessor",
            **kwargs
        )


class DetectionError(FootballAnalyticsError):
    """Raised when object detection fails"""
    
    def __init__(
        self, 
        message: str,
        detection_type: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if detection_type:
            context['detection_type'] = detection_type
        if confidence_threshold is not None:
            context['confidence_threshold'] = confidence_threshold
        
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.RETRY,
            context=context,
            component="ObjectDetector",
            **kwargs
        )


class TrackingError(FootballAnalyticsError):
    """Raised when object tracking fails"""
    
    def __init__(
        self, 
        message: str,
        tracker_type: Optional[str] = None,
        tracked_objects_count: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if tracker_type:
            context['tracker_type'] = tracker_type
        if tracked_objects_count is not None:
            context['tracked_objects_count'] = tracked_objects_count
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.RESET,
            context=context,
            component="PlayerTracker",
            **kwargs
        )


class ClassificationError(FootballAnalyticsError):
    """Raised when team classification fails"""
    
    def __init__(
        self, 
        message: str,
        classification_method: Optional[str] = None,
        players_count: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if classification_method:
            context['classification_method'] = classification_method
        if players_count is not None:
            context['players_count'] = players_count
        
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.RETRY,
            context=context,
            component="TeamClassifier",
            **kwargs
        )


class VideoError(FootballAnalyticsError):
    """Raised when video I/O operations fail"""
    
    def __init__(
        self, 
        message: str,
        video_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if video_path:
            context['video_path'] = video_path
        if operation:
            context['operation'] = operation
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.ABORT,
            context=context,
            component="VideoIO",
            recoverable=False,
            **kwargs
        )


class ConfigurationError(FootballAnalyticsError):
    """Raised when configuration is invalid"""
    
    def __init__(
        self, 
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = str(config_value)
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.ABORT,
            context=context,
            component="Configuration",
            recoverable=False,
            **kwargs
        )


class ExportError(FootballAnalyticsError):
    """Raised when data export fails"""
    
    def __init__(
        self, 
        message: str,
        export_format: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if export_format:
            context['export_format'] = export_format
        if file_path:
            context['file_path'] = file_path
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.RETRY,
            context=context,
            component="DataExporter",
            **kwargs
        )


class VisualizationError(FootballAnalyticsError):
    """Raised when visualization operations fail"""
    
    def __init__(
        self, 
        message: str,
        visualization_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if visualization_type:
            context['visualization_type'] = visualization_type
        
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.SKIP,
            context=context,
            component="Visualizer",
            **kwargs
        )


class AnalyticsError(FootballAnalyticsError):
    """Raised when analytics calculations fail"""
    
    def __init__(
        self, 
        message: str,
        analytics_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if analytics_type:
            context['analytics_type'] = analytics_type
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.SKIP,
            context=context,
            component="AnalyticsEngine",
            **kwargs
        )