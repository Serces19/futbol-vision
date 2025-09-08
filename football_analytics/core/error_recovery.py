"""
Error recovery mechanisms and graceful degradation for the football analytics system
"""

from typing import Optional, Dict, Any, Callable, List, Union
import time
import logging
from functools import wraps
from .exceptions import (
    FootballAnalyticsError, ErrorSeverity, RecoveryAction,
    ModelLoadError, CalibrationError, ProcessingError, DetectionError,
    TrackingError, ClassificationError, VideoError, ConfigurationError,
    ExportError, VisualizationError, AnalyticsError
)


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and graceful degradation
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[type, Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies for different error types"""
        self.recovery_strategies[ModelLoadError] = self._handle_model_load_error
        self.recovery_strategies[CalibrationError] = self._handle_calibration_error
        self.recovery_strategies[ProcessingError] = self._handle_processing_error
        self.recovery_strategies[DetectionError] = self._handle_detection_error
        self.recovery_strategies[TrackingError] = self._handle_tracking_error
        self.recovery_strategies[ClassificationError] = self._handle_classification_error
        self.recovery_strategies[VideoError] = self._handle_video_error
        self.recovery_strategies[ExportError] = self._handle_export_error
        self.recovery_strategies[VisualizationError] = self._handle_visualization_error
        self.recovery_strategies[AnalyticsError] = self._handle_analytics_error
    
    def register_fallback_handler(self, component: str, handler: Callable):
        """Register a fallback handler for a specific component"""
        self.fallback_handlers[component] = handler
    
    def handle_error(self, error: FootballAnalyticsError, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Handle an error using appropriate recovery strategy
        
        Args:
            error: The error to handle
            context: Additional context for recovery
            
        Returns:
            Recovery result or None if recovery failed
        """
        error_key = f"{error.__class__.__name__}_{error.component or 'unknown'}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        self.logger.error(f"Handling error: {error}")
        
        # Check if we've exceeded retry limits
        if self.error_counts[error_key] > self.max_retries:
            self.logger.critical(f"Max retries exceeded for {error_key}")
            if error.severity == ErrorSeverity.CRITICAL:
                raise error
            return None
        
        # Get recovery strategy
        strategy = self.recovery_strategies.get(type(error))
        if not strategy:
            self.logger.warning(f"No recovery strategy for {type(error).__name__}")
            return None
        
        try:
            return strategy(error, context)
        except Exception as e:
            self.logger.error(f"Recovery strategy failed: {e}")
            return None
    
    def _handle_model_load_error(self, error: ModelLoadError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle model loading errors with fallback strategies"""
        model_type = error.context.get('model_type', 'unknown')
        
        if model_type == 'detection':
            # Try CPU fallback if GPU failed
            if 'cuda' in error.message.lower():
                self.logger.info("Attempting CPU fallback for detection model")
                return {'device': 'cpu', 'fallback': True}
        
        elif model_type == 'calibration':
            # Use identity transformation as fallback
            self.logger.info("Using identity transformation fallback for calibration")
            return {'use_identity_transform': True, 'fallback': True}
        
        elif model_type == 'classification':
            # Use simple color-based classification
            self.logger.info("Using color-based classification fallback")
            return {'use_color_classification': True, 'fallback': True}
        
        return None
    
    def _handle_calibration_error(self, error: CalibrationError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle calibration errors with fallback mechanisms"""
        lines_detected = error.context.get('field_lines_detected', 0)
        
        if lines_detected < 4:
            # Not enough lines for proper calibration
            self.logger.warning("Insufficient field lines detected, using default field dimensions")
            return {
                'use_default_dimensions': True,
                'field_length': 105.0,
                'field_width': 68.0,
                'fallback': True
            }
        
        # Try alternative calibration method
        method = error.context.get('calibration_method', 'homography')
        if method == 'homography':
            self.logger.info("Trying perspective transformation fallback")
            return {'calibration_method': 'perspective', 'fallback': True}
        
        return None
    
    def _handle_processing_error(self, error: ProcessingError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle frame processing errors"""
        stage = error.context.get('processing_stage', 'unknown')
        
        if stage == 'detection':
            # Skip this frame and continue
            self.logger.info("Skipping frame due to detection error")
            return {'skip_frame': True}
        
        elif stage == 'tracking':
            # Reset tracker and continue
            self.logger.info("Resetting tracker due to processing error")
            return {'reset_tracker': True}
        
        return {'skip_frame': True}
    
    def _handle_detection_error(self, error: DetectionError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle object detection errors"""
        detection_type = error.context.get('detection_type', 'unknown')
        
        if detection_type == 'players':
            # Lower confidence threshold and retry
            current_threshold = error.context.get('confidence_threshold', 0.5)
            new_threshold = max(0.1, current_threshold - 0.1)
            self.logger.info(f"Lowering detection threshold to {new_threshold}")
            return {'confidence_threshold': new_threshold, 'retry': True}
        
        elif detection_type == 'ball':
            # Ball detection is optional, continue without it
            self.logger.info("Continuing without ball detection")
            return {'skip_ball_detection': True}
        
        return None
    
    def _handle_tracking_error(self, error: TrackingError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle tracking errors"""
        tracked_count = error.context.get('tracked_objects_count', 0)
        
        if tracked_count == 0:
            # No objects to track, wait for detections
            self.logger.info("No objects to track, waiting for detections")
            return {'wait_for_detections': True}
        
        # Reset tracker with current detections
        self.logger.info("Resetting tracker with current detections")
        return {'reset_tracker': True, 'reinitialize': True}
    
    def _handle_classification_error(self, error: ClassificationError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle team classification errors"""
        players_count = error.context.get('players_count', 0)
        
        if players_count < 2:
            # Not enough players for classification
            self.logger.info("Not enough players for team classification")
            return {'skip_classification': True}
        
        # Try simpler classification method
        method = error.context.get('classification_method', 'kmeans')
        if method == 'kmeans':
            self.logger.info("Trying color-based classification fallback")
            return {'classification_method': 'color_based', 'fallback': True}
        
        return None
    
    def _handle_video_error(self, error: VideoError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle video I/O errors"""
        operation = error.context.get('operation', 'unknown')
        
        if operation == 'read':
            # Try different video backend
            self.logger.info("Trying alternative video backend")
            return {'try_alternative_backend': True}
        
        elif operation == 'write':
            # Change output format
            self.logger.info("Trying alternative output format")
            return {'output_format': 'avi', 'fallback': True}
        
        return None
    
    def _handle_export_error(self, error: ExportError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle data export errors"""
        export_format = error.context.get('export_format', 'unknown')
        
        if export_format == 'json':
            # Try CSV format instead
            self.logger.info("Trying CSV export as fallback")
            return {'export_format': 'csv', 'fallback': True}
        
        elif export_format == 'csv':
            # Try simple text format
            self.logger.info("Trying text export as fallback")
            return {'export_format': 'txt', 'fallback': True}
        
        return None
    
    def _handle_visualization_error(self, error: VisualizationError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle visualization errors"""
        viz_type = error.context.get('visualization_type', 'unknown')
        
        # Visualization errors are usually non-critical, skip and continue
        self.logger.info(f"Skipping {viz_type} visualization due to error")
        return {'skip_visualization': True}
    
    def _handle_analytics_error(self, error: AnalyticsError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle analytics calculation errors"""
        analytics_type = error.context.get('analytics_type', 'unknown')
        
        if analytics_type == 'heatmap':
            # Use simplified heatmap calculation
            self.logger.info("Using simplified heatmap calculation")
            return {'use_simple_heatmap': True, 'fallback': True}
        
        elif analytics_type == 'velocity':
            # Skip velocity calculation
            self.logger.info("Skipping velocity calculation")
            return {'skip_velocity': True}
        
        return None
    
    def reset_error_counts(self):
        """Reset error counts (useful for new processing sessions)"""
        self.error_counts.clear()
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get current error statistics"""
        return self.error_counts.copy()


def with_error_recovery(recovery_manager: ErrorRecoveryManager, 
                       component: str = None,
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       recovery_action: RecoveryAction = RecoveryAction.RETRY):
    """
    Decorator for automatic error handling and recovery
    
    Args:
        recovery_manager: Error recovery manager instance
        component: Component name for error context
        severity: Default error severity
        recovery_action: Default recovery action
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FootballAnalyticsError as e:
                # Already a football analytics error, handle it
                recovery_result = recovery_manager.handle_error(e)
                if recovery_result and recovery_result.get('retry'):
                    # Retry with modified parameters
                    kwargs.update(recovery_result)
                    return func(*args, **kwargs)
                elif recovery_result and recovery_result.get('fallback'):
                    # Return fallback result
                    return recovery_result
                elif recovery_result and recovery_result.get('skip_frame'):
                    # Skip and return None
                    return None
                else:
                    # Recovery failed, re-raise only for high/critical severity
                    if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                        raise
                    return None
            except Exception as e:
                # Convert to football analytics error
                football_error = FootballAnalyticsError(
                    message=str(e),
                    severity=severity,
                    recovery_action=recovery_action,
                    component=component,
                    original_exception=e
                )
                recovery_result = recovery_manager.handle_error(football_error)
                if recovery_result and recovery_result.get('retry'):
                    kwargs.update(recovery_result)
                    return func(*args, **kwargs)
                elif recovery_result and recovery_result.get('fallback'):
                    return recovery_result
                else:
                    raise football_error
        return wrapper
    return decorator


def create_error_context(frame_id: int = None, 
                        processing_stage: str = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Create error context dictionary for consistent error reporting
    
    Args:
        frame_id: Current frame ID
        processing_stage: Current processing stage
        **kwargs: Additional context parameters
        
    Returns:
        Context dictionary
    """
    context = {}
    if frame_id is not None:
        context['frame_id'] = frame_id
    if processing_stage:
        context['processing_stage'] = processing_stage
    context.update(kwargs)
    return context