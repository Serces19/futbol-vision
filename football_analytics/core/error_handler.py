"""
Comprehensive error handling utilities for the football analytics system
"""

from typing import Optional, Dict, Any, List, Callable, Union, Type
import logging
import traceback
import sys
from contextlib import contextmanager
from functools import wraps
import time

from .exceptions import (
    FootballAnalyticsError, ErrorSeverity, RecoveryAction,
    ModelLoadError, CalibrationError, ProcessingError, DetectionError,
    TrackingError, ClassificationError, VideoError, ConfigurationError,
    ExportError, VisualizationError, AnalyticsError
)
from .error_recovery import ErrorRecoveryManager


class ErrorHandler:
    """
    Centralized error handling system for the football analytics pipeline
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 recovery_manager: Optional[ErrorRecoveryManager] = None,
                 enable_graceful_degradation: bool = True):
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_manager = recovery_manager or ErrorRecoveryManager()
        self.enable_graceful_degradation = enable_graceful_degradation
        self.error_history: List[Dict[str, Any]] = []
        self.component_states: Dict[str, str] = {}  # Track component health states
        
    def handle_exception(self, 
                        exception: Exception,
                        component: str = None,
                        context: Optional[Dict[str, Any]] = None,
                        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                        recovery_action: RecoveryAction = RecoveryAction.RETRY) -> Any:
        """
        Handle any exception with appropriate error handling strategy
        
        Args:
            exception: The exception to handle
            component: Component where error occurred
            context: Additional context information
            severity: Error severity level
            recovery_action: Suggested recovery action
            
        Returns:
            Recovery result or None if recovery failed
        """
        # Convert to FootballAnalyticsError if needed
        if not isinstance(exception, FootballAnalyticsError):
            football_error = FootballAnalyticsError(
                message=str(exception),
                severity=severity,
                recovery_action=recovery_action,
                context=context or {},
                component=component,
                original_exception=exception
            )
        else:
            football_error = exception
        
        # Log the error
        self._log_error(football_error)
        
        # Add to error history
        self._add_to_history(football_error)
        
        # Update component state
        if component:
            self._update_component_state(component, "error")
        
        # Attempt recovery if enabled
        if self.enable_graceful_degradation and football_error.recoverable:
            recovery_result = self.recovery_manager.handle_error(football_error, context)
            if recovery_result:
                self.logger.info(f"Recovery successful for {football_error.__class__.__name__}")
                if component:
                    self._update_component_state(component, "recovered")
                return recovery_result
        
        # If recovery failed or not enabled, decide whether to raise
        if football_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            if component:
                self._update_component_state(component, "failed")
            raise football_error
        
        # For low/medium severity, log and continue
        self.logger.warning(f"Continuing despite error: {football_error}")
        return None
    
    def _log_error(self, error: FootballAnalyticsError):
        """Log error with appropriate level based on severity"""
        log_message = f"[{error.component or 'Unknown'}] {error.message}"
        
        if error.context:
            context_str = ", ".join([f"{k}={v}" for k, v in error.context.items()])
            log_message += f" | Context: {context_str}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log traceback for debugging
        if error.traceback_str:
            self.logger.debug(f"Traceback: {error.traceback_str}")
    
    def _add_to_history(self, error: FootballAnalyticsError):
        """Add error to history for analysis"""
        error_dict = error.to_dict()
        # Flatten context into main dict for easier access
        if error_dict.get('context'):
            error_dict.update(error_dict['context'])
        self.error_history.append(error_dict)
        
        # Keep only last 100 errors to prevent memory issues
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def _update_component_state(self, component: str, state: str):
        """Update component health state"""
        self.component_states[component] = state
        self.logger.debug(f"Component {component} state updated to: {state}")
    
    def get_component_health(self) -> Dict[str, str]:
        """Get current health status of all components"""
        return self.component_states.copy()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        if not self.error_history:
            return {"total_errors": 0, "by_severity": {}, "by_component": {}}
        
        by_severity = {}
        by_component = {}
        
        for error in self.error_history:
            severity = error.get('severity', 'unknown')
            component = error.get('component', 'unknown')
            
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_component[component] = by_component.get(component, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_severity": by_severity,
            "by_component": by_component,
            "recent_errors": self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        }
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.component_states.clear()
        self.recovery_manager.reset_error_counts()
    
    @contextmanager
    def error_context(self, 
                     component: str,
                     operation: str = None,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     recovery_action: RecoveryAction = RecoveryAction.RETRY):
        """
        Context manager for handling errors in a specific operation
        
        Usage:
            with error_handler.error_context("ObjectDetector", "detect_players"):
                # Code that might raise exceptions
                detections = detector.detect(frame)
        """
        context = {"operation": operation} if operation else {}
        
        try:
            self._update_component_state(component, "running")
            yield
            self._update_component_state(component, "healthy")
        except Exception as e:
            recovery_result = self.handle_exception(
                e, component, context, severity, recovery_action
            )
            if recovery_result is None and severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                raise
            return recovery_result


def safe_execute(error_handler: ErrorHandler,
                component: str,
                operation: str = None,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                recovery_action: RecoveryAction = RecoveryAction.RETRY,
                default_return: Any = None):
    """
    Decorator for safe execution of functions with error handling
    
    Args:
        error_handler: Error handler instance
        component: Component name
        operation: Operation name
        severity: Error severity level
        recovery_action: Recovery action
        default_return: Default return value if error occurs
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with error_handler.error_context(component, operation, severity, recovery_action):
                    return func(*args, **kwargs)
            except Exception as e:
                error_handler.logger.warning(f"Function {func.__name__} failed, returning default value")
                return default_return
        return wrapper
    return decorator


class ErrorAggregator:
    """
    Aggregates and analyzes errors across the system for monitoring
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.errors: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def add_error(self, error: FootballAnalyticsError):
        """Add error to aggregation"""
        error_data = error.to_dict()
        error_data['relative_timestamp'] = error.timestamp - self.start_time
        
        self.errors.append(error_data)
        
        # Keep only recent errors
        if len(self.errors) > self.window_size:
            self.errors = self.errors[-self.window_size:]
    
    def get_error_rate(self, time_window: float = 60.0) -> float:
        """Get error rate (errors per second) in the last time window"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_errors = [
            e for e in self.errors 
            if e['timestamp'] >= cutoff_time
        ]
        
        return len(recent_errors) / time_window if time_window > 0 else 0
    
    def get_most_frequent_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most frequent error types"""
        error_counts = {}
        
        for error in self.errors:
            error_type = error.get('error_type', 'Unknown')
            component = error.get('component', 'Unknown')
            key = f"{component}:{error_type}"
            
            if key not in error_counts:
                error_counts[key] = {
                    'component': component,
                    'error_type': error_type,
                    'count': 0,
                    'last_occurrence': error['timestamp']
                }
            
            error_counts[key]['count'] += 1
            error_counts[key]['last_occurrence'] = max(
                error_counts[key]['last_occurrence'],
                error['timestamp']
            )
        
        # Sort by count and return top errors
        sorted_errors = sorted(
            error_counts.values(),
            key=lambda x: x['count'],
            reverse=True
        )
        
        return sorted_errors[:limit]
    
    def get_component_reliability(self) -> Dict[str, float]:
        """Calculate reliability score for each component (0-1, higher is better)"""
        component_stats = {}
        
        for error in self.errors:
            component = error.get('component', 'Unknown')
            severity = error.get('severity', 'medium')
            
            if component not in component_stats:
                component_stats[component] = {
                    'total_errors': 0,
                    'critical_errors': 0,
                    'high_errors': 0
                }
            
            component_stats[component]['total_errors'] += 1
            if severity == 'critical':
                component_stats[component]['critical_errors'] += 1
            elif severity == 'high':
                component_stats[component]['high_errors'] += 1
        
        # Calculate reliability scores
        reliability_scores = {}
        for component, stats in component_stats.items():
            # Penalize critical and high severity errors more
            penalty = (
                stats['critical_errors'] * 3 +
                stats['high_errors'] * 2 +
                (stats['total_errors'] - stats['critical_errors'] - stats['high_errors'])
            )
            
            # Normalize to 0-1 scale (assuming max 10 errors is very unreliable)
            reliability = max(0, 1 - (penalty / (self.window_size * 0.1)))
            reliability_scores[component] = reliability
        
        return reliability_scores
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        return {
            'total_errors': len(self.errors),
            'error_rate_1min': self.get_error_rate(60),
            'error_rate_5min': self.get_error_rate(300),
            'most_frequent_errors': self.get_most_frequent_errors(),
            'component_reliability': self.get_component_reliability(),
            'uptime_seconds': time.time() - self.start_time
        }


# Global error handler instance (can be configured by applications)
_global_error_handler: Optional[ErrorHandler] = None


def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def set_global_error_handler(handler: ErrorHandler):
    """Set global error handler instance"""
    global _global_error_handler
    _global_error_handler = handler