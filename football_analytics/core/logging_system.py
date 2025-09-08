"""
Comprehensive logging and monitoring system for the football analytics pipeline
"""

import logging
import logging.handlers
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os
from contextlib import contextmanager

from .exceptions import FootballAnalyticsError


class LogLevel(Enum):
    """Extended log levels for football analytics"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    PERFORMANCE = 25  # Custom level for performance metrics
    ANALYTICS = 15    # Custom level for analytics data


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    component: str
    operation: str
    duration: float
    timestamp: float
    frame_id: Optional[int] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class AnalyticsMetric:
    """Analytics metric data structure"""
    metric_type: str
    value: Union[int, float, str, Dict[str, Any]]
    timestamp: float
    frame_id: Optional[int] = None
    component: str = "analytics"
    metadata: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add thread information
        if hasattr(record, 'thread'):
            log_data['thread_id'] = record.thread
            log_data['thread_name'] = getattr(record, 'threadName', 'Unknown')
        
        # Add extra fields if enabled
        if self.include_extra_fields:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                              'filename', 'module', 'lineno', 'funcName', 'created',
                              'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'getMessage', 'exc_info',
                              'exc_text', 'stack_info']:
                    log_data[key] = value
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: List[PerformanceMetric] = []
        self.active_operations: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def measure_operation(self, component: str, operation: str, frame_id: Optional[int] = None):
        """Context manager for measuring operation performance"""
        operation_key = f"{component}:{operation}"
        start_time = time.time()
        
        with self._lock:
            self.active_operations[operation_key] = start_time
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            with self._lock:
                self.active_operations.pop(operation_key, None)
            
            # Create performance metric
            metric = PerformanceMetric(
                component=component,
                operation=operation,
                duration=duration,
                timestamp=end_time,
                frame_id=frame_id
            )
            
            self.log_performance_metric(metric)
    
    def log_performance_metric(self, metric: PerformanceMetric):
        """Log a performance metric"""
        with self._lock:
            self.metrics.append(metric)
            
            # Keep only last 1000 metrics to prevent memory issues
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
        
        # Log the metric
        self.logger.log(
            LogLevel.PERFORMANCE.value,
            f"Performance: {metric.component}.{metric.operation}",
            extra={
                'component': metric.component,
                'operation': metric.operation,
                'duration': metric.duration,
                'frame_id': metric.frame_id,
                'metric_type': 'performance'
            }
        )
    
    def get_performance_summary(self, time_window: float = 300) -> Dict[str, Any]:
        """Get performance summary for the last time window (seconds)"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"total_operations": 0, "average_duration": 0, "by_component": {}}
        
        # Aggregate by component and operation
        by_component = {}
        total_duration = 0
        
        for metric in recent_metrics:
            component = metric.component
            operation = metric.operation
            
            if component not in by_component:
                by_component[component] = {}
            
            if operation not in by_component[component]:
                by_component[component][operation] = {
                    'count': 0,
                    'total_duration': 0,
                    'min_duration': float('inf'),
                    'max_duration': 0
                }
            
            op_stats = by_component[component][operation]
            op_stats['count'] += 1
            op_stats['total_duration'] += metric.duration
            op_stats['min_duration'] = min(op_stats['min_duration'], metric.duration)
            op_stats['max_duration'] = max(op_stats['max_duration'], metric.duration)
            
            total_duration += metric.duration
        
        # Calculate averages
        for component_stats in by_component.values():
            for op_stats in component_stats.values():
                op_stats['average_duration'] = op_stats['total_duration'] / op_stats['count']
        
        return {
            "total_operations": len(recent_metrics),
            "average_duration": total_duration / len(recent_metrics),
            "by_component": by_component,
            "time_window": time_window
        }


class AnalyticsLogger:
    """Logger for analytics data and metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: List[AnalyticsMetric] = []
        self._lock = threading.Lock()
    
    def log_detection_metrics(self, frame_id: int, detections_count: int, 
                            confidence_scores: List[float], processing_time: float):
        """Log object detection metrics"""
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        metric = AnalyticsMetric(
            metric_type="detection",
            value={
                "detections_count": detections_count,
                "average_confidence": avg_confidence,
                "processing_time": processing_time
            },
            timestamp=time.time(),
            frame_id=frame_id
        )
        
        self._log_analytics_metric(metric)
    
    def log_tracking_metrics(self, frame_id: int, tracked_objects_count: int,
                           new_tracks: int, lost_tracks: int):
        """Log tracking metrics"""
        metric = AnalyticsMetric(
            metric_type="tracking",
            value={
                "tracked_objects": tracked_objects_count,
                "new_tracks": new_tracks,
                "lost_tracks": lost_tracks
            },
            timestamp=time.time(),
            frame_id=frame_id
        )
        
        self._log_analytics_metric(metric)
    
    def log_classification_metrics(self, frame_id: int, team_assignments: Dict[int, int],
                                 classification_confidence: float):
        """Log team classification metrics"""
        team_counts = {}
        for team_id in team_assignments.values():
            team_counts[team_id] = team_counts.get(team_id, 0) + 1
        
        metric = AnalyticsMetric(
            metric_type="classification",
            value={
                "team_counts": team_counts,
                "classification_confidence": classification_confidence,
                "total_players": len(team_assignments)
            },
            timestamp=time.time(),
            frame_id=frame_id
        )
        
        self._log_analytics_metric(metric)
    
    def log_calibration_metrics(self, success: bool, lines_detected: int,
                              calibration_quality: Optional[float] = None):
        """Log field calibration metrics"""
        metric = AnalyticsMetric(
            metric_type="calibration",
            value={
                "success": success,
                "lines_detected": lines_detected,
                "quality_score": calibration_quality
            },
            timestamp=time.time()
        )
        
        self._log_analytics_metric(metric)
    
    def log_custom_metric(self, metric_type: str, value: Any, 
                         frame_id: Optional[int] = None, **metadata):
        """Log custom analytics metric"""
        metric = AnalyticsMetric(
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            frame_id=frame_id,
            metadata=metadata
        )
        
        self._log_analytics_metric(metric)
    
    def _log_analytics_metric(self, metric: AnalyticsMetric):
        """Internal method to log analytics metric"""
        with self._lock:
            self.metrics.append(metric)
            
            # Keep only last 1000 metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
        
        self.logger.log(
            LogLevel.ANALYTICS.value,
            f"Analytics: {metric.metric_type}",
            extra={
                'metric_type': metric.metric_type,
                'value': metric.value,
                'frame_id': metric.frame_id,
                'metadata': metric.metadata,
                'analytics_data': True
            }
        )
    
    def get_analytics_summary(self, time_window: float = 300) -> Dict[str, Any]:
        """Get analytics summary for the last time window"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        summary = {"total_metrics": len(recent_metrics), "by_type": {}}
        
        for metric in recent_metrics:
            metric_type = metric.metric_type
            if metric_type not in summary["by_type"]:
                summary["by_type"][metric_type] = {"count": 0, "latest_value": None}
            
            summary["by_type"][metric_type]["count"] += 1
            summary["by_type"][metric_type]["latest_value"] = metric.value
        
        return summary


class FootballAnalyticsLogger:
    """Main logging system for football analytics"""
    
    def __init__(self, 
                 log_level: Union[str, int] = logging.INFO,
                 log_dir: Optional[Path] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_structured: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Add custom log levels
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        logging.addLevelName(LogLevel.PERFORMANCE.value, "PERFORMANCE")
        logging.addLevelName(LogLevel.ANALYTICS.value, "ANALYTICS")
        
        # Create main logger
        self.logger = logging.getLogger("football_analytics")
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Session info (set before handlers)
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup handlers
        if enable_console:
            self._setup_console_handler(enable_structured)
        
        if enable_file:
            self._setup_file_handlers(enable_structured, max_file_size, backup_count)
        
        # Initialize monitoring components
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.analytics_logger = AnalyticsLogger(self.logger)
        
        self.logger.info(f"Football Analytics Logger initialized - Session: {self.session_id}")
    
    def _setup_console_handler(self, structured: bool):
        """Setup console logging handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, structured: bool, max_size: int, backup_count: int):
        """Setup file logging handlers"""
        # Main log file
        main_log_file = self.log_dir / f"football_analytics_{self.session_id}.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=max_size, backupCount=backup_count
        )
        
        if structured:
            main_handler.setFormatter(StructuredFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            main_handler.setFormatter(formatter)
        
        self.logger.addHandler(main_handler)
        
        # Error-only log file
        error_log_file = self.log_dir / f"errors_{self.session_id}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=max_size, backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)
        
        # Performance log file
        perf_log_file = self.log_dir / f"performance_{self.session_id}.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file, maxBytes=max_size, backupCount=backup_count
        )
        perf_handler.setLevel(LogLevel.PERFORMANCE.value)
        perf_handler.addFilter(lambda record: record.levelno == LogLevel.PERFORMANCE.value)
        perf_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(perf_handler)
        
        # Analytics log file
        analytics_log_file = self.log_dir / f"analytics_{self.session_id}.log"
        analytics_handler = logging.handlers.RotatingFileHandler(
            analytics_log_file, maxBytes=max_size, backupCount=backup_count
        )
        analytics_handler.setLevel(LogLevel.ANALYTICS.value)
        analytics_handler.addFilter(lambda record: record.levelno == LogLevel.ANALYTICS.value)
        analytics_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(analytics_handler)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance"""
        if name:
            return logging.getLogger(f"football_analytics.{name}")
        return self.logger
    
    def log_system_info(self):
        """Log system information"""
        import platform
        import psutil
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "session_id": self.session_id
        }
        
        self.logger.info("System Information", extra=system_info)
    
    def create_component_logger(self, component_name: str) -> logging.Logger:
        """Create a logger for a specific component"""
        return self.get_logger(component_name)
    
    def log_error_with_context(self, error: FootballAnalyticsError, 
                              additional_context: Optional[Dict[str, Any]] = None):
        """Log error with full context"""
        context = error.to_dict()
        if additional_context:
            context.update(additional_context)
        
        self.logger.error(f"Error in {error.component}: {error.message}", extra=context)
    
    def generate_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive session report"""
        session_duration = time.time() - self.session_start
        
        return {
            "session_id": self.session_id,
            "session_duration": session_duration,
            "session_start": datetime.fromtimestamp(self.session_start).isoformat(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "analytics_summary": self.analytics_logger.get_analytics_summary(),
            "log_files": {
                "main_log": str(self.log_dir / f"football_analytics_{self.session_id}.log"),
                "error_log": str(self.log_dir / f"errors_{self.session_id}.log"),
                "performance_log": str(self.log_dir / f"performance_{self.session_id}.log"),
                "analytics_log": str(self.log_dir / f"analytics_{self.session_id}.log")
            }
        }
    
    def close(self):
        """Close all handlers and generate final report"""
        final_report = self.generate_session_report()
        self.logger.info("Session ending", extra=final_report)
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Global logger instance
_global_logger: Optional[FootballAnalyticsLogger] = None


def get_global_logger() -> FootballAnalyticsLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = FootballAnalyticsLogger()
    return _global_logger


def set_global_logger(logger: FootballAnalyticsLogger):
    """Set global logger instance"""
    global _global_logger
    _global_logger = logger


def setup_logging(log_level: Union[str, int] = logging.INFO,
                 log_dir: Optional[Path] = None,
                 **kwargs) -> FootballAnalyticsLogger:
    """Setup and configure logging system"""
    logger = FootballAnalyticsLogger(log_level=log_level, log_dir=log_dir, **kwargs)
    set_global_logger(logger)
    logger.log_system_info()
    return logger