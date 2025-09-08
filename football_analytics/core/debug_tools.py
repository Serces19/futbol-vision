"""
Debugging tools and diagnostic utilities for the football analytics system
"""

import time
import traceback
import inspect
import functools
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import json
import cv2
import numpy as np
from datetime import datetime

from .exceptions import FootballAnalyticsError
from .logging_system import get_global_logger


class DebugProfiler:
    """Profiler for debugging performance issues"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profiles: Dict[str, List[float]] = {}
        self.call_counts: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function execution time"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    if name not in self.profiles:
                        self.profiles[name] = []
                        self.call_counts[name] = 0
                    
                    self.profiles[name].append(duration)
                    self.call_counts[name] += 1
                    
                    # Keep only last 100 calls to prevent memory issues
                    if len(self.profiles[name]) > 100:
                        self.profiles[name] = self.profiles[name][-100:]
            
            return wrapper
        return decorator
    
    def get_profile_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all profiled functions"""
        summary = {}
        
        for func_name, durations in self.profiles.items():
            if durations:
                summary[func_name] = {
                    'call_count': self.call_counts[func_name],
                    'total_time': sum(durations),
                    'average_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations),
                    'recent_calls': len(durations)
                }
        
        return summary
    
    def reset_profiles(self):
        """Reset all profiling data"""
        self.profiles.clear()
        self.call_counts.clear()
    
    def log_profile_summary(self):
        """Log profiling summary"""
        summary = self.get_profile_summary()
        if summary:
            self.logger.info("Function profiling summary:")
            for func_name, stats in summary.items():
                self.logger.info(
                    f"  {func_name}: {stats['call_count']} calls, "
                    f"avg: {stats['average_time']:.4f}s, "
                    f"total: {stats['total_time']:.4f}s"
                )


class FrameDebugger:
    """Debug individual frame processing"""
    
    def __init__(self, output_dir: Optional[Path] = None, max_debug_frames: int = 10):
        self.output_dir = Path(output_dir) if output_dir else Path("debug_frames")
        self.output_dir.mkdir(exist_ok=True)
        self.max_debug_frames = max_debug_frames
        self.debug_frame_count = 0
        self.logger = logging.getLogger(__name__)
    
    def save_debug_frame(self, frame: np.ndarray, frame_id: int, 
                        stage: str, additional_data: Optional[Dict[str, Any]] = None):
        """Save frame with debug information"""
        if self.debug_frame_count >= self.max_debug_frames:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save frame image
        frame_filename = f"frame_{frame_id}_{stage}_{timestamp}.jpg"
        frame_path = self.output_dir / frame_filename
        
        try:
            cv2.imwrite(str(frame_path), frame)
            
            # Save metadata
            metadata = {
                'frame_id': frame_id,
                'stage': stage,
                'timestamp': timestamp,
                'frame_shape': frame.shape,
                'frame_path': str(frame_path)
            }
            
            if additional_data:
                metadata['additional_data'] = additional_data
            
            metadata_filename = f"frame_{frame_id}_{stage}_{timestamp}.json"
            metadata_path = self.output_dir / metadata_filename
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.debug_frame_count += 1
            self.logger.debug(f"Debug frame saved: {frame_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save debug frame: {e}")
    
    def save_detection_debug(self, frame: np.ndarray, detections: List[Any], 
                           frame_id: int, confidence_threshold: float = 0.5):
        """Save frame with detection debug information"""
        debug_frame = frame.copy()
        
        detection_data = []
        for i, detection in enumerate(detections):
            if hasattr(detection, 'bbox') and hasattr(detection, 'confidence'):
                bbox = detection.bbox
                conf = detection.confidence
                
                if conf >= confidence_threshold:
                    # Draw bounding box
                    cv2.rectangle(debug_frame, (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    
                    # Add confidence text
                    cv2.putText(debug_frame, f"{conf:.2f}", 
                              (int(bbox[0]), int(bbox[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                detection_data.append({
                    'id': i,
                    'bbox': bbox,
                    'confidence': conf,
                    'class_name': getattr(detection, 'class_name', 'unknown')
                })
        
        additional_data = {
            'detections': detection_data,
            'confidence_threshold': confidence_threshold,
            'total_detections': len(detections)
        }
        
        self.save_debug_frame(debug_frame, frame_id, "detection", additional_data)
    
    def save_tracking_debug(self, frame: np.ndarray, tracked_objects: List[Any], frame_id: int):
        """Save frame with tracking debug information"""
        debug_frame = frame.copy()
        
        tracking_data = []
        for obj in tracked_objects:
            if hasattr(obj, 'track_id') and hasattr(obj, 'detection'):
                bbox = obj.detection.bbox
                track_id = obj.track_id
                
                # Draw bounding box with track ID
                cv2.rectangle(debug_frame, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                
                # Add track ID text
                cv2.putText(debug_frame, f"ID:{track_id}", 
                          (int(bbox[0]), int(bbox[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                tracking_data.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'team_id': getattr(obj, 'team_id', None)
                })
        
        additional_data = {
            'tracked_objects': tracking_data,
            'total_tracked': len(tracked_objects)
        }
        
        self.save_debug_frame(debug_frame, frame_id, "tracking", additional_data)


class ComponentDebugger:
    """Debug individual components"""
    
    def __init__(self):
        self.component_states: Dict[str, Dict[str, Any]] = {}
        self.component_errors: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger(__name__)
    
    def log_component_state(self, component_name: str, state_data: Dict[str, Any]):
        """Log current state of a component"""
        self.component_states[component_name] = {
            'timestamp': time.time(),
            'state': state_data
        }
        
        self.logger.debug(f"Component {component_name} state updated", extra=state_data)
    
    def log_component_error(self, component_name: str, error: Exception, 
                          context: Optional[Dict[str, Any]] = None):
        """Log component error with context"""
        if component_name not in self.component_errors:
            self.component_errors[component_name] = []
        
        error_data = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.component_errors[component_name].append(error_data)
        
        # Keep only last 10 errors per component
        if len(self.component_errors[component_name]) > 10:
            self.component_errors[component_name] = self.component_errors[component_name][-10:]
        
        self.logger.error(f"Component {component_name} error: {error}", extra=error_data)
    
    def get_component_debug_info(self, component_name: str) -> Dict[str, Any]:
        """Get debug information for a specific component"""
        return {
            'current_state': self.component_states.get(component_name),
            'recent_errors': self.component_errors.get(component_name, []),
            'error_count': len(self.component_errors.get(component_name, []))
        }
    
    def get_all_debug_info(self) -> Dict[str, Any]:
        """Get debug information for all components"""
        return {
            'component_states': self.component_states,
            'component_errors': self.component_errors,
            'total_components': len(self.component_states),
            'components_with_errors': len(self.component_errors)
        }


class MemoryDebugger:
    """Debug memory usage and leaks"""
    
    def __init__(self):
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            self.logger.warning("psutil not available, memory debugging limited")
    
    def take_memory_snapshot(self, label: str = ""):
        """Take a memory usage snapshot"""
        if not self.psutil_available:
            return
        
        import psutil
        process = psutil.Process()
        
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'memory_info': process.memory_info()._asdict(),
            'memory_percent': process.memory_percent(),
            'num_threads': process.num_threads()
        }
        
        self.memory_snapshots.append(snapshot)
        
        # Keep only last 50 snapshots
        if len(self.memory_snapshots) > 50:
            self.memory_snapshots = self.memory_snapshots[-50:]
        
        self.logger.debug(f"Memory snapshot taken: {label}", extra=snapshot)
    
    def detect_memory_leaks(self, threshold_mb: float = 100.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks"""
        if len(self.memory_snapshots) < 2:
            return []
        
        leaks = []
        
        # Compare memory usage over time
        for i in range(1, len(self.memory_snapshots)):
            prev_snapshot = self.memory_snapshots[i-1]
            curr_snapshot = self.memory_snapshots[i]
            
            prev_memory = prev_snapshot['memory_info']['rss'] / (1024 * 1024)  # MB
            curr_memory = curr_snapshot['memory_info']['rss'] / (1024 * 1024)  # MB
            
            memory_increase = curr_memory - prev_memory
            
            if memory_increase > threshold_mb:
                leaks.append({
                    'from_label': prev_snapshot['label'],
                    'to_label': curr_snapshot['label'],
                    'memory_increase_mb': memory_increase,
                    'time_diff': curr_snapshot['timestamp'] - prev_snapshot['timestamp']
                })
        
        return leaks
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        if not self.memory_snapshots:
            return {}
        
        latest = self.memory_snapshots[-1]
        first = self.memory_snapshots[0]
        
        return {
            'current_memory_mb': latest['memory_info']['rss'] / (1024 * 1024),
            'memory_percent': latest['memory_percent'],
            'num_threads': latest['num_threads'],
            'total_increase_mb': (latest['memory_info']['rss'] - first['memory_info']['rss']) / (1024 * 1024),
            'snapshots_count': len(self.memory_snapshots),
            'potential_leaks': self.detect_memory_leaks()
        }


class DebugManager:
    """Central manager for all debugging tools"""
    
    def __init__(self, 
                 enable_profiling: bool = False,
                 enable_frame_debug: bool = False,
                 enable_component_debug: bool = True,
                 enable_memory_debug: bool = False,
                 debug_output_dir: Optional[Path] = None):
        
        self.profiler = DebugProfiler(enable_profiling) if enable_profiling else None
        self.frame_debugger = FrameDebugger(debug_output_dir) if enable_frame_debug else None
        self.component_debugger = ComponentDebugger() if enable_component_debug else None
        self.memory_debugger = MemoryDebugger() if enable_memory_debug else None
        
        self.logger = logging.getLogger(__name__)
        self.debug_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Debug manager initialized - Session: {self.debug_session_id}")
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator for function profiling"""
        if self.profiler:
            return self.profiler.profile_function(func_name)
        else:
            # Return pass-through decorator if profiling disabled
            def decorator(func):
                return func
            return decorator
    
    def debug_frame(self, frame: np.ndarray, frame_id: int, stage: str, **kwargs):
        """Debug frame processing"""
        if self.frame_debugger:
            self.frame_debugger.save_debug_frame(frame, frame_id, stage, kwargs)
    
    def debug_detections(self, frame: np.ndarray, detections: List[Any], frame_id: int, **kwargs):
        """Debug detection results"""
        if self.frame_debugger:
            self.frame_debugger.save_detection_debug(frame, detections, frame_id, **kwargs)
    
    def debug_tracking(self, frame: np.ndarray, tracked_objects: List[Any], frame_id: int):
        """Debug tracking results"""
        if self.frame_debugger:
            self.frame_debugger.save_tracking_debug(frame, tracked_objects, frame_id)
    
    def log_component_state(self, component_name: str, **state_data):
        """Log component state"""
        if self.component_debugger:
            self.component_debugger.log_component_state(component_name, state_data)
    
    def log_component_error(self, component_name: str, error: Exception, **context):
        """Log component error"""
        if self.component_debugger:
            self.component_debugger.log_component_error(component_name, error, context)
    
    def take_memory_snapshot(self, label: str = ""):
        """Take memory snapshot"""
        if self.memory_debugger:
            self.memory_debugger.take_memory_snapshot(label)
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        report = {
            'session_id': self.debug_session_id,
            'timestamp': datetime.now().isoformat(),
            'enabled_debuggers': {
                'profiler': self.profiler is not None,
                'frame_debugger': self.frame_debugger is not None,
                'component_debugger': self.component_debugger is not None,
                'memory_debugger': self.memory_debugger is not None
            }
        }
        
        if self.profiler:
            report['profiling'] = self.profiler.get_profile_summary()
        
        if self.component_debugger:
            report['components'] = self.component_debugger.get_all_debug_info()
        
        if self.memory_debugger:
            report['memory'] = self.memory_debugger.get_memory_summary()
        
        return report
    
    def save_debug_report(self, output_path: Optional[Path] = None):
        """Save debug report to file"""
        report = self.generate_debug_report()
        
        if output_path is None:
            output_path = Path(f"debug_report_{self.debug_session_id}.json")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Debug report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save debug report: {e}")


# Global debug manager instance
_global_debug_manager: Optional[DebugManager] = None


def get_global_debug_manager() -> Optional[DebugManager]:
    """Get global debug manager instance"""
    return _global_debug_manager


def set_global_debug_manager(manager: DebugManager):
    """Set global debug manager instance"""
    global _global_debug_manager
    _global_debug_manager = manager


def setup_debugging(enable_profiling: bool = False,
                   enable_frame_debug: bool = False,
                   enable_component_debug: bool = True,
                   enable_memory_debug: bool = False,
                   debug_output_dir: Optional[Path] = None) -> DebugManager:
    """Setup and configure debugging system"""
    manager = DebugManager(
        enable_profiling=enable_profiling,
        enable_frame_debug=enable_frame_debug,
        enable_component_debug=enable_component_debug,
        enable_memory_debug=enable_memory_debug,
        debug_output_dir=debug_output_dir
    )
    set_global_debug_manager(manager)
    return manager