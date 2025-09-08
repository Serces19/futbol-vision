"""
System monitoring and diagnostics for the football analytics pipeline
"""

import time
import threading
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_available: int
    disk_usage_percent: float
    gpu_metrics: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingMetrics:
    """Processing pipeline metrics"""
    timestamp: float
    frame_id: int
    fps: float
    processing_time: float
    queue_size: int
    active_threads: int
    component_states: Dict[str, str]


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    timestamp: float
    frame_id: int
    detection_quality: float
    tracking_quality: float
    calibration_quality: float
    overall_quality: float


class ResourceMonitor:
    """Monitor system resources (CPU, memory, GPU)"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_history: deque = deque(maxlen=300)  # Keep 5 minutes at 1s intervals
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Log warnings for high resource usage
                self._check_resource_warnings(metrics)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(self.update_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics if available
        gpu_metrics = None
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_metrics = {
                        'gpu_percent': gpu.load * 100,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_memory_used': gpu.memoryUsed,
                        'gpu_memory_total': gpu.memoryTotal,
                        'gpu_temperature': gpu.temperature
                    }
            except Exception as e:
                self.logger.debug(f"Could not collect GPU metrics: {e}")
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used=memory.used,
            memory_available=memory.available,
            disk_usage_percent=disk.percent,
            gpu_metrics=gpu_metrics
        )
    
    def _check_resource_warnings(self, metrics: SystemMetrics):
        """Check for resource usage warnings"""
        if metrics.cpu_percent > 90:
            self.logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > 90:
            self.logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_metrics and metrics.gpu_metrics['gpu_percent'] > 95:
            self.logger.warning(f"High GPU usage: {metrics.gpu_metrics['gpu_percent']:.1f}%")
        
        if metrics.gpu_metrics and metrics.gpu_metrics['gpu_memory_percent'] > 90:
            self.logger.warning(f"High GPU memory usage: {metrics.gpu_metrics['gpu_memory_percent']:.1f}%")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, time_window: float = 60.0) -> Dict[str, Any]:
        """Get summary of metrics over time window"""
        if not self.metrics_history:
            return {}
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        if not recent_metrics:
            return {}
        
        # Calculate averages and peaks
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        summary = {
            'time_window': time_window,
            'sample_count': len(recent_metrics),
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'peak': max(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory': {
                'average': sum(memory_values) / len(memory_values),
                'peak': max(memory_values),
                'current': memory_values[-1] if memory_values else 0
            }
        }
        
        # Add GPU metrics if available
        gpu_values = [m.gpu_metrics for m in recent_metrics if m.gpu_metrics]
        if gpu_values:
            gpu_usage = [g['gpu_percent'] for g in gpu_values]
            gpu_memory = [g['gpu_memory_percent'] for g in gpu_values]
            
            summary['gpu'] = {
                'usage': {
                    'average': sum(gpu_usage) / len(gpu_usage),
                    'peak': max(gpu_usage),
                    'current': gpu_usage[-1] if gpu_usage else 0
                },
                'memory': {
                    'average': sum(gpu_memory) / len(gpu_memory),
                    'peak': max(gpu_memory),
                    'current': gpu_memory[-1] if gpu_memory else 0
                }
            }
        
        return summary


class ProcessingMonitor:
    """Monitor processing pipeline performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.processing_history: deque = deque(maxlen=window_size)
        self.frame_times: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.total_frames = 0
        
    def log_frame_processed(self, frame_id: int, processing_time: float, 
                           queue_size: int = 0, component_states: Optional[Dict[str, str]] = None):
        """Log that a frame has been processed"""
        current_time = time.time()
        self.total_frames += 1
        
        # Calculate FPS
        self.frame_times.append(current_time)
        if len(self.frame_times) >= 2:
            time_span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / time_span if time_span > 0 else 0
        else:
            fps = 0
        
        # Create processing metrics
        metrics = ProcessingMetrics(
            timestamp=current_time,
            frame_id=frame_id,
            fps=fps,
            processing_time=processing_time,
            queue_size=queue_size,
            active_threads=threading.active_count(),
            component_states=component_states or {}
        )
        
        self.processing_history.append(metrics)
        
        # Log performance warnings
        if processing_time > 1.0:  # More than 1 second per frame
            self.logger.warning(f"Slow frame processing: {processing_time:.2f}s for frame {frame_id}")
        
        if fps < 10 and len(self.frame_times) > 10:
            self.logger.warning(f"Low FPS detected: {fps:.1f}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.processing_history:
            return {}
        
        recent_metrics = list(self.processing_history)
        processing_times = [m.processing_time for m in recent_metrics]
        fps_values = [m.fps for m in recent_metrics if m.fps > 0]
        
        total_runtime = time.time() - self.start_time
        
        return {
            'total_frames': self.total_frames,
            'total_runtime': total_runtime,
            'average_fps': self.total_frames / total_runtime if total_runtime > 0 else 0,
            'recent_fps': fps_values[-1] if fps_values else 0,
            'processing_time': {
                'average': sum(processing_times) / len(processing_times),
                'min': min(processing_times),
                'max': max(processing_times),
                'current': processing_times[-1] if processing_times else 0
            },
            'frames_in_window': len(recent_metrics)
        }


class QualityMonitor:
    """Monitor processing quality metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.quality_history: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)
        
    def log_quality_metrics(self, frame_id: int, detection_quality: float = 0.0,
                           tracking_quality: float = 0.0, calibration_quality: float = 0.0):
        """Log quality metrics for a frame"""
        overall_quality = (detection_quality + tracking_quality + calibration_quality) / 3
        
        metrics = QualityMetrics(
            timestamp=time.time(),
            frame_id=frame_id,
            detection_quality=detection_quality,
            tracking_quality=tracking_quality,
            calibration_quality=calibration_quality,
            overall_quality=overall_quality
        )
        
        self.quality_history.append(metrics)
        
        # Log quality warnings
        if overall_quality < 0.5:
            self.logger.warning(f"Low overall quality: {overall_quality:.2f} for frame {frame_id}")
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality statistics"""
        if not self.quality_history:
            return {}
        
        recent_metrics = list(self.quality_history)
        
        detection_scores = [m.detection_quality for m in recent_metrics]
        tracking_scores = [m.tracking_quality for m in recent_metrics]
        calibration_scores = [m.calibration_quality for m in recent_metrics]
        overall_scores = [m.overall_quality for m in recent_metrics]
        
        return {
            'detection_quality': {
                'average': sum(detection_scores) / len(detection_scores),
                'min': min(detection_scores),
                'max': max(detection_scores),
                'current': detection_scores[-1] if detection_scores else 0
            },
            'tracking_quality': {
                'average': sum(tracking_scores) / len(tracking_scores),
                'min': min(tracking_scores),
                'max': max(tracking_scores),
                'current': tracking_scores[-1] if tracking_scores else 0
            },
            'calibration_quality': {
                'average': sum(calibration_scores) / len(calibration_scores),
                'min': min(calibration_scores),
                'max': max(calibration_scores),
                'current': calibration_scores[-1] if calibration_scores else 0
            },
            'overall_quality': {
                'average': sum(overall_scores) / len(overall_scores),
                'min': min(overall_scores),
                'max': max(overall_scores),
                'current': overall_scores[-1] if overall_scores else 0
            }
        }


class DiagnosticCollector:
    """Collect diagnostic information for troubleshooting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.diagnostic_data: Dict[str, Any] = {}
        
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        import platform
        
        system_info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation(),
                'compiler': platform.python_compiler()
            },
            'resources': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total
            }
        }
        
        # Add GPU information if available
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                system_info['gpu'] = [
                    {
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'driver': gpu.driver
                    } for gpu in gpus
                ]
            except Exception as e:
                system_info['gpu'] = f"Error collecting GPU info: {e}"
        
        return system_info
    
    def collect_model_info(self, model_paths: Dict[str, str]) -> Dict[str, Any]:
        """Collect information about loaded models"""
        model_info = {}
        
        for model_name, model_path in model_paths.items():
            path_obj = Path(model_path)
            if path_obj.exists():
                model_info[model_name] = {
                    'path': str(path_obj),
                    'size': path_obj.stat().st_size,
                    'modified': datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat(),
                    'exists': True
                }
            else:
                model_info[model_name] = {
                    'path': str(path_obj),
                    'exists': False,
                    'error': 'File not found'
                }
        
        return model_info
    
    def collect_configuration_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect configuration information"""
        return {
            'configuration': config,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_diagnostic_report(self, 
                                 resource_monitor: Optional[ResourceMonitor] = None,
                                 processing_monitor: Optional[ProcessingMonitor] = None,
                                 quality_monitor: Optional[QualityMonitor] = None,
                                 additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.collect_system_info()
        }
        
        if resource_monitor:
            report['resource_metrics'] = resource_monitor.get_metrics_summary()
        
        if processing_monitor:
            report['processing_stats'] = processing_monitor.get_processing_stats()
        
        if quality_monitor:
            report['quality_stats'] = quality_monitor.get_quality_stats()
        
        if additional_data:
            report['additional_data'] = additional_data
        
        return report
    
    def save_diagnostic_report(self, report: Dict[str, Any], output_path: Path):
        """Save diagnostic report to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Diagnostic report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save diagnostic report: {e}")


class SystemMonitor:
    """Comprehensive system monitoring combining all monitors"""
    
    def __init__(self, 
                 resource_update_interval: float = 1.0,
                 enable_resource_monitoring: bool = True):
        
        self.resource_monitor = ResourceMonitor(resource_update_interval) if enable_resource_monitoring else None
        self.processing_monitor = ProcessingMonitor()
        self.quality_monitor = QualityMonitor()
        self.diagnostic_collector = DiagnosticCollector()
        self.logger = logging.getLogger(__name__)
        
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start all monitoring components"""
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        self.monitoring_active = True
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        self.monitoring_active = False
        self.logger.info("System monitoring stopped")
    
    def log_frame_processed(self, frame_id: int, processing_time: float, **kwargs):
        """Log frame processing metrics"""
        self.processing_monitor.log_frame_processed(frame_id, processing_time, **kwargs)
    
    def log_quality_metrics(self, frame_id: int, **quality_scores):
        """Log quality metrics"""
        self.quality_monitor.log_quality_metrics(frame_id, **quality_scores)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'monitoring_active': self.monitoring_active,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.resource_monitor:
            status['resources'] = self.resource_monitor.get_metrics_summary()
        
        status['processing'] = self.processing_monitor.get_processing_stats()
        status['quality'] = self.quality_monitor.get_quality_stats()
        
        return status
    
    def generate_health_check(self) -> Dict[str, Any]:
        """Generate system health check"""
        health = {'status': 'healthy', 'issues': []}
        
        # Check resource usage
        if self.resource_monitor:
            current_metrics = self.resource_monitor.get_current_metrics()
            if current_metrics:
                if current_metrics.cpu_percent > 90:
                    health['issues'].append('High CPU usage')
                if current_metrics.memory_percent > 90:
                    health['issues'].append('High memory usage')
        
        # Check processing performance
        processing_stats = self.processing_monitor.get_processing_stats()
        if processing_stats.get('recent_fps', 0) < 5:
            health['issues'].append('Low FPS performance')
        
        # Check quality
        quality_stats = self.quality_monitor.get_quality_stats()
        if quality_stats.get('overall_quality', {}).get('current', 1.0) < 0.5:
            health['issues'].append('Low processing quality')
        
        if health['issues']:
            health['status'] = 'warning' if len(health['issues']) < 3 else 'critical'
        
        return health