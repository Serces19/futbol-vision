"""
Main VideoProcessor class for orchestrating the football analytics pipeline
"""

import cv2
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
from pathlib import Path

from .interfaces import BaseProcessor
from .models import (
    Detection, TrackedObject, FieldLine, KeyPoint, 
    FrameResults, ProcessingResults
)
from .config import ConfigManager
from .factory import DefaultComponentFactory
from .exceptions import (
    VideoError, ProcessingError, ConfigurationError
)
from .video_io import VideoSource, VideoWriter, VideoStreamManager, FrameRateController

# Import component implementations
from ..detection import ObjectDetector, FieldDetector
from ..tracking import PlayerTracker
from ..classification import TeamClassifier
from ..calibration import HybridCalibrator
from ..visualization import VisualizationManager


class VideoProcessor(BaseProcessor):
    """
    Main orchestration class for the football analytics pipeline.
    Coordinates all components and manages the processing workflow.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize VideoProcessor with configuration
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self._is_processing = False
        self._should_stop = False
        self._callbacks: Dict[str, Callable] = {}
        
        # Component instances
        self._object_detector: Optional[ObjectDetector] = None
        self._field_detector: Optional[FieldDetector] = None
        self._player_tracker: Optional[PlayerTracker] = None
        self._team_classifier: Optional[TeamClassifier] = None
        self._field_calibrator: Optional[HybridCalibrator] = None
        self._visualization_manager: Optional[VisualizationManager] = None
        
        # Video I/O components
        self._video_source: Optional[VideoSource] = None
        self._video_writer: Optional[VideoWriter] = None
        self._stream_manager: VideoStreamManager = VideoStreamManager()
        self._frame_rate_controller: Optional[FrameRateController] = None
        
        # Processing statistics
        self._frame_count = 0
        self._processing_start_time = 0.0
        self._frame_times: List[float] = []
        
        # Results storage
        self._frame_results: List[FrameResults] = []
        
        # Performance optimization settings
        self._enable_multithreading = self.config_manager.processing_config.enable_multithreading
        self._memory_cleanup_interval = self.config_manager.processing_config.memory_cleanup_interval
        self._gc_collection_interval = self.config_manager.processing_config.gc_collection_interval
        self._last_cleanup_frame = 0
        self._last_gc_frame = 0
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all processing components"""
        try:
            self.logger.info("Initializing VideoProcessor components...")
            
            # Initialize object detector
            self._object_detector = ObjectDetector(
                model_path=self.config_manager.model_paths.yolo_player_model,
                device=self.config_manager.processing_config.device,
                confidence_threshold=self.config_manager.processing_config.confidence_threshold,
                nms_threshold=self.config_manager.processing_config.nms_threshold
            )
            
            # Initialize field detector
            self._field_detector = FieldDetector(
                lines_model_path=self.config_manager.model_paths.field_lines_model,
                keypoints_model_path=self.config_manager.model_paths.field_keypoints_model,
                device=self.config_manager.processing_config.device
            )
            
            # Initialize player tracker
            self._player_tracker = PlayerTracker(
                config=self.config_manager.tracker_config
            )
            
            # Initialize team classifier
            self._team_classifier = TeamClassifier(
                config=self.config_manager.processing_config
            )
            
            # Initialize field calibrator
            self._field_calibrator = HybridCalibrator(
                field_dimensions=self.config_manager.field_dimensions,
                min_quality_threshold=self.config_manager.processing_config.calibration_confidence_threshold
            )
            
            # Initialize visualization manager
            self._visualization_manager = VisualizationManager(
                config=self.config_manager.visualization_config,
                field_dimensions=self.config_manager.field_dimensions
            )
            
            # Performance optimization is now integrated directly
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise ConfigurationError(f"Component initialization failed: {e}")
    
    def process_video(self, video_source: str, output_path: Optional[str] = None) -> ProcessingResults:
        """
        Process entire video file or stream
        
        Args:
            video_source: Path to video file or stream URL
            output_path: Optional path for output video with overlays
            
        Returns:
            Complete processing results
        """
        self.logger.info(f"Starting video processing: {video_source}")
        
        # Initialize video source
        self._video_source = VideoSource(video_source)
        if not self._video_source.open():
            raise VideoError(f"Failed to open video source: {video_source}")
        
        try:
            # Get video properties
            fps = self._video_source.fps
            total_frames = self._video_source.total_frames
            frame_size = self._video_source.frame_size
            
            self.logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}, Size: {frame_size}")
            
            # Initialize video writer if output path provided
            if output_path:
                self._video_writer = VideoWriter(output_path, fps, frame_size)
                if not self._video_writer.open():
                    self.logger.warning(f"Failed to open video writer: {output_path}")
                    self._video_writer = None
            
            # Initialize frame rate controller for real-time processing
            if self.config_manager.processing_config.output_fps > 0:
                self._frame_rate_controller = FrameRateController(
                    self.config_manager.processing_config.output_fps
                )
            
            # Reset processing state
            self._reset_processing_state()
            self._processing_start_time = time.time()
            self._is_processing = True
            
            # Process frames
            frame_id = 0
            while not self._should_stop:
                ret, frame = self._video_source.read_frame()
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_id / fps if fps > 0 else frame_id * (1/30)  # Default 30 FPS
                
                # Process frame
                try:
                    frame_start_time = time.time()
                    frame_result = self.process_frame(frame, frame_id, timestamp)
                    frame_processing_time = time.time() - frame_start_time
                    
                    self._frame_results.append(frame_result)
                    self._frame_times.append(frame_processing_time)
                    
                    # Generate visualization if enabled
                    processed_frame = frame.copy()
                    if self._visualization_manager and self.config_manager.processing_config.enable_2d_visualization:
                        processed_frame = self._visualization_manager.draw_frame_overlays(
                            processed_frame, frame_result
                        )
                    
                    # Write to output video if writer available
                    if self._video_writer:
                        self._video_writer.write_frame(processed_frame)
                    
                    # Call progress callback
                    if 'progress' in self._callbacks:
                        progress = (frame_id + 1) / total_frames if total_frames > 0 else 0
                        self._callbacks['progress'](frame_id, progress, frame_result)
                    
                    # Call frame callback for real-time display
                    if 'frame_processed' in self._callbacks:
                        self._callbacks['frame_processed'](frame_result, processed_frame)
                    
                    # Frame rate control for real-time processing
                    if self._frame_rate_controller:
                        self._frame_rate_controller.wait_for_next_frame()
                    
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_id}: {e}")
                    if 'error' in self._callbacks:
                        self._callbacks['error'](frame_id, e)
                    continue
                
                frame_id += 1
                self._frame_count = frame_id
            
            # Calculate final statistics
            total_processing_time = time.time() - self._processing_start_time
            
            # Collect export paths
            export_paths = []
            if self._video_writer and output_path:
                export_paths.append(output_path)
            
            # Create processing results
            results = ProcessingResults(
                total_frames=frame_id,
                processing_time=total_processing_time,
                frame_results=self._frame_results,
                analytics_data=self._generate_analytics_summary(),
                export_paths=export_paths
            )
            
            self.logger.info(f"Video processing completed - {frame_id} frames in {total_processing_time:.2f}s")
            
            return results
            
        finally:
            self._cleanup_video_resources()
    
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> FrameResults:
        """
        Process single frame through the complete pipeline with optimizations
        
        Args:
            frame: Input frame as numpy array
            frame_id: Frame number
            timestamp: Frame timestamp in seconds
            
        Returns:
            Frame processing results
        """
        return self._process_frame_internal(frame, frame_id, timestamp)
    
    def _process_frame_internal(self, frame: np.ndarray, frame_id: int, timestamp: float) -> FrameResults:
        """
        Internal frame processing implementation
        
        Args:
            frame: Input frame as numpy array
            frame_id: Frame number
            timestamp: Frame timestamp in seconds
            
        Returns:
            Frame processing results
        """
        try:
            # Performance optimizations
            # Memory cleanup check
            if frame_id - self._last_cleanup_frame >= self._memory_cleanup_interval:
                self._cleanup_component_memory()
                self._last_cleanup_frame = frame_id
            
            # Garbage collection check
            if frame_id - self._last_gc_frame >= self._gc_collection_interval:
                import gc
                gc.collect()
                self._last_gc_frame = frame_id
            
            # Optimize frame for processing
            frame = self._optimize_frame_for_processing(frame)
            
            # Step 1: Object Detection
            detections = self._object_detector.detect(frame)
            
            # Step 2: Field Detection (if calibration enabled)
            field_lines = []
            key_points = []
            homography_matrix = None
            is_calibrated = False
            
            if self.config_manager.processing_config.enable_field_calibration:
                field_lines = self._field_detector.detect_lines(frame)
                key_points = self._field_detector.detect_key_points(frame)
                
                # Step 3: Field Calibration
                homography_matrix = self._field_calibrator.calibrate(field_lines, key_points)
                is_calibrated = self._field_calibrator.is_calibrated()
            
            # Step 4: Player Tracking with memory optimization
            tracked_objects = self._player_tracker.update(detections, frame.shape[:2])
            
            # Optimize tracking memory usage
            self._optimize_tracking_memory()
            
            # Step 5: Team Classification with caching
            if len(tracked_objects) >= 2:  # Need at least 2 players for team classification
                player_crops = self._extract_player_crops(frame, tracked_objects)
                if player_crops:
                    # Optimize classification caching
                    self._optimize_classification_caching()
                    
                    team_assignments = self._team_classifier.classify_teams(player_crops)
                    
                    # Update tracked objects with team assignments
                    for i, tracked_obj in enumerate(tracked_objects):
                        if i < len(team_assignments):
                            tracked_obj.team_id = team_assignments[i]
            
            # Step 6: Field Coordinate Transformation
            if is_calibrated:
                for tracked_obj in tracked_objects:
                    # Get center point of bounding box
                    bbox = tracked_obj.detection.bbox
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = bbox[3]  # Use bottom of bbox as player position
                    
                    # Transform to field coordinates
                    field_coords = self._field_calibrator.transform_to_field_coordinates([(center_x, center_y)])
                    if field_coords:
                        tracked_obj.field_position = field_coords[0]
            
            # Step 7: Ball Detection (separate from player detection)
            ball_position = self._detect_ball(detections)
            
            # Create frame results
            frame_result = FrameResults(
                frame_id=frame_id,
                timestamp=timestamp,
                tracked_objects=tracked_objects,
                field_lines=field_lines,
                key_points=key_points,
                ball_position=ball_position,
                is_calibrated=is_calibrated,
                homography_matrix=homography_matrix
            )
            
            # Call frame result callback
            if 'frame_result' in self._callbacks:
                self._callbacks['frame_result'](frame_result)
            
            return frame_result
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_id}: {e}")
            raise ProcessingError(f"Frame processing failed: {e}")
    
    def _extract_player_crops(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> List[np.ndarray]:
        """
        Extract cropped player images for team classification
        
        Args:
            frame: Input frame
            tracked_objects: List of tracked objects
            
        Returns:
            List of cropped player images
        """
        crops = []
        for tracked_obj in tracked_objects:
            if tracked_obj.detection.class_name == 'player':
                bbox = tracked_obj.detection.bbox
                x1, y1, x2, y2 = bbox
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops.append(crop)
        
        return crops
    
    def _detect_ball(self, detections: List[Detection]) -> Optional[Tuple[int, int]]:
        """
        Extract ball position from detections
        
        Args:
            detections: List of all detections
            
        Returns:
            Ball center position as (x, y) tuple, or None if not found
        """
        for detection in detections:
            if detection.class_name == 'ball':
                bbox = detection.bbox
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                return (center_x, center_y)
        
        return None
    
    def process_stream(self, video_source: str, output_path: Optional[str] = None) -> None:
        """
        Process video stream in real-time mode
        
        Args:
            video_source: Stream URL or camera index
            output_path: Optional path for output video with overlays
        """
        self.logger.info(f"Starting stream processing: {video_source}")
        
        # Initialize video source for streaming
        self._video_source = VideoSource(video_source, buffer_size=5)
        if not self._video_source.open():
            raise VideoError(f"Failed to open video stream: {video_source}")
        
        try:
            # Start streaming mode
            if not self._video_source.start_streaming():
                raise VideoError("Failed to start streaming mode")
            
            # Get video properties
            fps = self._video_source.fps
            frame_size = self._video_source.frame_size
            
            self.logger.info(f"Stream properties - FPS: {fps}, Size: {frame_size}")
            
            # Initialize video writer if output path provided
            if output_path:
                self._video_writer = VideoWriter(output_path, fps, frame_size)
                if not self._video_writer.open():
                    self.logger.warning(f"Failed to open video writer: {output_path}")
                    self._video_writer = None
            
            # Reset processing state
            self._reset_processing_state()
            self._processing_start_time = time.time()
            self._is_processing = True
            
            # Process streaming frames
            while not self._should_stop:
                # Get frame from buffer
                ret, frame, frame_id, timestamp = self._video_source.get_buffered_frame(timeout=0.5)
                if not ret:
                    continue
                
                # Process frame
                try:
                    frame_start_time = time.time()
                    frame_result = self.process_frame(frame, frame_id, timestamp)
                    frame_processing_time = time.time() - frame_start_time
                    
                    self._frame_results.append(frame_result)
                    self._frame_times.append(frame_processing_time)
                    
                    # Generate visualization if enabled
                    processed_frame = frame.copy()
                    if self._visualization_manager and self.config_manager.processing_config.enable_2d_visualization:
                        processed_frame = self._visualization_manager.draw_frame_overlays(
                            processed_frame, frame_result
                        )
                    
                    # Write to output video if writer available
                    if self._video_writer:
                        self._video_writer.write_frame(processed_frame)
                    
                    # Call streaming callback for real-time display
                    if 'stream_frame' in self._callbacks:
                        self._callbacks['stream_frame'](frame_result, processed_frame)
                    
                    self._frame_count = frame_id + 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing stream frame {frame_id}: {e}")
                    if 'error' in self._callbacks:
                        self._callbacks['error'](frame_id, e)
                    continue
            
            self.logger.info(f"Stream processing stopped - {self._frame_count} frames processed")
            
        finally:
            self._cleanup_video_resources()
    
    def _cleanup_video_resources(self) -> None:
        """Clean up video I/O resources"""
        if self._video_source:
            self._video_source.close()
            self._video_source = None
        
        if self._video_writer:
            self._video_writer.close()
            self._video_writer = None
        
        self._stream_manager.close_all()
        self._is_processing = False
    
    def _cleanup_component_memory(self) -> None:
        """Clean up component memory usage"""
        try:
            # Clear frame results history if too large
            if len(self._frame_results) > 1000:
                self._frame_results = self._frame_results[-500:]  # Keep last 500 frames
            
            # Clear frame timing history
            if len(self._frame_times) > 1000:
                self._frame_times = self._frame_times[-500:]
            
            # Reset tracker trajectories if too long
            if self._player_tracker and hasattr(self._player_tracker, 'trajectories'):
                max_length = 1000
                for track_id, trajectory in self._player_tracker.trajectories.items():
                    if len(trajectory) > max_length:
                        self._player_tracker.trajectories[track_id] = trajectory[-max_length:]
            
            # Clear team classifier cache if too large
            if self._team_classifier and hasattr(self._team_classifier, '_embedding_cache'):
                if len(self._team_classifier._embedding_cache) > 500:
                    # Keep only recent embeddings
                    cache_items = list(self._team_classifier._embedding_cache.items())
                    self._team_classifier._embedding_cache = dict(cache_items[-250:])
            
            self.logger.debug("Component memory cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during component memory cleanup: {e}")
    
    def _optimize_frame_for_processing(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame for processing"""
        # Ensure frame is contiguous for better cache performance
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        
        # Convert to optimal dtype if needed
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        return frame
    
    def _optimize_tracking_memory(self):
        """Optimize tracker memory usage by limiting trajectory length"""
        if self._player_tracker and hasattr(self._player_tracker, 'trajectories'):
            max_length = self.config_manager.processing_config.max_trajectory_length
            for track_id, trajectory in self._player_tracker.trajectories.items():
                if len(trajectory) > max_length:
                    # Keep only recent positions
                    self._player_tracker.trajectories[track_id] = trajectory[-max_length:]
    
    def _optimize_classification_caching(self):
        """Optimize classification by caching embeddings"""
        if self._team_classifier:
            cache_size = self.config_manager.processing_config.embedding_cache_size
            
            if not hasattr(self._team_classifier, '_embedding_cache'):
                self._team_classifier._embedding_cache = {}
                from collections import deque
                self._team_classifier._cache_order = deque(maxlen=cache_size)
            
            # Clean old cache entries
            while len(self._team_classifier._embedding_cache) > cache_size:
                if hasattr(self._team_classifier, '_cache_order') and self._team_classifier._cache_order:
                    old_key = self._team_classifier._cache_order.popleft()
                    self._team_classifier._embedding_cache.pop(old_key, None)
    
    def _reset_processing_state(self) -> None:
        """Reset processing state for new video"""
        self._frame_count = 0
        self._frame_results.clear()
        self._frame_times.clear()
        self._should_stop = False
        
        # Reset component states
        if self._player_tracker:
            self._player_tracker.reset()
        if self._field_calibrator:
            self._field_calibrator.reset_calibration()
        
        # Reset performance optimization counters
        self._last_cleanup_frame = 0
        self._last_gc_frame = 0
    
    def _generate_analytics_summary(self) -> Dict[str, Any]:
        """
        Generate analytics summary from processed frames
        
        Returns:
            Dictionary containing analytics data
        """
        if not self._frame_results:
            return {}
        
        # Calculate processing statistics
        avg_frame_time = np.mean(self._frame_times) if self._frame_times else 0
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Count detections and tracks
        total_detections = sum(len(fr.tracked_objects) for fr in self._frame_results)
        unique_tracks = set()
        for frame_result in self._frame_results:
            for obj in frame_result.tracked_objects:
                unique_tracks.add(obj.track_id)
        
        # Calculate calibration success rate
        calibrated_frames = sum(1 for fr in self._frame_results if fr.is_calibrated)
        calibration_rate = calibrated_frames / len(self._frame_results) if self._frame_results else 0
        
        return {
            'processing_stats': {
                'total_frames': len(self._frame_results),
                'average_fps': fps,
                'average_frame_time': avg_frame_time,
                'total_processing_time': sum(self._frame_times)
            },
            'detection_stats': {
                'total_detections': total_detections,
                'unique_tracks': len(unique_tracks),
                'average_detections_per_frame': total_detections / len(self._frame_results) if self._frame_results else 0
            },
            'calibration_stats': {
                'calibrated_frames': calibrated_frames,
                'calibration_success_rate': calibration_rate
            }
        }
    
    def set_callbacks(self, callbacks: Dict[str, Callable]) -> None:
        """
        Set callback functions for real-time updates and monitoring
        
        Args:
            callbacks: Dictionary of callback functions
                - 'progress': Called with (frame_id, progress, frame_result)
                - 'frame_processed': Called with (frame_result, frame)
                - 'frame_result': Called with (frame_result)
                - 'error': Called with (frame_id, exception)
        """
        self._callbacks = callbacks.copy()
        self.logger.info(f"Set {len(callbacks)} callbacks: {list(callbacks.keys())}")
    
    def stop_processing(self) -> None:
        """Stop video processing gracefully"""
        self._should_stop = True
        self.logger.info("Processing stop requested")
    
    def is_processing(self) -> bool:
        """Check if currently processing video"""
        return self._is_processing
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics
        
        Returns:
            Dictionary with current stats
        """
        if not self._frame_times:
            return {'frames_processed': 0, 'fps': 0, 'avg_frame_time': 0}
        
        recent_times = self._frame_times[-30:]  # Last 30 frames
        avg_time = np.mean(recent_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'frames_processed': self._frame_count,
            'current_fps': current_fps,
            'average_frame_time': avg_time,
            'total_processing_time': time.time() - self._processing_start_time if self._is_processing else 0
        }
    
    def get_component_status(self) -> Dict[str, bool]:
        """
        Get status of all components
        
        Returns:
            Dictionary with component initialization status
        """
        return {
            'object_detector': self._object_detector is not None,
            'field_detector': self._field_detector is not None,
            'player_tracker': self._player_tracker is not None,
            'team_classifier': self._team_classifier is not None,
            'field_calibrator': self._field_calibrator is not None,
            'visualization_manager': self._visualization_manager is not None,
            'video_source': self._video_source is not None,
            'video_writer': self._video_writer is not None
        }
    
    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to specific frame (file sources only)
        
        Args:
            frame_number: Target frame number
            
        Returns:
            True if seek successful
        """
        if not self._video_source:
            return False
        
        return self._video_source.seek_frame(frame_number)
    
    def get_video_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about current video source
        
        Returns:
            Dictionary with video information or None if no source
        """
        if not self._video_source:
            return None
        
        return {
            'fps': self._video_source.fps,
            'total_frames': self._video_source.total_frames,
            'frame_size': self._video_source.frame_size,
            'current_frame': self._video_source.current_frame_id,
            'is_streaming': self._video_source.is_streaming
        }
    
    def set_output_video(self, output_path: str, fps: Optional[float] = None, 
                        frame_size: Optional[Tuple[int, int]] = None, 
                        codec: str = 'mp4v') -> bool:
        """
        Set up video output writer
        
        Args:
            output_path: Output video file path
            fps: Output FPS (uses source FPS if None)
            frame_size: Output frame size (uses source size if None)
            codec: Video codec
            
        Returns:
            True if writer setup successful
        """
        # Get properties from source if not provided
        if self._video_source:
            fps = fps or self._video_source.fps
            frame_size = frame_size or self._video_source.frame_size
        else:
            fps = fps or 30.0
            frame_size = frame_size or (1920, 1080)
        
        # Close existing writer
        if self._video_writer:
            self._video_writer.close()
        
        # Create new writer
        self._video_writer = VideoWriter(output_path, fps, frame_size, codec)
        return self._video_writer.open()
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get supported video formats
        
        Returns:
            Dictionary with input and output format lists
        """
        # Common input formats supported by OpenCV
        input_formats = [
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
            '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'
        ]
        
        # Common output formats
        output_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Stream protocols
        stream_protocols = ['http://', 'https://', 'rtmp://', 'rtsp://']
        
        return {
            'input_formats': input_formats,
            'output_formats': output_formats,
            'stream_protocols': stream_protocols,
            'codecs': ['mp4v', 'XVID', 'MJPG', 'X264']
        }
    
    def validate_video_source(self, video_source: str) -> Dict[str, Any]:
        """
        Validate video source and get detailed information
        
        Args:
            video_source: Video source to validate
            
        Returns:
            Dictionary with validation results and source info
        """
        result = {
            'is_valid': False,
            'source_type': 'unknown',
            'error': None,
            'properties': {}
        }
        
        try:
            # Check if it's a file path
            if Path(video_source).exists():
                result['source_type'] = 'file'
                result['is_valid'] = True
                
                # Try to get file properties
                temp_cap = cv2.VideoCapture(video_source)
                if temp_cap.isOpened():
                    result['properties'] = {
                        'fps': temp_cap.get(cv2.CAP_PROP_FPS),
                        'total_frames': int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        'width': int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    }
                    temp_cap.release()
                
            # Check if it's a URL or stream
            elif video_source.startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
                result['source_type'] = 'stream'
                result['is_valid'] = True
                
            # Check if it's a camera index
            else:
                try:
                    camera_index = int(video_source)
                    if camera_index >= 0:
                        result['source_type'] = 'camera'
                        result['is_valid'] = True
                        
                        # Try to access camera
                        temp_cap = cv2.VideoCapture(camera_index)
                        if temp_cap.isOpened():
                            result['properties'] = {
                                'fps': temp_cap.get(cv2.CAP_PROP_FPS),
                                'width': int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                'height': int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            }
                            temp_cap.release()
                        else:
                            result['error'] = f"Cannot access camera {camera_index}"
                            result['is_valid'] = False
                except ValueError:
                    result['error'] = "Invalid camera index"
            
            if not result['is_valid'] and not result['error']:
                result['error'] = "Unsupported video source format"
                
        except Exception as e:
            result['error'] = str(e)
        
        return result