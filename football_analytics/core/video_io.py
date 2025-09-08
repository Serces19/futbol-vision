"""
Video I/O and streaming support for the football analytics system
"""

import cv2
import time
import threading
from typing import Optional, Tuple, Iterator, Dict, Any, List
import numpy as np
from pathlib import Path
from queue import Queue, Empty
import logging

from .exceptions import VideoError
from .models import FrameResults


class VideoSource:
    """
    Handles video input from various sources (files, streams, cameras)
    """
    
    def __init__(self, source: str, buffer_size: int = 10):
        """
        Initialize video source
        
        Args:
            source: Video source (file path, URL, or camera index)
            buffer_size: Size of frame buffer for streaming
        """
        self.source = source
        self.buffer_size = buffer_size
        self.logger = logging.getLogger(__name__)
        
        # Video capture object
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Video properties
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._frame_width: int = 0
        self._frame_height: int = 0
        
        # Streaming support
        self._frame_buffer: Queue = Queue(maxsize=buffer_size)
        self._streaming_thread: Optional[threading.Thread] = None
        self._is_streaming = False
        self._should_stop_streaming = False
        
        # Frame control
        self._current_frame_id = 0
        self._start_time = 0.0
        
    def open(self) -> bool:
        """
        Open video source
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine source type and open accordingly
            if self._is_camera_source():
                camera_index = int(self.source)
                self._cap = cv2.VideoCapture(camera_index)
            elif self._is_stream_source():
                self._cap = cv2.VideoCapture(self.source)
                # Set buffer size for streaming
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                # File source
                if not Path(self.source).exists():
                    raise VideoError(f"Video file not found: {self.source}")
                self._cap = cv2.VideoCapture(self.source)
            
            if not self._cap.isOpened():
                raise VideoError(f"Failed to open video source: {self.source}")
            
            # Get video properties
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Opened video source: {self.source}")
            self.logger.info(f"Properties - FPS: {self._fps}, Frames: {self._total_frames}, "
                           f"Resolution: {self._frame_width}x{self._frame_height}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open video source {self.source}: {e}")
            return False
    
    def close(self) -> None:
        """Close video source and cleanup resources"""
        self.stop_streaming()
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        self.logger.info(f"Closed video source: {self.source}")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video source
        
        Returns:
            Tuple of (success, frame) where frame is None if read failed
        """
        if not self._cap:
            return False, None
        
        ret, frame = self._cap.read()
        if ret:
            self._current_frame_id += 1
        
        return ret, frame
    
    def start_streaming(self) -> bool:
        """
        Start streaming mode with buffered frame reading
        
        Returns:
            True if streaming started successfully
        """
        if self._is_streaming:
            return True
        
        if not self._cap:
            return False
        
        self._should_stop_streaming = False
        self._streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self._streaming_thread.start()
        self._is_streaming = True
        self._start_time = time.time()
        
        self.logger.info("Started streaming mode")
        return True
    
    def stop_streaming(self) -> None:
        """Stop streaming mode"""
        if not self._is_streaming:
            return
        
        self._should_stop_streaming = True
        if self._streaming_thread:
            self._streaming_thread.join(timeout=1.0)
        
        # Clear buffer
        while not self._frame_buffer.empty():
            try:
                self._frame_buffer.get_nowait()
            except Empty:
                break
        
        self._is_streaming = False
        self.logger.info("Stopped streaming mode")
    
    def get_buffered_frame(self, timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """
        Get frame from buffer (streaming mode)
        
        Args:
            timeout: Timeout for getting frame from buffer
            
        Returns:
            Tuple of (success, frame, frame_id, timestamp)
        """
        try:
            frame_data = self._frame_buffer.get(timeout=timeout)
            return True, frame_data['frame'], frame_data['frame_id'], frame_data['timestamp']
        except Empty:
            return False, None, 0, 0.0
    
    def _streaming_worker(self) -> None:
        """Worker thread for streaming mode"""
        while not self._should_stop_streaming and self._cap:
            ret, frame = self._cap.read()
            if not ret:
                break
            
            current_time = time.time()
            timestamp = current_time - self._start_time
            
            frame_data = {
                'frame': frame,
                'frame_id': self._current_frame_id,
                'timestamp': timestamp
            }
            
            # Add to buffer, drop oldest if full
            try:
                self._frame_buffer.put_nowait(frame_data)
                self._current_frame_id += 1
            except:
                # Buffer full, drop oldest frame
                try:
                    self._frame_buffer.get_nowait()
                    self._frame_buffer.put_nowait(frame_data)
                    self._current_frame_id += 1
                except Empty:
                    pass
    
    def seek_frame(self, frame_number: int) -> bool:
        """
        Seek to specific frame (file sources only)
        
        Args:
            frame_number: Target frame number
            
        Returns:
            True if seek successful
        """
        if not self._cap or self._is_stream_source() or self._is_camera_source():
            return False
        
        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if success:
            self._current_frame_id = frame_number
        
        return success
    
    def _is_camera_source(self) -> bool:
        """Check if source is a camera index"""
        try:
            int(self.source)
            return True
        except ValueError:
            return False
    
    def _is_stream_source(self) -> bool:
        """Check if source is a stream URL"""
        return self.source.startswith(('http://', 'https://', 'rtmp://', 'rtsp://'))
    
    @property
    def fps(self) -> float:
        """Get video FPS"""
        return self._fps
    
    @property
    def total_frames(self) -> int:
        """Get total number of frames (0 for streams/cameras)"""
        return self._total_frames
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame size as (width, height)"""
        return (self._frame_width, self._frame_height)
    
    @property
    def current_frame_id(self) -> int:
        """Get current frame ID"""
        return self._current_frame_id
    
    @property
    def is_streaming(self) -> bool:
        """Check if in streaming mode"""
        return self._is_streaming


class VideoWriter:
    """
    Handles video output with processed frames
    """
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], 
                 codec: str = 'mp4v'):
        """
        Initialize video writer
        
        Args:
            output_path: Output video file path
            fps: Output video FPS
            frame_size: Frame size as (width, height)
            codec: Video codec (default: mp4v)
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.logger = logging.getLogger(__name__)
        
        # Video writer object
        self._writer: Optional[cv2.VideoWriter] = None
        self._is_opened = False
        self._frames_written = 0
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def open(self) -> bool:
        """
        Open video writer
        
        Returns:
            True if successful
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, self.frame_size
            )
            
            if not self._writer.isOpened():
                raise VideoError(f"Failed to open video writer: {self.output_path}")
            
            self._is_opened = True
            self.logger.info(f"Opened video writer: {self.output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open video writer: {e}")
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write frame to video
        
        Args:
            frame: Frame to write
            
        Returns:
            True if successful
        """
        if not self._is_opened or not self._writer:
            return False
        
        try:
            # Ensure frame is correct size
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            
            self._writer.write(frame)
            self._frames_written += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write frame: {e}")
            return False
    
    def close(self) -> None:
        """Close video writer"""
        if self._writer:
            self._writer.release()
            self._writer = None
        
        self._is_opened = False
        self.logger.info(f"Closed video writer: {self.output_path} ({self._frames_written} frames)")
    
    @property
    def is_opened(self) -> bool:
        """Check if writer is opened"""
        return self._is_opened
    
    @property
    def frames_written(self) -> int:
        """Get number of frames written"""
        return self._frames_written


class FrameRateController:
    """
    Controls frame rate for real-time processing
    """
    
    def __init__(self, target_fps: float):
        """
        Initialize frame rate controller
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self.last_frame_time = 0.0
        
    def wait_for_next_frame(self) -> None:
        """Wait to maintain target frame rate"""
        if self.frame_interval <= 0:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        
        self.last_frame_time = time.time()
    
    def reset(self) -> None:
        """Reset frame rate controller"""
        self.last_frame_time = time.time()


class VideoStreamManager:
    """
    Manages multiple video streams and synchronization
    """
    
    def __init__(self):
        """Initialize video stream manager"""
        self.logger = logging.getLogger(__name__)
        self._sources: Dict[str, VideoSource] = {}
        self._writers: Dict[str, VideoWriter] = {}
        self._frame_rate_controller: Optional[FrameRateController] = None
    
    def add_source(self, name: str, source: str, buffer_size: int = 10) -> bool:
        """
        Add video source
        
        Args:
            name: Source identifier
            source: Video source path/URL
            buffer_size: Buffer size for streaming
            
        Returns:
            True if added successfully
        """
        try:
            video_source = VideoSource(source, buffer_size)
            if video_source.open():
                self._sources[name] = video_source
                self.logger.info(f"Added video source '{name}': {source}")
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Failed to add video source '{name}': {e}")
            return False
    
    def add_writer(self, name: str, output_path: str, fps: float, 
                   frame_size: Tuple[int, int], codec: str = 'mp4v') -> bool:
        """
        Add video writer
        
        Args:
            name: Writer identifier
            output_path: Output file path
            fps: Output FPS
            frame_size: Frame size
            codec: Video codec
            
        Returns:
            True if added successfully
        """
        try:
            writer = VideoWriter(output_path, fps, frame_size, codec)
            if writer.open():
                self._writers[name] = writer
                self.logger.info(f"Added video writer '{name}': {output_path}")
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Failed to add video writer '{name}': {e}")
            return False
    
    def set_frame_rate_control(self, target_fps: float) -> None:
        """
        Set frame rate control for synchronized playback
        
        Args:
            target_fps: Target FPS for playback
        """
        self._frame_rate_controller = FrameRateController(target_fps)
    
    def read_frame(self, source_name: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from specific source
        
        Args:
            source_name: Name of source to read from
            
        Returns:
            Tuple of (success, frame)
        """
        if source_name not in self._sources:
            return False, None
        
        return self._sources[source_name].read_frame()
    
    def write_frame(self, writer_name: str, frame: np.ndarray) -> bool:
        """
        Write frame to specific writer
        
        Args:
            writer_name: Name of writer
            frame: Frame to write
            
        Returns:
            True if successful
        """
        if writer_name not in self._writers:
            return False
        
        return self._writers[writer_name].write_frame(frame)
    
    def start_streaming(self, source_name: str) -> bool:
        """
        Start streaming mode for specific source
        
        Args:
            source_name: Name of source
            
        Returns:
            True if successful
        """
        if source_name not in self._sources:
            return False
        
        return self._sources[source_name].start_streaming()
    
    def get_buffered_frame(self, source_name: str, timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """
        Get buffered frame from streaming source
        
        Args:
            source_name: Name of source
            timeout: Timeout for frame retrieval
            
        Returns:
            Tuple of (success, frame, frame_id, timestamp)
        """
        if source_name not in self._sources:
            return False, None, 0, 0.0
        
        return self._sources[source_name].get_buffered_frame(timeout)
    
    def wait_for_next_frame(self) -> None:
        """Wait for next frame based on frame rate control"""
        if self._frame_rate_controller:
            self._frame_rate_controller.wait_for_next_frame()
    
    def close_all(self) -> None:
        """Close all sources and writers"""
        for source in self._sources.values():
            source.close()
        
        for writer in self._writers.values():
            writer.close()
        
        self._sources.clear()
        self._writers.clear()
        
        self.logger.info("Closed all video sources and writers")
    
    def get_source_info(self, source_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a video source
        
        Args:
            source_name: Name of source
            
        Returns:
            Dictionary with source information or None if not found
        """
        if source_name not in self._sources:
            return None
        
        source = self._sources[source_name]
        return {
            'fps': source.fps,
            'total_frames': source.total_frames,
            'frame_size': source.frame_size,
            'current_frame': source.current_frame_id,
            'is_streaming': source.is_streaming
        }