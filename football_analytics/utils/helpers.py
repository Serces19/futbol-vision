"""
Utility functions and helper classes
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
import time
import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("football_analytics")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_video_source(video_source: str) -> bool:
    """
    Validate video source (file or stream)
    
    Args:
        video_source: Path to video file or stream URL
        
    Returns:
        True if valid, False otherwise
    """
    if video_source.startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
        # Stream URL - assume valid for now
        return True
    
    # File path
    return Path(video_source).exists()


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image using bounding box
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Clamp coordinates to image bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return image[y1:y2, x1:x2]


def resize_image(image: np.ndarray, target_size: Tuple[int, int], keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if not keep_aspect_ratio:
        return cv2.resize(image, target_size)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return padded


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance between points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_velocity(positions: List[Tuple[float, float, float]]) -> float:
    """
    Calculate velocity from position history
    
    Args:
        positions: List of (x, y, timestamp) tuples
        
    Returns:
        Average velocity in units per second
    """
    if len(positions) < 2:
        return 0.0
    
    total_distance = 0.0
    total_time = 0.0
    
    for i in range(1, len(positions)):
        prev_x, prev_y, prev_t = positions[i-1]
        curr_x, curr_y, curr_t = positions[i]
        
        distance = calculate_distance((prev_x, prev_y), (curr_x, curr_y))
        time_diff = curr_t - prev_t
        
        if time_diff > 0:
            total_distance += distance
            total_time += time_diff
    
    return total_distance / total_time if total_time > 0 else 0.0


def smooth_trajectory(positions: List[Tuple[float, float]], window_size: int = 5) -> List[Tuple[float, float]]:
    """
    Smooth trajectory using moving average
    
    Args:
        positions: List of (x, y) positions
        window_size: Size of smoothing window
        
    Returns:
        Smoothed positions
    """
    if len(positions) < window_size:
        return positions
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(positions)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(positions), i + half_window + 1)
        
        window_positions = positions[start_idx:end_idx]
        avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
        avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
        
        smoothed.append((avg_x, avg_y))
    
    return smoothed


class Timer:
    """Simple timer for performance measurement"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer"""
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FPSCounter:
    """FPS counter for performance monitoring"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """Update FPS counter and return current FPS"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def reset(self):
        """Reset FPS counter"""
        self.frame_times.clear()
        self.last_time = time.time()


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_video_info(video_path: str) -> dict:
    """
    Get video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {}
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info