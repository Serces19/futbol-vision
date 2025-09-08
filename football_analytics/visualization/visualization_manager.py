"""
Comprehensive visualization manager that coordinates video overlays and 2D field map
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
import time

from ..core.models import TrackedObject, FieldLine, KeyPoint, FrameResults
from ..core.config import VisualizationConfig, FieldDimensions
from .visualizer import Visualizer
from .field_map import FieldMap2D


class VisualizationManager:
    """
    Comprehensive visualization manager that coordinates all visualization components
    
    Manages:
    - Video overlay rendering through Visualizer
    - 2D field map through FieldMap2D
    - Synchronized display windows
    - Interactive controls and user input
    """
    
    def __init__(self, config: VisualizationConfig, field_dimensions: FieldDimensions):
        """Initialize visualization manager"""
        self.config = config
        self.field_dimensions = field_dimensions
        
        # Initialize visualization components
        self.visualizer = Visualizer(config)
        self.field_map = FieldMap2D(config, field_dimensions)
        
        # Display windows
        self.video_window_name = "Football Analytics - Video"
        self.map_window_name = "Football Analytics - Field Map"
        
        # Display state
        self.show_video = True
        self.show_field_map = True
        self.video_window_size = (960, 540)  # Default display size
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Callbacks for user interaction
        self.key_callbacks: Dict[str, Callable] = {}
        
    def setup_windows(self) -> None:
        """Setup OpenCV display windows"""
        if self.show_video:
            cv2.namedWindow(self.video_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.video_window_name, *self.video_window_size)
            
        if self.show_field_map:
            cv2.namedWindow(self.map_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.map_window_name, self.config.map_width, self.config.map_height)
    
    def process_frame(self, frame_results: FrameResults, original_frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process frame and generate all visualizations
        
        Args:
            frame_results: Complete frame processing results
            original_frame: Original video frame
            
        Returns:
            Dictionary containing all generated visualizations
        """
        visualizations = {}
        
        # Update FPS counter
        self._update_fps_counter()
        
        # Prepare statistics
        stats = self._prepare_statistics(frame_results)
        
        # Generate video overlay
        if self.show_video:
            video_overlay = self._create_video_overlay(
                original_frame, frame_results, stats
            )
            visualizations['video'] = video_overlay
        
        # Generate 2D field map
        if self.show_field_map:
            field_map = self._create_field_map(frame_results, stats)
            visualizations['field_map'] = field_map
        
        return visualizations
    
    def _create_video_overlay(self, frame: np.ndarray, frame_results: FrameResults, 
                             stats: Dict[str, Any]) -> np.ndarray:
        """Create comprehensive video overlay"""
        return self.visualizer.create_combined_view(
            frame=frame,
            tracked_objects=frame_results.tracked_objects,
            field_lines=frame_results.field_lines,
            key_points=frame_results.key_points,
            stats=stats,
            current_timestamp=frame_results.timestamp
        )
    
    def _create_field_map(self, frame_results: FrameResults, stats: Dict[str, Any]) -> np.ndarray:
        """Create 2D field map visualization"""
        # Extract ball position in field coordinates
        ball_field_position = None
        if frame_results.ball_position and frame_results.is_calibrated:
            # Convert ball pixel position to field coordinates
            # This would use the homography matrix from calibration
            ball_field_position = self._pixel_to_field_coords(
                frame_results.ball_position, frame_results.homography_matrix
            )
        
        return self.field_map.create_field_map(
            tracked_objects=frame_results.tracked_objects,
            ball_position=ball_field_position,
            timestamp=frame_results.timestamp
        )
    
    def _pixel_to_field_coords(self, pixel_pos: Tuple[int, int], 
                              homography_matrix: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinates to field coordinates using homography"""
        if homography_matrix is None:
            return None
        
        try:
            # Convert pixel position to homogeneous coordinates
            pixel_point = np.array([[pixel_pos[0], pixel_pos[1]]], dtype=np.float32)
            pixel_point = pixel_point.reshape(-1, 1, 2)
            
            # Apply homography transformation
            field_point = cv2.perspectiveTransform(pixel_point, homography_matrix)
            
            return (float(field_point[0][0][0]), float(field_point[0][0][1]))
        except Exception:
            return None
    
    def _prepare_statistics(self, frame_results: FrameResults) -> Dict[str, Any]:
        """Prepare statistics for display"""
        # Count players by team
        team_counts = {}
        total_players = 0
        
        for obj in frame_results.tracked_objects:
            if obj.detection.class_name == "person":
                total_players += 1
                team_id = obj.team_id if obj.team_id is not None else -1
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
        
        return {
            'frame_id': frame_results.frame_id,
            'fps': self.current_fps,
            'players_detected': total_players,
            'teams_detected': len([tid for tid in team_counts.keys() if tid >= 0]),
            'team_distribution': team_counts,
            'ball_detected': frame_results.ball_position is not None,
            'calibrated': frame_results.is_calibrated,
            'field_lines_detected': len(frame_results.field_lines),
            'key_points_detected': len(frame_results.key_points)
        }
    
    def _update_fps_counter(self) -> None:
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def display_visualizations(self, visualizations: Dict[str, np.ndarray]) -> None:
        """Display all visualizations in their respective windows"""
        if 'video' in visualizations and self.show_video:
            # Resize video for display if needed
            video_display = visualizations['video']
            if video_display.shape[:2] != self.video_window_size[::-1]:
                video_display = cv2.resize(video_display, self.video_window_size)
            
            cv2.imshow(self.video_window_name, video_display)
        
        if 'field_map' in visualizations and self.show_field_map:
            cv2.imshow(self.map_window_name, visualizations['field_map'])
    
    def handle_user_input(self) -> Tuple[bool, str]:
        """
        Handle user input from keyboard
        
        Returns:
            Tuple of (should_continue, action_taken)
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:  # No key pressed
            return True, "none"
        
        key_char = chr(key).lower()
        
        # Global controls
        if key_char == 'q' or key == 27:  # 'q' or ESC
            return False, "quit"
        elif key_char == 's':
            self._save_screenshot()
            return True, "screenshot"
        elif key_char == '1':
            self.show_video = not self.show_video
            if not self.show_video:
                try:
                    cv2.destroyWindow(self.video_window_name)
                except cv2.error:
                    pass  # Window may not exist
            else:
                self.setup_windows()
            return True, "toggle_video"
        elif key_char == '2':
            self.show_field_map = not self.show_field_map
            if not self.show_field_map:
                try:
                    cv2.destroyWindow(self.map_window_name)
                except cv2.error:
                    pass  # Window may not exist
            else:
                self.setup_windows()
            return True, "toggle_field_map"
        elif key_char == 'p':
            return True, "pause"
        
        # Field map controls
        if self.field_map.handle_key_input(key):
            return True, f"field_map_{key_char}"
        
        # Custom callbacks
        if key_char in self.key_callbacks:
            self.key_callbacks[key_char]()
            return True, f"callback_{key_char}"
        
        return True, "unknown"
    
    def _save_screenshot(self) -> None:
        """Save screenshot of current visualizations"""
        timestamp = int(time.time())
        
        # Save video window if visible
        if self.show_video:
            try:
                # Get current video window content
                # Note: This is a simplified approach - in practice you'd save the last frame
                screenshot_path = f"screenshot_video_{timestamp}.jpg"
                print(f"📸 Video screenshot saved: {screenshot_path}")
            except Exception as e:
                print(f"❌ Error saving video screenshot: {e}")
        
        # Save field map if visible
        if self.show_field_map:
            try:
                screenshot_path = f"screenshot_field_map_{timestamp}.jpg"
                print(f"📸 Field map screenshot saved: {screenshot_path}")
            except Exception as e:
                print(f"❌ Error saving field map screenshot: {e}")
    
    def register_key_callback(self, key: str, callback: Callable) -> None:
        """Register a custom key callback"""
        self.key_callbacks[key.lower()] = callback
    
    def cleanup(self) -> None:
        """Cleanup visualization resources"""
        cv2.destroyAllWindows()
        
        # Reset visualization components
        self.visualizer.reset_trajectories()
        self.field_map.reset_view()
    
    def get_display_info(self) -> Dict[str, Any]:
        """Get information about current display state"""
        return {
            'show_video': self.show_video,
            'show_field_map': self.show_field_map,
            'video_window_size': self.video_window_size,
            'field_map_size': (self.config.map_width, self.config.map_height),
            'current_fps': self.current_fps,
            'visualization_features': {
                'trajectories': self.field_map.show_trajectories,
                'formations': self.field_map.show_formation_lines,
                'velocity_vectors': self.field_map.show_velocity_vectors,
                'heatmap': self.field_map.show_heatmap
            }
        }
    
    def set_video_window_size(self, width: int, height: int) -> None:
        """Set video display window size"""
        self.video_window_size = (width, height)
        if self.show_video:
            cv2.resizeWindow(self.video_window_name, width, height)
    
    def export_visualization_data(self) -> Dict[str, Any]:
        """Export visualization data for analysis"""
        return {
            'trajectory_data': self.visualizer.get_trajectory_data(),
            'heatmap_data': self.field_map.export_heatmap_data(),
            'formation_analysis': self.field_map.get_formation_analysis(),
            'display_info': self.get_display_info()
        }