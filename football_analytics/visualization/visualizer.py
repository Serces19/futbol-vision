"""
Real-time visualization system for football analytics
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import math

from ..core.models import TrackedObject, FieldLine, KeyPoint, FrameResults
from ..core.config import VisualizationConfig


@dataclass
class TrajectoryPoint:
    """Represents a point in a player's trajectory"""
    position: Tuple[int, int]
    timestamp: float
    alpha: float = 1.0  # For fading effect


class Visualizer:
    """
    Real-time visualization system for video overlay rendering
    
    Handles:
    - Bounding box drawing with team colors and player IDs
    - Trajectory visualization and ball tracking overlays  
    - Field line overlay and calibration visualization
    """
    
    def __init__(self, config: VisualizationConfig):
        """Initialize visualizer with configuration"""
        self.config = config
        self.trajectories: Dict[int, List[TrajectoryPoint]] = {}
        self.max_trajectory_length = 50
        self.trajectory_fade_time = 3.0  # seconds
        
    def draw_detections(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """
        Draw bounding boxes with team colors and player IDs
        
        Args:
            frame: Input frame to draw on
            tracked_objects: List of tracked objects to visualize
            
        Returns:
            Frame with detection overlays
        """
        result_frame = frame.copy()
        
        for obj in tracked_objects:
            if obj.detection.class_name == "person":
                self._draw_player_detection(result_frame, obj)
            elif obj.detection.class_name == "ball":
                self._draw_ball_detection(result_frame, obj)
                
        return result_frame
    
    def _draw_player_detection(self, frame: np.ndarray, obj: TrackedObject) -> None:
        """Draw player detection with team colors and ID"""
        x1, y1, x2, y2 = obj.detection.bbox
        
        # Get team color
        team_id = obj.team_id if obj.team_id is not None else -1
        color = self.config.team_colors.get(team_id, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.bbox_thickness)
        
        # Prepare label text
        if obj.team_id is not None:
            label = f"P{obj.track_id} T{obj.team_id + 1}"
        else:
            label = f"P{obj.track_id}"
            
        conf_text = f"({obj.detection.confidence:.2f})"
        
        # Add velocity if available
        if obj.velocity is not None:
            velocity_text = f" {obj.velocity:.1f}m/s"
            label += velocity_text
        
        # Draw label background and text
        self._draw_label(frame, (x1, y1), label, color)
        
        # Draw confidence below the main label
        conf_y = y1 - 5 if y1 > 25 else y2 + 20
        self._draw_label(frame, (x1, conf_y), conf_text, color, font_scale=0.4)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
    def _draw_ball_detection(self, frame: np.ndarray, obj: TrackedObject) -> None:
        """Draw ball detection with special highlighting"""
        x1, y1, x2, y2 = obj.detection.bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Draw ball with special color and style
        cv2.circle(frame, (center_x, center_y), 8, self.config.ball_color, -1)
        cv2.circle(frame, (center_x, center_y), 12, self.config.ball_color, 2)
        
        # Draw ball label
        label = f"Ball ({obj.detection.confidence:.2f})"
        self._draw_label(frame, (x1, y1 - 10), label, self.config.ball_color)
        
    def _draw_label(self, frame: np.ndarray, position: Tuple[int, int], 
                   text: str, color: Tuple[int, int, int], font_scale: float = None) -> None:
        """Draw text label with background"""
        x, y = position
        font_scale = font_scale or self.config.text_font_scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = self.config.text_thickness
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Adjust position if text would go outside frame
        if y - text_height - 5 < 0:
            y = text_height + 10
        if x + text_width > frame.shape[1]:
            x = frame.shape[1] - text_width - 5
            
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (x - 2, y - text_height - 5), 
                     (x + text_width + 2, y + 5), 
                     color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
    def draw_trajectories(self, frame: np.ndarray, tracked_objects: List[TrackedObject], 
                         current_timestamp: float) -> np.ndarray:
        """
        Draw player trajectories with fading effect
        
        Args:
            frame: Input frame to draw on
            tracked_objects: Current tracked objects
            current_timestamp: Current frame timestamp
            
        Returns:
            Frame with trajectory overlays
        """
        result_frame = frame.copy()
        
        # Update trajectories with current positions
        self._update_trajectories(tracked_objects, current_timestamp)
        
        # Draw trajectories for each player
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
                
            # Get team color for this player
            team_id = self._get_team_id_for_track(tracked_objects, track_id)
            color = self.config.team_colors.get(team_id, (128, 128, 128))
            
            # Draw trajectory lines with fading effect
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i-1]
                pt2 = trajectory[i]
                
                # Calculate alpha based on age
                age = current_timestamp - pt2.timestamp
                alpha = max(0, 1 - (age / self.trajectory_fade_time))
                
                if alpha > 0:
                    # Apply alpha to color
                    faded_color = tuple(int(c * alpha) for c in color)
                    thickness = max(1, int(self.config.trajectory_thickness * alpha))
                    
                    cv2.line(result_frame, pt1.position, pt2.position, faded_color, thickness)
        
        return result_frame
    
    def _update_trajectories(self, tracked_objects: List[TrackedObject], timestamp: float) -> None:
        """Update trajectory data with current positions"""
        current_tracks = set()
        
        for obj in tracked_objects:
            if obj.detection.class_name == "person":
                track_id = obj.track_id
                current_tracks.add(track_id)
                
                # Get center position
                x1, y1, x2, y2 = obj.detection.bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Initialize trajectory if new
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = []
                
                # Add new point
                self.trajectories[track_id].append(
                    TrajectoryPoint(center, timestamp)
                )
                
                # Limit trajectory length
                if len(self.trajectories[track_id]) > self.max_trajectory_length:
                    self.trajectories[track_id].pop(0)
        
        # Clean up old trajectories
        self._cleanup_old_trajectories(current_tracks, timestamp)
    
    def _cleanup_old_trajectories(self, current_tracks: set, timestamp: float) -> None:
        """Remove old trajectory points and inactive tracks"""
        tracks_to_remove = []
        
        for track_id, trajectory in self.trajectories.items():
            # Remove old points
            self.trajectories[track_id] = [
                pt for pt in trajectory 
                if timestamp - pt.timestamp <= self.trajectory_fade_time
            ]
            
            # Mark empty trajectories for removal
            if not self.trajectories[track_id]:
                tracks_to_remove.append(track_id)
        
        # Remove empty trajectories
        for track_id in tracks_to_remove:
            del self.trajectories[track_id]
    
    def _get_team_id_for_track(self, tracked_objects: List[TrackedObject], track_id: int) -> Optional[int]:
        """Get team ID for a specific track ID"""
        for obj in tracked_objects:
            if obj.track_id == track_id:
                return obj.team_id
        return None
    
    def draw_field_overlay(self, frame: np.ndarray, field_lines: List[FieldLine], 
                          key_points: List[KeyPoint] = None) -> np.ndarray:
        """
        Draw field lines and calibration visualization
        
        Args:
            frame: Input frame to draw on
            field_lines: Detected field lines
            key_points: Detected field key points (optional)
            
        Returns:
            Frame with field overlay
        """
        result_frame = frame.copy()
        
        # Draw field lines
        for line in field_lines:
            start_pt = line.start_point
            end_pt = line.end_point
            
            # Use different colors for different line types
            if line.line_type == "center_line":
                color = (0, 255, 255)  # Yellow
                thickness = 3
            elif line.line_type in ["goal_line", "sideline"]:
                color = (255, 255, 255)  # White
                thickness = 2
            else:
                color = self.config.line_color
                thickness = 2
            
            cv2.line(result_frame, start_pt, end_pt, color, thickness)
            
            # Draw line type label
            mid_x = (start_pt[0] + end_pt[0]) // 2
            mid_y = (start_pt[1] + end_pt[1]) // 2
            self._draw_label(result_frame, (mid_x, mid_y), line.line_type, color, font_scale=0.4)
        
        # Draw key points if provided
        if key_points:
            for kp in key_points:
                x, y = kp.position
                
                # Use different markers for different keypoint types
                if kp.keypoint_type == "corner":
                    cv2.circle(result_frame, (x, y), 8, (0, 255, 0), -1)  # Green
                elif kp.keypoint_type == "penalty_spot":
                    cv2.circle(result_frame, (x, y), 6, (255, 0, 255), -1)  # Magenta
                elif kp.keypoint_type == "center_circle":
                    cv2.circle(result_frame, (x, y), 10, (0, 255, 255), 2)  # Yellow circle
                else:
                    cv2.circle(result_frame, (x, y), 5, (255, 255, 255), -1)  # White
                
                # Draw keypoint label
                self._draw_label(result_frame, (x + 10, y), kp.keypoint_type, (255, 255, 255), font_scale=0.4)
        
        return result_frame
    
    def draw_statistics(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """
        Draw processing statistics on frame
        
        Args:
            frame: Input frame to draw on
            stats: Statistics dictionary
            
        Returns:
            Frame with statistics overlay
        """
        result_frame = frame.copy()
        
        # Prepare statistics text
        stats_text = []
        
        if 'frame_id' in stats:
            stats_text.append(f"Frame: {stats['frame_id']}")
        if 'fps' in stats:
            stats_text.append(f"FPS: {stats['fps']:.1f}")
        if 'players_detected' in stats:
            stats_text.append(f"Players: {stats['players_detected']}")
        if 'teams_detected' in stats:
            stats_text.append(f"Teams: {stats['teams_detected']}")
        if 'ball_detected' in stats:
            ball_status = "Yes" if stats['ball_detected'] else "No"
            stats_text.append(f"Ball: {ball_status}")
        if 'calibrated' in stats:
            cal_status = "Yes" if stats['calibrated'] else "No"
            stats_text.append(f"Calibrated: {cal_status}")
        
        # Draw statistics box
        if stats_text:
            self._draw_statistics_box(result_frame, stats_text)
        
        return result_frame
    
    def _draw_statistics_box(self, frame: np.ndarray, stats_text: List[str]) -> None:
        """Draw statistics in a box overlay"""
        if not stats_text:
            return
            
        # Calculate box dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 25
        padding = 10
        
        max_width = 0
        for text in stats_text:
            (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, text_width)
        
        box_width = max_width + 2 * padding
        box_height = len(stats_text) * line_height + 2 * padding
        
        # Draw background box
        cv2.rectangle(frame, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + box_width, 10 + box_height), (255, 255, 255), 2)
        
        # Draw text
        for i, text in enumerate(stats_text):
            y_pos = 10 + padding + (i + 1) * line_height - 5
            cv2.putText(frame, text, (10 + padding, y_pos), font, font_scale, (255, 255, 255), thickness)
    
    def create_combined_view(self, frame: np.ndarray, tracked_objects: List[TrackedObject],
                           field_lines: List[FieldLine], key_points: List[KeyPoint],
                           stats: Dict[str, Any], current_timestamp: float) -> np.ndarray:
        """
        Create a comprehensive visualization combining all overlays
        
        Args:
            frame: Input frame
            tracked_objects: Current tracked objects
            field_lines: Detected field lines
            key_points: Detected field key points
            stats: Processing statistics
            current_timestamp: Current frame timestamp
            
        Returns:
            Frame with all visualization overlays
        """
        result_frame = frame.copy()
        
        # Apply overlays in order
        result_frame = self.draw_field_overlay(result_frame, field_lines, key_points)
        result_frame = self.draw_trajectories(result_frame, tracked_objects, current_timestamp)
        result_frame = self.draw_detections(result_frame, tracked_objects)
        result_frame = self.draw_statistics(result_frame, stats)
        
        return result_frame
    
    def reset_trajectories(self) -> None:
        """Reset all trajectory data"""
        self.trajectories.clear()
    
    def get_trajectory_data(self) -> Dict[int, List[TrajectoryPoint]]:
        """Get current trajectory data for export"""
        return self.trajectories.copy()