"""
2D field map visualization for real-time player positioning and team analysis
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import math

from ..core.models import TrackedObject, FieldLine, KeyPoint
from ..core.config import VisualizationConfig, FieldDimensions


@dataclass
class PlayerPosition:
    """Represents a player position on the 2D field map"""
    track_id: int
    team_id: Optional[int]
    field_position: Tuple[float, float]  # (x, y) in field coordinates
    velocity: Optional[float] = None
    direction: Optional[float] = None  # Direction in radians


@dataclass
class TeamFormation:
    """Represents team formation analysis"""
    team_id: int
    centroid: Tuple[float, float]
    spread: float  # Average distance from centroid
    compactness: float  # Measure of how compact the formation is
    defensive_line: Optional[float] = None  # Y-coordinate of defensive line
    offensive_line: Optional[float] = None  # Y-coordinate of offensive line


class FieldMap2D:
    """
    Real-time 2D field map visualization
    
    Handles:
    - Real-time 2D field map with player positions
    - Team formation visualization and movement indicators
    - Interactive controls for map viewing and analysis
    """
    
    def __init__(self, config: VisualizationConfig, field_dimensions: FieldDimensions):
        """Initialize 2D field map visualizer"""
        self.config = config
        self.field_dimensions = field_dimensions
        
        # Map dimensions and scaling
        self.map_width = config.map_width
        self.map_height = config.map_height
        
        # Calculate scaling factors
        self.scale_x = self.map_width / field_dimensions.length
        self.scale_y = self.map_height / field_dimensions.width
        
        # Interactive controls state
        self.show_trajectories = True
        self.show_formation_lines = True
        self.show_velocity_vectors = True
        self.show_heatmap = False
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        
        # Data storage for analysis
        self.player_history: Dict[int, List[PlayerPosition]] = {}
        self.formation_history: Dict[int, List[TeamFormation]] = {}
        self.heatmap_data: Dict[int, np.ndarray] = {}
        
        # Initialize heatmaps
        self._initialize_heatmaps()
        
    def _initialize_heatmaps(self) -> None:
        """Initialize heatmap arrays for each team"""
        heatmap_resolution = (100, 68)  # Lower resolution for performance
        for team_id in range(4):  # Support up to 4 teams
            self.heatmap_data[team_id] = np.zeros(heatmap_resolution, dtype=np.float32)
    
    def create_field_map(self, tracked_objects: List[TrackedObject], 
                        ball_position: Optional[Tuple[float, float]] = None,
                        timestamp: float = 0.0) -> np.ndarray:
        """
        Create 2D field map with current player positions
        
        Args:
            tracked_objects: Current tracked objects with field positions
            ball_position: Ball position in field coordinates (optional)
            timestamp: Current timestamp
            
        Returns:
            2D field map image
        """
        # Create base field image
        field_map = self._create_base_field()
        
        # Convert tracked objects to player positions
        player_positions = self._extract_player_positions(tracked_objects)
        
        # Update player history
        self._update_player_history(player_positions, timestamp)
        
        # Update heatmaps
        self._update_heatmaps(player_positions)
        
        # Draw heatmap if enabled
        if self.show_heatmap:
            field_map = self._draw_heatmap_overlay(field_map)
        
        # Draw formation analysis
        if self.show_formation_lines:
            formations = self._analyze_formations(player_positions)
            field_map = self._draw_formations(field_map, formations)
        
        # Draw player trajectories
        if self.show_trajectories:
            field_map = self._draw_trajectories(field_map)
        
        # Draw current player positions
        field_map = self._draw_players(field_map, player_positions)
        
        # Draw ball position
        if ball_position:
            field_map = self._draw_ball(field_map, ball_position)
        
        # Draw velocity vectors
        if self.show_velocity_vectors:
            field_map = self._draw_velocity_vectors(field_map, player_positions)
        
        # Add interactive controls overlay
        field_map = self._draw_controls_overlay(field_map)
        
        return field_map
    
    def _create_base_field(self) -> np.ndarray:
        """Create base football field visualization"""
        # Create field background
        field_map = np.full((self.map_height, self.map_width, 3), 
                           self.config.map_background_color, dtype=np.uint8)
        
        # Field dimensions in map coordinates
        field_length = self.map_width
        field_width = self.map_height
        
        # Draw field outline
        cv2.rectangle(field_map, (0, 0), (field_length-1, field_width-1), (255, 255, 255), 2)
        
        # Draw center line
        center_x = field_length // 2
        cv2.line(field_map, (center_x, 0), (center_x, field_width), (255, 255, 255), 2)
        
        # Draw center circle
        center_y = field_width // 2
        radius = int(self.field_dimensions.center_circle_radius * self.scale_x)
        cv2.circle(field_map, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Draw penalty areas
        penalty_length = int(self.field_dimensions.penalty_area_length * self.scale_x)
        penalty_width = int(self.field_dimensions.penalty_area_width * self.scale_y)
        penalty_y_offset = (field_width - penalty_width) // 2
        
        # Left penalty area
        cv2.rectangle(field_map, (0, penalty_y_offset), 
                     (penalty_length, penalty_y_offset + penalty_width), (255, 255, 255), 2)
        
        # Right penalty area
        cv2.rectangle(field_map, (field_length - penalty_length, penalty_y_offset), 
                     (field_length, penalty_y_offset + penalty_width), (255, 255, 255), 2)
        
        # Draw goal areas
        goal_length = int(self.field_dimensions.goal_area_length * self.scale_x)
        goal_width = int(self.field_dimensions.goal_area_width * self.scale_y)
        goal_y_offset = (field_width - goal_width) // 2
        
        # Left goal area
        cv2.rectangle(field_map, (0, goal_y_offset), 
                     (goal_length, goal_y_offset + goal_width), (255, 255, 255), 2)
        
        # Right goal area
        cv2.rectangle(field_map, (field_length - goal_length, goal_y_offset), 
                     (field_length, goal_y_offset + goal_width), (255, 255, 255), 2)
        
        # Draw penalty spots
        penalty_spot_x = int(self.field_dimensions.penalty_spot_distance * self.scale_x)
        cv2.circle(field_map, (penalty_spot_x, center_y), 3, (255, 255, 255), -1)
        cv2.circle(field_map, (field_length - penalty_spot_x, center_y), 3, (255, 255, 255), -1)
        
        # Draw corner arcs
        corner_radius = int(self.field_dimensions.corner_arc_radius * self.scale_x)
        # Top-left corner
        cv2.ellipse(field_map, (0, 0), (corner_radius, corner_radius), 0, 0, 90, (255, 255, 255), 2)
        # Top-right corner
        cv2.ellipse(field_map, (field_length, 0), (corner_radius, corner_radius), 0, 90, 180, (255, 255, 255), 2)
        # Bottom-left corner
        cv2.ellipse(field_map, (0, field_width), (corner_radius, corner_radius), 0, 270, 360, (255, 255, 255), 2)
        # Bottom-right corner
        cv2.ellipse(field_map, (field_length, field_width), (corner_radius, corner_radius), 0, 180, 270, (255, 255, 255), 2)
        
        return field_map
    
    def _extract_player_positions(self, tracked_objects: List[TrackedObject]) -> List[PlayerPosition]:
        """Extract player positions from tracked objects"""
        player_positions = []
        
        for obj in tracked_objects:
            if obj.detection.class_name == "person" and obj.field_position:
                # Calculate direction from trajectory if available
                direction = None
                if obj.trajectory and len(obj.trajectory) >= 2:
                    last_pos = obj.trajectory[-1]
                    prev_pos = obj.trajectory[-2]
                    dx = last_pos[0] - prev_pos[0]
                    dy = last_pos[1] - prev_pos[1]
                    direction = math.atan2(dy, dx)
                
                player_pos = PlayerPosition(
                    track_id=obj.track_id,
                    team_id=obj.team_id,
                    field_position=obj.field_position,
                    velocity=obj.velocity,
                    direction=direction
                )
                player_positions.append(player_pos)
        
        return player_positions
    
    def _field_to_map_coords(self, field_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert field coordinates to map pixel coordinates"""
        field_x, field_y = field_pos
        
        # Convert to map coordinates
        map_x = int(field_x * self.scale_x)
        map_y = int(field_y * self.scale_y)
        
        # Apply zoom and pan
        map_x = int((map_x - self.pan_offset[0]) * self.zoom_factor)
        map_y = int((map_y - self.pan_offset[1]) * self.zoom_factor)
        
        return (map_x, map_y)
    
    def _draw_players(self, field_map: np.ndarray, player_positions: List[PlayerPosition]) -> np.ndarray:
        """Draw player positions on the field map"""
        for player in player_positions:
            map_x, map_y = self._field_to_map_coords(player.field_position)
            
            # Skip if outside visible area
            if not (0 <= map_x < self.map_width and 0 <= map_y < self.map_height):
                continue
            
            # Get team color
            team_id = player.team_id if player.team_id is not None else -1
            color = self.config.team_colors.get(team_id, (128, 128, 128))
            
            # Draw player circle
            cv2.circle(field_map, (map_x, map_y), 8, color, -1)
            cv2.circle(field_map, (map_x, map_y), 10, (255, 255, 255), 2)
            
            # Draw player ID
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            text = str(player.track_id)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_x = map_x - text_width // 2
            text_y = map_y + text_height // 2
            
            cv2.putText(field_map, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return field_map
    
    def _draw_ball(self, field_map: np.ndarray, ball_position: Tuple[float, float]) -> np.ndarray:
        """Draw ball position on the field map"""
        map_x, map_y = self._field_to_map_coords(ball_position)
        
        if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
            cv2.circle(field_map, (map_x, map_y), 6, self.config.ball_color, -1)
            cv2.circle(field_map, (map_x, map_y), 8, (255, 255, 255), 2)
        
        return field_map
    
    def _update_player_history(self, player_positions: List[PlayerPosition], timestamp: float) -> None:
        """Update player position history for trajectory visualization"""
        max_history_length = 50
        
        for player in player_positions:
            track_id = player.track_id
            
            if track_id not in self.player_history:
                self.player_history[track_id] = []
            
            self.player_history[track_id].append(player)
            
            # Limit history length
            if len(self.player_history[track_id]) > max_history_length:
                self.player_history[track_id].pop(0)
    
    def _draw_trajectories(self, field_map: np.ndarray) -> np.ndarray:
        """Draw player trajectories on the field map"""
        for track_id, history in self.player_history.items():
            if len(history) < 2:
                continue
            
            # Get team color
            team_id = history[-1].team_id if history[-1].team_id is not None else -1
            color = self.config.team_colors.get(team_id, (128, 128, 128))
            
            # Draw trajectory line
            points = []
            for i, player_pos in enumerate(history):
                map_x, map_y = self._field_to_map_coords(player_pos.field_position)
                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    points.append((map_x, map_y))
            
            if len(points) >= 2:
                # Draw trajectory with fading effect
                for i in range(1, len(points)):
                    alpha = i / len(points)  # Fade from old to new
                    faded_color = tuple(int(c * alpha) for c in color)
                    thickness = max(1, int(2 * alpha))
                    
                    cv2.line(field_map, points[i-1], points[i], faded_color, thickness)
        
        return field_map
    
    def _analyze_formations(self, player_positions: List[PlayerPosition]) -> List[TeamFormation]:
        """Analyze team formations"""
        formations = []
        
        # Group players by team
        teams = {}
        for player in player_positions:
            if player.team_id is not None:
                if player.team_id not in teams:
                    teams[player.team_id] = []
                teams[player.team_id].append(player)
        
        # Analyze each team
        for team_id, players in teams.items():
            if len(players) < 3:  # Need at least 3 players for formation analysis
                continue
            
            # Calculate centroid
            positions = [p.field_position for p in players]
            centroid_x = sum(pos[0] for pos in positions) / len(positions)
            centroid_y = sum(pos[1] for pos in positions) / len(positions)
            centroid = (centroid_x, centroid_y)
            
            # Calculate spread (average distance from centroid)
            distances = [math.sqrt((pos[0] - centroid_x)**2 + (pos[1] - centroid_y)**2) 
                        for pos in positions]
            spread = sum(distances) / len(distances)
            
            # Calculate compactness (inverse of spread, normalized)
            compactness = 1.0 / (1.0 + spread / 10.0)
            
            # Find defensive and offensive lines
            y_positions = [pos[1] for pos in positions]
            defensive_line = min(y_positions) if team_id == 0 else max(y_positions)
            offensive_line = max(y_positions) if team_id == 0 else min(y_positions)
            
            formation = TeamFormation(
                team_id=team_id,
                centroid=centroid,
                spread=spread,
                compactness=compactness,
                defensive_line=defensive_line,
                offensive_line=offensive_line
            )
            formations.append(formation)
        
        return formations
    
    def _draw_formations(self, field_map: np.ndarray, formations: List[TeamFormation]) -> np.ndarray:
        """Draw team formation analysis on the field map"""
        for formation in formations:
            team_color = self.config.team_colors.get(formation.team_id, (128, 128, 128))
            
            # Draw centroid
            centroid_map = self._field_to_map_coords(formation.centroid)
            if 0 <= centroid_map[0] < self.map_width and 0 <= centroid_map[1] < self.map_height:
                cv2.circle(field_map, centroid_map, 12, team_color, 2)
                cv2.circle(field_map, centroid_map, 4, team_color, -1)
            
            # Draw formation spread circle
            spread_radius = int(formation.spread * self.scale_x)
            cv2.circle(field_map, centroid_map, spread_radius, team_color, 1)
            
            # Draw defensive and offensive lines
            if formation.defensive_line is not None:
                def_y = int(formation.defensive_line * self.scale_y)
                if 0 <= def_y < self.map_height:
                    cv2.line(field_map, (0, def_y), (self.map_width, def_y), team_color, 2)
            
            if formation.offensive_line is not None:
                off_y = int(formation.offensive_line * self.scale_y)
                if 0 <= off_y < self.map_height:
                    cv2.line(field_map, (0, off_y), (self.map_width, off_y), team_color, 1)
        
        return field_map
    
    def _draw_velocity_vectors(self, field_map: np.ndarray, player_positions: List[PlayerPosition]) -> np.ndarray:
        """Draw velocity vectors for players"""
        for player in player_positions:
            if player.velocity is None or player.direction is None:
                continue
            
            map_x, map_y = self._field_to_map_coords(player.field_position)
            
            if not (0 <= map_x < self.map_width and 0 <= map_y < self.map_height):
                continue
            
            # Calculate vector end point
            vector_length = min(player.velocity * 5, 30)  # Scale and limit vector length
            end_x = int(map_x + vector_length * math.cos(player.direction))
            end_y = int(map_y + vector_length * math.sin(player.direction))
            
            # Get team color
            team_id = player.team_id if player.team_id is not None else -1
            color = self.config.team_colors.get(team_id, (128, 128, 128))
            
            # Draw velocity vector
            cv2.arrowedLine(field_map, (map_x, map_y), (end_x, end_y), color, 2, tipLength=0.3)
        
        return field_map
    
    def _update_heatmaps(self, player_positions: List[PlayerPosition]) -> None:
        """Update heatmap data with current player positions"""
        for player in player_positions:
            if player.team_id is None:
                continue
            
            # Convert to heatmap coordinates
            heatmap_shape = self.heatmap_data[player.team_id].shape
            heatmap_x = int((player.field_position[0] / self.field_dimensions.length) * heatmap_shape[1])
            heatmap_y = int((player.field_position[1] / self.field_dimensions.width) * heatmap_shape[0])
            
            # Add to heatmap (with bounds checking)
            if 0 <= heatmap_x < heatmap_shape[1] and 0 <= heatmap_y < heatmap_shape[0]:
                self.heatmap_data[player.team_id][heatmap_y, heatmap_x] += 1.0
    
    def _draw_heatmap_overlay(self, field_map: np.ndarray) -> np.ndarray:
        """Draw heatmap overlay on the field map"""
        for team_id, heatmap in self.heatmap_data.items():
            if np.sum(heatmap) == 0:
                continue
            
            # Normalize heatmap
            normalized_heatmap = heatmap / np.max(heatmap)
            
            # Resize to field map dimensions
            resized_heatmap = cv2.resize(normalized_heatmap, (self.map_width, self.map_height))
            
            # Apply color map
            team_color = self.config.team_colors.get(team_id, (128, 128, 128))
            colored_heatmap = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
            
            for i in range(3):
                colored_heatmap[:, :, i] = (resized_heatmap * team_color[i]).astype(np.uint8)
            
            # Blend with field map
            alpha = 0.3
            field_map = cv2.addWeighted(field_map, 1 - alpha, colored_heatmap, alpha, 0)
        
        return field_map
    
    def _draw_controls_overlay(self, field_map: np.ndarray) -> np.ndarray:
        """Draw interactive controls overlay"""
        controls_text = [
            "Controls:",
            "T - Toggle trajectories",
            "F - Toggle formations", 
            "V - Toggle velocity vectors",
            "H - Toggle heatmap",
            "R - Reset view"
        ]
        
        # Draw controls box
        box_width = 200
        box_height = len(controls_text) * 20 + 20
        box_x = self.map_width - box_width - 10
        box_y = 10
        
        # Semi-transparent background
        overlay = field_map.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
        field_map = cv2.addWeighted(field_map, 0.7, overlay, 0.3, 0)
        
        # Draw border
        cv2.rectangle(field_map, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 1)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        for i, text in enumerate(controls_text):
            y_pos = box_y + 15 + i * 20
            cv2.putText(field_map, text, (box_x + 10, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        return field_map
    
    def handle_key_input(self, key: int) -> bool:
        """
        Handle keyboard input for interactive controls
        
        Args:
            key: OpenCV key code
            
        Returns:
            True if key was handled, False otherwise
        """
        key_char = chr(key & 0xFF).lower()
        
        if key_char == 't':
            self.show_trajectories = not self.show_trajectories
            return True
        elif key_char == 'f':
            self.show_formation_lines = not self.show_formation_lines
            return True
        elif key_char == 'v':
            self.show_velocity_vectors = not self.show_velocity_vectors
            return True
        elif key_char == 'h':
            self.show_heatmap = not self.show_heatmap
            return True
        elif key_char == 'r':
            self.reset_view()
            return True
        
        return False
    
    def reset_view(self) -> None:
        """Reset view parameters and clear data"""
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        self.player_history.clear()
        self.formation_history.clear()
        self._initialize_heatmaps()
    
    def get_formation_analysis(self) -> Dict[int, TeamFormation]:
        """Get current formation analysis for all teams"""
        # This would be called with current player positions
        # For now, return empty dict as placeholder
        return {}
    
    def export_heatmap_data(self) -> Dict[int, np.ndarray]:
        """Export heatmap data for analysis"""
        return self.heatmap_data.copy()