"""
Analytics engine for generating football match statistics and metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time
import json

from ..core.models import TrackedObject, Detection
from ..core.config import ProcessingConfig, FieldDimensions

@dataclass
class PlayerStats:
    """Statistics for a single player"""
    player_id: int
    team_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    total_distance: float = 0.0
    max_velocity: float = 0.0
    avg_velocity: float = 0.0
    time_on_field: float = 0.0
    last_seen_frame: int = 0

@dataclass
class TeamStats:
    """Statistics for a team"""
    team_id: int
    player_count: int = 0
    total_distance: float = 0.0
    avg_velocity: float = 0.0
    formation_center: Tuple[float, float] = (0.0, 0.0)
    formation_spread: float = 0.0
    possession_time: float = 0.0

@dataclass
class MatchStats:
    """Overall match statistics"""
    frame_count: int = 0
    processing_time: float = 0.0
    players_detected: Dict[int, PlayerStats] = field(default_factory=dict)
    teams: Dict[int, TeamStats] = field(default_factory=dict)
    ball_positions: List[Tuple[float, float]] = field(default_factory=list)
    heatmap_data: Dict[int, np.ndarray] = field(default_factory=dict)

class AnalyticsEngine:
    """
    Analytics engine for generating football match statistics and metrics
    """
    
    def __init__(self, config: ProcessingConfig, field_dims: FieldDimensions):
        self.config = config
        self.field_dims = field_dims
        
        # Statistics storage
        self.match_stats = MatchStats()
        self.player_trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.team_assignments: Dict[int, int] = {}
        
        # Timing and frame tracking
        self.start_time = time.time()
        self.last_frame_time = 0.0
        self.fps = 30.0  # Default FPS
        
        # Heatmap resolution (field divided into grid)
        self.heatmap_resolution = (50, 30)  # 50x30 grid
        self._initialize_heatmaps()
        
    def _initialize_heatmaps(self):
        """Initialize heatmap arrays for each team"""
        for team_id in [0, 1]:  # Assuming 2 teams
            self.match_stats.heatmap_data[team_id] = np.zeros(self.heatmap_resolution)
    
    def update_frame_data(self, tracked_objects: List[TrackedObject], 
                         ball_detection: Optional[Detection] = None,
                         frame_number: int = 0,
                         timestamp: float = 0.0):
        """
        Update analytics with new frame data
        
        Args:
            tracked_objects: List of tracked players
            ball_detection: Ball detection if available
            frame_number: Current frame number
            timestamp: Frame timestamp
        """
        current_time = time.time()
        frame_dt = timestamp - self.last_frame_time if self.last_frame_time > 0 else 1.0/self.fps
        self.last_frame_time = timestamp
        
        self.match_stats.frame_count = frame_number
        
        # Process player data
        self._process_players(tracked_objects, frame_dt, frame_number)
        
        # Process ball data
        if ball_detection:
            self._process_ball(ball_detection)
        
        # Update team statistics
        self._update_team_stats()
        
        # Update processing time
        self.match_stats.processing_time = current_time - self.start_time
    
    def _process_players(self, tracked_objects: List[TrackedObject], 
                        frame_dt: float, frame_number: int):
        """Process player tracking data and calculate metrics"""
        
        for obj in tracked_objects:
            player_id = obj.track_id
            
            # Initialize player stats if new
            if player_id not in self.match_stats.players_detected:
                team_id = self.team_assignments.get(player_id, 0)
                self.match_stats.players_detected[player_id] = PlayerStats(
                    player_id=player_id,
                    team_id=team_id
                )
            
            player_stats = self.match_stats.players_detected[player_id]
            
            # Calculate center coordinates from bounding box
            bbox = obj.detection.bbox
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Convert pixel coordinates to field coordinates if calibration available
            field_pos = self._pixel_to_field_coords(center_x, center_y)
            
            # Update position history
            player_stats.positions.append(field_pos)
            player_stats.last_seen_frame = frame_number
            player_stats.time_on_field += frame_dt
            
            # Add to trajectory for velocity calculation
            self.player_trajectories[player_id].append((field_pos, time.time()))
            
            # Calculate velocity and distance if we have previous positions
            if len(self.player_trajectories[player_id]) >= 2:
                velocity, distance = self._calculate_movement_metrics(player_id, frame_dt)
                
                player_stats.velocities.append(velocity)
                player_stats.distances.append(distance)
                player_stats.total_distance += distance
                player_stats.max_velocity = max(player_stats.max_velocity, velocity)
                
                # Update average velocity
                if len(player_stats.velocities) > 0:
                    player_stats.avg_velocity = np.mean(player_stats.velocities)
            
            # Update heatmap
            self._update_heatmap(player_id, field_pos)
    
    def _process_ball(self, ball_detection: Detection):
        """Process ball detection data"""
        # Calculate center coordinates from bounding box
        bbox = ball_detection.bbox
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        field_pos = self._pixel_to_field_coords(center_x, center_y)
        self.match_stats.ball_positions.append(field_pos)
    
    def _calculate_movement_metrics(self, player_id: int, frame_dt: float) -> Tuple[float, float]:
        """Calculate velocity and distance for a player"""
        trajectory = self.player_trajectories[player_id]
        
        if len(trajectory) < 2:
            return 0.0, 0.0
        
        # Get last two positions
        (x1, y1), t1 = trajectory[-2]
        (x2, y2), t2 = trajectory[-1]
        
        # Calculate distance in meters
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate velocity in m/s
        time_diff = max(t2 - t1, frame_dt)  # Avoid division by zero
        velocity = distance / time_diff if time_diff > 0 else 0.0
        
        return velocity, distance
    
    def _pixel_to_field_coords(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to field coordinates
        This is a placeholder - should use actual homography matrix from calibration
        """
        # For now, return normalized coordinates
        # In real implementation, this would use the homography matrix
        field_x = (pixel_x / 1920.0) * self.field_dims.length  # Assuming 1920px width
        field_y = (pixel_y / 1080.0) * self.field_dims.width   # Assuming 1080px height
        
        return field_x, field_y
    
    def _update_heatmap(self, player_id: int, field_pos: Tuple[float, float]):
        """Update heatmap data for player position"""
        team_id = self.team_assignments.get(player_id, 0)
        
        if team_id not in self.match_stats.heatmap_data:
            self.match_stats.heatmap_data[team_id] = np.zeros(self.heatmap_resolution)
        
        # Convert field coordinates to heatmap grid indices
        x, y = field_pos
        grid_x = int((x / self.field_dims.length) * self.heatmap_resolution[0])
        grid_y = int((y / self.field_dims.width) * self.heatmap_resolution[1])
        
        # Ensure indices are within bounds
        grid_x = max(0, min(grid_x, self.heatmap_resolution[0] - 1))
        grid_y = max(0, min(grid_y, self.heatmap_resolution[1] - 1))
        
        # Increment heatmap value with Gaussian smoothing for better visualization
        self._add_gaussian_to_heatmap(team_id, grid_x, grid_y, intensity=1.0, sigma=1.5)
    
    def _add_gaussian_to_heatmap(self, team_id: int, center_x: int, center_y: int, 
                                intensity: float = 1.0, sigma: float = 1.5):
        """Add Gaussian distribution around position for smoother heatmaps"""
        heatmap = self.match_stats.heatmap_data[team_id]
        
        # Create Gaussian kernel
        kernel_size = int(3 * sigma)
        y_indices, x_indices = np.ogrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
        gaussian = np.exp(-(x_indices**2 + y_indices**2) / (2 * sigma**2))
        gaussian = gaussian * intensity / gaussian.sum()
        
        # Apply kernel to heatmap with bounds checking
        for dy in range(-kernel_size, kernel_size + 1):
            for dx in range(-kernel_size, kernel_size + 1):
                hm_x = center_x + dx
                hm_y = center_y + dy
                
                if (0 <= hm_x < self.heatmap_resolution[0] and 
                    0 <= hm_y < self.heatmap_resolution[1]):
                    kernel_x = dx + kernel_size
                    kernel_y = dy + kernel_size
                    heatmap[hm_x, hm_y] += gaussian[kernel_y, kernel_x]
    
    def _update_team_stats(self):
        """Update team-level statistics"""
        team_players = defaultdict(list)
        
        # Group players by team
        for player_id, player_stats in self.match_stats.players_detected.items():
            team_id = player_stats.team_id
            team_players[team_id].append(player_stats)
        
        # Calculate team statistics
        for team_id, players in team_players.items():
            if team_id not in self.match_stats.teams:
                self.match_stats.teams[team_id] = TeamStats(team_id=team_id)
            
            team_stats = self.match_stats.teams[team_id]
            team_stats.player_count = len(players)
            
            if players:
                # Calculate total distance and average velocity
                team_stats.total_distance = sum(p.total_distance for p in players)
                team_stats.avg_velocity = np.mean([p.avg_velocity for p in players if p.avg_velocity > 0])
                
                # Calculate formation center (average position)
                if any(p.positions for p in players):
                    recent_positions = []
                    for player in players:
                        if player.positions:
                            recent_positions.append(player.positions[-1])
                    
                    if recent_positions:
                        center_x = np.mean([pos[0] for pos in recent_positions])
                        center_y = np.mean([pos[1] for pos in recent_positions])
                        team_stats.formation_center = (center_x, center_y)
                        
                        # Calculate formation spread (standard deviation of positions)
                        distances_from_center = [
                            np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
                            for pos in recent_positions
                        ]
                        team_stats.formation_spread = np.std(distances_from_center) if distances_from_center else 0.0
    
    def update_team_assignments(self, team_assignments: Dict[int, int]):
        """Update team assignments for players"""
        self.team_assignments.update(team_assignments)
        
        # Update existing player stats with new team assignments
        for player_id, team_id in team_assignments.items():
            if player_id in self.match_stats.players_detected:
                self.match_stats.players_detected[player_id].team_id = team_id
    
    def get_player_stats(self, player_id: int) -> Optional[PlayerStats]:
        """Get statistics for a specific player"""
        return self.match_stats.players_detected.get(player_id)
    
    def get_team_stats(self, team_id: int) -> Optional[TeamStats]:
        """Get statistics for a specific team"""
        return self.match_stats.teams.get(team_id)
    
    def get_match_summary(self) -> Dict[str, Any]:
        """Get comprehensive match statistics summary"""
        return {
            'match_info': {
                'total_frames': self.match_stats.frame_count,
                'processing_time': self.match_stats.processing_time,
                'fps': self.fps,
                'duration_minutes': self.match_stats.processing_time / 60.0
            },
            'players': {
                pid: {
                    'team_id': stats.team_id,
                    'total_distance_m': stats.total_distance,
                    'max_velocity_ms': stats.max_velocity,
                    'avg_velocity_ms': stats.avg_velocity,
                    'time_on_field_s': stats.time_on_field,
                    'positions_count': len(stats.positions)
                }
                for pid, stats in self.match_stats.players_detected.items()
            },
            'teams': {
                tid: {
                    'player_count': stats.player_count,
                    'total_distance_m': stats.total_distance,
                    'avg_velocity_ms': stats.avg_velocity,
                    'formation_center': stats.formation_center,
                    'formation_spread': stats.formation_spread
                }
                for tid, stats in self.match_stats.teams.items()
            },
            'ball': {
                'positions_count': len(self.match_stats.ball_positions),
                'last_position': self.match_stats.ball_positions[-1] if self.match_stats.ball_positions else None
            }
        }
    
    def get_heatmap_data(self, team_id: int) -> Optional[np.ndarray]:
        """Get heatmap data for a specific team"""
        return self.match_stats.heatmap_data.get(team_id)
    
    def generate_normalized_heatmap(self, team_id: int) -> Optional[np.ndarray]:
        """Generate normalized heatmap (0-1 range) for visualization"""
        heatmap = self.get_heatmap_data(team_id)
        if heatmap is None or heatmap.max() == 0:
            return None
        
        # Normalize to 0-1 range
        normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return normalized
    
    def generate_player_heatmap(self, player_id: int) -> Optional[np.ndarray]:
        """Generate heatmap for individual player"""
        if player_id not in self.match_stats.players_detected:
            return None
        
        player_stats = self.match_stats.players_detected[player_id]
        heatmap = np.zeros(self.heatmap_resolution)
        
        # Add all player positions to heatmap
        for pos in player_stats.positions:
            x, y = pos
            grid_x = int((x / self.field_dims.length) * self.heatmap_resolution[0])
            grid_y = int((y / self.field_dims.width) * self.heatmap_resolution[1])
            
            grid_x = max(0, min(grid_x, self.heatmap_resolution[0] - 1))
            grid_y = max(0, min(grid_y, self.heatmap_resolution[1] - 1))
            
            # Add Gaussian distribution
            self._add_gaussian_to_individual_heatmap(heatmap, grid_x, grid_y)
        
        return heatmap
    
    def _add_gaussian_to_individual_heatmap(self, heatmap: np.ndarray, center_x: int, center_y: int, 
                                          intensity: float = 1.0, sigma: float = 1.5):
        """Add Gaussian distribution to individual player heatmap"""
        kernel_size = int(3 * sigma)
        y_indices, x_indices = np.ogrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
        gaussian = np.exp(-(x_indices**2 + y_indices**2) / (2 * sigma**2))
        gaussian = gaussian * intensity / gaussian.sum()
        
        for dy in range(-kernel_size, kernel_size + 1):
            for dx in range(-kernel_size, kernel_size + 1):
                hm_x = center_x + dx
                hm_y = center_y + dy
                
                if (0 <= hm_x < heatmap.shape[0] and 0 <= hm_y < heatmap.shape[1]):
                    kernel_x = dx + kernel_size
                    kernel_y = dy + kernel_size
                    heatmap[hm_x, hm_y] += gaussian[kernel_y, kernel_x]
    
    def analyze_team_formation(self, team_id: int, time_window: int = 30) -> Dict[str, Any]:
        """
        Analyze team formation and positioning patterns
        
        Args:
            team_id: Team to analyze
            time_window: Number of recent frames to analyze
            
        Returns:
            Dictionary with formation analysis data
        """
        team_players = [
            player for player in self.match_stats.players_detected.values()
            if player.team_id == team_id and len(player.positions) > 0
        ]
        
        if len(team_players) < 2:
            return {'error': 'Insufficient player data for formation analysis'}
        
        # Get recent positions for each player
        recent_positions = []
        for player in team_players:
            if len(player.positions) > 0:
                # Take last position or average of recent positions
                if len(player.positions) >= time_window:
                    recent_pos = player.positions[-time_window:]
                    avg_x = np.mean([pos[0] for pos in recent_pos])
                    avg_y = np.mean([pos[1] for pos in recent_pos])
                    recent_positions.append((avg_x, avg_y, player.player_id))
                else:
                    pos = player.positions[-1]
                    recent_positions.append((pos[0], pos[1], player.player_id))
        
        if len(recent_positions) < 2:
            return {'error': 'Insufficient recent position data'}
        
        # Calculate formation metrics
        positions_array = np.array([(pos[0], pos[1]) for pos in recent_positions])
        
        # Formation center (centroid)
        center_x, center_y = np.mean(positions_array, axis=0)
        
        # Formation spread (average distance from center)
        distances_from_center = [
            np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
            for pos in positions_array
        ]
        avg_spread = np.mean(distances_from_center)
        max_spread = np.max(distances_from_center)
        
        # Formation compactness (standard deviation of distances)
        compactness = np.std(distances_from_center)
        
        # Formation width and length
        x_coords = positions_array[:, 0]
        y_coords = positions_array[:, 1]
        formation_width = np.max(y_coords) - np.min(y_coords)
        formation_length = np.max(x_coords) - np.min(x_coords)
        
        # Defensive/offensive positioning (relative to field center)
        field_center_x = self.field_dims.length / 2
        avg_field_position = np.mean(x_coords)
        positioning_bias = avg_field_position - field_center_x  # Positive = offensive, negative = defensive
        
        return {
            'team_id': team_id,
            'player_count': len(recent_positions),
            'formation_center': (float(center_x), float(center_y)),
            'formation_spread': {
                'average': float(avg_spread),
                'maximum': float(max_spread),
                'compactness': float(compactness)
            },
            'formation_dimensions': {
                'width': float(formation_width),
                'length': float(formation_length),
                'area': float(formation_width * formation_length)
            },
            'positioning': {
                'field_position': float(avg_field_position),
                'bias': float(positioning_bias),
                'tendency': 'offensive' if positioning_bias > 5 else 'defensive' if positioning_bias < -5 else 'neutral'
            },
            'player_positions': [
                {'player_id': pos[2], 'x': float(pos[0]), 'y': float(pos[1])}
                for pos in recent_positions
            ]
        }
    
    def analyze_spatial_dominance(self, team_id: int) -> Dict[str, Any]:
        """
        Analyze spatial dominance patterns for a team
        
        Args:
            team_id: Team to analyze
            
        Returns:
            Dictionary with spatial dominance analysis
        """
        heatmap = self.get_heatmap_data(team_id)
        if heatmap is None or heatmap.sum() == 0:
            return {'error': 'No heatmap data available for team'}
        
        # Divide field into zones for analysis
        zones = self._divide_field_into_zones(heatmap)
        
        # Calculate dominance metrics
        total_presence = heatmap.sum()
        zone_dominance = {}
        
        for zone_name, zone_data in zones.items():
            zone_presence = zone_data.sum()
            dominance_percentage = (zone_presence / total_presence) * 100 if total_presence > 0 else 0
            zone_dominance[zone_name] = {
                'presence': float(zone_presence),
                'dominance_percentage': float(dominance_percentage),
                'intensity': float(zone_data.mean())
            }
        
        # Find most and least dominated zones
        sorted_zones = sorted(zone_dominance.items(), key=lambda x: x[1]['dominance_percentage'], reverse=True)
        
        return {
            'team_id': team_id,
            'total_presence': float(total_presence),
            'zone_analysis': zone_dominance,
            'most_dominated_zone': sorted_zones[0][0] if sorted_zones else None,
            'least_dominated_zone': sorted_zones[-1][0] if sorted_zones else None,
            'dominance_distribution': {
                'attacking_third': sum(zone_dominance.get(zone, {}).get('dominance_percentage', 0) 
                                     for zone in ['attacking_left', 'attacking_center', 'attacking_right']),
                'middle_third': sum(zone_dominance.get(zone, {}).get('dominance_percentage', 0) 
                                  for zone in ['middle_left', 'middle_center', 'middle_right']),
                'defensive_third': sum(zone_dominance.get(zone, {}).get('dominance_percentage', 0) 
                                     for zone in ['defensive_left', 'defensive_center', 'defensive_right'])
            }
        }
    
    def _divide_field_into_zones(self, heatmap: np.ndarray) -> Dict[str, np.ndarray]:
        """Divide field heatmap into tactical zones"""
        height, width = heatmap.shape
        
        # Divide into 3x3 grid (defensive, middle, attacking thirds)
        third_h = height // 3
        third_w = width // 3
        
        zones = {
            'defensive_left': heatmap[0:third_h, 0:third_w],
            'defensive_center': heatmap[0:third_h, third_w:2*third_w],
            'defensive_right': heatmap[0:third_h, 2*third_w:width],
            'middle_left': heatmap[third_h:2*third_h, 0:third_w],
            'middle_center': heatmap[third_h:2*third_h, third_w:2*third_w],
            'middle_right': heatmap[third_h:2*third_h, 2*third_w:width],
            'attacking_left': heatmap[2*third_h:height, 0:third_w],
            'attacking_center': heatmap[2*third_h:height, third_w:2*third_w],
            'attacking_right': heatmap[2*third_h:height, 2*third_w:width]
        }
        
        return zones
    
    def generate_movement_patterns(self, player_id: int) -> Dict[str, Any]:
        """
        Analyze movement patterns for a specific player
        
        Args:
            player_id: Player to analyze
            
        Returns:
            Dictionary with movement pattern analysis
        """
        if player_id not in self.match_stats.players_detected:
            return {'error': 'Player not found'}
        
        player_stats = self.match_stats.players_detected[player_id]
        
        if len(player_stats.positions) < 10:
            return {'error': 'Insufficient position data for pattern analysis'}
        
        positions = np.array(player_stats.positions)
        
        # Calculate movement vectors
        movement_vectors = np.diff(positions, axis=0)
        
        # Movement characteristics
        total_movement = np.sum(np.linalg.norm(movement_vectors, axis=1))
        avg_movement_per_frame = total_movement / len(movement_vectors) if len(movement_vectors) > 0 else 0
        
        # Direction analysis
        angles = np.arctan2(movement_vectors[:, 1], movement_vectors[:, 0])
        
        # Dominant movement direction
        avg_angle = np.mean(angles)
        direction_consistency = 1 - (np.std(angles) / np.pi)  # 0-1, higher = more consistent direction
        
        # Activity zones (areas where player spends most time)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        activity_center = (np.mean(x_coords), np.mean(y_coords))
        activity_radius = np.std(np.linalg.norm(positions - activity_center, axis=1))
        
        # Movement intensity over time
        movement_intensities = np.linalg.norm(movement_vectors, axis=1)
        
        return {
            'player_id': player_id,
            'team_id': player_stats.team_id,
            'movement_summary': {
                'total_distance': float(total_movement),
                'avg_movement_per_frame': float(avg_movement_per_frame),
                'position_samples': len(positions)
            },
            'direction_analysis': {
                'dominant_angle_rad': float(avg_angle),
                'dominant_direction': self._angle_to_direction(avg_angle),
                'consistency': float(direction_consistency)
            },
            'activity_zone': {
                'center': (float(activity_center[0]), float(activity_center[1])),
                'radius': float(activity_radius),
                'field_coverage': float(activity_radius / (self.field_dims.length * 0.5))  # Normalized coverage
            },
            'intensity_stats': {
                'max_intensity': float(np.max(movement_intensities)) if len(movement_intensities) > 0 else 0,
                'avg_intensity': float(np.mean(movement_intensities)) if len(movement_intensities) > 0 else 0,
                'intensity_variance': float(np.var(movement_intensities)) if len(movement_intensities) > 0 else 0
            }
        }
    
    def _angle_to_direction(self, angle_rad: float) -> str:
        """Convert angle in radians to cardinal direction"""
        angle_deg = np.degrees(angle_rad) % 360
        
        if angle_deg < 22.5 or angle_deg >= 337.5:
            return 'East'
        elif angle_deg < 67.5:
            return 'Northeast'
        elif angle_deg < 112.5:
            return 'North'
        elif angle_deg < 157.5:
            return 'Northwest'
        elif angle_deg < 202.5:
            return 'West'
        elif angle_deg < 247.5:
            return 'Southwest'
        elif angle_deg < 292.5:
            return 'South'
        else:
            return 'Southeast'
    
    def export_analytics_data(self, filepath: str):
        """Export all analytics data to JSON file"""
        # Generate comprehensive spatial analysis for all teams
        team_formations = {}
        spatial_dominance = {}
        
        for team_id in self.match_stats.teams.keys():
            team_formations[str(team_id)] = self.analyze_team_formation(team_id)
            spatial_dominance[str(team_id)] = self.analyze_spatial_dominance(team_id)
        
        # Generate movement patterns for all players
        movement_patterns = {}
        individual_heatmaps = {}
        
        for player_id in self.match_stats.players_detected.keys():
            movement_patterns[str(player_id)] = self.generate_movement_patterns(player_id)
            
            # Generate individual player heatmap
            player_heatmap = self.generate_player_heatmap(player_id)
            if player_heatmap is not None:
                individual_heatmaps[str(player_id)] = player_heatmap.tolist()
        
        # Generate normalized team heatmaps for visualization
        normalized_heatmaps = {}
        for team_id in self.match_stats.heatmap_data.keys():
            normalized = self.generate_normalized_heatmap(team_id)
            if normalized is not None:
                normalized_heatmaps[str(team_id)] = normalized.tolist()
        
        export_data = {
            'match_summary': self.get_match_summary(),
            'detailed_trajectories': {
                str(pid): [
                    {'x': pos[0], 'y': pos[1], 'frame': i}
                    for i, pos in enumerate(stats.positions)
                ]
                for pid, stats in self.match_stats.players_detected.items()
            },
            'ball_trajectory': [
                {'x': pos[0], 'y': pos[1], 'frame': i}
                for i, pos in enumerate(self.match_stats.ball_positions)
            ],
            'heatmaps': {
                'raw_team_heatmaps': {
                    str(team_id): heatmap.tolist()
                    for team_id, heatmap in self.match_stats.heatmap_data.items()
                },
                'normalized_team_heatmaps': normalized_heatmaps,
                'individual_player_heatmaps': individual_heatmaps,
                'heatmap_resolution': self.heatmap_resolution
            },
            'spatial_analysis': {
                'team_formations': team_formations,
                'spatial_dominance': spatial_dominance,
                'movement_patterns': movement_patterns
            },
            'field_dimensions': {
                'length': self.field_dims.length,
                'width': self.field_dims.width,
                'goal_width': self.field_dims.goal_width,
                'goal_area_length': self.field_dims.goal_area_length,
                'penalty_area_length': self.field_dims.penalty_area_length
            },
            'metadata': {
                'export_timestamp': time.time(),
                'processing_config': {
                    'confidence_threshold': self.config.confidence_threshold,
                    'n_teams': self.config.n_teams,
                    'device': self.config.device
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_zone_heatmap(self, team_id: int, zone_name: str) -> Optional[np.ndarray]:
        """
        Get heatmap data for a specific field zone
        
        Args:
            team_id: Team ID
            zone_name: Zone name (e.g., 'attacking_center', 'defensive_left')
            
        Returns:
            Heatmap array for the specified zone
        """
        heatmap = self.get_heatmap_data(team_id)
        if heatmap is None:
            return None
        
        zones = self._divide_field_into_zones(heatmap)
        return zones.get(zone_name)
    
    def compare_team_heatmaps(self, team1_id: int, team2_id: int) -> Dict[str, Any]:
        """
        Compare heatmaps between two teams
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            
        Returns:
            Dictionary with comparison metrics
        """
        heatmap1 = self.get_heatmap_data(team1_id)
        heatmap2 = self.get_heatmap_data(team2_id)
        
        if heatmap1 is None or heatmap2 is None:
            return {'error': 'Heatmap data not available for one or both teams'}
        
        # Normalize heatmaps for comparison
        norm1 = heatmap1 / heatmap1.sum() if heatmap1.sum() > 0 else heatmap1
        norm2 = heatmap2 / heatmap2.sum() if heatmap2.sum() > 0 else heatmap2
        
        # Calculate overlap and differences
        overlap = np.minimum(norm1, norm2).sum()
        difference = np.abs(norm1 - norm2).sum()
        
        # Calculate center of mass for each team
        y_indices, x_indices = np.mgrid[0:heatmap1.shape[0], 0:heatmap1.shape[1]]
        
        if norm1.sum() > 0:
            com1_x = (x_indices * norm1).sum() / norm1.sum()
            com1_y = (y_indices * norm1).sum() / norm1.sum()
        else:
            com1_x = com1_y = 0
        
        if norm2.sum() > 0:
            com2_x = (x_indices * norm2).sum() / norm2.sum()
            com2_y = (y_indices * norm2).sum() / norm2.sum()
        else:
            com2_x = com2_y = 0
        
        # Distance between centers of mass
        com_distance = np.sqrt((com1_x - com2_x)**2 + (com1_y - com2_y)**2)
        
        return {
            'team1_id': team1_id,
            'team2_id': team2_id,
            'overlap_coefficient': float(overlap),
            'difference_metric': float(difference),
            'centers_of_mass': {
                'team1': (float(com1_x), float(com1_y)),
                'team2': (float(com2_x), float(com2_y)),
                'distance': float(com_distance)
            },
            'spatial_separation': float(com_distance / max(heatmap1.shape))  # Normalized distance
        }
    
    def get_temporal_heatmap(self, team_id: int, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
        """
        Generate heatmap for a specific time period
        
        Args:
            team_id: Team ID
            start_frame: Starting frame number
            end_frame: Ending frame number
            
        Returns:
            Heatmap for the specified time period
        """
        temporal_heatmap = np.zeros(self.heatmap_resolution)
        
        team_players = [
            player for player in self.match_stats.players_detected.values()
            if player.team_id == team_id
        ]
        
        for player in team_players:
            # Get positions within the time frame
            for i, pos in enumerate(player.positions):
                if start_frame <= i <= end_frame:
                    x, y = pos
                    grid_x = int((x / self.field_dims.length) * self.heatmap_resolution[0])
                    grid_y = int((y / self.field_dims.width) * self.heatmap_resolution[1])
                    
                    grid_x = max(0, min(grid_x, self.heatmap_resolution[0] - 1))
                    grid_y = max(0, min(grid_y, self.heatmap_resolution[1] - 1))
                    
                    self._add_gaussian_to_individual_heatmap(temporal_heatmap, grid_x, grid_y)
        
        return temporal_heatmap if temporal_heatmap.sum() > 0 else None
    
    def export_heatmap_images(self, output_dir: str):
        """
        Export heatmaps as image files for visualization
        Note: This method requires matplotlib, which should be imported when needed
        
        Args:
            output_dir: Directory to save heatmap images
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import os
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Export team heatmaps
            for team_id in self.match_stats.heatmap_data.keys():
                heatmap = self.generate_normalized_heatmap(team_id)
                if heatmap is not None:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(heatmap.T, cmap='hot', interpolation='bilinear', origin='lower')
                    plt.colorbar(label='Normalized Presence')
                    plt.title(f'Team {team_id} Heatmap')
                    plt.xlabel('Field Length')
                    plt.ylabel('Field Width')
                    plt.savefig(os.path.join(output_dir, f'team_{team_id}_heatmap.png'), 
                              dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Export individual player heatmaps
            for player_id in self.match_stats.players_detected.keys():
                player_heatmap = self.generate_player_heatmap(player_id)
                if player_heatmap is not None and player_heatmap.sum() > 0:
                    normalized = (player_heatmap - player_heatmap.min()) / (player_heatmap.max() - player_heatmap.min())
                    
                    plt.figure(figsize=(12, 8))
                    plt.imshow(normalized.T, cmap='viridis', interpolation='bilinear', origin='lower')
                    plt.colorbar(label='Normalized Presence')
                    plt.title(f'Player {player_id} Heatmap')
                    plt.xlabel('Field Length')
                    plt.ylabel('Field Width')
                    plt.savefig(os.path.join(output_dir, f'player_{player_id}_heatmap.png'), 
                              dpi=300, bbox_inches='tight')
                    plt.close()
                    
        except ImportError:
            print("Warning: matplotlib not available. Cannot export heatmap images.")
        except Exception as e:
            print(f"Error exporting heatmap images: {e}")
    
    def reset_analytics(self):
        """Reset all analytics data for new match"""
        self.match_stats = MatchStats()
        self.player_trajectories.clear()
        self.team_assignments.clear()
        self.start_time = time.time()
        self.last_frame_time = 0.0
        self._initialize_heatmaps()