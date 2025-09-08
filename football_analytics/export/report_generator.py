"""
Report generator for comprehensive match analysis and visualization exports
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ..core.models import ProcessingResults, PlayerStats, TeamStats
from ..core.exceptions import ExportError
from ..analytics.engine import AnalyticsEngine


class ReportGenerator:
    """
    Comprehensive report generator for football match analysis
    Generates match summaries, player performance reports, and visualization exports
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for report files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        if HAS_SEABORN:
            sns.set_palette("husl")
    
    def generate_match_summary(self, engine: AnalyticsEngine, 
                             match_name: str = "Football Match") -> Dict[str, Any]:
        """
        Generate comprehensive match summary with key statistics
        
        Args:
            engine: Analytics engine with match data
            match_name: Name of the match
            
        Returns:
            Dictionary containing match summary data
        """
        try:
            match_summary = engine.get_match_summary()
            
            # Enhanced summary with additional analysis
            enhanced_summary = {
                'match_info': {
                    'match_name': match_name,
                    'analysis_timestamp': datetime.now().isoformat(),
                    **match_summary['match_info']
                },
                'overview': self._generate_match_overview(match_summary),
                'team_comparison': self._generate_team_comparison(match_summary),
                'top_performers': self._identify_top_performers(match_summary),
                'tactical_analysis': self._generate_tactical_analysis(engine),
                'key_statistics': self._calculate_key_statistics(match_summary),
                'recommendations': self._generate_recommendations(match_summary)
            }
            
            return enhanced_summary
            
        except Exception as e:
            raise ExportError(f"Failed to generate match summary: {str(e)}")
    
    def generate_player_performance_report(self, engine: AnalyticsEngine, 
                                         player_id: int) -> Dict[str, Any]:
        """
        Generate detailed performance report for individual player
        
        Args:
            engine: Analytics engine with match data
            player_id: ID of player to analyze
            
        Returns:
            Dictionary containing player performance data
        """
        try:
            player_stats = engine.get_player_stats(player_id)
            if not player_stats:
                raise ValueError(f"Player {player_id} not found in match data")
            
            # Generate movement patterns analysis
            movement_analysis = engine.generate_movement_patterns(player_id)
            
            # Generate individual heatmap
            player_heatmap = engine.generate_player_heatmap(player_id)
            
            performance_report = {
                'player_info': {
                    'player_id': player_id,
                    'team_id': player_stats.team_id,
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'performance_metrics': {
                    'total_distance_m': player_stats.total_distance,
                    'max_velocity_ms': player_stats.max_velocity,
                    'avg_velocity_ms': player_stats.avg_velocity,
                    'time_on_field_s': player_stats.time_on_field,
                    'time_on_field_min': player_stats.time_on_field / 60.0,
                    'positions_recorded': len(player_stats.positions)
                },
                'movement_analysis': movement_analysis,
                'activity_zones': self._analyze_player_activity_zones(player_stats),
                'performance_rating': self._calculate_player_rating(player_stats),
                'comparison_to_team': self._compare_player_to_team(engine, player_id),
                'recommendations': self._generate_player_recommendations(player_stats, movement_analysis)
            }
            
            return performance_report
            
        except Exception as e:
            raise ExportError(f"Failed to generate player performance report: {str(e)}")
    
    def generate_team_analysis_report(self, engine: AnalyticsEngine, 
                                    team_id: int) -> Dict[str, Any]:
        """
        Generate comprehensive team analysis report
        
        Args:
            engine: Analytics engine with match data
            team_id: ID of team to analyze
            
        Returns:
            Dictionary containing team analysis data
        """
        try:
            team_stats = engine.get_team_stats(team_id)
            if not team_stats:
                raise ValueError(f"Team {team_id} not found in match data")
            
            # Generate formation analysis
            formation_analysis = engine.analyze_team_formation(team_id)
            
            # Generate spatial dominance analysis
            spatial_analysis = engine.analyze_spatial_dominance(team_id)
            
            team_report = {
                'team_info': {
                    'team_id': team_id,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'player_count': team_stats.player_count
                },
                'team_metrics': {
                    'total_distance_m': team_stats.total_distance,
                    'avg_velocity_ms': team_stats.avg_velocity,
                    'formation_center': team_stats.formation_center,
                    'formation_spread': team_stats.formation_spread
                },
                'formation_analysis': formation_analysis,
                'spatial_dominance': spatial_analysis,
                'player_distribution': self._analyze_player_distribution(engine, team_id),
                'tactical_insights': self._generate_tactical_insights(formation_analysis, spatial_analysis),
                'team_rating': self._calculate_team_rating(team_stats, formation_analysis),
                'recommendations': self._generate_team_recommendations(team_stats, formation_analysis)
            }
            
            return team_report
            
        except Exception as e:
            raise ExportError(f"Failed to generate team analysis report: {str(e)}")
    
    def create_match_visualization_exports(self, engine: AnalyticsEngine, 
                                         base_filename: str = "match_viz") -> Dict[str, str]:
        """
        Create comprehensive visualization exports for match analysis
        
        Args:
            engine: Analytics engine with match data
            base_filename: Base filename for visualization files
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_paths = {}
            
            # Team heatmaps
            for team_id in engine.match_stats.teams.keys():
                heatmap_path = self._create_team_heatmap(
                    engine, team_id, f"{base_filename}_team_{team_id}_heatmap_{timestamp}.png"
                )
                if heatmap_path:
                    viz_paths[f'team_{team_id}_heatmap'] = heatmap_path
            
            # Formation comparison
            formation_path = self._create_formation_comparison(
                engine, f"{base_filename}_formations_{timestamp}.png"
            )
            if formation_path:
                viz_paths['formations'] = formation_path
            
            # Player performance comparison
            performance_path = self._create_player_performance_chart(
                engine, f"{base_filename}_player_performance_{timestamp}.png"
            )
            if performance_path:
                viz_paths['player_performance'] = performance_path
            
            # Team statistics comparison
            team_stats_path = self._create_team_statistics_chart(
                engine, f"{base_filename}_team_stats_{timestamp}.png"
            )
            if team_stats_path:
                viz_paths['team_statistics'] = team_stats_path
            
            # Movement patterns visualization
            movement_path = self._create_movement_patterns_viz(
                engine, f"{base_filename}_movement_patterns_{timestamp}.png"
            )
            if movement_path:
                viz_paths['movement_patterns'] = movement_path
            
            return viz_paths
            
        except Exception as e:
            raise ExportError(f"Failed to create visualization exports: {str(e)}")
    
    def create_player_visualization_exports(self, engine: AnalyticsEngine, 
                                          player_id: int,
                                          base_filename: str = "player_viz") -> Dict[str, str]:
        """
        Create visualization exports for individual player analysis
        
        Args:
            engine: Analytics engine with match data
            player_id: ID of player to visualize
            base_filename: Base filename for visualization files
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_paths = {}
            
            # Individual player heatmap
            heatmap_path = self._create_player_heatmap(
                engine, player_id, f"{base_filename}_player_{player_id}_heatmap_{timestamp}.png"
            )
            if heatmap_path:
                viz_paths['heatmap'] = heatmap_path
            
            # Player trajectory visualization
            trajectory_path = self._create_player_trajectory(
                engine, player_id, f"{base_filename}_player_{player_id}_trajectory_{timestamp}.png"
            )
            if trajectory_path:
                viz_paths['trajectory'] = trajectory_path
            
            # Player performance metrics
            metrics_path = self._create_player_metrics_chart(
                engine, player_id, f"{base_filename}_player_{player_id}_metrics_{timestamp}.png"
            )
            if metrics_path:
                viz_paths['metrics'] = metrics_path
            
            return viz_paths
            
        except Exception as e:
            raise ExportError(f"Failed to create player visualization exports: {str(e)}")
    
    def export_comprehensive_report(self, engine: AnalyticsEngine, 
                                  match_name: str = "Football Match",
                                  include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Export comprehensive match report with all analysis and visualizations
        
        Args:
            engine: Analytics engine with match data
            match_name: Name of the match
            include_visualizations: Whether to include visualization exports
            
        Returns:
            Dictionary containing all export paths and summary
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_summary = {
                'export_timestamp': datetime.now().isoformat(),
                'match_name': match_name,
                'reports': {},
                'visualizations': {}
            }
            
            # Generate match summary report
            match_summary = self.generate_match_summary(engine, match_name)
            summary_path = self.output_dir / f"match_summary_{timestamp}.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(match_summary, f, indent=2, ensure_ascii=False)
            export_summary['reports']['match_summary'] = str(summary_path)
            
            # Generate team analysis reports
            for team_id in engine.match_stats.teams.keys():
                team_report = self.generate_team_analysis_report(engine, team_id)
                team_path = self.output_dir / f"team_{team_id}_analysis_{timestamp}.json"
                with open(team_path, 'w', encoding='utf-8') as f:
                    json.dump(team_report, f, indent=2, ensure_ascii=False)
                export_summary['reports'][f'team_{team_id}'] = str(team_path)
            
            # Generate player performance reports
            player_reports = {}
            for player_id in engine.match_stats.players_detected.keys():
                try:
                    player_report = self.generate_player_performance_report(engine, player_id)
                    player_path = self.output_dir / f"player_{player_id}_performance_{timestamp}.json"
                    with open(player_path, 'w', encoding='utf-8') as f:
                        json.dump(player_report, f, indent=2, ensure_ascii=False)
                    player_reports[str(player_id)] = str(player_path)
                except Exception as e:
                    print(f"Warning: Failed to generate report for player {player_id}: {e}")
            
            export_summary['reports']['players'] = player_reports
            
            # Generate visualizations if requested
            if include_visualizations:
                match_viz = self.create_match_visualization_exports(engine, f"match_{timestamp}")
                export_summary['visualizations']['match'] = match_viz
                
                # Individual player visualizations for top performers
                top_performers = self._identify_top_performers(engine.get_match_summary())
                player_viz = {}
                for category, players in top_performers.items():
                    if isinstance(players, dict) and 'player_id' in players:
                        player_id = players['player_id']
                        try:
                            viz_paths = self.create_player_visualization_exports(
                                engine, player_id, f"top_performer_{category}_{timestamp}"
                            )
                            player_viz[f"{category}_player_{player_id}"] = viz_paths
                        except Exception as e:
                            print(f"Warning: Failed to create visualizations for player {player_id}: {e}")
                
                export_summary['visualizations']['players'] = player_viz
            
            return export_summary
            
        except Exception as e:
            raise ExportError(f"Failed to export comprehensive report: {str(e)}")
    
    # Helper methods for analysis
    
    def _generate_match_overview(self, match_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level match overview"""
        match_info = match_summary.get('match_info', {})
        players_data = match_summary.get('players', {})
        teams_data = match_summary.get('teams', {})
        
        total_players = len(players_data)
        total_distance = sum(p.get('total_distance_m', 0) for p in players_data.values())
        avg_match_velocity = np.mean([p.get('avg_velocity_ms', 0) for p in players_data.values()])
        
        return {
            'total_players_detected': total_players,
            'total_distance_covered_m': total_distance,
            'average_match_velocity_ms': avg_match_velocity,
            'match_duration_min': match_info.get('duration_minutes', 0),
            'processing_fps': match_info.get('fps', 0),
            'teams_detected': len(teams_data)
        }
    
    def _generate_team_comparison(self, match_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate team comparison analysis"""
        teams_data = match_summary.get('teams', {})
        
        if len(teams_data) < 2:
            return {'error': 'Insufficient team data for comparison'}
        
        team_metrics = {}
        for team_id, stats in teams_data.items():
            team_metrics[team_id] = {
                'player_count': stats.get('player_count', 0),
                'total_distance': stats.get('total_distance_m', 0),
                'avg_velocity': stats.get('avg_velocity_ms', 0),
                'formation_spread': stats.get('formation_spread', 0)
            }
        
        # Calculate differences
        team_ids = list(team_metrics.keys())
        if len(team_ids) >= 2:
            team1, team2 = team_ids[0], team_ids[1]
            comparison = {
                'distance_difference_m': team_metrics[team1]['total_distance'] - team_metrics[team2]['total_distance'],
                'velocity_difference_ms': team_metrics[team1]['avg_velocity'] - team_metrics[team2]['avg_velocity'],
                'formation_spread_difference': team_metrics[team1]['formation_spread'] - team_metrics[team2]['formation_spread']
            }
            
            return {
                'team_metrics': team_metrics,
                'comparison': comparison,
                'dominant_team': team1 if comparison['distance_difference_m'] > 0 else team2
            }
        
        return {'team_metrics': team_metrics}
    
    def _identify_top_performers(self, match_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Identify top performing players in various categories"""
        players_data = match_summary.get('players', {})
        
        if not players_data:
            return {}
        
        # Find top performers in different categories
        top_performers = {}
        
        # Most distance covered
        max_distance_player = max(players_data.items(), 
                                key=lambda x: x[1].get('total_distance_m', 0))
        top_performers['most_distance'] = {
            'player_id': int(max_distance_player[0]),
            'value': max_distance_player[1].get('total_distance_m', 0),
            'metric': 'meters'
        }
        
        # Highest max velocity
        max_velocity_player = max(players_data.items(), 
                               key=lambda x: x[1].get('max_velocity_ms', 0))
        top_performers['highest_velocity'] = {
            'player_id': int(max_velocity_player[0]),
            'value': max_velocity_player[1].get('max_velocity_ms', 0),
            'metric': 'm/s'
        }
        
        # Most consistent (highest average velocity)
        most_consistent_player = max(players_data.items(), 
                                   key=lambda x: x[1].get('avg_velocity_ms', 0))
        top_performers['most_consistent'] = {
            'player_id': int(most_consistent_player[0]),
            'value': most_consistent_player[1].get('avg_velocity_ms', 0),
            'metric': 'm/s average'
        }
        
        return top_performers
    
    def _generate_tactical_analysis(self, engine: AnalyticsEngine) -> Dict[str, Any]:
        """Generate tactical analysis from formation and spatial data"""
        tactical_analysis = {}
        
        for team_id in engine.match_stats.teams.keys():
            formation_analysis = engine.analyze_team_formation(team_id)
            spatial_analysis = engine.analyze_spatial_dominance(team_id)
            
            if 'error' not in formation_analysis and 'error' not in spatial_analysis:
                tactical_analysis[f'team_{team_id}'] = {
                    'formation_style': self._classify_formation_style(formation_analysis),
                    'playing_style': self._classify_playing_style(spatial_analysis),
                    'tactical_discipline': self._assess_tactical_discipline(formation_analysis),
                    'field_dominance': spatial_analysis.get('dominance_distribution', {})
                }
        
        return tactical_analysis
    
    def _calculate_key_statistics(self, match_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key match statistics"""
        players_data = match_summary.get('players', {})
        
        if not players_data:
            return {}
        
        distances = [p.get('total_distance_m', 0) for p in players_data.values()]
        velocities = [p.get('avg_velocity_ms', 0) for p in players_data.values()]
        max_velocities = [p.get('max_velocity_ms', 0) for p in players_data.values()]
        
        return {
            'distance_stats': {
                'total': sum(distances),
                'average': np.mean(distances),
                'std_dev': np.std(distances),
                'min': min(distances),
                'max': max(distances)
            },
            'velocity_stats': {
                'average': np.mean(velocities),
                'std_dev': np.std(velocities),
                'min': min(velocities),
                'max': max(velocities)
            },
            'max_velocity_stats': {
                'overall_max': max(max_velocities),
                'average_max': np.mean(max_velocities),
                'std_dev': np.std(max_velocities)
            }
        }
    
    def _generate_recommendations(self, match_summary: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on match analysis"""
        recommendations = []
        
        players_data = match_summary.get('players', {})
        teams_data = match_summary.get('teams', {})
        
        if not players_data or not teams_data:
            return ["Insufficient data for recommendations"]
        
        # Analyze team performance differences
        if len(teams_data) >= 2:
            team_ids = list(teams_data.keys())
            team1_distance = teams_data[team_ids[0]].get('total_distance_m', 0)
            team2_distance = teams_data[team_ids[1]].get('total_distance_m', 0)
            
            if abs(team1_distance - team2_distance) > 1000:  # Significant difference
                dominant_team = team_ids[0] if team1_distance > team2_distance else team_ids[1]
                recommendations.append(
                    f"Team {dominant_team} showed significantly higher work rate. "
                    f"The other team should focus on improving fitness and movement intensity."
                )
        
        # Analyze velocity distribution
        velocities = [p.get('avg_velocity_ms', 0) for p in players_data.values()]
        velocity_std = np.std(velocities)
        
        if velocity_std > 1.0:  # High variation in player velocities
            recommendations.append(
                "High variation in player movement speeds detected. "
                "Consider tactical adjustments to ensure more consistent team movement patterns."
            )
        
        # Check for low-activity players
        low_activity_players = [
            pid for pid, stats in players_data.items()
            if stats.get('avg_velocity_ms', 0) < np.mean(velocities) * 0.7
        ]
        
        if len(low_activity_players) > 2:
            recommendations.append(
                f"Multiple players ({len(low_activity_players)}) showing below-average movement. "
                f"Focus on improving positioning and involvement in play."
            )
        
        return recommendations
    
    def _analyze_player_activity_zones(self, player_stats) -> Dict[str, Any]:
        """Analyze player activity zones and positioning"""
        if not player_stats.positions:
            return {'error': 'No position data available'}
        
        positions = np.array(player_stats.positions)
        
        # Calculate activity center and spread
        center = np.mean(positions, axis=0)
        distances_from_center = np.linalg.norm(positions - center, axis=1)
        
        return {
            'activity_center': center.tolist(),
            'activity_radius': float(np.mean(distances_from_center)),
            'max_distance_from_center': float(np.max(distances_from_center)),
            'position_consistency': float(1.0 - (np.std(distances_from_center) / np.mean(distances_from_center)))
        }
    
    def _calculate_player_rating(self, player_stats) -> Dict[str, Any]:
        """Calculate overall player performance rating"""
        # Normalize metrics (these would be calibrated based on match data)
        distance_score = min(player_stats.total_distance / 1000.0, 10.0)  # Max 10 for 1km+
        velocity_score = min(player_stats.avg_velocity * 2.0, 10.0)  # Max 10 for 5 m/s avg
        consistency_score = min(len(player_stats.positions) / 100.0, 10.0)  # Max 10 for 100+ positions
        
        overall_rating = (distance_score + velocity_score + consistency_score) / 3.0
        
        return {
            'overall_rating': float(overall_rating),
            'distance_score': float(distance_score),
            'velocity_score': float(velocity_score),
            'consistency_score': float(consistency_score),
            'rating_scale': '0-10 (10 being excellent)'
        }
    
    def _compare_player_to_team(self, engine: AnalyticsEngine, player_id: int) -> Dict[str, Any]:
        """Compare individual player performance to team averages"""
        player_stats = engine.get_player_stats(player_id)
        if not player_stats:
            return {'error': 'Player not found'}
        
        # Get team players for comparison
        team_players = [
            stats for stats in engine.match_stats.players_detected.values()
            if stats.team_id == player_stats.team_id and stats.player_id != player_id
        ]
        
        if not team_players:
            return {'error': 'No team data for comparison'}
        
        team_avg_distance = np.mean([p.total_distance for p in team_players])
        team_avg_velocity = np.mean([p.avg_velocity for p in team_players])
        
        return {
            'distance_vs_team_avg': {
                'player': player_stats.total_distance,
                'team_avg': float(team_avg_distance),
                'difference_pct': float((player_stats.total_distance - team_avg_distance) / team_avg_distance * 100)
            },
            'velocity_vs_team_avg': {
                'player': player_stats.avg_velocity,
                'team_avg': float(team_avg_velocity),
                'difference_pct': float((player_stats.avg_velocity - team_avg_velocity) / team_avg_velocity * 100)
            }
        }
    
    def _generate_player_recommendations(self, player_stats, movement_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for individual player improvement"""
        recommendations = []
        
        # Check movement intensity
        if player_stats.avg_velocity < 2.0:  # Low average velocity
            recommendations.append(
                "Increase movement intensity and involvement in play. "
                "Focus on making more runs and positioning adjustments."
            )
        
        # Check activity consistency
        if 'activity_zone' in movement_analysis:
            coverage = movement_analysis['activity_zone'].get('field_coverage', 0)
            if coverage < 0.3:  # Low field coverage
                recommendations.append(
                    "Expand field coverage and positioning variety. "
                    "Consider making more diverse runs across different areas."
                )
        
        # Check direction consistency
        if 'direction_analysis' in movement_analysis:
            consistency = movement_analysis['direction_analysis'].get('consistency', 0)
            if consistency > 0.8:  # Very predictable movement
                recommendations.append(
                    "Movement patterns are highly predictable. "
                    "Vary running directions and positioning to be less predictable to opponents."
                )
        
        return recommendations
    
    def _analyze_player_distribution(self, engine: AnalyticsEngine, team_id: int) -> Dict[str, Any]:
        """Analyze distribution of players across the field"""
        team_players = [
            player for player in engine.match_stats.players_detected.values()
            if player.team_id == team_id and len(player.positions) > 0
        ]
        
        if len(team_players) < 2:
            return {'error': 'Insufficient player data'}
        
        # Calculate position statistics
        all_positions = []
        for player in team_players:
            all_positions.extend(player.positions)
        
        if not all_positions:
            return {'error': 'No position data available'}
        
        positions_array = np.array(all_positions)
        
        return {
            'field_coverage': {
                'x_range': float(np.max(positions_array[:, 0]) - np.min(positions_array[:, 0])),
                'y_range': float(np.max(positions_array[:, 1]) - np.min(positions_array[:, 1])),
                'center_of_mass': np.mean(positions_array, axis=0).tolist()
            },
            'player_spread': {
                'avg_distance_between_players': float(self._calculate_avg_player_distance(team_players)),
                'formation_compactness': float(np.std(np.linalg.norm(positions_array - np.mean(positions_array, axis=0), axis=1)))
            }
        }
    
    def _calculate_avg_player_distance(self, team_players: List) -> float:
        """Calculate average distance between team players"""
        if len(team_players) < 2:
            return 0.0
        
        # Get latest positions for each player
        latest_positions = []
        for player in team_players:
            if player.positions:
                latest_positions.append(player.positions[-1])
        
        if len(latest_positions) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(latest_positions)):
            for j in range(i + 1, len(latest_positions)):
                pos1, pos2 = latest_positions[i], latest_positions[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _generate_tactical_insights(self, formation_analysis: Dict[str, Any], 
                                  spatial_analysis: Dict[str, Any]) -> List[str]:
        """Generate tactical insights from formation and spatial data"""
        insights = []
        
        if 'error' in formation_analysis or 'error' in spatial_analysis:
            return ["Insufficient data for tactical analysis"]
        
        # Formation insights
        if 'positioning' in formation_analysis:
            tendency = formation_analysis['positioning'].get('tendency', 'neutral')
            if tendency == 'offensive':
                insights.append("Team shows offensive positioning bias - high attacking intent")
            elif tendency == 'defensive':
                insights.append("Team shows defensive positioning bias - focus on defensive stability")
        
        # Spatial dominance insights
        if 'dominance_distribution' in spatial_analysis:
            dominance = spatial_analysis['dominance_distribution']
            attacking_dominance = dominance.get('attacking_third', 0)
            defensive_dominance = dominance.get('defensive_third', 0)
            
            if attacking_dominance > 40:
                insights.append("Strong attacking third presence - effective offensive positioning")
            if defensive_dominance > 50:
                insights.append("Heavy defensive third focus - may indicate defensive strategy or pressure")
        
        return insights
    
    def _calculate_team_rating(self, team_stats, formation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall team performance rating"""
        # Base metrics
        distance_score = min(team_stats.total_distance / 10000.0, 10.0)  # Max 10 for 10km+ total
        velocity_score = min(team_stats.avg_velocity * 2.0, 10.0)  # Max 10 for 5 m/s avg
        
        # Formation metrics
        formation_score = 5.0  # Default neutral
        if 'error' not in formation_analysis:
            compactness = formation_analysis.get('formation_spread', {}).get('compactness', 0)
            formation_score = min(10.0 - compactness, 10.0)  # Lower compactness = better organization
        
        overall_rating = (distance_score + velocity_score + formation_score) / 3.0
        
        return {
            'overall_rating': float(overall_rating),
            'distance_score': float(distance_score),
            'velocity_score': float(velocity_score),
            'formation_score': float(formation_score),
            'rating_scale': '0-10 (10 being excellent)'
        }
    
    def _generate_team_recommendations(self, team_stats, formation_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for team improvement"""
        recommendations = []
        
        # Check team work rate
        if team_stats.avg_velocity < 2.5:
            recommendations.append(
                "Team average velocity is low. Focus on increasing overall movement intensity and work rate."
            )
        
        # Check formation discipline
        if 'error' not in formation_analysis:
            spread = formation_analysis.get('formation_spread', {}).get('compactness', 0)
            if spread > 15.0:  # High spread indicates poor organization
                recommendations.append(
                    "Formation shows high spread/low compactness. "
                    "Work on maintaining better positional discipline and team shape."
                )
        
        return recommendations
    
    # Visualization methods
    
    def _create_team_heatmap(self, engine: AnalyticsEngine, team_id: int, filename: str) -> Optional[str]:
        """Create team heatmap visualization"""
        try:
            heatmap_data = engine.generate_normalized_heatmap(team_id)
            if heatmap_data is None:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap
            im = ax.imshow(heatmap_data.T, cmap='Reds', aspect='auto', origin='lower')
            
            # Add field markings (simplified)
            self._add_field_markings(ax, heatmap_data.shape)
            
            ax.set_title(f'Team {team_id} Position Heatmap', fontsize=16, fontweight='bold')
            ax.set_xlabel('Field Length', fontsize=12)
            ax.set_ylabel('Field Width', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Activity Intensity', fontsize=12)
            
            plt.tight_layout()
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"Warning: Failed to create team heatmap: {e}")
            return None
    
    def _create_formation_comparison(self, engine: AnalyticsEngine, filename: str) -> Optional[str]:
        """Create formation comparison visualization"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            team_ids = list(engine.match_stats.teams.keys())
            if len(team_ids) < 2:
                return None
            
            for i, team_id in enumerate(team_ids[:2]):
                formation_analysis = engine.analyze_team_formation(team_id)
                if 'error' in formation_analysis:
                    continue
                
                ax = axes[i]
                
                # Plot player positions
                if 'player_positions' in formation_analysis:
                    positions = formation_analysis['player_positions']
                    x_coords = [p['x'] for p in positions]
                    y_coords = [p['y'] for p in positions]
                    
                    ax.scatter(x_coords, y_coords, s=100, alpha=0.7, 
                             c=f'C{team_id}', label=f'Team {team_id}')
                    
                    # Add player IDs
                    for pos in positions:
                        ax.annotate(str(pos['player_id']), 
                                  (pos['x'], pos['y']), 
                                  xytext=(5, 5), textcoords='offset points')
                
                # Add formation center
                if 'formation_center' in formation_analysis:
                    center = formation_analysis['formation_center']
                    ax.plot(center[0], center[1], 'x', markersize=15, 
                           markeredgewidth=3, color='red', label='Formation Center')
                
                ax.set_title(f'Team {team_id} Formation', fontsize=14, fontweight='bold')
                ax.set_xlabel('Field Length (m)', fontsize=12)
                ax.set_ylabel('Field Width (m)', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"Warning: Failed to create formation comparison: {e}")
            return None
    
    def _create_player_performance_chart(self, engine: AnalyticsEngine, filename: str) -> Optional[str]:
        """Create player performance comparison chart"""
        try:
            players_data = []
            for player_id, stats in engine.match_stats.players_detected.items():
                players_data.append({
                    'player_id': player_id,
                    'team_id': stats.team_id,
                    'total_distance': stats.total_distance,
                    'avg_velocity': stats.avg_velocity,
                    'max_velocity': stats.max_velocity
                })
            
            if not players_data:
                return None
            
            df = pd.DataFrame(players_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Distance by team
            df.boxplot(column='total_distance', by='team_id', ax=axes[0, 0])
            axes[0, 0].set_title('Total Distance by Team')
            axes[0, 0].set_ylabel('Distance (m)')
            
            # Average velocity by team
            df.boxplot(column='avg_velocity', by='team_id', ax=axes[0, 1])
            axes[0, 1].set_title('Average Velocity by Team')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            
            # Max velocity by team
            df.boxplot(column='max_velocity', by='team_id', ax=axes[1, 0])
            axes[1, 0].set_title('Max Velocity by Team')
            axes[1, 0].set_ylabel('Velocity (m/s)')
            
            # Scatter plot: distance vs velocity
            for team_id in df['team_id'].unique():
                team_data = df[df['team_id'] == team_id]
                axes[1, 1].scatter(team_data['total_distance'], team_data['avg_velocity'], 
                                 label=f'Team {team_id}', alpha=0.7, s=60)
            
            axes[1, 1].set_xlabel('Total Distance (m)')
            axes[1, 1].set_ylabel('Average Velocity (m/s)')
            axes[1, 1].set_title('Distance vs Velocity')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"Warning: Failed to create player performance chart: {e}")
            return None
    
    def _create_team_statistics_chart(self, engine: AnalyticsEngine, filename: str) -> Optional[str]:
        """Create team statistics comparison chart"""
        try:
            team_data = []
            for team_id, stats in engine.match_stats.teams.items():
                team_data.append({
                    'team_id': team_id,
                    'player_count': stats.player_count,
                    'total_distance': stats.total_distance,
                    'avg_velocity': stats.avg_velocity,
                    'formation_spread': stats.formation_spread
                })
            
            if len(team_data) < 2:
                return None
            
            df = pd.DataFrame(team_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Team comparison metrics
            metrics = ['player_count', 'total_distance', 'avg_velocity', 'formation_spread']
            titles = ['Player Count', 'Total Distance (m)', 'Average Velocity (m/s)', 'Formation Spread']
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[i // 2, i % 2]
                bars = ax.bar(df['team_id'].astype(str), df[metric], alpha=0.7)
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Team ID')
                ax.set_ylabel(title)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"Warning: Failed to create team statistics chart: {e}")
            return None
    
    def _create_movement_patterns_viz(self, engine: AnalyticsEngine, filename: str) -> Optional[str]:
        """Create movement patterns visualization"""
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Plot trajectories for a sample of players
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            plotted_players = 0
            for player_id, stats in engine.match_stats.players_detected.items():
                if plotted_players >= 8:  # Limit to avoid clutter
                    break
                
                if len(stats.positions) > 10:  # Only plot players with sufficient data
                    positions = np.array(stats.positions)
                    color = colors[plotted_players % len(colors)]
                    
                    ax.plot(positions[:, 0], positions[:, 1], 
                           color=color, alpha=0.6, linewidth=2,
                           label=f'Player {player_id} (Team {stats.team_id})')
                    
                    # Mark start and end positions
                    ax.scatter(positions[0, 0], positions[0, 1], 
                             color=color, s=100, marker='o', edgecolor='black')
                    ax.scatter(positions[-1, 0], positions[-1, 1], 
                             color=color, s=100, marker='s', edgecolor='black')
                    
                    plotted_players += 1
            
            ax.set_title('Player Movement Patterns', fontsize=16, fontweight='bold')
            ax.set_xlabel('Field Length (m)', fontsize=12)
            ax.set_ylabel('Field Width (m)', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"Warning: Failed to create movement patterns visualization: {e}")
            return None
    
    def _create_player_heatmap(self, engine: AnalyticsEngine, player_id: int, filename: str) -> Optional[str]:
        """Create individual player heatmap"""
        try:
            heatmap_data = engine.generate_player_heatmap(player_id)
            if heatmap_data is None:
                return None
            
            # Normalize heatmap
            if heatmap_data.max() > 0:
                heatmap_data = heatmap_data / heatmap_data.max()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            im = ax.imshow(heatmap_data.T, cmap='Blues', aspect='auto', origin='lower')
            
            self._add_field_markings(ax, heatmap_data.shape)
            
            player_stats = engine.get_player_stats(player_id)
            team_id = player_stats.team_id if player_stats else "Unknown"
            
            ax.set_title(f'Player {player_id} (Team {team_id}) Position Heatmap', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Field Length', fontsize=12)
            ax.set_ylabel('Field Width', fontsize=12)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Activity Intensity', fontsize=12)
            
            plt.tight_layout()
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"Warning: Failed to create player heatmap: {e}")
            return None
    
    def _create_player_trajectory(self, engine: AnalyticsEngine, player_id: int, filename: str) -> Optional[str]:
        """Create player trajectory visualization"""
        try:
            player_stats = engine.get_player_stats(player_id)
            if not player_stats or len(player_stats.positions) < 2:
                return None
            
            positions = np.array(player_stats.positions)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Plot trajectory with color gradient
            for i in range(len(positions) - 1):
                alpha = (i + 1) / len(positions)  # Fade from start to end
                ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], 
                       color='blue', alpha=alpha, linewidth=2)
            
            # Mark start and end
            ax.scatter(positions[0, 0], positions[0, 1], 
                      color='green', s=150, marker='o', 
                      edgecolor='black', linewidth=2, label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], 
                      color='red', s=150, marker='s', 
                      edgecolor='black', linewidth=2, label='End')
            
            # Add direction arrows at intervals
            arrow_interval = max(1, len(positions) // 10)
            for i in range(0, len(positions) - 1, arrow_interval):
                if i + 1 < len(positions):
                    dx = positions[i+1, 0] - positions[i, 0]
                    dy = positions[i+1, 1] - positions[i, 1]
                    ax.arrow(positions[i, 0], positions[i, 1], dx, dy,
                            head_width=2, head_length=2, fc='red', ec='red', alpha=0.6)
            
            ax.set_title(f'Player {player_id} (Team {player_stats.team_id}) Movement Trajectory', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Field Length (m)', fontsize=12)
            ax.set_ylabel('Field Width (m)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"Warning: Failed to create player trajectory: {e}")
            return None
    
    def _create_player_metrics_chart(self, engine: AnalyticsEngine, player_id: int, filename: str) -> Optional[str]:
        """Create player metrics visualization"""
        try:
            player_stats = engine.get_player_stats(player_id)
            if not player_stats:
                return None
            
            # Get movement analysis
            movement_analysis = engine.generate_movement_patterns(player_id)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Velocity over time (if available)
            if player_stats.velocities:
                axes[0, 0].plot(player_stats.velocities, color='blue', linewidth=2)
                axes[0, 0].set_title('Velocity Over Time')
                axes[0, 0].set_xlabel('Time Points')
                axes[0, 0].set_ylabel('Velocity (m/s)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Distance accumulation
            if player_stats.distances:
                cumulative_distance = np.cumsum(player_stats.distances)
                axes[0, 1].plot(cumulative_distance, color='green', linewidth=2)
                axes[0, 1].set_title('Cumulative Distance')
                axes[0, 1].set_xlabel('Time Points')
                axes[0, 1].set_ylabel('Distance (m)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Performance metrics bar chart
            metrics = {
                'Total Distance (m)': player_stats.total_distance,
                'Max Velocity (m/s)': player_stats.max_velocity,
                'Avg Velocity (m/s)': player_stats.avg_velocity,
                'Time on Field (min)': player_stats.time_on_field / 60.0
            }
            
            bars = axes[1, 0].bar(range(len(metrics)), list(metrics.values()), 
                                 color=['blue', 'red', 'green', 'orange'], alpha=0.7)
            axes[1, 0].set_xticks(range(len(metrics)))
            axes[1, 0].set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
            axes[1, 0].set_title('Performance Metrics')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics.values()):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}', ha='center', va='bottom')
            
            # Activity zone visualization
            if 'activity_zone' in movement_analysis and 'error' not in movement_analysis:
                activity_data = movement_analysis['activity_zone']
                center = activity_data['center']
                radius = activity_data['radius']
                
                # Create a simple activity zone plot
                circle = plt.Circle(center, radius, fill=False, color='red', linewidth=2)
                axes[1, 1].add_patch(circle)
                axes[1, 1].scatter(*center, color='red', s=100, marker='x')
                axes[1, 1].set_xlim(center[0] - radius * 1.5, center[0] + radius * 1.5)
                axes[1, 1].set_ylim(center[1] - radius * 1.5, center[1] + radius * 1.5)
                axes[1, 1].set_title('Activity Zone')
                axes[1, 1].set_xlabel('Field Length (m)')
                axes[1, 1].set_ylabel('Field Width (m)')
                axes[1, 1].set_aspect('equal')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"Warning: Failed to create player metrics chart: {e}")
            return None
    
    def _add_field_markings(self, ax, heatmap_shape: Tuple[int, int]):
        """Add simplified field markings to heatmap"""
        height, width = heatmap_shape
        
        # Center line
        ax.axvline(x=width/2, color='white', linestyle='--', alpha=0.7, linewidth=2)
        
        # Goal areas (approximate)
        goal_width = width * 0.1
        ax.axvline(x=goal_width, color='white', linestyle='-', alpha=0.5)
        ax.axvline(x=width - goal_width, color='white', linestyle='-', alpha=0.5)
        
        # Penalty areas (approximate)
        penalty_width = width * 0.2
        ax.axvline(x=penalty_width, color='white', linestyle='-', alpha=0.3)
        ax.axvline(x=width - penalty_width, color='white', linestyle='-', alpha=0.3)
    
    def _classify_formation_style(self, formation_analysis: Dict[str, Any]) -> str:
        """Classify team formation style based on analysis"""
        if 'error' in formation_analysis:
            return 'Unknown'
        
        # Simple classification based on formation spread and positioning
        spread = formation_analysis.get('formation_spread', {}).get('average', 0)
        positioning = formation_analysis.get('positioning', {}).get('tendency', 'neutral')
        
        if spread > 20:
            if positioning == 'offensive':
                return 'Attacking/Wide'
            else:
                return 'Spread/Defensive'
        elif spread < 10:
            return 'Compact/Organized'
        else:
            if positioning == 'offensive':
                return 'Balanced/Attacking'
            elif positioning == 'defensive':
                return 'Balanced/Defensive'
            else:
                return 'Balanced'
    
    def _classify_playing_style(self, spatial_analysis: Dict[str, Any]) -> str:
        """Classify team playing style based on spatial dominance"""
        if 'error' in spatial_analysis:
            return 'Unknown'
        
        dominance = spatial_analysis.get('dominance_distribution', {})
        attacking = dominance.get('attacking_third', 0)
        middle = dominance.get('middle_third', 0)
        defensive = dominance.get('defensive_third', 0)
        
        if attacking > 40:
            return 'Attacking/High Press'
        elif defensive > 50:
            return 'Defensive/Counter-Attack'
        elif middle > 40:
            return 'Possession/Build-up'
        else:
            return 'Balanced'
    
    def _assess_tactical_discipline(self, formation_analysis: Dict[str, Any]) -> str:
        """Assess tactical discipline based on formation consistency"""
        if 'error' in formation_analysis:
            return 'Unknown'
        
        compactness = formation_analysis.get('formation_spread', {}).get('compactness', 0)
        
        if compactness < 5:
            return 'Excellent'
        elif compactness < 10:
            return 'Good'
        elif compactness < 15:
            return 'Average'
        else:
            return 'Poor'