"""
Data exporter for structured output of analytics data
"""

import json
import csv
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from ..core.interfaces import BaseExporter
from ..core.models import ProcessingResults, FrameResults, TrackedObject, PlayerStats, TeamStats
from ..core.exceptions import ExportError
from ..analytics.engine import AnalyticsEngine


class DataExporter(BaseExporter):
    """
    Data exporter for structured output of football analytics data
    Supports JSON, CSV, and video export formats
    """
    
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize data exporter
        
        Args:
            output_dir: Directory for export files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_json(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Export data to JSON format with validation
        
        Args:
            data: Data dictionary to export
            filepath: Output file path
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Ensure filepath is within output directory
            full_path = self.output_dir / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = self._make_json_serializable(data)
            
            # Add metadata
            export_metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '1.0',
                'data_type': 'football_analytics'
            }
            
            final_data = {
                'metadata': export_metadata,
                'data': serializable_data
            }
            
            # Validate data structure
            self._validate_json_data(final_data)
            
            # Write to file
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            raise ExportError(f"Failed to export JSON to {filepath}: {str(e)}")
    
    def export_csv(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Export data to CSV format for statistical analysis
        
        Args:
            data: Data dictionary to export
            filepath: Output file path
            
        Raises:
            ExportError: If export fails
        """
        try:
            full_path = self.output_dir / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert data to DataFrame format
            df = self._convert_to_dataframe(data)
            
            # Validate DataFrame
            self._validate_csv_data(df)
            
            # Export to CSV
            df.to_csv(full_path, index=False, encoding='utf-8')
            
        except Exception as e:
            raise ExportError(f"Failed to export CSV to {filepath}: {str(e)}")
    
    def export_video(self, frames: List[np.ndarray], filepath: str, fps: int = 30) -> None:
        """
        Export processed frames as video
        
        Args:
            frames: List of processed frames
            filepath: Output video path
            fps: Frames per second
            
        Raises:
            ExportError: If export fails
        """
        try:
            import cv2
            
            if not frames:
                raise ValueError("No frames to export")
            
            full_path = self.output_dir / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(full_path), fourcc, fps, (width, height))
            
            # Write frames
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
            
            out.release()
            
        except Exception as e:
            raise ExportError(f"Failed to export video to {filepath}: {str(e)}")
    
    def export_processing_results(self, results: ProcessingResults, 
                                base_filename: str = "match_analysis") -> Dict[str, str]:
        """
        Export complete processing results in multiple formats
        
        Args:
            results: Processing results to export
            base_filename: Base filename for exports
            
        Returns:
            Dictionary mapping format to exported file path
            
        Raises:
            ExportError: If export fails
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_paths = {}
            
            # Export comprehensive JSON
            json_filename = f"{base_filename}_{timestamp}.json"
            comprehensive_data = self._extract_comprehensive_data(results)
            self.export_json(comprehensive_data, json_filename)
            export_paths['json'] = str(self.output_dir / json_filename)
            
            # Export player statistics CSV
            player_csv_filename = f"{base_filename}_players_{timestamp}.csv"
            player_data = self._extract_player_data(results)
            self.export_csv(player_data, player_csv_filename)
            export_paths['player_csv'] = str(self.output_dir / player_csv_filename)
            
            # Export team statistics CSV
            team_csv_filename = f"{base_filename}_teams_{timestamp}.csv"
            team_data = self._extract_team_data(results)
            self.export_csv(team_data, team_csv_filename)
            export_paths['team_csv'] = str(self.output_dir / team_csv_filename)
            
            # Export frame-by-frame data CSV
            frames_csv_filename = f"{base_filename}_frames_{timestamp}.csv"
            frames_data = self._extract_frames_data(results)
            self.export_csv(frames_data, frames_csv_filename)
            export_paths['frames_csv'] = str(self.output_dir / frames_csv_filename)
            
            return export_paths
            
        except Exception as e:
            raise ExportError(f"Failed to export processing results: {str(e)}")
    
    def export_analytics_engine_data(self, engine: AnalyticsEngine, 
                                   base_filename: str = "analytics") -> Dict[str, str]:
        """
        Export analytics engine data in multiple formats
        
        Args:
            engine: Analytics engine instance
            base_filename: Base filename for exports
            
        Returns:
            Dictionary mapping format to exported file path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_paths = {}
            
            # Export match summary
            match_summary = engine.get_match_summary()
            json_filename = f"{base_filename}_summary_{timestamp}.json"
            self.export_json(match_summary, json_filename)
            export_paths['summary_json'] = str(self.output_dir / json_filename)
            
            # Export detailed player statistics
            player_stats_data = self._extract_engine_player_data(engine)
            if player_stats_data:
                player_csv_filename = f"{base_filename}_detailed_players_{timestamp}.csv"
                self.export_csv(player_stats_data, player_csv_filename)
                export_paths['detailed_player_csv'] = str(self.output_dir / player_csv_filename)
            
            # Export heatmap data
            heatmap_data = self._extract_heatmap_data(engine)
            if heatmap_data:
                heatmap_filename = f"{base_filename}_heatmaps_{timestamp}.json"
                self.export_json(heatmap_data, heatmap_filename)
                export_paths['heatmaps_json'] = str(self.output_dir / heatmap_filename)
            
            # Export formation analysis
            formation_data = self._extract_formation_data(engine)
            if formation_data:
                formation_filename = f"{base_filename}_formations_{timestamp}.json"
                self.export_json(formation_data, formation_filename)
                export_paths['formations_json'] = str(self.output_dir / formation_filename)
            
            return export_paths
            
        except Exception as e:
            raise ExportError(f"Failed to export analytics engine data: {str(e)}")
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        else:
            return data
    
    def _validate_json_data(self, data: Dict[str, Any]) -> None:
        """Validate JSON data structure"""
        if not isinstance(data, dict):
            raise ValueError("JSON data must be a dictionary")
        
        if 'metadata' not in data:
            raise ValueError("JSON data must contain metadata")
        
        if 'data' not in data:
            raise ValueError("JSON data must contain data section")
    
    def _validate_csv_data(self, df: pd.DataFrame) -> None:
        """Validate CSV DataFrame"""
        if df.empty:
            raise ValueError("CSV data cannot be empty")
        
        # Check for required columns based on data type
        if 'player_id' in df.columns:
            required_cols = ['player_id', 'team_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _convert_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert data dictionary to pandas DataFrame"""
        # Handle different data structures
        if 'players' in data:
            return self._players_to_dataframe(data['players'])
        elif 'teams' in data:
            return self._teams_to_dataframe(data['teams'])
        elif 'frames' in data:
            return self._frames_to_dataframe(data['frames'])
        else:
            # Generic conversion for flat dictionaries
            return pd.DataFrame([data])
    
    def _players_to_dataframe(self, players_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert players data to DataFrame"""
        rows = []
        for player_id, stats in players_data.items():
            row = {'player_id': player_id}
            row.update(stats)
            rows.append(row)
        return pd.DataFrame(rows)
    
    def _teams_to_dataframe(self, teams_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert teams data to DataFrame"""
        rows = []
        for team_id, stats in teams_data.items():
            row = {'team_id': team_id}
            row.update(stats)
            rows.append(row)
        return pd.DataFrame(rows)
    
    def _frames_to_dataframe(self, frames_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert frames data to DataFrame"""
        return pd.DataFrame(frames_data)
    
    def _extract_comprehensive_data(self, results: ProcessingResults) -> Dict[str, Any]:
        """Extract comprehensive data from processing results"""
        return {
            'processing_info': {
                'total_frames': results.total_frames,
                'processing_time': results.processing_time,
                'fps': results.total_frames / results.processing_time if results.processing_time > 0 else 0
            },
            'analytics_data': results.analytics_data,
            'frame_count': len(results.frame_results),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def _extract_player_data(self, results: ProcessingResults) -> Dict[str, Any]:
        """Extract player statistics for CSV export"""
        players_data = {}
        
        if 'players' in results.analytics_data:
            players_data = {'players': results.analytics_data['players']}
        
        return players_data
    
    def _extract_team_data(self, results: ProcessingResults) -> Dict[str, Any]:
        """Extract team statistics for CSV export"""
        teams_data = {}
        
        if 'teams' in results.analytics_data:
            teams_data = {'teams': results.analytics_data['teams']}
        
        return teams_data
    
    def _extract_frames_data(self, results: ProcessingResults) -> Dict[str, Any]:
        """Extract frame-by-frame data for CSV export"""
        frames_data = []
        
        for frame_result in results.frame_results:
            frame_data = {
                'frame_id': frame_result.frame_id,
                'timestamp': frame_result.timestamp,
                'players_detected': len(frame_result.tracked_objects),
                'field_lines_detected': len(frame_result.field_lines),
                'keypoints_detected': len(frame_result.key_points),
                'ball_detected': frame_result.ball_position is not None,
                'is_calibrated': frame_result.is_calibrated
            }
            
            # Add individual player data
            for i, obj in enumerate(frame_result.tracked_objects):
                frame_data[f'player_{i}_id'] = obj.track_id
                frame_data[f'player_{i}_team'] = obj.team_id
                frame_data[f'player_{i}_confidence'] = obj.detection.confidence
                if obj.field_position:
                    frame_data[f'player_{i}_field_x'] = obj.field_position[0]
                    frame_data[f'player_{i}_field_y'] = obj.field_position[1]
                if obj.velocity:
                    frame_data[f'player_{i}_velocity'] = obj.velocity
            
            frames_data.append(frame_data)
        
        return {'frames': frames_data}
    
    def _extract_engine_player_data(self, engine: AnalyticsEngine) -> Optional[Dict[str, Any]]:
        """Extract detailed player data from analytics engine"""
        if not engine.match_stats.players_detected:
            return None
        
        players_data = {}
        for player_id, stats in engine.match_stats.players_detected.items():
            players_data[str(player_id)] = {
                'team_id': stats.team_id,
                'total_distance': stats.total_distance,
                'max_velocity': stats.max_velocity,
                'avg_velocity': stats.avg_velocity,
                'time_on_field': stats.time_on_field,
                'positions_count': len(stats.positions),
                'velocities_count': len(stats.velocities),
                'last_seen_frame': stats.last_seen_frame
            }
        
        return {'players': players_data}
    
    def _extract_heatmap_data(self, engine: AnalyticsEngine) -> Optional[Dict[str, Any]]:
        """Extract heatmap data from analytics engine"""
        heatmap_data = {}
        
        # Team heatmaps
        for team_id in engine.match_stats.heatmap_data.keys():
            normalized_heatmap = engine.generate_normalized_heatmap(team_id)
            if normalized_heatmap is not None:
                heatmap_data[f'team_{team_id}'] = normalized_heatmap.tolist()
        
        # Individual player heatmaps
        for player_id in engine.match_stats.players_detected.keys():
            player_heatmap = engine.generate_player_heatmap(player_id)
            if player_heatmap is not None:
                heatmap_data[f'player_{player_id}'] = player_heatmap.tolist()
        
        return heatmap_data if heatmap_data else None
    
    def _extract_formation_data(self, engine: AnalyticsEngine) -> Optional[Dict[str, Any]]:
        """Extract formation analysis data from analytics engine"""
        formation_data = {}
        
        for team_id in engine.match_stats.teams.keys():
            formation_analysis = engine.analyze_team_formation(team_id)
            if 'error' not in formation_analysis:
                formation_data[f'team_{team_id}'] = formation_analysis
        
        return formation_data if formation_data else None
    
    def get_export_summary(self, export_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate summary of exported files
        
        Args:
            export_paths: Dictionary of exported file paths
            
        Returns:
            Summary information about exports
        """
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_files': len(export_paths),
            'files': {}
        }
        
        for format_type, filepath in export_paths.items():
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                summary['files'][format_type] = {
                    'filepath': filepath,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                }
            else:
                summary['files'][format_type] = {
                    'filepath': filepath,
                    'error': 'File not found'
                }
        
        return summary