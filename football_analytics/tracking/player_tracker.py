"""
PlayerTracker class with improved ByteTrack integration for persistent player tracking
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time

from ..core.models import Detection, TrackedObject, Position
from ..core.config import TrackerConfig

try:
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import TrackState
except ImportError:
    print("Warning: YOLOX ByteTracker not found. Please install yolox package.")
    BYTETracker = None
    TrackState = None


@dataclass
class TrajectoryData:
    """Stores trajectory information for a tracked player"""
    positions: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    velocities: deque = field(default_factory=lambda: deque(maxlen=100))
    field_positions: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_position(self, position: Tuple[float, float], timestamp: float, 
                    field_position: Optional[Tuple[float, float]] = None):
        """Add a new position to the trajectory"""
        self.positions.append(position)
        self.timestamps.append(timestamp)
        if field_position:
            self.field_positions.append(field_position)
        
        # Calculate velocity if we have previous position
        if len(self.positions) >= 2 and len(self.timestamps) >= 2:
            prev_pos = self.positions[-2]
            curr_pos = self.positions[-1]
            time_diff = self.timestamps[-1] - self.timestamps[-2]
            
            if time_diff > 0:
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                 (curr_pos[1] - prev_pos[1])**2)
                velocity = distance / time_diff
                self.velocities.append(velocity)
    
    def get_recent_positions(self, n: int = 10) -> List[Tuple[float, float]]:
        """Get the n most recent positions"""
        return list(self.positions)[-n:]
    
    def get_average_velocity(self, n: int = 10) -> float:
        """Get average velocity over the last n measurements"""
        if not self.velocities:
            return 0.0
        recent_velocities = list(self.velocities)[-n:]
        return np.mean(recent_velocities) if recent_velocities else 0.0
    
    def get_total_distance(self) -> float:
        """Calculate total distance traveled"""
        if len(self.positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.positions)):
            prev_pos = self.positions[i-1]
            curr_pos = self.positions[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance
        
        return total_distance


class PlayerTracker:
    """
    Enhanced ByteTrack wrapper for persistent player tracking with trajectory management
    """
    
    def __init__(self, config: TrackerConfig):
        """
        Initialize the PlayerTracker
        
        Args:
            config: TrackerConfig instance with tracking parameters
        """
        self.config = config
        self.trajectories: Dict[int, TrajectoryData] = {}
        self.lost_tracks: Dict[int, TrajectoryData] = {}
        self.track_id_mapping: Dict[int, int] = {}  # Maps internal IDs to persistent IDs
        self.reverse_id_mapping: Dict[int, int] = {}  # Maps persistent IDs to internal IDs
        self.next_persistent_id = 1
        self.frame_count = 0
        
        # Persistence and recovery parameters
        self.max_lost_frames = config.track_buffer  # Frames to keep lost tracks
        self.lost_track_timeout = {}  # Track when tracks were lost
        self.interpolation_max_gap = 10  # Max frames to interpolate
        self.recovery_iou_threshold = 0.5  # IoU threshold for track recovery
        
        # Initialize ByteTracker if available
        if BYTETracker is not None:
            # Create args object for ByteTracker
            from types import SimpleNamespace
            tracker_args = SimpleNamespace(
                track_thresh=config.track_thresh,
                match_thresh=config.match_thresh,
                track_buffer=config.track_buffer,
                frame_rate=config.frame_rate,
                use_byte=True,
                mot20=False
            )
            self.byte_tracker = BYTETracker(tracker_args)
        else:
            self.byte_tracker = None
            print("Warning: ByteTracker not available, using fallback tracking")
        
        # Statistics
        self.stats = {
            'total_tracks_created': 0,
            'active_tracks': 0,
            'lost_tracks_recovered': 0,
            'frames_processed': 0
        }
    
    def update(self, detections: List[Detection], frame_shape: Tuple[int, int], 
               timestamp: Optional[float] = None) -> List[TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            frame_shape: (height, width) of the frame
            timestamp: Current timestamp (defaults to frame count)
            
        Returns:
            List of TrackedObject instances with persistent IDs
        """
        if timestamp is None:
            timestamp = self.frame_count
        
        self.frame_count += 1
        self.stats['frames_processed'] = self.frame_count
        
        if not detections:
            return []
        
        # Convert detections to ByteTracker format
        if self.byte_tracker is not None:
            dets_for_tracker = self._convert_detections_to_bytetrack_format(detections)
            
            # Update ByteTracker
            byte_tracks = self.byte_tracker.update(dets_for_tracker, frame_shape, frame_shape)
            
            # Convert back to TrackedObject format with persistent IDs
            tracked_objects = self._convert_bytetrack_to_tracked_objects(
                byte_tracks, detections, timestamp
            )
        else:
            # Fallback simple tracking
            tracked_objects = self._fallback_tracking(detections, timestamp)
        
        # Update trajectories
        self._update_trajectories(tracked_objects, timestamp)
        
        # Update statistics
        self.stats['active_tracks'] = len([t for t in tracked_objects if t.track_id > 0])
        
        return tracked_objects
    
    def _convert_detections_to_bytetrack_format(self, detections: List[Detection]) -> np.ndarray:
        """Convert Detection objects to ByteTracker format"""
        if not detections:
            return np.empty((0, 6))
        
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # ByteTracker expects [x1, y1, x2, y2, score, class_id]
            dets.append([x1, y1, x2, y2, det.confidence, det.class_id])
        
        return np.array(dets)
    
    def _convert_bytetrack_to_tracked_objects(self, byte_tracks: List, 
                                            detections: List[Detection], 
                                            timestamp: float) -> List[TrackedObject]:
        """Convert ByteTracker results to TrackedObject format"""
        tracked_objects = []
        
        for track in byte_tracks:
            # Get persistent ID for this track
            persistent_id = self._get_persistent_id(track.track_id)
            
            # Find matching detection using IoU
            best_detection = self._find_matching_detection(track, detections)
            
            if best_detection:
                # Calculate center position
                x1, y1, x2, y2 = best_detection.bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                tracked_obj = TrackedObject(
                    track_id=persistent_id,
                    detection=best_detection,
                    team_id=None,  # Will be set by team classifier
                    field_position=None,  # Will be set by field calibrator
                    velocity=self._get_current_velocity(persistent_id),
                    trajectory=self._get_recent_trajectory(persistent_id)
                )
                
                tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def _find_matching_detection(self, track, detections: List[Detection]) -> Optional[Detection]:
        """Find the detection that best matches the track using IoU"""
        if not detections:
            return None
        
        track_bbox = track.tlwh  # [x, y, w, h]
        track_x1, track_y1 = track_bbox[0], track_bbox[1]
        track_x2, track_y2 = track_x1 + track_bbox[2], track_y1 + track_bbox[3]
        
        best_detection = None
        best_iou = 0.0
        
        for detection in detections:
            det_x1, det_y1, det_x2, det_y2 = detection.bbox
            iou = self._calculate_iou(
                [track_x1, track_y1, track_x2, track_y2],
                [det_x1, det_y1, det_x2, det_y2]
            )
            
            if iou > best_iou:
                best_iou = iou
                best_detection = detection
        
        # Return detection if IoU is above threshold
        return best_detection if best_iou > 0.3 else None
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _get_persistent_id(self, byte_track_id: int) -> int:
        """Get or create persistent ID for a ByteTrack ID"""
        if byte_track_id not in self.track_id_mapping:
            # Check if this could be a recovered track
            recovered_id = self._attempt_track_recovery(byte_track_id)
            if recovered_id is not None:
                self.track_id_mapping[byte_track_id] = recovered_id
                self.reverse_id_mapping[recovered_id] = byte_track_id
                self.stats['lost_tracks_recovered'] += 1
                return recovered_id
            
            # Create new persistent ID
            self.track_id_mapping[byte_track_id] = self.next_persistent_id
            self.reverse_id_mapping[self.next_persistent_id] = byte_track_id
            self.next_persistent_id += 1
            self.stats['total_tracks_created'] += 1
        
        return self.track_id_mapping[byte_track_id]
    
    def _fallback_tracking(self, detections: List[Detection], timestamp: float) -> List[TrackedObject]:
        """Simple fallback tracking when ByteTracker is not available"""
        tracked_objects = []
        
        for i, detection in enumerate(detections):
            # Simple ID assignment based on detection order
            track_id = i + 1
            
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            tracked_obj = TrackedObject(
                track_id=track_id,
                detection=detection,
                team_id=None,
                field_position=None,
                velocity=0.0,
                trajectory=[]
            )
            
            tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def _update_trajectories(self, tracked_objects: List[TrackedObject], timestamp: float):
        """Update trajectory data for all tracked objects"""
        current_track_ids = set()
        
        for obj in tracked_objects:
            track_id = obj.track_id
            current_track_ids.add(track_id)
            
            # Initialize trajectory if new track
            if track_id not in self.trajectories:
                self.trajectories[track_id] = TrajectoryData()
            
            # Calculate center position
            x1, y1, x2, y2 = obj.detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check if we need to interpolate missing positions
            trajectory = self.trajectories[track_id]
            if trajectory.timestamps and len(trajectory.timestamps) > 0:
                last_timestamp = trajectory.timestamps[-1]
                frame_gap = timestamp - last_timestamp
                
                # Interpolate if gap is reasonable
                if 1 < frame_gap <= self.interpolation_max_gap:
                    self._interpolate_missing_positions(
                        track_id, last_timestamp, timestamp, 
                        trajectory.positions[-1], (center_x, center_y)
                    )
            
            # Add position to trajectory
            trajectory.add_position(
                (center_x, center_y), 
                timestamp, 
                obj.field_position
            )
            
            # Remove from lost tracks if it was there
            if track_id in self.lost_tracks:
                del self.lost_tracks[track_id]
            if track_id in self.lost_track_timeout:
                del self.lost_track_timeout[track_id]
        
        # Handle lost tracks
        self._handle_lost_tracks(current_track_ids, timestamp)
    
    def _get_current_velocity(self, track_id: int) -> float:
        """Get current velocity for a track"""
        if track_id in self.trajectories:
            return self.trajectories[track_id].get_average_velocity(n=5)
        return 0.0
    
    def _get_recent_trajectory(self, track_id: int, n: int = 20) -> List[Tuple[float, float]]:
        """Get recent trajectory points for a track"""
        if track_id in self.trajectories:
            return self.trajectories[track_id].get_recent_positions(n)
        return []
    
    def get_trajectory_history(self, track_id: int) -> Optional[TrajectoryData]:
        """Get complete trajectory history for a track"""
        return self.trajectories.get(track_id)
    
    def get_all_trajectories(self) -> Dict[int, TrajectoryData]:
        """Get all trajectory data"""
        return self.trajectories.copy()
    
    def get_track_statistics(self, track_id: int) -> Dict[str, Any]:
        """Get statistics for a specific track"""
        if track_id not in self.trajectories:
            return {}
        
        trajectory = self.trajectories[track_id]
        
        return {
            'track_id': track_id,
            'total_positions': len(trajectory.positions),
            'total_distance': trajectory.get_total_distance(),
            'average_velocity': trajectory.get_average_velocity(),
            'max_velocity': max(trajectory.velocities) if trajectory.velocities else 0.0,
            'first_seen': trajectory.timestamps[0] if trajectory.timestamps else None,
            'last_seen': trajectory.timestamps[-1] if trajectory.timestamps else None,
            'duration': (trajectory.timestamps[-1] - trajectory.timestamps[0]) 
                       if len(trajectory.timestamps) >= 2 else 0.0
        }
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global tracking statistics"""
        active_trajectories = len(self.trajectories)
        total_positions = sum(len(traj.positions) for traj in self.trajectories.values())
        
        stats = self.stats.copy()
        stats.update({
            'active_trajectories': active_trajectories,
            'total_positions_recorded': total_positions,
            'average_trajectory_length': total_positions / active_trajectories if active_trajectories > 0 else 0
        })
        
        return stats
    
    def reset(self):
        """Reset tracker state"""
        self.trajectories.clear()
        self.lost_tracks.clear()
        self.track_id_mapping.clear()
        self.reverse_id_mapping.clear()
        self.lost_track_timeout.clear()
        self.next_persistent_id = 1
        self.frame_count = 0
        
        if self.byte_tracker is not None:
            # Reset ByteTracker
            from types import SimpleNamespace
            tracker_args = SimpleNamespace(
                track_thresh=self.config.track_thresh,
                match_thresh=self.config.match_thresh,
                track_buffer=self.config.track_buffer,
                frame_rate=self.config.frame_rate,
                use_byte=True,
                mot20=False
            )
            self.byte_tracker = BYTETracker(tracker_args)
        
        # Reset statistics
        self.stats = {
            'total_tracks_created': 0,
            'active_tracks': 0,
            'lost_tracks_recovered': 0,
            'frames_processed': 0
        }
    
    def _attempt_track_recovery(self, new_byte_track_id: int) -> Optional[int]:
        """
        Attempt to recover a lost track by matching with recently lost tracks
        
        Args:
            new_byte_track_id: New ByteTrack ID to potentially match
            
        Returns:
            Persistent ID if recovery successful, None otherwise
        """
        if not self.lost_tracks:
            return None
        
        # For now, implement simple recovery based on position proximity
        # In a more sophisticated implementation, you could use appearance features
        
        # Get the current position from ByteTracker (this would need access to current detections)
        # For now, return None as we need more context for proper recovery
        return None
    
    def _interpolate_missing_positions(self, track_id: int, start_timestamp: float, 
                                     end_timestamp: float, start_pos: Tuple[float, float], 
                                     end_pos: Tuple[float, float]):
        """
        Interpolate positions for missing frames using linear interpolation
        
        Args:
            track_id: Track ID to interpolate for
            start_timestamp: Starting timestamp
            end_timestamp: Ending timestamp  
            start_pos: Starting position (x, y)
            end_pos: Ending position (x, y)
        """
        if track_id not in self.trajectories:
            return
        
        trajectory = self.trajectories[track_id]
        frame_gap = int(end_timestamp - start_timestamp)
        
        if frame_gap <= 1:
            return
        
        # Linear interpolation
        for i in range(1, frame_gap):
            alpha = i / frame_gap
            interp_x = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
            interp_y = start_pos[1] + alpha * (end_pos[1] - start_pos[1])
            interp_timestamp = start_timestamp + i
            
            # Add interpolated position (marked as interpolated)
            trajectory.positions.append((interp_x, interp_y))
            trajectory.timestamps.append(interp_timestamp)
            
            # Calculate interpolated velocity
            if len(trajectory.positions) >= 2:
                prev_pos = trajectory.positions[-2]
                curr_pos = trajectory.positions[-1]
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                 (curr_pos[1] - prev_pos[1])**2)
                trajectory.velocities.append(distance)  # Assuming 1 frame = 1 time unit
    
    def _handle_lost_tracks(self, current_track_ids: set, timestamp: float):
        """
        Handle tracks that are no longer being detected
        
        Args:
            current_track_ids: Set of currently active track IDs
            timestamp: Current timestamp
        """
        # Find tracks that were active but are now missing
        all_known_tracks = set(self.trajectories.keys())
        lost_this_frame = all_known_tracks - current_track_ids
        
        for track_id in lost_this_frame:
            if track_id not in self.lost_track_timeout:
                # Mark track as lost
                self.lost_track_timeout[track_id] = timestamp
                self.lost_tracks[track_id] = self.trajectories[track_id]
        
        # Clean up tracks that have been lost too long
        tracks_to_remove = []
        for track_id, lost_timestamp in self.lost_track_timeout.items():
            if timestamp - lost_timestamp > self.max_lost_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.lost_track_timeout:
                del self.lost_track_timeout[track_id]
            if track_id in self.lost_tracks:
                del self.lost_tracks[track_id]
            # Keep trajectory data for analysis but remove from active tracking
    
    def smooth_trajectory(self, track_id: int, window_size: int = 5) -> bool:
        """
        Apply smoothing to a trajectory to reduce noise
        
        Args:
            track_id: Track ID to smooth
            window_size: Size of smoothing window
            
        Returns:
            True if smoothing was applied, False otherwise
        """
        if track_id not in self.trajectories:
            return False
        
        trajectory = self.trajectories[track_id]
        if len(trajectory.positions) < window_size:
            return False
        
        # Apply moving average smoothing
        positions = list(trajectory.positions)
        smoothed_positions = []
        
        for i in range(len(positions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            
            window_positions = positions[start_idx:end_idx]
            avg_x = np.mean([pos[0] for pos in window_positions])
            avg_y = np.mean([pos[1] for pos in window_positions])
            
            smoothed_positions.append((avg_x, avg_y))
        
        # Update trajectory with smoothed positions
        trajectory.positions = deque(smoothed_positions, maxlen=trajectory.positions.maxlen)
        
        # Recalculate velocities after smoothing
        trajectory.velocities.clear()
        for i in range(1, len(trajectory.positions)):
            prev_pos = trajectory.positions[i-1]
            curr_pos = trajectory.positions[i]
            if i < len(trajectory.timestamps):
                time_diff = trajectory.timestamps[i] - trajectory.timestamps[i-1]
                if time_diff > 0:
                    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                     (curr_pos[1] - prev_pos[1])**2)
                    velocity = distance / time_diff
                    trajectory.velocities.append(velocity)
        
        return True
    
    def get_lost_tracks(self) -> Dict[int, TrajectoryData]:
        """Get currently lost tracks that might be recoverable"""
        return self.lost_tracks.copy()
    
    def force_track_recovery(self, lost_track_id: int, new_detection: Detection, 
                           timestamp: float) -> bool:
        """
        Manually force recovery of a lost track with a new detection
        
        Args:
            lost_track_id: ID of the lost track to recover
            new_detection: New detection to associate with the lost track
            timestamp: Current timestamp
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if lost_track_id not in self.lost_tracks:
            return False
        
        # Move track back to active trajectories
        self.trajectories[lost_track_id] = self.lost_tracks[lost_track_id]
        
        # Add new position
        x1, y1, x2, y2 = new_detection.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        self.trajectories[lost_track_id].add_position((center_x, center_y), timestamp)
        
        # Clean up lost track records
        if lost_track_id in self.lost_tracks:
            del self.lost_tracks[lost_track_id]
        if lost_track_id in self.lost_track_timeout:
            del self.lost_track_timeout[lost_track_id]
        
        self.stats['lost_tracks_recovered'] += 1
        return True
    
    def export_trajectories(self, output_path: str, format: str = 'json'):
        """Export trajectory data to file"""
        import json
        import csv
        
        if format == 'json':
            export_data = {}
            for track_id, trajectory in self.trajectories.items():
                export_data[str(track_id)] = {
                    'positions': list(trajectory.positions),
                    'timestamps': list(trajectory.timestamps),
                    'velocities': list(trajectory.velocities),
                    'field_positions': list(trajectory.field_positions),
                    'statistics': self.get_track_statistics(track_id)
                }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == 'csv':
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['track_id', 'timestamp', 'x', 'y', 'velocity', 'field_x', 'field_y'])
                
                for track_id, trajectory in self.trajectories.items():
                    for i, (pos, timestamp) in enumerate(zip(trajectory.positions, trajectory.timestamps)):
                        velocity = trajectory.velocities[i] if i < len(trajectory.velocities) else 0.0
                        field_pos = trajectory.field_positions[i] if i < len(trajectory.field_positions) else (None, None)
                        
                        writer.writerow([
                            track_id, timestamp, pos[0], pos[1], velocity,
                            field_pos[0] if field_pos[0] is not None else '',
                            field_pos[1] if field_pos[1] is not None else ''
                        ])