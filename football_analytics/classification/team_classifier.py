"""
Team classification system for football analytics
Integrates embedding models and K-means clustering for team assignment
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any
from sklearn.cluster import KMeans, MiniBatchKMeans
import timm
import torchreid
import torchvision.models as models
import torchvision.transforms as T
from torchvision import transforms
import torch.nn as nn

from ..core.models import TrackedObject, Detection
from ..core.config import ProcessingConfig


class TeamClassifier:
    """
    Team classification system that integrates multiple embedding models
    and K-means clustering for consistent team assignment
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the team classifier
        
        Args:
            config: Processing configuration containing embedding model and team settings
        """
        self.config = config
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.n_teams = config.n_teams
        self.embedding_model_name = config.embedding_model
        
        # Model and transform placeholders
        self.embedding_model = None
        self.transform = None
        
        # Clustering
        self.kmeans_model: Optional[MiniBatchKMeans] = None
        self.is_initialized = False
        
        # Team assignment tracking
        self.team_assignments: Dict[int, int] = {}  # track_id -> team_id
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.embedding_history: Dict[int, List[np.ndarray]] = {}  # track_id -> embeddings
        
        # Configuration
        self.crop_size = 224
        self.oversampling_margin = 0.1
        self.min_embeddings_for_init = max(self.n_teams, 4)
        
        # Load the specified embedding model
        self._load_embedding_model()
        self._initialize_team_colors()
    
    def _load_embedding_model(self) -> None:
        """Load the specified embedding model"""
        try:
            if self.embedding_model_name.lower() == "osnet":
                self.embedding_model, self.transform = self._load_osnet_model()
            elif self.embedding_model_name.lower() == "resnet50":
                self.embedding_model, self.transform = self._load_resnet50_model()
            elif self.embedding_model_name.lower() == "dinov2":
                self.embedding_model, self.transform = self._load_dinov2_model()
            else:
                raise ValueError(f"Unsupported embedding model: {self.embedding_model_name}")
                
            print(f"✅ Loaded embedding model: {self.embedding_model_name}")
            
        except Exception as e:
            print(f"❌ Error loading embedding model {self.embedding_model_name}: {e}")
            # Fallback to ResNet50
            print("🔄 Falling back to ResNet50...")
            self.embedding_model, self.transform = self._load_resnet50_model()
            self.embedding_model_name = "resnet50"
    
    def _load_osnet_model(self) -> Tuple[Any, Any]:
        """Load OSNet model for person re-identification"""
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        model.eval().to(self.device)
        
        transform = torchreid.data.transforms.build_transforms(
            height=256, 
            width=128, 
            norm_mean=[0.485, 0.456, 0.406], 
            norm_std=[0.229, 0.224, 0.225]
        )[0]
        
        return model, transform
    
    def _load_resnet50_model(self) -> Tuple[Any, Any]:
        """Load ResNet50 model for feature extraction"""
        model = models.resnet50(pretrained=True)
        # Remove final classification layer to get embeddings
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval().to(self.device)
        
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])
        
        return model, transform
    
    def _load_dinov2_model(self) -> Tuple[Any, Any]:
        """Load DINOv2 model for self-supervised feature extraction"""
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        model = model.to(self.device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(self.crop_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        return model, transform
    
    def _initialize_team_colors(self) -> None:
        """Initialize default team colors"""
        default_colors = [
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i in range(self.n_teams):
            self.team_colors[i] = default_colors[i % len(default_colors)]
    
    def extract_player_crop(self, frame: np.ndarray, detection: Detection) -> Optional[np.ndarray]:
        """
        Extract player crop from frame using detection bounding box
        
        Args:
            frame: Input frame
            detection: Detection object with bounding box
            
        Returns:
            Cropped player image or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = detection.bbox
            h, w = frame.shape[:2]
            
            # Add margin around bounding box
            margin_x = int((x2 - x1) * self.oversampling_margin)
            margin_y = int((y2 - y1) * self.oversampling_margin)
            
            # Expand bounding box with margin
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
                
            return crop
            
        except Exception as e:
            print(f"Error extracting player crop: {e}")
            return None
    
    def get_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Generate embeddings for player crops using the loaded model
        
        Args:
            crops: List of player crop images
            
        Returns:
            Numpy array of embeddings (N, embedding_dim)
        """
        if not crops or self.embedding_model is None:
            return np.array([])
        
        try:
            if self.embedding_model_name.lower() == "osnet":
                return self._get_osnet_embeddings(crops)
            elif self.embedding_model_name.lower() == "resnet50":
                return self._get_resnet50_embeddings(crops)
            elif self.embedding_model_name.lower() == "dinov2":
                return self._get_dinov2_embeddings(crops)
            else:
                raise ValueError(f"Unknown embedding model: {self.embedding_model_name}")
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.array([])
    
    def _get_osnet_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Generate embeddings using OSNet"""
        embeddings = []
        
        with torch.no_grad():
            for crop in crops:
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                img = self.transform(img).unsqueeze(0).to(self.device)
                feat = self.embedding_model(img)
                embeddings.append(feat.cpu().numpy())
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def _get_resnet50_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Generate embeddings using ResNet50"""
        if not crops:
            return np.array([])
        
        with torch.no_grad():
            # Preprocess crops
            transformed_crops = []
            for crop in crops:
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                transformed_crops.append(self.transform(img))
            
            input_tensors = torch.stack(transformed_crops).to(self.device)
            
            # Forward pass
            features = self.embedding_model(input_tensors)  # (N, 2048, 1, 1)
            features = features.view(features.size(0), -1)  # (N, 2048)
            
            embeddings = features.cpu().numpy()
        
        return embeddings
    
    def _get_dinov2_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Generate embeddings using DINOv2"""
        if not crops:
            return np.array([])
        
        embeddings = []
        
        with torch.no_grad():
            # Transform crops
            transformed_crops = []
            for crop in crops:
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                transformed_crops.append(self.transform(img))
            
            input_tensors = torch.stack(transformed_crops).to(self.device)
            
            # Forward pass
            patch_tokens = self.embedding_model.forward_features(input_tensors)
            
            # Extract embeddings
            if patch_tokens.dim() == 3:
                # Use CLS token (first token)
                embeddings_tensor = patch_tokens[:, 0, :]
            elif patch_tokens.dim() == 2:
                embeddings_tensor = patch_tokens
            else:
                raise ValueError(f"Unexpected tensor shape: {patch_tokens.shape}")
            
            embeddings = embeddings_tensor.cpu().numpy()
        
        return embeddings
    
    def classify_teams(self, tracked_objects: List[TrackedObject], frame: np.ndarray) -> List[TrackedObject]:
        """
        Classify players into teams using K-means clustering on embeddings
        
        Args:
            tracked_objects: List of tracked objects to classify
            frame: Current frame for crop extraction
            
        Returns:
            Updated tracked objects with team assignments
        """
        if not tracked_objects:
            return tracked_objects
        
        # Filter for player detections only
        players = [obj for obj in tracked_objects if obj.detection.class_name in ["player"]]
        
        if len(players) < 2:
            return tracked_objects
        
        # Extract crops and generate embeddings
        crops = []
        valid_players = []
        
        for player in players:
            crop = self.extract_player_crop(frame, player.detection)
            if crop is not None:
                crops.append(crop)
                valid_players.append(player)
        
        if len(crops) < 2:
            return tracked_objects
        
        # Generate embeddings
        embeddings = self.get_embeddings(crops)
        
        if embeddings.size == 0:
            return tracked_objects
        
        # Initialize or update K-means model
        if not self.is_initialized and len(embeddings) >= self.min_embeddings_for_init:
            self._initialize_kmeans(embeddings)
        
        if self.kmeans_model is not None:
            # Predict team assignments
            team_predictions = self.kmeans_model.predict(embeddings)
            
            # Update team assignments with consistency checking
            updated_assignments = []
            for i, player in enumerate(valid_players):
                predicted_team = int(team_predictions[i])
                consistent_team = self._get_consistent_team_assignment(
                    player.track_id, 
                    predicted_team, 
                    embeddings[i]
                )
                player.team_id = consistent_team
                updated_assignments.append(consistent_team)
            
            # Update team colors based on current crops (every 30 frames to avoid too frequent updates)
            if len(self.team_assignments) % 30 == 0 and len(crops) > 0:
                try:
                    self.update_team_colors_from_crops(crops, updated_assignments)
                except Exception as e:
                    print(f"Warning: Could not update team colors: {e}")
        
        return tracked_objects
    
    def _initialize_kmeans(self, embeddings: np.ndarray) -> None:
        """Initialize K-means clustering model"""
        try:
            self.kmeans_model = MiniBatchKMeans(
                n_clusters=self.n_teams,
                init='k-means++',
                max_iter=300,
                random_state=42,
                reassignment_ratio=0.01,
                n_init=20,
                batch_size=len(embeddings)
            )
            
            self.kmeans_model.fit(embeddings)
            self.is_initialized = True
            
            print(f"✅ K-means initialized with {self.n_teams} teams")
            
        except Exception as e:
            print(f"❌ Error initializing K-means: {e}")
    
    def _get_consistent_team_assignment(self, track_id: int, predicted_team: int, embedding: np.ndarray) -> int:
        """
        Get consistent team assignment using tracking history and temporal consistency
        
        Args:
            track_id: Player tracking ID
            predicted_team: Current frame prediction
            embedding: Current embedding
            
        Returns:
            Consistent team assignment
        """
        # Store embedding history
        if track_id not in self.embedding_history:
            self.embedding_history[track_id] = []
        
        self.embedding_history[track_id].append(embedding)
        
        # Keep only recent embeddings (last 15 frames for better consistency)
        if len(self.embedding_history[track_id]) > 15:
            self.embedding_history[track_id] = self.embedding_history[track_id][-15:]
        
        # If this is a new player, use current prediction
        if track_id not in self.team_assignments:
            self.team_assignments[track_id] = predicted_team
            return predicted_team
        
        # Check consistency with previous assignments
        previous_team = self.team_assignments[track_id]
        
        # If prediction matches previous assignment, keep it
        if predicted_team == previous_team:
            return predicted_team
        
        # If prediction differs, use temporal consistency check
        if len(self.embedding_history[track_id]) >= 5:
            # Use weighted majority vote with recent frames having more weight
            recent_embeddings = np.vstack(self.embedding_history[track_id][-10:])
            recent_predictions = self.kmeans_model.predict(recent_embeddings)
            
            # Apply temporal weighting (more recent predictions have higher weight)
            weights = np.linspace(0.5, 1.0, len(recent_predictions))
            
            # Calculate weighted votes for each team
            team_votes = {}
            for i, team in enumerate(recent_predictions):
                if team not in team_votes:
                    team_votes[team] = 0
                team_votes[team] += weights[i]
            
            # Choose team with highest weighted vote
            best_team = max(team_votes.items(), key=lambda x: x[1])[0]
            
            # Only change assignment if confidence is high enough
            total_weight = sum(team_votes.values())
            confidence = team_votes[best_team] / total_weight
            
            if confidence > 0.6:  # 60% confidence threshold
                self.team_assignments[track_id] = int(best_team)
                return int(best_team)
        
        # Not enough history or low confidence, keep previous assignment
        return previous_team
    
    def extract_dominant_colors(self, crops: List[np.ndarray], team_assignments: List[int]) -> Dict[int, Tuple[int, int, int]]:
        """
        Extract dominant colors for each team from player crops
        
        Args:
            crops: List of player crop images
            team_assignments: Team assignments for each crop
            
        Returns:
            Dictionary mapping team_id to dominant color (BGR)
        """
        team_colors = {}
        
        # Group crops by team
        team_crops = {}
        for crop, team_id in zip(crops, team_assignments):
            if team_id not in team_crops:
                team_crops[team_id] = []
            team_crops[team_id].append(crop)
        
        # Extract dominant color for each team
        for team_id, crops_list in team_crops.items():
            if not crops_list:
                continue
                
            # Combine all crops for this team
            all_pixels = []
            for crop in crops_list:
                # Focus on the torso area (middle part of the crop)
                h, w = crop.shape[:2]
                torso_crop = crop[h//4:3*h//4, w//4:3*w//4]
                
                # Reshape to get all pixels
                pixels = torso_crop.reshape(-1, 3)
                all_pixels.append(pixels)
            
            if all_pixels:
                combined_pixels = np.vstack(all_pixels)
                
                # Use K-means to find dominant color
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    kmeans.fit(combined_pixels)
                    
                    # Get the most frequent cluster center as dominant color
                    labels = kmeans.labels_
                    unique, counts = np.unique(labels, return_counts=True)
                    dominant_cluster = unique[np.argmax(counts)]
                    dominant_color = kmeans.cluster_centers_[dominant_cluster]
                    
                    # Convert to BGR integers
                    team_colors[team_id] = tuple(map(int, dominant_color))
                    
                except Exception as e:
                    print(f"Error extracting color for team {team_id}: {e}")
                    # Fallback to default colors
                    team_colors[team_id] = self.team_colors.get(team_id, (128, 128, 128))
        
        return team_colors
    
    def update_team_colors_from_crops(self, crops: List[np.ndarray], team_assignments: List[int]) -> None:
        """
        Update team colors based on extracted dominant colors from crops
        
        Args:
            crops: List of player crop images
            team_assignments: Team assignments for each crop
        """
        if len(crops) != len(team_assignments):
            return
            
        extracted_colors = self.extract_dominant_colors(crops, team_assignments)
        
        # Update team colors with extracted colors
        for team_id, color in extracted_colors.items():
            self.team_colors[team_id] = color
        
        print(f"✅ Updated team colors: {self.team_colors}")
    
    def get_team_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """Get team color mapping"""
        return self.team_colors.copy()
    
    def update_team_colors(self, team_colors: Dict[int, Tuple[int, int, int]]) -> None:
        """Update team color mapping"""
        self.team_colors.update(team_colors)
    
    def reset_classifier(self) -> None:
        """Reset the classifier state"""
        self.kmeans_model = None
        self.is_initialized = False
        self.team_assignments.clear()
        self.embedding_history.clear()
        print("🔄 Team classifier reset")
    
    def validate_team_classification_consistency(self) -> Dict[str, Any]:
        """
        Validate team classification consistency across frames
        
        Returns:
            Dictionary with consistency metrics
        """
        validation_results = {
            "consistency_score": 0.0,
            "team_stability": {},
            "assignment_changes": 0,
            "total_assignments": len(self.team_assignments),
            "embedding_history_length": {}
        }
        
        if not self.team_assignments:
            return validation_results
        
        # Calculate team stability (how often players stay in the same team)
        stable_assignments = 0
        total_history_points = 0
        assignment_changes = 0
        
        for track_id, current_team in self.team_assignments.items():
            if track_id in self.embedding_history:
                history_length = len(self.embedding_history[track_id])
                validation_results["embedding_history_length"][track_id] = history_length
                
                if history_length > 1 and self.kmeans_model is not None:
                    # Check consistency across history
                    embeddings = np.array(self.embedding_history[track_id])
                    predictions = self.kmeans_model.predict(embeddings)
                    
                    # Count how many predictions match current assignment
                    matches = np.sum(predictions == current_team)
                    stable_assignments += matches
                    total_history_points += len(predictions)
                    
                    # Count assignment changes
                    changes = np.sum(np.diff(predictions) != 0)
                    assignment_changes += changes
                    
                    # Team-specific stability
                    team_stability = matches / len(predictions)
                    validation_results["team_stability"][track_id] = team_stability
        
        # Calculate overall consistency score
        if total_history_points > 0:
            validation_results["consistency_score"] = stable_assignments / total_history_points
        
        validation_results["assignment_changes"] = assignment_changes
        
        return validation_results
    
    def validate_team_balance(self) -> Dict[str, Any]:
        """
        Validate team balance and distribution
        
        Returns:
            Dictionary with balance metrics
        """
        balance_results = {
            "team_distribution": {},
            "balance_score": 0.0,
            "is_balanced": False,
            "imbalance_ratio": 0.0
        }
        
        if not self.team_assignments:
            return balance_results
        
        # Calculate team distribution
        unique, counts = np.unique(list(self.team_assignments.values()), return_counts=True)
        for team_id, count in zip(unique, counts):
            balance_results["team_distribution"][int(team_id)] = int(count)
        
        # Calculate balance score (how evenly distributed the teams are)
        if len(counts) >= 2:
            min_count = np.min(counts)
            max_count = np.max(counts)
            
            # Balance score: 1.0 = perfectly balanced, 0.0 = completely imbalanced
            balance_results["balance_score"] = min_count / max_count if max_count > 0 else 0.0
            balance_results["imbalance_ratio"] = max_count / min_count if min_count > 0 else float('inf')
            
            # Consider balanced if ratio is less than 2:1
            balance_results["is_balanced"] = balance_results["imbalance_ratio"] <= 2.0
        
        return balance_results
    
    def validate_classification_quality(self) -> Dict[str, Any]:
        """
        Comprehensive validation of classification quality
        
        Returns:
            Dictionary with quality metrics
        """
        quality_results = {
            "overall_quality_score": 0.0,
            "consistency_validation": self.validate_team_classification_consistency(),
            "balance_validation": self.validate_team_balance(),
            "model_confidence": 0.0,
            "recommendations": []
        }
        
        # Calculate overall quality score
        consistency_score = quality_results["consistency_validation"]["consistency_score"]
        balance_score = quality_results["balance_validation"]["balance_score"]
        
        # Weighted combination of consistency and balance
        quality_results["overall_quality_score"] = (0.7 * consistency_score + 0.3 * balance_score)
        
        # Model confidence based on K-means inertia if available
        if self.kmeans_model is not None and hasattr(self.kmeans_model, 'inertia_'):
            # Normalize inertia to a 0-1 confidence score (lower inertia = higher confidence)
            # This is a rough approximation
            max_inertia = 1000.0  # Assumed maximum for normalization
            quality_results["model_confidence"] = max(0.0, 1.0 - (self.kmeans_model.inertia_ / max_inertia))
        
        # Generate recommendations
        recommendations = []
        
        if consistency_score < 0.7:
            recommendations.append("Low consistency detected. Consider increasing embedding history length or adjusting confidence threshold.")
        
        if not quality_results["balance_validation"]["is_balanced"]:
            recommendations.append("Team imbalance detected. Check if all teams are properly represented in the video.")
        
        if quality_results["overall_quality_score"] < 0.6:
            recommendations.append("Overall classification quality is low. Consider using a different embedding model or adjusting parameters.")
        
        if len(self.team_assignments) < 4:
            recommendations.append("Few players detected. Classification accuracy may be limited with small sample size.")
        
        quality_results["recommendations"] = recommendations
        
        return quality_results
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics"""
        stats = {
            "is_initialized": self.is_initialized,
            "n_teams": self.n_teams,
            "embedding_model": self.embedding_model_name,
            "tracked_players": len(self.team_assignments),
            "team_distribution": {},
            "quality_validation": self.validate_classification_quality()
        }
        
        if self.team_assignments:
            unique, counts = np.unique(list(self.team_assignments.values()), return_counts=True)
            for team_id, count in zip(unique, counts):
                stats["team_distribution"][int(team_id)] = int(count)
        
        return stats