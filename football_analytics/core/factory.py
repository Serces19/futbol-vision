"""
Factory class for creating system components
"""

from typing import Dict, Any, Type
from .interfaces import (
    BaseDetector, BaseTracker, BaseClassifier, BaseCalibrator,
    BaseAnalyticsEngine, BaseVisualizer, ComponentFactory
)
from .config import ConfigManager
from .exceptions import ConfigurationError


class DefaultComponentFactory(ComponentFactory):
    """Default implementation of component factory"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._detector_registry: Dict[str, Type[BaseDetector]] = {}
        self._tracker_registry: Dict[str, Type[BaseTracker]] = {}
        self._classifier_registry: Dict[str, Type[BaseClassifier]] = {}
        self._calibrator_registry: Dict[str, Type[BaseCalibrator]] = {}
        self._analytics_registry: Dict[str, Type[BaseAnalyticsEngine]] = {}
        self._visualizer_registry: Dict[str, Type[BaseVisualizer]] = {}
    
    def register_detector(self, name: str, detector_class: Type[BaseDetector]) -> None:
        """Register a detector implementation"""
        self._detector_registry[name] = detector_class
    
    def register_tracker(self, name: str, tracker_class: Type[BaseTracker]) -> None:
        """Register a tracker implementation"""
        self._tracker_registry[name] = tracker_class
    
    def register_classifier(self, name: str, classifier_class: Type[BaseClassifier]) -> None:
        """Register a classifier implementation"""
        self._classifier_registry[name] = classifier_class
    
    def register_calibrator(self, name: str, calibrator_class: Type[BaseCalibrator]) -> None:
        """Register a calibrator implementation"""
        self._calibrator_registry[name] = calibrator_class
    
    def register_analytics_engine(self, name: str, engine_class: Type[BaseAnalyticsEngine]) -> None:
        """Register an analytics engine implementation"""
        self._analytics_registry[name] = engine_class
    
    def register_visualizer(self, name: str, visualizer_class: Type[BaseVisualizer]) -> None:
        """Register a visualizer implementation"""
        self._visualizer_registry[name] = visualizer_class
    
    def create_detector(self, detector_type: str, **kwargs) -> BaseDetector:
        """Create detector instance"""
        if detector_type not in self._detector_registry:
            raise ConfigurationError(f"Unknown detector type: {detector_type}")
        
        detector_class = self._detector_registry[detector_type]
        return detector_class(
            config=self.config_manager.processing_config,
            model_paths=self.config_manager.model_paths,
            **kwargs
        )
    
    def create_tracker(self, tracker_type: str, **kwargs) -> BaseTracker:
        """Create tracker instance"""
        if tracker_type not in self._tracker_registry:
            raise ConfigurationError(f"Unknown tracker type: {tracker_type}")
        
        tracker_class = self._tracker_registry[tracker_type]
        return tracker_class(
            config=self.config_manager.tracker_config,
            **kwargs
        )
    
    def create_classifier(self, classifier_type: str, **kwargs) -> BaseClassifier:
        """Create classifier instance"""
        if classifier_type not in self._classifier_registry:
            raise ConfigurationError(f"Unknown classifier type: {classifier_type}")
        
        classifier_class = self._classifier_registry[classifier_type]
        return classifier_class(
            config=self.config_manager.processing_config,
            **kwargs
        )
    
    def create_calibrator(self, calibrator_type: str, **kwargs) -> BaseCalibrator:
        """Create calibrator instance"""
        if calibrator_type not in self._calibrator_registry:
            raise ConfigurationError(f"Unknown calibrator type: {calibrator_type}")
        
        calibrator_class = self._calibrator_registry[calibrator_type]
        return calibrator_class(
            field_dimensions=self.config_manager.field_dimensions,
            **kwargs
        )
    
    def create_analytics_engine(self, engine_type: str, **kwargs) -> BaseAnalyticsEngine:
        """Create analytics engine instance"""
        if engine_type not in self._analytics_registry:
            raise ConfigurationError(f"Unknown analytics engine type: {engine_type}")
        
        engine_class = self._analytics_registry[engine_type]
        return engine_class(
            config=self.config_manager.processing_config,
            **kwargs
        )
    
    def create_visualizer(self, visualizer_type: str, **kwargs) -> BaseVisualizer:
        """Create visualizer instance"""
        if visualizer_type not in self._visualizer_registry:
            raise ConfigurationError(f"Unknown visualizer type: {visualizer_type}")
        
        visualizer_class = self._visualizer_registry[visualizer_type]
        return visualizer_class(
            config=self.config_manager.visualization_config,
            **kwargs
        )
    
    def get_available_components(self) -> Dict[str, list]:
        """Get list of available component types"""
        return {
            'detectors': list(self._detector_registry.keys()),
            'trackers': list(self._tracker_registry.keys()),
            'classifiers': list(self._classifier_registry.keys()),
            'calibrators': list(self._calibrator_registry.keys()),
            'analytics_engines': list(self._analytics_registry.keys()),
            'visualizers': list(self._visualizer_registry.keys()),
        }