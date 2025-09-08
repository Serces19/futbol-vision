"""
Detection module for football analytics
"""

from .object_detector import ObjectDetector, ObjectDetectionError, ModelLoadError, InferenceError, create_object_detector_from_config
from .field_detector import FieldDetector, FieldDetectionError, FieldModelLoadError, FieldInferenceError, create_field_detector_from_config

__all__ = [
    'ObjectDetector',
    'ObjectDetectionError', 
    'ModelLoadError',
    'InferenceError',
    'create_object_detector_from_config',
    'FieldDetector',
    'FieldDetectionError',
    'FieldModelLoadError', 
    'FieldInferenceError',
    'create_field_detector_from_config'
]