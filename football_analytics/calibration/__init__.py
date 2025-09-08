"""
Field calibration and coordinate transformation components
"""

from .field_calibrator import FieldCalibrator, create_field_calibrator_from_config
from .fallback_calibrator import (
    FallbackCalibrator, 
    CalibrationQualityAssessment,
    create_fallback_calibrator_from_config
)
from .hybrid_calibrator import HybridCalibrator, create_hybrid_calibrator_from_config

# PnLCalib integration (optional import)
try:
    from .pnl_calibrator import PnLCalibrator, create_pnl_calibrator_from_config
    PNL_AVAILABLE = True
except ImportError:
    PNL_AVAILABLE = False

__all__ = [
    'FieldCalibrator',
    'create_field_calibrator_from_config',
    'FallbackCalibrator',
    'CalibrationQualityAssessment', 
    'create_fallback_calibrator_from_config',
    'HybridCalibrator',
    'create_hybrid_calibrator_from_config'
]

if PNL_AVAILABLE:
    __all__.extend(['PnLCalibrator', 'create_pnl_calibrator_from_config'])