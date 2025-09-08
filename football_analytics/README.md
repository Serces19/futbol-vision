# Football Analytics MVP

A modular real-time football video analysis system that processes streaming video of football matches and generates useful data for player and team performance analysis.

## Project Structure

```
football_analytics/
├── core/                   # Core interfaces and data models
│   ├── __init__.py
│   ├── models.py          # Data models (Detection, TrackedObject, etc.)
│   ├── config.py          # Configuration management
│   ├── interfaces.py      # Abstract base classes
│   ├── exceptions.py      # Custom exceptions
│   └── factory.py         # Component factory
├── detection/             # Object detection components
├── tracking/              # Player tracking components
├── classification/        # Team classification components
├── calibration/           # Field calibration components
├── analytics/             # Analytics engine components
├── visualization/         # Real-time visualization components
└── utils/                 # Utility functions and helpers
```

## Key Features

- **Object Detection**: Detect players and ball using YOLO models
- **Player Tracking**: Persistent tracking with ByteTrack
- **Team Classification**: Automatic team assignment using K-means clustering
- **Field Calibration**: Automatic field calibration and coordinate transformation
- **Real-time Visualization**: Live video overlay and 2D field map
- **Analytics Generation**: Position tracking, velocity calculation, heatmaps
- **Data Export**: JSON, CSV, and video export capabilities

## Configuration

The system uses a YAML configuration file (`config.yaml`) to manage all parameters:

```yaml
processing:
  confidence_threshold: 0.5
  device: "cuda"
  n_teams: 2
  
field_dimensions:
  length: 105.0  # meters
  width: 68.0    # meters
  
model_paths:
  yolo_player_model: "models/yolov8m-football.pt"
  field_lines_model: "models/SV_lines"
```

## Usage

```python
from football_analytics.core import ConfigManager, DefaultComponentFactory

# Load configuration
config_manager = ConfigManager()
config_manager.load_config("config.yaml")

# Create component factory
factory = DefaultComponentFactory(config_manager)

# Create and use components
detector = factory.create_detector("yolo")
tracker = factory.create_tracker("bytetrack")
```

## Requirements

See `requirements.txt` for full dependency list. Key requirements:
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy, SciPy, scikit-learn

## Installation

```bash
pip install -r requirements.txt
```

## Development

This is a modular system designed for easy extension. Each component implements a base interface from `core.interfaces`, making it easy to swap implementations or add new functionality.

To add a new component:
1. Implement the appropriate base interface
2. Register it with the component factory
3. Update configuration as needed