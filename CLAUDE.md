# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Dependencies
```bash
pip install -r requirements.txt
```

### Testing Face Detection
Test the node by importing it in a ComfyUI environment or create a simple test script to validate face detection functionality.

### Environment Configuration
Set logging level via environment variable:
```bash
export COMFYUI_FACE_DETECTION_LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
```

## Architecture Overview

This is a ComfyUI custom node for face detection and cropping that provides dual compatibility:

### Core Architecture
- **Dual Version Support**: Automatically detects ComfyUI version and loads appropriate implementation
- **ComfyUI v3**: Uses modern `ComfyNode` class with async execution and schema definitions
- **ComfyUI v1/v2**: Falls back to legacy class structure for backward compatibility
- **Stateless Design**: Face detection logic is implemented as static methods for better performance

### Key Components

#### Version Detection (`face_detection_node.py:9-28`)
```python
try:
    from comfy_api.v0_0_3_io import ComfyNode, Schema, ...
    COMFY_V3_AVAILABLE = True
except ImportError:
    COMFY_V3_AVAILABLE = False
```

#### Node Implementation Structure
- **FaceDetectionNode** (v3): Modern implementation with schema-based configuration
- **FaceDetectionNodeV1** (v1/v2): Legacy compatibility wrapper with INPUT_TYPES method
- **NODE_CLASS_MAPPINGS** (line 447-457): Runtime selection of appropriate class

#### Face Detection Logic
- **Cascade Classifiers**: Uses OpenCV Haar cascades with dual classifier support (default/alternative)
- **Stateless Execution**: Core detection methods are static for v3 compatibility
- **Image Processing Pipeline**: 
  - Tensor → NumPy conversion with proper format handling
  - Grayscale conversion for detection
  - Face cropping with configurable padding
  - Multi-face handling (largest face or all faces)

### Input/Output Handling
- **Input**: RGB images as PyTorch tensors in [B, H, W, C] format
- **Processing**: OpenCV operations on NumPy arrays
- **Output**: Cropped face images as tensors, properly formatted for ComfyUI pipeline

### ComfyUI Integration Files
- **`__init__.py`**: Exports node mappings for ComfyUI discovery
- **`comfyui-manager-entry.json`**: ComfyUI Manager metadata
- **`node_list.json`**: Node registry information

## Development Guidelines

### Face Detection Parameters
- `detection_threshold`: 0.1-1.0 (confidence threshold)
- `min_face_size`: 32-512 pixels (minimum face size)
- `padding`: 0-256 pixels (padding around faces)
- `output_mode`: "largest_face" or "all_faces"
- `classifier_type`: "default" or "alternative" Haar cascade

### Error Handling Patterns
- Cascade classifier validation with fallback mechanisms
- Comprehensive input tensor validation and format conversion
- Graceful degradation when no faces are detected (returns zero tensor)

### Logging Configuration
Uses environment-configurable logging levels via `COMFYUI_FACE_DETECTION_LOG_LEVEL`.