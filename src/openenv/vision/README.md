# OpenEnv Vision Module

Vision-friendly utilities for OpenEnv environments, particularly useful for vision-first counterfactual environments like Robot City.

## Overview

This module provides:
- **Image encoding/decoding**: Standard base64 PNG/JPEG encoding for transport and logging
- **Frame buffering**: Utilities for managing frame sequences
- **Vision protocols**: Optional conventions for vision observations
- **Trajectory logging**: JSONL export with frame summaries

## Quick Start

```python
from openenv.vision import encode_frame, decode_frame, FrameBuffer
from openenv.vision.protocols import VisionObservation, extract_frame

# Encode a numpy array to base64 PNG
import numpy as np
frame = np.zeros((100, 100, 3), dtype=np.uint8)
encoded = encode_frame(frame, format="png")

# Decode back
decoded = decode_frame(encoded, return_numpy=True)

# Use frame buffer for trajectory logging
buffer = FrameBuffer(max_frames=100, store_thumbnails=True)
for obs in observations:
    buffer.add(obs.frame, metadata={"step": step})
```

## Image Encoding

The `encoding` module provides utilities for converting images to/from base64:

```python
from openenv.vision import (
    encode_frame,           # Encode to base64
    decode_frame,           # Decode from base64
    encode_frame_to_data_uri,  # Encode to data:image/... URI
    decode_data_uri,        # Decode data URI
    get_image_info,         # Get image dimensions without full decode
)

# Supports numpy arrays, PIL Images, and raw bytes
encoded = encode_frame(numpy_array, format="jpeg", quality=85)
encoded = encode_frame(pil_image, format="png")
```

## Frame Buffering

The `FrameBuffer` class manages frame sequences for trajectory logging:

```python
from openenv.vision import FrameBuffer, summarize_frame

# Create buffer with limits
buffer = FrameBuffer(
    max_frames=50,           # Keep last 50 frames
    store_encoded=False,     # Don't store full frames (saves memory)
    store_thumbnails=True,   # Store small thumbnails
    thumbnail_size=(64, 64),
    format="png",
)

# Add frames during episode
for step, obs in enumerate(episode):
    summary = buffer.add(obs.frame, metadata={"reward": obs.reward})

# Export for logging
log_data = buffer.to_dict()
```

## Vision Observation Protocol

The `VisionObservation` protocol is an **optional** convention for environments that want to expose visual data in a standard way:

```python
from openenv.vision.protocols import VisionObservation, is_vision_observation, extract_frame

# Check if observation implements protocol
if is_vision_observation(obs):
    frame = obs.get_frame()
    info = obs.get_frame_info()

# Or use extract_frame() which tries multiple strategies
frame = extract_frame(obs)  # Tries .frame, .image, ["frame"], etc.
```

### Implementing VisionObservation

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class RobotCityObs:
    frame: np.ndarray  # RGB image (H, W, 3)
    reward: float
    done: bool
    info: dict
    
    def get_frame(self):
        return self.frame
    
    def get_frame_info(self):
        return {
            "width": self.frame.shape[1],
            "height": self.frame.shape[0],
            "channels": self.frame.shape[2],
        }
```

## Trajectory Logging with Frames

Combine with the counterfactual module for vision-aware trajectory logging:

```python
from openenv.counterfactual import (
    summary_to_log_entries,
    tree_to_log_entries,
    write_jsonl,
    make_vision_frame_processor,
)

# Create frame processor
processor = make_vision_frame_processor(
    include_thumbnail=True,
    include_full=False,  # Don't store full frames
)

# Convert trajectory to log entries with frame summaries
entries = summary_to_log_entries(
    trajectory_summary,
    trajectory_id="episode_001",
    frame_processor=processor,
)

# Write to JSONL
write_jsonl(entries, "trajectories.jsonl")
```

## Dependencies

- **Required**: None (base functionality works without extra deps)
- **For image encoding**: `Pillow` (`pip install Pillow`)
- **For numpy arrays**: `numpy` (`pip install numpy`)

Tests skip gracefully if optional dependencies are not installed.

## Design Principles

1. **Non-invasive**: Does not modify OpenEnv's core protocol or server/client
2. **Optional**: Environments don't need to implement VisionObservation
3. **Efficient**: Thumbnails and lazy encoding for memory efficiency
4. **JSON-friendly**: All outputs are JSON-serializable for logging
