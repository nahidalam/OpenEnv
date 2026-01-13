# openenv/vision/__init__.py
"""Vision-friendly utilities for OpenEnv.

Provides standard encoding/decoding for image frames in observations,
video bundling utilities, and trajectory logging helpers for vision-first
counterfactual environments like Robot City.

Usage:
    from openenv.vision import encode_image_to_base64, decode_image_from_base64
    from openenv.vision import pack_frames, unpack_frames
    from openenv.vision import summarize_frame, safe_jsonify_observation
"""

from .frames import (
    encode_image_to_base64,
    decode_image_from_base64,
    pack_frames,
    unpack_frames,
    summarize_frame,
    safe_jsonify_observation,
)

# Keep legacy exports for backwards compatibility
from .encoding import (
    encode_frame,
    decode_frame,
    encode_frame_to_data_uri,
    decode_data_uri,
    get_image_info,
)
from .protocols import (
    VisionObservation,
    is_vision_observation,
    extract_frame,
    make_vision_observation,
)

__all__ = [
    # Primary frame utilities (new)
    "encode_image_to_base64",
    "decode_image_from_base64",
    "pack_frames",
    "unpack_frames",
    "summarize_frame",
    "safe_jsonify_observation",
    # Legacy encoding utilities
    "encode_frame",
    "decode_frame",
    "encode_frame_to_data_uri",
    "decode_data_uri",
    "get_image_info",
    # Protocols
    "VisionObservation",
    "is_vision_observation",
    "extract_frame",
    "make_vision_observation",
]
