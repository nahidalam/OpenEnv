"""Frame encoding/decoding and observation utilities for vision environments.

Provides utilities for base64 image encoding, frame packing/unpacking,
and safe JSON serialization of observations containing images.

Vision + Counterfactual Conventions
-----------------------------------
Vision observations may include an "rgb" key containing HxWx3 uint8 numpy arrays.
Other common keys: "image", "frame". These are checked by serialization helpers.

Encoding/logging workflow:
    from openenv.vision import encode_image_to_base64, safe_jsonify_observation
    from openenv.counterfactual.serialize import summary_to_json
    from openenv.counterfactual.logging import make_real_step_row

    # Encode frame for transport
    encoded = encode_image_to_base64(obs["rgb"], format="jpg", quality=85)

    # Serialize full observation (auto-encodes rgb/image/frame keys)
    json_safe = safe_jsonify_observation(obs)

    # Log trajectory step with optional frame encoding
    row = make_real_step_row(step, obs, action, reward, done, include_frame=True)

Counterfactual support:
    Environments can implement snapshot()/restore() for imagination-based planning.
    See openenv.counterfactual.protocols.Snapshotable for the protocol definition.
    Use compare() to evaluate candidate actions without mutating the real env state.
"""

import base64
import io
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union


def _require_numpy():
    """Import numpy or raise clear error."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError(
            "numpy is required for vision frame operations. "
            "Install with: pip install numpy"
        )


def _require_pil():
    """Import PIL or raise clear error."""
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise ImportError(
            "Pillow is required for vision frame operations. "
            "Install with: pip install Pillow"
        )


def encode_image_to_base64(
    image: Any,  # np.ndarray | PIL.Image
    format: str = "png",
    quality: int = 90,
) -> str:
    """Encode an image to base64 string.
    
    Args:
        image: numpy array (HxWxC or HxW) or PIL Image
        format: Output format ("png", "jpg", "jpeg", "webp")
        quality: JPEG/WebP quality (1-100), ignored for PNG
        
    Returns:
        Base64 encoded string
        
    Raises:
        ImportError: If numpy or PIL not installed
        ValueError: If image format is unsupported
    """
    np = _require_numpy()
    Image = _require_pil()
    
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        arr = image
        
        # Ensure uint8
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        
        # Handle different shapes
        if arr.ndim == 2:
            # Grayscale (H, W)
            img = Image.fromarray(arr, mode="L")
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                # Grayscale with channel dim (H, W, 1)
                img = Image.fromarray(arr.squeeze(axis=2), mode="L")
            elif arr.shape[2] == 3:
                # RGB (H, W, 3)
                img = Image.fromarray(arr, mode="RGB")
            elif arr.shape[2] == 4:
                # RGBA (H, W, 4)
                img = Image.fromarray(arr, mode="RGBA")
            else:
                raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")
    elif hasattr(image, "save"):  # PIL Image duck typing
        img = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Normalize format
    fmt = format.lower()
    if fmt == "jpg":
        fmt = "jpeg"
    
    # Convert RGBA to RGB for JPEG
    if fmt == "jpeg" and img.mode == "RGBA":
        img = img.convert("RGB")
    
    # Encode to bytes
    buffer = io.BytesIO()
    save_kwargs = {}
    if fmt in ("jpeg", "webp"):
        save_kwargs["quality"] = quality
    
    img.save(buffer, format=fmt.upper(), **save_kwargs)
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_from_base64(data: str) -> Any:  # np.ndarray uint8 HxWxC
    """Decode a base64 string to numpy array.
    
    Args:
        data: Base64 encoded image string
        
    Returns:
        numpy array (uint8, HxWxC for color, HxW for grayscale)
        
    Raises:
        ImportError: If numpy or PIL not installed
    """
    np = _require_numpy()
    Image = _require_pil()
    
    # Decode base64
    image_bytes = base64.b64decode(data)
    
    # Open as PIL Image and convert to numpy
    buffer = io.BytesIO(image_bytes)
    img = Image.open(buffer)
    
    # Convert to numpy array
    arr = np.array(img, dtype=np.uint8)
    
    return arr


def pack_frames(
    frames: List[Any],  # list[np.ndarray]
    *,
    max_frames: Optional[int] = None,
    stride: int = 1,
    format: str = "jpg",
    quality: int = 85,
) -> List[str]:
    """Pack a list of frames into base64 encoded strings.
    
    Args:
        frames: List of numpy arrays (HxWxC)
        max_frames: Maximum number of frames to include (None = all)
        stride: Sample every Nth frame (1 = all frames)
        format: Encoding format ("png", "jpg", "jpeg")
        quality: JPEG quality (1-100)
        
    Returns:
        List of base64 encoded strings
        
    Raises:
        ImportError: If numpy or PIL not installed
    """
    if not frames:
        return []
    
    # Apply stride
    if stride > 1:
        frames = frames[::stride]
    
    # Apply max_frames limit
    if max_frames is not None and len(frames) > max_frames:
        frames = frames[:max_frames]
    
    # Encode each frame
    return [encode_image_to_base64(f, format=format, quality=quality) for f in frames]


def unpack_frames(encoded: List[str]) -> List[Any]:  # list[np.ndarray]
    """Unpack base64 encoded strings to numpy arrays.
    
    Args:
        encoded: List of base64 encoded image strings
        
    Returns:
        List of numpy arrays (uint8, HxWxC)
        
    Raises:
        ImportError: If numpy or PIL not installed
    """
    return [decode_image_from_base64(e) for e in encoded]


def summarize_frame(frame: Any) -> Dict[str, Any]:  # frame: np.ndarray
    """Create a summary dictionary for a frame.
    
    Args:
        frame: numpy array (HxWxC or HxW)
        
    Returns:
        Dictionary with keys: height, width, channels, mean, std
        
    Raises:
        ImportError: If numpy not installed
    """
    np = _require_numpy()
    
    arr = np.asarray(frame)
    
    height = arr.shape[0]
    width = arr.shape[1]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    
    # Compute statistics
    mean = float(arr.mean())
    std = float(arr.std())
    
    return {
        "height": height,
        "width": width,
        "channels": channels,
        "mean": mean,
        "std": std,
    }


def safe_jsonify_observation(obj: Any) -> Any:
    """Safely convert an observation to JSON-serializable format.
    
    Handles:
    - numpy image (HxWx3 uint8) → base64 jpg
    - small numpy arrays (<=2048 elements) → list
    - large numpy arrays → summary dict {shape, dtype, mean, std}
    - dataclasses → dict
    - pydantic models → dict
    - dict/list → recursive safe conversion
    
    Args:
        obj: Any object to convert
        
    Returns:
        JSON-serializable object
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle basic types
    if isinstance(obj, (bool, int, float, str)):
        return obj
    
    # Try to import numpy (optional)
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
        np = None
    
    # Handle numpy arrays
    if has_numpy and isinstance(obj, np.ndarray):
        # Check if it's an image (HxWx3 uint8)
        if (obj.ndim == 3 and 
            obj.shape[2] == 3 and 
            obj.dtype == np.uint8 and
            obj.shape[0] >= 8 and obj.shape[1] >= 8):  # Reasonable image size
            # Encode as base64 jpg
            return encode_image_to_base64(obj, format="jpg", quality=85)
        
        # Small arrays → list
        if obj.size <= 2048:
            return obj.tolist()
        
        # Large arrays → summary
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "mean": float(obj.mean()),
            "std": float(obj.std()),
        }
    
    # Handle pydantic models (check before dataclass since pydantic uses dataclass internally)
    if hasattr(obj, "model_dump"):
        # Pydantic v2
        return safe_jsonify_observation(obj.model_dump())
    if hasattr(obj, "dict") and hasattr(obj, "__fields__"):
        # Pydantic v1
        return safe_jsonify_observation(obj.dict())
    
    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: safe_jsonify_observation(getattr(obj, f.name))
            for f in fields(obj)
        }
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {
            str(k): safe_jsonify_observation(v) 
            for k, v in obj.items()
        }
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [safe_jsonify_observation(item) for item in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return [safe_jsonify_observation(item) for item in sorted(obj, key=str)]
    
    # Handle bytes
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    
    # Fallback: try str()
    try:
        return str(obj)
    except Exception:
        return f"<unserializable: {type(obj).__name__}>"
