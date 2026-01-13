"""Image encoding/decoding utilities for vision observations.

Provides standard base64 PNG/JPEG encoding for image frames in observations,
suitable for transport (HTTP/WebSocket) and logging (JSONL).
"""

import base64
import io
from typing import Any, Dict, Optional, Tuple, Union, Literal

# Type alias for image format
ImageFormat = Literal["png", "jpeg", "jpg", "webp"]

# Default format for encoding
DEFAULT_FORMAT: ImageFormat = "png"
DEFAULT_JPEG_QUALITY = 85


def encode_frame(
    frame: Any,
    format: ImageFormat = DEFAULT_FORMAT,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> str:
    """Encode an image frame to base64 string.
    
    Supports:
    - numpy arrays (H, W, C) or (H, W) for grayscale
    - PIL Image objects
    - bytes (raw image data, passed through)
    
    Args:
        frame: Image frame (numpy array, PIL Image, or bytes)
        format: Output format ("png", "jpeg", "jpg", "webp")
        quality: JPEG/WebP quality (1-100), ignored for PNG
        
    Returns:
        Base64 encoded string
        
    Raises:
        ValueError: If frame type is not supported
    """
    # Handle bytes directly
    if isinstance(frame, bytes):
        return base64.b64encode(frame).decode("utf-8")
    
    # Handle base64 string (already encoded)
    if isinstance(frame, str):
        # Assume it's already base64 encoded
        return frame
    
    # Try to import PIL
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL (Pillow) is required for image encoding. "
            "Install with: pip install Pillow"
        )
    
    # Convert numpy array to PIL Image
    if hasattr(frame, "shape") and hasattr(frame, "dtype"):
        # numpy array
        import numpy as np
        
        arr = np.asarray(frame)
        
        # Handle different array shapes
        if arr.ndim == 2:
            # Grayscale (H, W)
            mode = "L"
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                # Grayscale with channel dim (H, W, 1)
                arr = arr.squeeze(axis=2)
                mode = "L"
            elif arr.shape[2] == 3:
                # RGB (H, W, 3)
                mode = "RGB"
            elif arr.shape[2] == 4:
                # RGBA (H, W, 4)
                mode = "RGBA"
            else:
                raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")
        
        # Ensure uint8
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        
        img = Image.fromarray(arr, mode=mode)
    elif isinstance(frame, Image.Image):
        img = frame
    else:
        raise ValueError(f"Unsupported frame type: {type(frame)}")
    
    # Encode to bytes
    buffer = io.BytesIO()
    
    # Normalize format
    fmt = format.lower()
    if fmt == "jpg":
        fmt = "jpeg"
    
    save_kwargs = {}
    if fmt in ("jpeg", "webp"):
        save_kwargs["quality"] = quality
    if fmt == "png":
        save_kwargs["compress_level"] = 6  # Balanced compression
    
    # Convert RGBA to RGB for JPEG
    if fmt == "jpeg" and img.mode == "RGBA":
        img = img.convert("RGB")
    
    img.save(buffer, format=fmt.upper(), **save_kwargs)
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_frame(
    encoded: str,
    return_numpy: bool = True,
) -> Any:
    """Decode a base64 encoded image frame.
    
    Args:
        encoded: Base64 encoded string
        return_numpy: If True, return numpy array; else return PIL Image
        
    Returns:
        Decoded image (numpy array or PIL Image)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL (Pillow) is required for image decoding. "
            "Install with: pip install Pillow"
        )
    
    # Decode base64
    image_bytes = base64.b64decode(encoded)
    
    # Open as PIL Image
    buffer = io.BytesIO(image_bytes)
    img = Image.open(buffer)
    
    if return_numpy:
        import numpy as np
        return np.array(img)
    
    return img


def encode_frame_to_data_uri(
    frame: Any,
    format: ImageFormat = DEFAULT_FORMAT,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> str:
    """Encode an image frame to a data URI string.
    
    Data URIs can be embedded directly in HTML/JSON and displayed in browsers.
    
    Args:
        frame: Image frame (numpy array, PIL Image, or bytes)
        format: Output format ("png", "jpeg", "jpg", "webp")
        quality: JPEG/WebP quality (1-100)
        
    Returns:
        Data URI string (e.g., "data:image/png;base64,...")
    """
    b64 = encode_frame(frame, format=format, quality=quality)
    
    # Normalize format for MIME type
    fmt = format.lower()
    if fmt == "jpg":
        fmt = "jpeg"
    
    return f"data:image/{fmt};base64,{b64}"


def decode_data_uri(data_uri: str, return_numpy: bool = True) -> Any:
    """Decode a data URI to an image.
    
    Args:
        data_uri: Data URI string (e.g., "data:image/png;base64,...")
        return_numpy: If True, return numpy array; else return PIL Image
        
    Returns:
        Decoded image
    """
    # Parse data URI
    if not data_uri.startswith("data:"):
        raise ValueError("Invalid data URI: must start with 'data:'")
    
    # Extract base64 part
    parts = data_uri.split(",", 1)
    if len(parts) != 2:
        raise ValueError("Invalid data URI: missing comma separator")
    
    b64_data = parts[1]
    return decode_frame(b64_data, return_numpy=return_numpy)


def get_image_info(encoded: str) -> Dict[str, Any]:
    """Get information about an encoded image without fully decoding it.
    
    Args:
        encoded: Base64 encoded string (or data URI)
        
    Returns:
        Dictionary with image info (width, height, format, mode, size_bytes)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL (Pillow) is required for image info. "
            "Install with: pip install Pillow"
        )
    
    # Handle data URI
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    
    # Decode base64
    image_bytes = base64.b64decode(encoded)
    
    # Open as PIL Image
    buffer = io.BytesIO(image_bytes)
    img = Image.open(buffer)
    
    return {
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "mode": img.mode,
        "size_bytes": len(image_bytes),
    }
