"""Tests for openenv.vision.frames module.

Tests:
- PNG roundtrip exact match
- JPG roundtrip approximate match
- pack/unpack length + shape
- summarize_frame keys exist
- safe_jsonify_observation handles image and large array
"""

import pytest
from dataclasses import dataclass

# Check if dependencies are available
NUMPY_AVAILABLE = False
PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not NUMPY_AVAILABLE or not PIL_AVAILABLE, reason="numpy and PIL required")
class TestEncodeDecode:
    """Test encode/decode roundtrips."""
    
    def test_png_roundtrip_exact_match(self):
        """PNG encoding should produce exact pixel match on roundtrip."""
        from openenv.vision import encode_image_to_base64, decode_image_from_base64
        
        # Create test image with specific pixel values
        original = np.zeros((64, 64, 3), dtype=np.uint8)
        original[10:20, 10:20, 0] = 255  # Red square
        original[30:40, 30:40, 1] = 128  # Green square (specific value)
        original[50:60, 50:60, 2] = 64   # Blue square (specific value)
        
        # Encode as PNG (lossless)
        encoded = encode_image_to_base64(original, format="png")
        
        # Decode back
        decoded = decode_image_from_base64(encoded)
        
        # Should be exact match
        assert decoded.shape == original.shape
        assert decoded.dtype == np.uint8
        np.testing.assert_array_equal(decoded, original)
    
    def test_jpg_roundtrip_approximate_match(self):
        """JPEG encoding should produce approximate match (lossy)."""
        from openenv.vision import encode_image_to_base64, decode_image_from_base64
        
        # Create test image
        original = np.zeros((64, 64, 3), dtype=np.uint8)
        original[20:40, 20:40] = [200, 100, 50]  # Orange square
        
        # Encode as JPEG (lossy)
        encoded = encode_image_to_base64(original, format="jpg", quality=95)
        
        # Decode back
        decoded = decode_image_from_base64(encoded)
        
        # Should have same shape
        assert decoded.shape == original.shape
        assert decoded.dtype == np.uint8
        
        # Values should be close but not exact due to compression
        # Check that the orange region is approximately correct
        center_original = original[30, 30]
        center_decoded = decoded[30, 30]
        
        # Allow some tolerance for JPEG artifacts
        assert np.allclose(center_original, center_decoded, atol=20)
    
    def test_grayscale_roundtrip(self):
        """Grayscale images should roundtrip correctly."""
        from openenv.vision import encode_image_to_base64, decode_image_from_base64
        
        # Create grayscale image
        original = np.zeros((32, 32), dtype=np.uint8)
        original[10:20, 10:20] = 128
        
        # Encode and decode
        encoded = encode_image_to_base64(original, format="png")
        decoded = decode_image_from_base64(encoded)
        
        # Grayscale might come back as HxW or HxWx1
        if decoded.ndim == 3:
            decoded = decoded.squeeze()
        
        assert decoded.shape == original.shape
        np.testing.assert_array_equal(decoded, original)
    
    def test_pil_image_input(self):
        """Should accept PIL Image as input."""
        from openenv.vision import encode_image_to_base64, decode_image_from_base64
        from PIL import Image
        
        # Create PIL image
        pil_img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        
        # Encode and decode
        encoded = encode_image_to_base64(pil_img, format="png")
        decoded = decode_image_from_base64(encoded)
        
        assert decoded.shape == (50, 50, 3)
        assert decoded[25, 25, 0] == 255  # Red channel
        assert decoded[25, 25, 1] == 0    # Green channel
        assert decoded[25, 25, 2] == 0    # Blue channel


@pytest.mark.skipif(not NUMPY_AVAILABLE or not PIL_AVAILABLE, reason="numpy and PIL required")
class TestPackUnpack:
    """Test pack_frames and unpack_frames."""
    
    def test_pack_unpack_length(self):
        """pack/unpack should preserve frame count."""
        from openenv.vision import pack_frames, unpack_frames
        
        # Create 5 frames
        frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
        
        # Pack and unpack
        packed = pack_frames(frames, format="jpg", quality=85)
        unpacked = unpack_frames(packed)
        
        assert len(packed) == 5
        assert len(unpacked) == 5
    
    def test_pack_unpack_shape(self):
        """pack/unpack should preserve frame shapes."""
        from openenv.vision import pack_frames, unpack_frames
        
        # Create frames of specific shape
        frames = [np.zeros((64, 48, 3), dtype=np.uint8) for _ in range(3)]
        
        # Pack and unpack
        packed = pack_frames(frames, format="png")
        unpacked = unpack_frames(packed)
        
        for frame in unpacked:
            assert frame.shape == (64, 48, 3)
    
    def test_pack_max_frames(self):
        """max_frames should limit output count."""
        from openenv.vision import pack_frames
        
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(10)]
        
        packed = pack_frames(frames, max_frames=3)
        assert len(packed) == 3
    
    def test_pack_stride(self):
        """stride should sample every Nth frame."""
        from openenv.vision import pack_frames
        
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(10)]
        
        # stride=2 should give 5 frames (0, 2, 4, 6, 8)
        packed = pack_frames(frames, stride=2)
        assert len(packed) == 5
        
        # stride=3 should give 4 frames (0, 3, 6, 9)
        packed = pack_frames(frames, stride=3)
        assert len(packed) == 4
    
    def test_pack_stride_and_max_frames(self):
        """stride and max_frames should work together."""
        from openenv.vision import pack_frames
        
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(20)]
        
        # stride=2 gives 10 frames, max_frames=3 limits to 3
        packed = pack_frames(frames, stride=2, max_frames=3)
        assert len(packed) == 3
    
    def test_pack_empty_list(self):
        """Empty list should return empty list."""
        from openenv.vision import pack_frames, unpack_frames
        
        packed = pack_frames([])
        assert packed == []
        
        unpacked = unpack_frames([])
        assert unpacked == []


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
class TestSummarizeFrame:
    """Test summarize_frame function."""
    
    def test_summarize_frame_keys_exist(self):
        """summarize_frame should return dict with required keys."""
        from openenv.vision import summarize_frame
        
        frame = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        
        summary = summarize_frame(frame)
        
        assert "height" in summary
        assert "width" in summary
        assert "channels" in summary
        assert "mean" in summary
        assert "std" in summary
    
    def test_summarize_frame_values(self):
        """summarize_frame should return correct values."""
        from openenv.vision import summarize_frame
        
        # Create known image
        frame = np.full((64, 48, 3), 100, dtype=np.uint8)
        
        summary = summarize_frame(frame)
        
        assert summary["height"] == 64
        assert summary["width"] == 48
        assert summary["channels"] == 3
        assert summary["mean"] == 100.0
        assert summary["std"] == 0.0
    
    def test_summarize_grayscale_frame(self):
        """summarize_frame should handle grayscale."""
        from openenv.vision import summarize_frame
        
        frame = np.zeros((32, 32), dtype=np.uint8)
        
        summary = summarize_frame(frame)
        
        assert summary["height"] == 32
        assert summary["width"] == 32
        assert summary["channels"] == 1


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
class TestSafeJsonifyObservation:
    """Test safe_jsonify_observation function."""
    
    def test_handles_image(self):
        """Should convert HxWx3 uint8 array to base64 jpg."""
        from openenv.vision import safe_jsonify_observation
        
        # Create image-like array
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[20:40, 20:40] = [255, 0, 0]
        
        result = safe_jsonify_observation(img)
        
        # Should be a base64 string
        assert isinstance(result, str)
        # Base64 strings contain only these characters
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in result)
    
    def test_handles_large_array(self):
        """Should convert large array to summary dict."""
        from openenv.vision import safe_jsonify_observation
        
        # Create large 1D array (> 2048 elements)
        arr = np.zeros(5000, dtype=np.float32)
        
        result = safe_jsonify_observation(arr)
        
        # Should be a summary dict
        assert isinstance(result, dict)
        assert "shape" in result
        assert "dtype" in result
        assert "mean" in result
        assert "std" in result
        assert result["shape"] == [5000]
    
    def test_handles_small_array(self):
        """Should convert small array to list."""
        from openenv.vision import safe_jsonify_observation
        
        # Create small array (<= 2048 elements)
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        
        result = safe_jsonify_observation(arr)
        
        # Should be a list
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]
    
    def test_handles_dataclass(self):
        """Should convert dataclass to dict."""
        from openenv.vision import safe_jsonify_observation
        
        @dataclass
        class Observation:
            x: int
            y: float
            name: str
        
        obs = Observation(x=10, y=3.14, name="test")
        
        result = safe_jsonify_observation(obs)
        
        assert isinstance(result, dict)
        assert result["x"] == 10
        assert result["y"] == 3.14
        assert result["name"] == "test"
    
    def test_handles_nested_dataclass(self):
        """Should handle nested dataclasses."""
        from openenv.vision import safe_jsonify_observation
        
        @dataclass
        class Inner:
            value: int
        
        @dataclass
        class Outer:
            inner: Inner
            label: str
        
        obs = Outer(inner=Inner(value=42), label="test")
        
        result = safe_jsonify_observation(obs)
        
        assert isinstance(result, dict)
        assert isinstance(result["inner"], dict)
        assert result["inner"]["value"] == 42
    
    def test_handles_dict_with_array(self):
        """Should handle dict containing numpy array."""
        from openenv.vision import safe_jsonify_observation
        
        obs = {
            "frame": np.zeros((32, 32, 3), dtype=np.uint8),
            "reward": 1.5,
            "done": False,
        }
        
        result = safe_jsonify_observation(obs)
        
        assert isinstance(result, dict)
        assert isinstance(result["frame"], str)  # base64 encoded
        assert result["reward"] == 1.5
        assert result["done"] is False
    
    def test_handles_list_of_arrays(self):
        """Should handle list containing arrays."""
        from openenv.vision import safe_jsonify_observation
        
        obs = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        ]
        
        result = safe_jsonify_observation(obs)
        
        assert isinstance(result, list)
        assert result == [[1, 2, 3], [4, 5, 6]]
    
    def test_handles_primitives(self):
        """Should pass through primitive types."""
        from openenv.vision import safe_jsonify_observation
        
        assert safe_jsonify_observation(None) is None
        assert safe_jsonify_observation(42) == 42
        assert safe_jsonify_observation(3.14) == 3.14
        assert safe_jsonify_observation("hello") == "hello"
        assert safe_jsonify_observation(True) is True


@pytest.mark.skipif(not NUMPY_AVAILABLE or not PIL_AVAILABLE, reason="numpy and PIL required")
def test_import_works():
    """Verify that the main import path works."""
    from openenv.vision import encode_image_to_base64
    
    # Should be callable
    assert callable(encode_image_to_base64)
    
    # Quick smoke test
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    result = encode_image_to_base64(img)
    assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
