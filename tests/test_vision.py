"""Tests for vision module.

Tests core functionality:
- Image encoding/decoding (base64)
- Frame buffering and summaries
- Vision observation protocols
- Counterfactual logging with frames
"""

import pytest
import json
import tempfile
import os

# Skip all tests if PIL is not available
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass

NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass


# --- Encoding tests ---

@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_encode_decode_numpy_rgb():
    """Test encoding and decoding numpy RGB array."""
    from openenv.vision import encode_frame, decode_frame
    
    # Create test image (100x100 RGB)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80, 0] = 255  # Red square
    
    # Encode
    encoded = encode_frame(img, format="png")
    assert isinstance(encoded, str)
    assert len(encoded) > 0
    
    # Decode
    decoded = decode_frame(encoded, return_numpy=True)
    assert decoded.shape == (100, 100, 3)
    assert decoded[50, 50, 0] == 255  # Red channel in center


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_encode_decode_grayscale():
    """Test encoding and decoding grayscale array."""
    from openenv.vision import encode_frame, decode_frame
    
    # Create test image (64x64 grayscale)
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 128
    
    # Encode and decode
    encoded = encode_frame(img, format="png")
    decoded = decode_frame(encoded, return_numpy=True)
    
    assert decoded.shape[:2] == (64, 64)


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_encode_jpeg_quality():
    """Test JPEG encoding with quality setting."""
    from openenv.vision import encode_frame
    
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # High quality = larger file
    encoded_high = encode_frame(img, format="jpeg", quality=95)
    encoded_low = encode_frame(img, format="jpeg", quality=10)
    
    assert len(encoded_high) > len(encoded_low)


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_data_uri():
    """Test data URI encoding and decoding."""
    from openenv.vision import encode_frame_to_data_uri, decode_data_uri
    
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 1] = 255  # Green
    
    # Encode to data URI
    uri = encode_frame_to_data_uri(img, format="png")
    assert uri.startswith("data:image/png;base64,")
    
    # Decode
    decoded = decode_data_uri(uri, return_numpy=True)
    assert decoded.shape == (32, 32, 3)
    assert decoded[0, 0, 1] == 255


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_get_image_info():
    """Test getting image info without full decode."""
    from openenv.vision import encode_frame, get_image_info
    
    img = np.zeros((200, 150, 3), dtype=np.uint8)
    encoded = encode_frame(img, format="png")
    
    info = get_image_info(encoded)
    assert info["width"] == 150
    assert info["height"] == 200
    assert info["format"] == "PNG"
    assert info["mode"] == "RGB"
    assert info["size_bytes"] > 0


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
def test_encode_pil_image():
    """Test encoding PIL Image directly."""
    from openenv.vision import encode_frame, decode_frame
    
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    
    encoded = encode_frame(img, format="png")
    decoded = decode_frame(encoded, return_numpy=False)
    
    assert isinstance(decoded, Image.Image)
    assert decoded.size == (50, 50)


# --- Frame buffer tests ---

@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_frame_buffer_basic():
    """Test basic frame buffer operations."""
    from openenv.vision import FrameBuffer
    
    buffer = FrameBuffer(max_frames=5, store_thumbnails=True)
    
    # Add frames
    for i in range(3):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = i * 80  # Different red levels
        summary = buffer.add(img, metadata={"frame_idx": i})
        assert summary.step == i
        assert summary.width == 64
        assert summary.height == 64
    
    assert len(buffer) == 3
    assert buffer.get_latest(1)[0].step == 2


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_frame_buffer_max_frames():
    """Test frame buffer respects max_frames limit."""
    from openenv.vision import FrameBuffer
    
    buffer = FrameBuffer(max_frames=3)
    
    # Add more than max
    for i in range(5):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        buffer.add(img)
    
    assert len(buffer) == 3
    # Should have steps 2, 3, 4 (oldest dropped)
    steps = [f.step for f in buffer.get_frames()]
    assert steps == [2, 3, 4]


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_frame_summary():
    """Test frame summary creation."""
    from openenv.vision import summarize_frame
    
    img = np.zeros((128, 96, 3), dtype=np.uint8)
    
    summary = summarize_frame(
        img,
        step=5,
        include_encoded=True,
        include_thumbnail=True,
        metadata={"custom": "value"},
    )
    
    assert summary.step == 5
    assert summary.width == 96
    assert summary.height == 128
    assert summary.encoded is not None
    assert summary.thumbnail is not None
    assert summary.metadata["custom"] == "value"
    
    # to_dict should be JSON-serializable
    d = summary.to_dict()
    json.dumps(d)  # Should not raise


# --- Protocol tests ---

def test_vision_observation_protocol():
    """Test VisionObservation protocol checking."""
    from openenv.vision.protocols import VisionObservation, is_vision_observation
    
    class GoodObs:
        def get_frame(self):
            return "frame_data"
        
        def get_frame_info(self):
            return {"width": 100, "height": 100}
    
    class BadObs:
        pass
    
    assert is_vision_observation(GoodObs())
    assert not is_vision_observation(BadObs())
    assert not is_vision_observation({"frame": "data"})


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_extract_frame():
    """Test frame extraction from various observation types."""
    from openenv.vision.protocols import extract_frame
    
    # From dict
    frame = np.zeros((10, 10), dtype=np.uint8)
    obs_dict = {"frame": frame, "reward": 1.0}
    extracted = extract_frame(obs_dict)
    assert extracted is frame
    
    # From object with attribute
    class Obs:
        def __init__(self):
            self.image = frame
    
    extracted = extract_frame(Obs())
    assert extracted is frame
    
    # From observation with VisionObservation protocol
    class VisionObs:
        def get_frame(self):
            return frame
        
        def get_frame_info(self):
            return {}
    
    extracted = extract_frame(VisionObs())
    assert extracted is frame


def test_make_vision_observation():
    """Test creating standard vision observation dict."""
    from openenv.vision.protocols import make_vision_observation
    
    if NUMPY_AVAILABLE:
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
    else:
        frame = "mock_frame"
    
    obs = make_vision_observation(
        frame=frame,
        reward=1.5,
        done=False,
        info={"step": 0},
    )
    
    assert obs["frame"] is frame
    assert obs["reward"] == 1.5
    assert obs["done"] is False
    assert obs["info"]["step"] == 0
    
    if NUMPY_AVAILABLE:
        assert obs["frame_info"]["width"] == 80
        assert obs["frame_info"]["height"] == 100


# --- Logging tests ---

def test_trajectory_log_entry():
    """Test TrajectoryLogEntry serialization."""
    from openenv.counterfactual.logging import TrajectoryLogEntry
    
    entry = TrajectoryLogEntry(
        entry_type="step",
        trajectory_id="traj_001",
        timestamp=1234567890.0,
        step=0,
        kind="real",
        data={"observation": {"x": 1}},
    )
    
    # to_dict should be JSON-serializable
    d = entry.to_dict()
    json.dumps(d)
    
    # to_json should produce valid JSON
    j = entry.to_json()
    parsed = json.loads(j)
    assert parsed["trajectory_id"] == "traj_001"


def test_summary_to_log_entries():
    """Test converting TrajectorySummary to log entries."""
    from openenv.counterfactual.simulate import TrajectorySummary
    from openenv.counterfactual.logging import summary_to_log_entries
    
    summary = TrajectorySummary(
        horizon=10,
        final_observation={"x": 5},
        steps=3,
        observations=[{"x": 1}, {"x": 2}, {"x": 3}],
        total_reward=6.0,
        done=False,
    )
    
    entries = summary_to_log_entries(
        summary,
        trajectory_id="test_traj",
        kind="counterfactual",
    )
    
    # Should have 3 step entries + 1 summary entry
    assert len(entries) == 4
    
    step_entries = [e for e in entries if e.entry_type == "step"]
    summary_entries = [e for e in entries if e.entry_type == "summary"]
    
    assert len(step_entries) == 3
    assert len(summary_entries) == 1
    
    # Check summary entry
    assert summary_entries[0].data["total_reward"] == 6.0


def test_write_read_jsonl():
    """Test writing and reading JSONL files."""
    from openenv.counterfactual.logging import (
        TrajectoryLogEntry,
        write_jsonl,
        read_jsonl,
    )
    
    entries = [
        TrajectoryLogEntry(
            entry_type="step",
            trajectory_id="t1",
            timestamp=1.0,
            step=i,
            data={"value": i},
        )
        for i in range(3)
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        filepath = f.name
    
    try:
        # Write
        count = write_jsonl(entries, filepath)
        assert count == 3
        
        # Read back
        read_entries = list(read_jsonl(filepath))
        assert len(read_entries) == 3
        
        for i, entry in enumerate(read_entries):
            assert entry.step == i
            assert entry.data["value"] == i
    finally:
        os.unlink(filepath)


def test_tree_to_log_entries():
    """Test converting TrajectoryTree to log entries."""
    from openenv.counterfactual.tree import TrajectoryTree
    from openenv.counterfactual.simulate import TrajectorySummary
    from openenv.counterfactual.logging import tree_to_log_entries
    
    tree = TrajectoryTree()
    
    # Add root
    root_summary = TrajectorySummary(
        horizon=5,
        final_observation={"x": 0},
        steps=1,
        total_reward=0.0,
    )
    root_id = tree.add_root(root_summary, snapshot_id="snap_0", kind="real")
    
    # Add children (counterfactual branches)
    for i in range(2):
        child_summary = TrajectorySummary(
            horizon=5,
            final_observation={"x": i + 1},
            steps=2,
            total_reward=float(i + 1),
        )
        tree.add_child(
            parent_id=root_id,
            action=f"action_{i}",
            summary=child_summary,
            kind="counterfactual",
        )
    
    entries = tree_to_log_entries(tree)
    
    assert len(entries) == 3  # 1 root + 2 children
    
    # Check all entries are branches
    assert all(e.entry_type == "branch" for e in entries)
    
    # Root should have no parent
    root_entries = [e for e in entries if e.parent_trajectory_id is None]
    assert len(root_entries) == 1


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
def test_vision_frame_processor():
    """Test frame processor for vision-aware logging."""
    from openenv.counterfactual.logging import make_vision_frame_processor
    from openenv.counterfactual.simulate import TrajectorySummary
    from openenv.counterfactual.logging import summary_to_log_entries
    
    # Create a simple observation with frame
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    observations = [
        {"frame": frame, "reward": 1.0},
        {"frame": frame, "reward": 2.0},
    ]
    
    summary = TrajectorySummary(
        horizon=5,
        final_observation=observations[-1],
        steps=2,
        observations=observations,
        total_reward=3.0,
    )
    
    # Create frame processor
    processor = make_vision_frame_processor(
        include_thumbnail=True,
        include_full=False,
    )
    
    entries = summary_to_log_entries(
        summary,
        trajectory_id="vision_test",
        frame_processor=processor,
    )
    
    # Check that frame data was added
    step_entries = [e for e in entries if e.entry_type == "step"]
    assert len(step_entries) == 2
    
    # Each step should have frame data
    for entry in step_entries:
        assert "frame" in entry.data
        frame_data = entry.data["frame"]
        assert frame_data["width"] == 64
        assert frame_data["height"] == 64
        assert frame_data["thumbnail"] is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
