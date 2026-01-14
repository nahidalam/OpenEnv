"""Trajectory logging helpers for counterfactual environments.

Provides utilities for exporting trajectories to JSONL format,
with support for vision observations and frame summaries.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator, Union, Callable, Tuple, Sequence

from .simulate import TrajectorySummary
from .tree import TrajectoryTree, TrajectoryNode
from .compare import CandidateResult


# Default keys to look for image frames in observations
DEFAULT_FRAME_KEYS: Tuple[str, ...] = ("rgb", "image", "frame")


@dataclass
class TrajectoryLogEntry:
    """A single log entry for trajectory logging.
    
    Designed for JSONL output where each line is one entry.
    """
    entry_type: str  # "step", "branch", "summary", "metadata"
    trajectory_id: str
    timestamp: float
    step: Optional[int] = None
    parent_trajectory_id: Optional[str] = None
    kind: str = "real"  # "real" or "counterfactual"
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "entry_type": self.entry_type,
            "trajectory_id": self.trajectory_id,
            "timestamp": self.timestamp,
            "step": self.step,
            "parent_trajectory_id": self.parent_trajectory_id,
            "kind": self.kind,
            "data": self.data,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


def summary_to_log_entries(
    summary: TrajectorySummary,
    trajectory_id: str,
    parent_id: Optional[str] = None,
    kind: str = "real",
    frame_processor: Optional[Callable[[Any, int], Dict[str, Any]]] = None,
) -> List[TrajectoryLogEntry]:
    """Convert a TrajectorySummary to log entries.
    
    Args:
        summary: TrajectorySummary to convert
        trajectory_id: Unique ID for this trajectory
        parent_id: Parent trajectory ID (for counterfactual branches)
        kind: "real" or "counterfactual"
        frame_processor: Optional function to process observation frames.
                        Called as frame_processor(observation, step) -> dict.
                        Use this to extract and encode frames from observations.
    
    Returns:
        List of TrajectoryLogEntry objects
    """
    entries = []
    ts = time.time()
    
    # Add step entries if observations are recorded
    for i, obs in enumerate(summary.observations):
        step_data = {"observation": _safe_serialize(obs)}
        
        # Process frame if processor provided
        if frame_processor is not None:
            try:
                frame_data = frame_processor(obs, i)
                if frame_data:
                    step_data["frame"] = frame_data
            except Exception:
                pass  # Skip frame processing on error
        
        entries.append(TrajectoryLogEntry(
            entry_type="step",
            trajectory_id=trajectory_id,
            timestamp=ts,
            step=i,
            parent_trajectory_id=parent_id,
            kind=kind,
            data=step_data,
        ))
    
    # Add summary entry
    summary_data = {
        "horizon": summary.horizon,
        "steps": summary.steps,
        "total_reward": summary.total_reward,
        "done": summary.done,
        "info": summary.info,
        "final_observation": _safe_serialize(summary.final_observation),
    }
    
    # Process final frame if processor provided
    if frame_processor is not None and summary.final_observation is not None:
        try:
            frame_data = frame_processor(summary.final_observation, summary.steps - 1)
            if frame_data:
                summary_data["final_frame"] = frame_data
        except Exception:
            pass
    
    entries.append(TrajectoryLogEntry(
        entry_type="summary",
        trajectory_id=trajectory_id,
        timestamp=ts,
        step=summary.steps,
        parent_trajectory_id=parent_id,
        kind=kind,
        data=summary_data,
    ))
    
    return entries


def tree_to_log_entries(
    tree: TrajectoryTree,
    frame_processor: Optional[Callable[[Any, int], Dict[str, Any]]] = None,
) -> List[TrajectoryLogEntry]:
    """Convert a TrajectoryTree to log entries.
    
    Produces entries for all nodes in the tree, preserving
    the parent-child relationships for reconstruction.
    
    Args:
        tree: TrajectoryTree to convert
        frame_processor: Optional function to process frames
        
    Returns:
        List of TrajectoryLogEntry objects
    """
    entries = []
    ts = time.time()
    
    # Convert each node
    for row in tree.to_jsonl_rows():
        entry = TrajectoryLogEntry(
            entry_type="branch",
            trajectory_id=row["node_id"],
            timestamp=row.get("timestamp", ts),
            step=row.get("depth", 0),
            parent_trajectory_id=row.get("parent_id"),
            kind=row.get("kind", "real"),
            data={
                "snapshot_id": row.get("snapshot_id"),
                "action_repr": row.get("action_repr"),
                "summary": row.get("summary", {}),
            },
        )
        entries.append(entry)
    
    return entries


def write_jsonl(
    entries: List[TrajectoryLogEntry],
    filepath: str,
    append: bool = False,
) -> int:
    """Write log entries to a JSONL file.
    
    Args:
        entries: List of TrajectoryLogEntry objects
        filepath: Path to output file
        append: If True, append to existing file
        
    Returns:
        Number of entries written
    """
    mode = "a" if append else "w"
    with open(filepath, mode) as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")
    return len(entries)


def read_jsonl(filepath: str) -> Iterator[TrajectoryLogEntry]:
    """Read log entries from a JSONL file.
    
    Args:
        filepath: Path to input file
        
    Yields:
        TrajectoryLogEntry objects
    """
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield TrajectoryLogEntry(
                entry_type=data["entry_type"],
                trajectory_id=data["trajectory_id"],
                timestamp=data["timestamp"],
                step=data.get("step"),
                parent_trajectory_id=data.get("parent_trajectory_id"),
                kind=data.get("kind", "real"),
                data=data.get("data", {}),
            )


def _safe_serialize(obj: Any) -> Any:
    """Safely serialize an object to JSON-compatible format.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation
    """
    if obj is None:
        return None
    
    # Try direct serialization
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    
    # Try dict conversion
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    
    if hasattr(obj, "__dict__"):
        try:
            d = {k: _safe_serialize(v) for k, v in obj.__dict__.items() 
                 if not k.startswith("_")}
            json.dumps(d)
            return d
        except Exception:
            pass
    
    # Fall back to string representation
    return str(obj)


# Convenience function for vision-aware logging
def make_vision_frame_processor(
    include_thumbnail: bool = True,
    include_full: bool = False,
    thumbnail_size: tuple = (64, 64),
    format: str = "png",
) -> Callable[[Any, int], Dict[str, Any]]:
    """Create a frame processor for vision observations.
    
    This creates a processor function that can be passed to
    summary_to_log_entries() to extract and encode frames.
    
    Args:
        include_thumbnail: Include small thumbnail (encoded as base64)
        include_full: Include full encoded frame
        thumbnail_size: Size for thumbnails
        format: Encoding format ("png" or "jpg")
        
    Returns:
        Frame processor function
    """
    def processor(obs: Any, step: int) -> Dict[str, Any]:
        # Try to import vision module
        try:
            from openenv.vision import summarize_frame, encode_image_to_base64
            from openenv.vision.protocols import extract_frame
        except ImportError:
            return {}
        
        # Extract frame from observation
        frame = extract_frame(obs)
        if frame is None:
            return {}
        
        # Create summary with new API
        result = summarize_frame(frame)
        result["step"] = step
        
        # Optionally add encoded frame
        if include_full:
            result["encoded"] = encode_image_to_base64(frame, format=format)
        
        # Optionally add thumbnail
        if include_thumbnail:
            try:
                from PIL import Image
                import numpy as np
                
                # Resize frame to thumbnail
                if isinstance(frame, np.ndarray):
                    img = Image.fromarray(frame)
                else:
                    img = frame
                thumb = img.resize(thumbnail_size, Image.Resampling.LANCZOS)
                result["thumbnail"] = encode_image_to_base64(np.array(thumb), format=format)
            except Exception:
                pass
        
        return result
    
    return processor


def make_real_step_row(
    step_idx: int,
    observation: Any,
    action: Any,
    reward: Optional[float],
    done: Optional[bool],
    *,
    include_frame: bool = False,
    frame_key_candidates: Sequence[str] = DEFAULT_FRAME_KEYS,
) -> Dict[str, Any]:
    """Create a JSON-serializable row for a real environment step.
    
    Args:
        step_idx: Step index in the episode
        observation: Observation from the environment
        action: Action taken
        reward: Reward received (or None)
        done: Done flag (or None)
        include_frame: If True, encode frames found in observation
        frame_key_candidates: Keys to check for image frames
        
    Returns:
        JSON-serializable dictionary
    """
    from openenv.vision import safe_jsonify_observation
    
    row: Dict[str, Any] = {
        "type": "real_step",
        "step_idx": step_idx,
        "timestamp": time.time(),
        "reward": reward,
        "done": done,
    }
    
    # Serialize action
    row["action"] = _serialize_action(action)
    
    # Serialize observation
    if include_frame:
        row["observation"] = _serialize_observation_with_frames(
            observation, frame_key_candidates
        )
    else:
        row["observation"] = safe_jsonify_observation(observation)
    
    return row


def make_counterfactual_row(
    step_idx: int,
    branch_id: str,
    candidate: Any,
    result: CandidateResult,
    *,
    include_frame: bool = False,
) -> Dict[str, Any]:
    """Create a JSON-serializable row for a counterfactual branch.
    
    Args:
        step_idx: Step index where branching occurred
        branch_id: Unique identifier for this branch
        candidate: The candidate action or action sequence
        result: CandidateResult from comparison
        include_frame: If True, encode frames in final observation
        
    Returns:
        JSON-serializable dictionary
    """
    from openenv.vision import safe_jsonify_observation
    from .serialize import summary_to_json
    
    row: Dict[str, Any] = {
        "type": "counterfactual",
        "step_idx": step_idx,
        "branch_id": branch_id,
        "timestamp": time.time(),
        "candidate_id": result.candidate_id,
        "score": result.score,
    }
    
    # Serialize candidate
    row["candidate"] = _serialize_action(candidate)
    
    # Serialize summary (without full observations for efficiency)
    row["summary"] = summary_to_json(
        result.summary,
        include_observations=False,
    )
    
    # Optionally include final observation with frame
    if include_frame and result.summary.final_observation is not None:
        row["final_observation"] = _serialize_observation_with_frames(
            result.summary.final_observation,
            DEFAULT_FRAME_KEYS,
        )
    
    return row


def rows_from_tree(tree: TrajectoryTree) -> List[Dict[str, Any]]:
    """Convert a TrajectoryTree to a list of JSON-serializable rows.
    
    Each node in the tree becomes one row, preserving parent-child
    relationships via parent_id field.
    
    Args:
        tree: TrajectoryTree to convert
        
    Returns:
        List of JSON-serializable dictionaries
    """
    rows = []
    
    for node_dict in tree.to_jsonl_rows():
        row: Dict[str, Any] = {
            "type": "tree_node",
            "node_id": node_dict["node_id"],
            "parent_id": node_dict.get("parent_id"),
            "kind": node_dict.get("kind", "real"),
            "depth": node_dict.get("depth", 0),
            "timestamp": node_dict.get("timestamp", time.time()),
            "snapshot_id": node_dict.get("snapshot_id"),
            "action_repr": node_dict.get("action_repr", ""),
            "summary": node_dict.get("summary", {}),
        }
        rows.append(row)
    
    return rows


def _serialize_observation_with_frames(
    obs: Any,
    frame_key_candidates: Sequence[str],
) -> Any:
    """Serialize observation, encoding any frames found.
    
    Args:
        obs: Observation to serialize
        frame_key_candidates: Keys to check for image frames
        
    Returns:
        JSON-serializable representation
    """
    from openenv.vision import safe_jsonify_observation, encode_image_to_base64
    
    # Check if observation is a dict with frame keys
    if isinstance(obs, dict):
        result = {}
        for key, value in obs.items():
            if key in frame_key_candidates and _is_image_array(value):
                result[key] = encode_image_to_base64(value, format="jpg", quality=85)
            else:
                result[key] = safe_jsonify_observation(value)
        return result
    
    # Check if observation has frame attributes
    if hasattr(obs, "__dict__"):
        result = {}
        has_frames = False
        for key in frame_key_candidates:
            if hasattr(obs, key):
                value = getattr(obs, key)
                if _is_image_array(value):
                    result[key] = encode_image_to_base64(value, format="jpg", quality=85)
                    has_frames = True
        
        if has_frames:
            base = safe_jsonify_observation(obs)
            if isinstance(base, dict):
                base.update(result)
                return base
            return result
    
    # Fallback
    return safe_jsonify_observation(obs)


def _is_image_array(value: Any) -> bool:
    """Check if value looks like an image array."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.ndim == 3 and value.shape[2] in (1, 3, 4):
                return value.dtype == np.uint8
            if value.ndim == 2:
                return value.dtype == np.uint8
    except ImportError:
        pass
    return False


def _serialize_action(action: Any) -> Any:
    """Serialize action to JSON-safe representation.
    
    Args:
        action: Action to serialize
        
    Returns:
        JSON-serializable representation
    """
    if action is None:
        return None
    
    if isinstance(action, (bool, int, float, str)):
        return action
    
    if isinstance(action, dict):
        return {str(k): _serialize_action(v) for k, v in action.items()}
    
    if isinstance(action, (list, tuple)):
        return [_serialize_action(item) for item in action]
    
    # Try dataclass/object conversion
    if hasattr(action, "__dict__"):
        try:
            return {k: _serialize_action(v) for k, v in action.__dict__.items()
                    if not k.startswith("_")}
        except Exception:
            pass
    
    return str(action)
