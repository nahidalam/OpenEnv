"""Serialization utilities for counterfactual trajectories.

Provides JSON-safe conversion of TrajectorySummary and CandidateResult
objects, with special handling for vision observations containing frames.
"""

from typing import Any, Dict, List, Tuple, Sequence

from .simulate import TrajectorySummary
from .compare import CandidateResult


# Default keys to look for image frames in observations
DEFAULT_FRAME_KEYS: Tuple[str, ...] = ("rgb", "image", "frame")


def summary_to_json(
    summary: TrajectorySummary,
    *,
    include_observations: bool = False,
    frame_key_candidates: Sequence[str] = DEFAULT_FRAME_KEYS,
) -> Dict[str, Any]:
    """Convert TrajectorySummary to a JSON-serializable dictionary.
    
    Args:
        summary: TrajectorySummary to convert
        include_observations: If True, include serialized observations list
        frame_key_candidates: Keys to check for image frames in observations.
            If found, frames are encoded to base64. Default: ("rgb", "image", "frame")
    
    Returns:
        JSON-serializable dictionary with keys:
        - steps, horizon, done, total_reward, info (always included)
        - observations (if include_observations=True)
        - final_observation (always included, safely jsonified)
    """
    from openenv.vision import safe_jsonify_observation
    
    result: Dict[str, Any] = {
        "steps": summary.steps,
        "horizon": summary.horizon,
        "done": summary.done,
        "total_reward": summary.total_reward,
        "info": summary.info,
    }
    
    # Always include final_observation (safely converted)
    result["final_observation"] = _serialize_observation(
        summary.final_observation, 
        frame_key_candidates
    )
    
    # Optionally include observations list
    if include_observations and summary.observations:
        result["observations"] = [
            _serialize_observation(obs, frame_key_candidates)
            for obs in summary.observations
        ]
    
    return result


def candidate_results_to_json(results: List[CandidateResult]) -> List[Dict[str, Any]]:
    """Convert a list of CandidateResult to JSON-serializable dictionaries.
    
    Each result includes score, candidate/action representation, and summary
    (without observations for efficiency).
    
    Args:
        results: List of CandidateResult objects
        
    Returns:
        List of JSON-serializable dictionaries
    """
    output = []
    
    for r in results:
        entry: Dict[str, Any] = {
            "candidate_id": r.candidate_id,
            "score": r.score,
            "summary": summary_to_json(r.summary, include_observations=False),
        }
        
        # Include action representation
        if r.action is not None:
            entry["action"] = _action_to_repr(r.action)
        if r.action_seq is not None:
            entry["action_seq"] = [_action_to_repr(a) for a in r.action_seq]
        
        output.append(entry)
    
    return output


def _serialize_observation(
    obs: Any, 
    frame_key_candidates: Sequence[str],
) -> Any:
    """Serialize a single observation, encoding frames if found.
    
    Args:
        obs: Observation to serialize
        frame_key_candidates: Keys to check for image frames
        
    Returns:
        JSON-serializable representation
    """
    from openenv.vision import safe_jsonify_observation, encode_image_to_base64
    
    # Check if observation is a dict-like with frame keys
    if isinstance(obs, dict):
        result = {}
        for key, value in obs.items():
            if key in frame_key_candidates and _is_image_array(value):
                # Encode frame to base64
                result[key] = encode_image_to_base64(value, format="jpg", quality=85)
            else:
                result[key] = safe_jsonify_observation(value)
        return result
    
    # Check if observation has frame attributes
    if hasattr(obs, "__dict__"):
        result = {}
        for key in frame_key_candidates:
            if hasattr(obs, key):
                value = getattr(obs, key)
                if _is_image_array(value):
                    result[key] = encode_image_to_base64(value, format="jpg", quality=85)
        
        # If we found frames, merge with safe_jsonify of the rest
        if result:
            base = safe_jsonify_observation(obs)
            if isinstance(base, dict):
                base.update(result)
                return base
            return result
    
    # Fallback to safe_jsonify
    return safe_jsonify_observation(obs)


def _is_image_array(value: Any) -> bool:
    """Check if value looks like an image array (HxWxC or HxW, uint8)."""
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


def _action_to_repr(action: Any) -> Any:
    """Convert action to JSON-safe representation.
    
    Args:
        action: Action to convert
        
    Returns:
        JSON-serializable representation (str, dict, or primitive)
    """
    # Handle None
    if action is None:
        return None
    
    # Handle primitives
    if isinstance(action, (bool, int, float, str)):
        return action
    
    # Handle dict
    if isinstance(action, dict):
        return {str(k): _action_to_repr(v) for k, v in action.items()}
    
    # Handle list/tuple
    if isinstance(action, (list, tuple)):
        return [_action_to_repr(item) for item in action]
    
    # Try to convert dataclass/object to dict
    if hasattr(action, "__dict__"):
        try:
            return {k: _action_to_repr(v) for k, v in action.__dict__.items() 
                    if not k.startswith("_")}
        except Exception:
            pass
    
    # Fallback to string
    return str(action)
