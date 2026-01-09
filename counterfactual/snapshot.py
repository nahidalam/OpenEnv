"""Snapshot functionality for counterfactual environments.

Provides state snapshotting and restoration capabilities for environments
that support counterfactual reasoning, including RNG state capture/restore.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import uuid


@dataclass
class Snapshot:
    """Represents a snapshot of an environment state.
    
    The payload is opaque - it can be a Python object, bytes, or any
    serializable format. The environment is responsible for serializing
    its own state.
    
    Attributes:
        snapshot_id: Unique identifier for this snapshot
        payload: Opaque environment state (can be any type)
        rng: Captured RNG states (dict mapping RNG names to states)
        meta: Optional metadata (timestamp, step_count, notes, etc.)
    """
    snapshot_id: str
    payload: Any
    rng: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Snapshot(snapshot_id={self.snapshot_id!r}, rng={'captured' if self.rng else 'none'})"


def make_snapshot_id() -> str:
    """Generate a short UUID string for snapshot identification.
    
    Returns:
        A short UUID string (e.g., "a1b2c3d4")
    """
    return uuid.uuid4().hex[:8]


def capture_rng_state(env: Optional[Any] = None) -> Dict[str, Any]:
    """Capture RNG state from global random, numpy, and optionally env-local RNGs.
    
    Supports:
    - Python random global state (random.getstate())
    - NumPy RNG state (if numpy is installed)
    - Environment-local RNGs:
      - env.rng with getstate()/setstate() (like random.Random)
      - env.np_rng with bit_generator.state (like numpy.random.Generator)
    
    If something doesn't exist, it's safely skipped.
    
    Args:
        env: Optional environment object that may have local RNG attributes
        
    Returns:
        Dictionary mapping RNG names to their captured states
    """
    rng_state: Dict[str, Any] = {}
    
    # Capture Python global random state
    try:
        rng_state["random"] = random.getstate()
    except Exception:
        pass  # Skip if not available
    
    # Capture NumPy global RNG state (if numpy is installed)
    try:
        import numpy as np
        # NumPy uses a global RNG state that can be captured
        # Note: numpy.random.get_state() captures the legacy RNG state
        # For newer Generator API, we'd need to capture the generator itself
        try:
            rng_state["numpy"] = np.random.get_state()
        except Exception:
            pass
        
        # Also try to capture numpy.random.default_rng() state if available
        try:
            # This is tricky - default_rng() returns a new generator each time
            # We can't capture it globally, but we can check if env has it
            pass
        except Exception:
            pass
    except ImportError:
        pass  # NumPy not installed, skip
    
    # Capture environment-local RNGs if env is provided
    if env is not None:
        # Check for env.rng (like random.Random instance)
        if hasattr(env, "rng"):
            rng_obj = getattr(env, "rng")
            if hasattr(rng_obj, "getstate"):
                try:
                    rng_state["env.rng"] = rng_obj.getstate()
                except Exception:
                    pass
        
        # Check for env.np_rng (like numpy.random.Generator)
        if hasattr(env, "np_rng"):
            np_rng = getattr(env, "np_rng")
            # NumPy Generator has bit_generator.state
            if hasattr(np_rng, "bit_generator"):
                try:
                    bit_gen = np_rng.bit_generator
                    if hasattr(bit_gen, "state"):
                        rng_state["env.np_rng"] = bit_gen.state
                except Exception:
                    pass
    
    return rng_state


def restore_rng_state(state: Dict[str, Any], env: Optional[Any] = None) -> None:
    """Restore RNG state from a captured state dictionary.
    
    Restores:
    - Python random global state
    - NumPy RNG state (if numpy is installed)
    - Environment-local RNGs (if env provided and state contains them)
    
    If something doesn't exist or fails, it's safely skipped.
    
    Args:
        state: Dictionary mapping RNG names to their states (from capture_rng_state)
        env: Optional environment object that may have local RNG attributes
    """
    # Restore Python global random state
    if "random" in state:
        try:
            random.setstate(state["random"])
        except Exception:
            pass  # Skip if restore fails
    
    # Restore NumPy global RNG state
    if "numpy" in state:
        try:
            import numpy as np
            try:
                np.random.set_state(state["numpy"])
            except Exception:
                pass
        except ImportError:
            pass  # NumPy not installed, skip
    
    # Restore environment-local RNGs if env is provided
    if env is not None:
        # Restore env.rng
        if "env.rng" in state:
            if hasattr(env, "rng"):
                rng_obj = getattr(env, "rng")
                if hasattr(rng_obj, "setstate"):
                    try:
                        rng_obj.setstate(state["env.rng"])
                    except Exception:
                        pass
        
        # Restore env.np_rng
        if "env.np_rng" in state:
            if hasattr(env, "np_rng"):
                np_rng = getattr(env, "np_rng")
                if hasattr(np_rng, "bit_generator"):
                    try:
                        bit_gen = np_rng.bit_generator
                        if hasattr(bit_gen, "state"):
                            bit_gen.state = state["env.np_rng"]
                    except Exception:
                        pass


def deepcopy_payload(payload: Any) -> Any:
    """Create a deep copy of a snapshot payload.
    
    This is a fallback helper for environments that don't provide
    their own snapshot/restore methods. Use copy.deepcopy() to create
    an independent copy of the payload.
    
    Warning: This may be too heavy for large environments. Prefer
    environments that implement their own snapshot/restore methods.
    
    Args:
        payload: The payload to deep copy
        
    Returns:
        Deep copy of the payload
    """
    return copy.deepcopy(payload)
