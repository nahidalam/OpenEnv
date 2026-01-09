"""Protocol definitions for counterfactual environments.

Defines the interfaces and protocols that counterfactual environments
should implement to support snapshotting, simulation, and comparison.

Design note: These protocols do not modify OpenEnv's Environment base class.
This avoids breaking existing environments. Environments optionally implement
these protocols to gain counterfactual capabilities.
"""

from typing import Protocol, runtime_checkable, Any, Optional
from .snapshot import Snapshot


@runtime_checkable
class Snapshotable(Protocol):
    """Protocol for environments that can be snapshotted and restored.
    
    Environments optionally implement this protocol to support counterfactual
    operations. The snapshot payload is opaque - it can be a Python object,
    bytes, or any serializable format.
    
    Example:
        from counterfactual.snapshot import (
            Snapshot, make_snapshot_id, capture_rng_state, restore_rng_state
        )
        
        class MyEnv:
            def snapshot(self) -> Snapshot:
                return Snapshot(
                    snapshot_id=make_snapshot_id(),
                    payload=self._state,
                    rng=capture_rng_state(env=self),
                    meta={"step_count": self.step_count}
                )
            
            def restore(self, snapshot: Snapshot) -> None:
                self._state = snapshot.payload
                if snapshot.rng:
                    restore_rng_state(snapshot.rng, env=self)
    """
    
    def snapshot(self) -> Snapshot:
        """Create a snapshot of the current state.
        
        Returns:
            Snapshot object containing the current state
        """
        ...
    
    def restore(self, snapshot: Snapshot) -> None:
        """Restore state from a snapshot.
        
        Args:
            snapshot: Snapshot object to restore from
        """
        ...


@runtime_checkable
class Stepable(Protocol):
    """Protocol for environments that support step operations.
    
    This is a minimal protocol to make types happy. It does not require
    environments to implement reset(), though it's recommended.
    
    Example:
        class MyEnv:
            def step(self, action: Any) -> Any:
                # ... step logic ...
                return observation
            
            def reset(self, **kwargs) -> Any:  # Optional
                # ... reset logic ...
                return observation
    """
    
    def step(self, action: Any) -> Any:
        """Step the environment with an action.
        
        Args:
            action: Action to take
            
        Returns:
            Observation (or dict with observation, reward, done, info)
        """
        ...
    
    def reset(self, **kwargs) -> Any:
        """Reset the environment to an initial state.
        
        This method is optional but recommended.
        
        Args:
            **kwargs: Optional reset parameters
            
        Returns:
            Initial observation (or dict with observation, info)
        """
        ...
