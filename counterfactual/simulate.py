"""Simulation functionality for counterfactual environments.

Provides the ability to simulate action sequences in a forked environment
state without affecting the main trajectory. Ensures no mutation leakage
by supporting restoration to a "real state" snapshot.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Dict, Union, Tuple
from .protocols import Snapshotable, Stepable
from .snapshot import Snapshot, restore_rng_state


@dataclass
class TrajectorySummary:
    """Summary of a simulated trajectory.
    
    Return values are generic to support different environment observation types.
    
    Attributes:
        horizon: Maximum horizon that was requested
        observations: List of observations (optional, can be empty for efficiency)
        final_observation: Last observation from the trajectory
        total_reward: Cumulative reward (None if rewards not available)
        done: Whether episode ended (None if done flag not available)
        info: Extra statistics dictionary (collapse_count, max_stress, etc.)
        steps: Actual number of steps taken
    """
    horizon: int
    final_observation: Any
    steps: int
    observations: List[Any] = field(default_factory=list)
    total_reward: Optional[float] = None
    done: Optional[bool] = None
    info: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        reward_str = f"{self.total_reward:.2f}" if self.total_reward is not None else "N/A"
        done_str = str(self.done) if self.done is not None else "N/A"
        return (
            f"TrajectorySummary(horizon={self.horizon}, steps={self.steps}, "
            f"reward={reward_str}, done={done_str})"
        )


def _extract_reward_done(obs: Any) -> Tuple[Optional[float], Optional[bool]]:
    """Extract reward and done from observation.
    
    Supports:
    - Dictionary with "reward" and "done" keys
    - Object with reward/done attributes
    - Tuple/list with (observation, reward, done) or (observation, reward, done, info)
    
    Args:
        obs: Observation that may contain reward/done information
        
    Returns:
        Tuple of (reward, done), either may be None if not found
    """
    reward = None
    done = None
    
    # Try dictionary access
    if isinstance(obs, dict):
        reward = obs.get("reward")
        done = obs.get("done")
    # Try attribute access
    elif hasattr(obs, "reward") or hasattr(obs, "done"):
        reward = getattr(obs, "reward", None)
        done = getattr(obs, "done", None)
    # Try tuple/list unpacking (common in gym-style APIs)
    elif isinstance(obs, (tuple, list)) and len(obs) >= 3:
        # Format: (observation, reward, done) or (observation, reward, done, info)
        reward = obs[1] if len(obs) > 1 else None
        done = obs[2] if len(obs) > 2 else None
    
    return reward, done


def simulate(
    env: Union[Snapshotable, Stepable, Any],
    snapshot: Snapshot,
    action_seq: Optional[List[Any]] = None,
    horizon: int = 1,
    *,
    policy: Optional[Callable[[int, Any], Any]] = None,
    record_observations: bool = False,
    restore_to: Optional[Snapshot] = None,
    stop_on_done: bool = True,
) -> TrajectorySummary:
    """Simulate a trajectory in a counterfactual environment.
    
    This function:
    1. Restores the environment from the provided snapshot
    2. Runs simulation for up to horizon steps
    3. Uses either action_seq or policy to determine actions
    4. Extracts reward/done from observations
    5. Optionally restores to a "real state" snapshot at the end
    
    To avoid mutation leakage, the caller should:
    - Take a snapshot of the real state before calling simulate
    - Pass that snapshot as restore_to parameter
    - Or manually restore after simulation
    
    Args:
        env: Environment that implements Snapshotable (for restore) and Stepable (for step)
        snapshot: Snapshot to restore from at the start
        action_seq: Optional sequence of actions to use. If shorter than horizon,
                   padded with None (meaning "noop" - policy decides).
        horizon: Maximum number of steps to simulate
        policy: Optional callable policy(t, last_obs) -> action.
                Called when action_seq is None or exhausted.
        record_observations: If True, store all observations in summary.
                            If False, only store final_observation (more efficient).
        restore_to: Optional snapshot to restore to at the end (prevents mutation leakage).
                   If None, environment state is left as-is after simulation.
        stop_on_done: If True, stop early when done flag is True
        
    Returns:
        TrajectorySummary with trajectory results
        
    Raises:
        ValueError: If neither action_seq nor policy is provided
    """
    # Validate inputs
    if action_seq is None and policy is None:
        raise ValueError("Must provide either action_seq or policy")
    
    # Restore snapshot at the start
    if isinstance(env, Snapshotable):
        env.restore(snapshot)
        # Restore RNG state if present
        if snapshot.rng:
            restore_rng_state(snapshot.rng, env=env)
    
    observations: List[Any] = []
    total_reward: Optional[float] = 0.0
    done: Optional[bool] = None
    info: Dict[str, Any] = {}
    last_obs: Any = None
    
    # Run simulation
    for t in range(horizon):
        # Determine action for this step
        if action_seq is not None:
            if t < len(action_seq):
                action = action_seq[t]
            else:
                # Pad with None (noop) - let policy decide if available
                if policy is not None:
                    action = policy(t, last_obs)
                else:
                    action = None  # Noop
        else:
            # Use policy
            action = policy(t, last_obs)
        
        # Step the environment
        step_result = env.step(action)
        
        # Extract observation, reward, done from step_result
        # step_result might be just observation, or dict, or tuple
        obs = step_result
        reward = None
        step_done = None
        step_info: Dict[str, Any] = {}
        
        if isinstance(step_result, dict):
            obs = step_result.get("observation", step_result)
            reward = step_result.get("reward")
            step_done = step_result.get("done")
            step_info = step_result.get("info", {})
        elif isinstance(step_result, (tuple, list)):
            # Gym-style: (obs, reward, done) or (obs, reward, done, info)
            if len(step_result) >= 1:
                obs = step_result[0]
            if len(step_result) >= 2:
                reward = step_result[1]
            if len(step_result) >= 3:
                step_done = step_result[2]
            if len(step_result) >= 4:
                step_info = step_result[3] if isinstance(step_result[3], dict) else {}
        else:
            # Try to extract reward/done from observation itself
            reward, step_done = _extract_reward_done(obs)
        
        # Store observation if requested
        if record_observations:
            observations.append(obs)
        
        last_obs = obs
        
        # Accumulate reward
        if reward is not None:
            if total_reward is None:
                total_reward = 0.0
            total_reward += reward
        
        # Update done flag
        if step_done is not None:
            done = step_done
        
        # Merge info dictionaries
        if isinstance(step_info, dict):
            info.update(step_info)
        
        # Early termination if done
        if stop_on_done and done is True:
            break
    
    # Create summary
    summary = TrajectorySummary(
        horizon=horizon,
        observations=observations,
        final_observation=last_obs,
        total_reward=total_reward,
        done=done,
        info=info,
        steps=t + 1  # t is 0-indexed, steps is count
    )
    
    # Restore to real state if requested (prevents mutation leakage)
    if restore_to is not None and isinstance(env, Snapshotable):
        env.restore(restore_to)
        # Restore RNG state if present
        if restore_to.rng:
            restore_rng_state(restore_to.rng, env=env)
    
    return summary
