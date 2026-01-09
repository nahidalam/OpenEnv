"""Comparison functionality for counterfactual trajectories.

Provides the ability to compare multiple counterfactual action sequences
and evaluate their outcomes. Ensures no mutation leakage by always
restoring the real environment state after comparison.
"""

from dataclasses import dataclass
from typing import List, Any, Callable, Union
from .protocols import Snapshotable
from .simulate import simulate, TrajectorySummary
from .snapshot import Snapshot, make_snapshot_id


@dataclass
class CandidateResult:
    """Result of evaluating a single candidate action or sequence.
    
    Attributes:
        candidate_id: Unique identifier for this candidate
        action: Single action (if candidate was a single action)
        action_seq: Action sequence (if candidate was a sequence)
        summary: TrajectorySummary from simulation
        score: Computed score for this candidate
    """
    candidate_id: str
    summary: TrajectorySummary
    score: float
    action: Any = None
    action_seq: List[Any] = None
    
    def __repr__(self) -> str:
        action_str = f"action={self.action}" if self.action is not None else f"action_seq={self.action_seq}"
        return (
            f"CandidateResult(candidate_id={self.candidate_id!r}, "
            f"{action_str}, score={self.score:.2f})"
        )


def default_scorer(summary: TrajectorySummary) -> float:
    """Default scoring function for trajectory summaries.
    
    Uses total_reward if available, otherwise returns 0.0.
    
    Args:
        summary: TrajectorySummary to score
        
    Returns:
        Score value (total_reward if available, else 0.0)
    """
    if summary.total_reward is not None:
        return summary.total_reward
    return 0.0


def compare(
    env: Snapshotable,
    base_snapshot: Snapshot,
    candidates: Union[List[Any], List[List[Any]]],
    horizon: int,
    *,
    scorer: Callable[[TrajectorySummary], float] = default_scorer,
    record_observations: bool = False,
    stop_on_done: bool = True,
) -> List[CandidateResult]:
    """Compare multiple counterfactual action candidates.
    
    This function:
    1. Takes a snapshot of the real environment state
    2. For each candidate: restores base_snapshot, simulates, scores
    3. Always restores the real environment state at the end
    
    This ensures compare() never mutates the live episode.
    
    Args:
        env: Environment that implements Snapshotable (for restore) and Stepable (for step)
        base_snapshot: Snapshot to start all simulations from (counterfactual starting point)
        candidates: List of candidates. Each candidate can be:
                   - A single action (Any)
                   - An action sequence (List[Any])
        horizon: Maximum number of steps per simulation
        scorer: Function to score a TrajectorySummary. Default uses total_reward.
        record_observations: If True, record all observations in summaries
        stop_on_done: If True, stop simulation early when done=True
        
    Returns:
        List of CandidateResult objects, sorted by score descending (best first)
    """
    # Take snapshot of real environment state (current live episode)
    real_snapshot = env.snapshot()
    
    # Normalize candidates: determine if they're single actions or sequences
    if not candidates:
        # No candidates, return empty list
        env.restore(real_snapshot)
        return []
    
    # Determine format:
    # - list[list[Any]]: each candidate is a sequence (list of actions)
    # - list[Any]: each candidate is a single action
    # Check first candidate to determine format
    first_candidate = candidates[0]
    
    # If first candidate is a list, check if it contains lists (nested) or actions
    if isinstance(first_candidate, list):
        # Check if nested: list[list[Any]] vs flat: list[Any]
        # If first element of first candidate is also a list, it's nested
        if len(first_candidate) > 0 and isinstance(first_candidate[0], list):
            # Nested: list[list[Any]] - each candidate is a sequence
            is_sequence_format = True
        else:
            # Could be list[list[Any]] (each inner list is a sequence) or list[Any] (single actions)
            # If all candidates are lists, assume they're sequences
            # Otherwise, treat as single actions
            all_are_lists = all(isinstance(c, list) for c in candidates)
            is_sequence_format = all_are_lists
    else:
        # First candidate is not a list - treat all as single actions (list[Any])
        is_sequence_format = False
    
    results: List[CandidateResult] = []
    
    try:
        for idx, candidate in enumerate(candidates):
            # Normalize candidate to action sequence
            if is_sequence_format:
                # Candidate is already a sequence
                action_seq = candidate if isinstance(candidate, list) else [candidate]
                single_action = None
            else:
                # Candidate is a single action, wrap it
                action_seq = [candidate]
                single_action = candidate
            
            # Generate candidate ID
            candidate_id = f"candidate_{idx:03d}"
            
            # Restore base snapshot for this simulation
            env.restore(base_snapshot)
            # Restore RNG state if present
            if base_snapshot.rng:
                from .snapshot import restore_rng_state
                restore_rng_state(base_snapshot.rng, env=env)
            
            # Simulate this candidate
            summary = simulate(
                env=env,
                snapshot=base_snapshot,
                action_seq=action_seq,
                horizon=horizon,
                record_observations=record_observations,
                restore_to=None,  # Don't restore here, we'll restore real_snapshot at end
                stop_on_done=stop_on_done,
            )
            
            # Compute score
            score = scorer(summary)
            
            # Create result
            result = CandidateResult(
                candidate_id=candidate_id,
                action=single_action,
                action_seq=action_seq if is_sequence_format else None,
                summary=summary,
                score=score,
            )
            results.append(result)
    
    finally:
        # Always restore real environment state (prevents mutation leakage)
        env.restore(real_snapshot)
        # Restore RNG state if present
        if real_snapshot.rng:
            from .snapshot import restore_rng_state
            restore_rng_state(real_snapshot.rng, env=env)
    
    # Sort by score descending (best first)
    results.sort(key=lambda r: r.score, reverse=True)
    
    return results
