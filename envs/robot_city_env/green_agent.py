# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Green Agent Wrapper for Robot City.

A compute-aware action selection helper that:
- Uses greedy forward movement toward goal by default
- Only runs expensive counterfactual sampling when uncertainty is high

This is NOT a full RL agent - it's a lightweight policy wrapper
for efficient action selection with minimal compute.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from .models import RobotCityAction, RobotCityObservation
except ImportError:
    from models import RobotCityAction, RobotCityObservation


def compute_goal_direction(robot_state: Dict[str, Any]) -> Tuple[float, float]:
    """Compute unit vector toward goal.
    
    Args:
        robot_state: Robot dict with x, y, goal_x, goal_y
        
    Returns:
        (dx, dy) unit vector toward goal
    """
    dx = robot_state["goal_x"] - robot_state["x"]
    dy = robot_state["goal_y"] - robot_state["y"]
    dist = math.sqrt(dx * dx + dy * dy)
    
    if dist < 1e-6:
        return 0.0, 0.0
    
    return dx / dist, dy / dist


def greedy_action(robot_state: Dict[str, Any]) -> str:
    """Select greedy action toward goal.
    
    Args:
        robot_state: Robot dict with pose and goal
        
    Returns:
        Move string ("forward", "turn_left", "turn_right", etc.)
    """
    goal_dx, goal_dy = compute_goal_direction(robot_state)
    
    if abs(goal_dx) < 0.01 and abs(goal_dy) < 0.01:
        return "noop"  # At goal
    
    # Current heading
    theta = robot_state.get("theta", 0.0)
    heading_x = math.cos(theta)
    heading_y = math.sin(theta)
    
    # Dot product with goal direction
    alignment = heading_x * goal_dx + heading_y * goal_dy
    
    # Cross product for turn direction
    cross = heading_x * goal_dy - heading_y * goal_dx
    
    # If well-aligned, go forward
    if alignment > 0.7:
        return "forward"
    
    # Otherwise, turn toward goal
    if cross > 0:
        return "turn_left"
    else:
        return "turn_right"


def estimate_risk_from_overlay(uncertainty_overlay: Optional[str],
                                threshold: float = 0.3) -> bool:
    """Estimate if uncertainty overlay indicates high risk.
    
    Args:
        uncertainty_overlay: Base64 PNG of uncertainty heatmap
        threshold: Risk threshold (fraction of high-value pixels)
        
    Returns:
        True if risk is above threshold
    """
    if uncertainty_overlay is None:
        return False
    
    try:
        import base64
        import io
        from PIL import Image
        import numpy as np
        
        # Decode image
        img_bytes = base64.b64decode(uncertainty_overlay)
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        
        # Check center region (where robot is)
        h, w = arr.shape[:2]
        center = arr[h//3:2*h//3, w//3:2*w//3]
        
        # For RGBA, check alpha channel (intensity)
        if len(center.shape) == 3 and center.shape[2] >= 4:
            alpha = center[:, :, 3]
            risk_pixels = np.sum(alpha > 100) / alpha.size
            return risk_pixels > threshold
        
        # For RGB, check brightness
        if len(center.shape) == 3:
            brightness = center.mean(axis=2)
            risk_pixels = np.sum(brightness > 128) / brightness.size
            return risk_pixels > threshold
        
        return False
        
    except Exception:
        return False


def counterfactual_select(
    env,  # RobotCityEnvironment with snapshot/restore
    robot_id: int = 0,
    candidates: Optional[List[str]] = None,
    horizon: int = 5,
) -> str:
    """Select action via counterfactual rollouts.
    
    Simulates multiple action candidates and picks the best.
    This is expensive - only use when uncertainty is high.
    
    Args:
        env: Environment instance with snapshot/restore
        robot_id: Robot to control
        candidates: List of moves to try (default: common moves)
        horizon: Steps to simulate ahead
        
    Returns:
        Best move string
    """
    if candidates is None:
        candidates = ["forward", "turn_left", "turn_right", "noop"]
    
    # Snapshot current state
    snapshot = env.snapshot()
    
    best_move = "noop"
    best_score = float("-inf")
    
    for move in candidates:
        # Restore to snapshot
        env.restore(snapshot)
        
        # Simulate this action
        total_reward = 0.0
        
        for t in range(horizon):
            action = RobotCityAction(robot_id=robot_id, move=move if t == 0 else "forward")
            obs = env.step(action)
            total_reward += obs.reward or 0.0
            
            if obs.done:
                break
        
        if total_reward > best_score:
            best_score = total_reward
            best_move = move
    
    # Restore original state
    env.restore(snapshot)
    
    return best_move


def select_action_green(
    env_state: Dict[str, Any],
    last_obs: RobotCityObservation,
    robot_id: int = 0,
    risk_threshold: float = 0.3,
    env=None,  # Optional environment for counterfactual
) -> RobotCityAction:
    """Select action with minimal compute (Green Agent).
    
    Strategy:
    1. Default: use greedy forward toward goal (cheap)
    2. If uncertainty is high: run counterfactual sampling (expensive)
    
    Args:
        env_state: Current environment state dict
        last_obs: Last observation from environment
        robot_id: Robot to control
        risk_threshold: When to trigger expensive planning
        env: Optional environment instance for counterfactual (if not provided, only greedy)
        
    Returns:
        RobotCityAction with selected move
    """
    # Get robot state
    robots = env_state.get("robots", [])
    if robot_id >= len(robots):
        return RobotCityAction(robot_id=robot_id, move="noop")
    
    robot = robots[robot_id]
    
    # Check uncertainty
    high_risk = estimate_risk_from_overlay(
        last_obs.uncertainty_overlay,
        threshold=risk_threshold,
    )
    
    if high_risk and env is not None:
        # Expensive path: counterfactual planning
        move = counterfactual_select(env, robot_id=robot_id)
    else:
        # Cheap path: greedy toward goal
        move = greedy_action(robot)
    
    return RobotCityAction(robot_id=robot_id, move=move, speed=1.0)


# Convenience wrapper class
class GreenAgent:
    """Compute-aware agent wrapper for Robot City.
    
    Example:
        >>> agent = GreenAgent(risk_threshold=0.4)
        >>> action = agent.act(obs, env)
    """
    
    def __init__(
        self,
        robot_id: int = 0,
        risk_threshold: float = 0.3,
        use_counterfactual: bool = True,
    ):
        self.robot_id = robot_id
        self.risk_threshold = risk_threshold
        self.use_counterfactual = use_counterfactual
    
    def act(
        self,
        obs: RobotCityObservation,
        env=None,
    ) -> RobotCityAction:
        """Select action given observation.
        
        Args:
            obs: Current observation
            env: Optional environment for counterfactual planning
            
        Returns:
            Selected action
        """
        return select_action_green(
            env_state=obs.state,
            last_obs=obs,
            robot_id=self.robot_id,
            risk_threshold=self.risk_threshold,
            env=env if self.use_counterfactual else None,
        )
