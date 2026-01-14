# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Robot City Environment Implementation.

A visual multi-robot simulation environment with pedestrians, obstacles,
failure injection, and counterfactual planning support.
"""

import base64
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from ..models import RobotCityAction, RobotCityObservation
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from models import RobotCityAction, RobotCityObservation

from .sim.dynamics import (
    SimState, update_state, check_collisions, 
    compute_forward_progress, robot_at_goal
)
from .sim.render import (
    render_global_view, render_ego_view, render_heatmap,
    compute_uncertainty_heatmap, compute_future_heatmap
)
from .sim.scenarios import get_scenario, SCENARIOS
from .sim.failures import FailureConfig, apply_failures, apply_gps_drift, create_dropout_image


class RobotCityEnvironment(Environment):
    """
    Robot City: A visual multi-robot simulation environment.
    
    Features:
    - Multi-robot control with discrete actions
    - Visual RGB observations (128x128 top-down view)
    - Pedestrian dynamics with stochastic motion
    - Collision detection and near-miss events
    - Uncertainty overlays (collision risk heatmaps)
    - Future prediction heatmaps (pedestrian occupancy)
    - Failure injection modes for robustness testing
    - Snapshot/restore for counterfactual planning
    
    Example:
        >>> env = RobotCityEnvironment()
        >>> obs = env.reset(scenario="intersection_crosswalk", num_robots=2)
        >>> obs = env.step(RobotCityAction(robot_id=0, move="forward"))
    """
    
    # Environment does not support concurrent sessions by default
    SUPPORTS_CONCURRENT_SESSIONS = False
    
    def __init__(self):
        """Initialize the Robot City environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sim: Optional[SimState] = None
        self._failure_config = FailureConfig()
        self._max_steps = 200
        self._render_size = 128
        self._compute_overlays = True
        self._end_on_collision = False
        self._episode_reward = 0.0
        self._episode_events: List[str] = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario: str = "intersection_crosswalk",
        num_robots: int = 1,
        num_peds: int = 4,
        max_steps: int = 200,
        compute_overlays: bool = True,
        end_on_collision: bool = False,
        # Failure injection kwargs
        camera_dropout_prob: float = 0.0,
        gps_drift_sigma: float = 0.0,
        pedestrian_rush_prob: float = 0.0,
        sudden_obstacle_prob: float = 0.0,
        action_delay_prob: float = 0.0,
        **kwargs,
    ) -> RobotCityObservation:
        """
        Reset the environment to a new episode.
        
        Args:
            seed: Random seed for reproducibility
            episode_id: Optional episode identifier
            scenario: Scenario name ("intersection_crosswalk", "warehouse_aisles", 
                     "sidewalk_delivery", or "random")
            num_robots: Number of robots (1-4)
            num_peds: Number of pedestrians (0-10)
            max_steps: Maximum steps before episode ends
            compute_overlays: Whether to compute uncertainty/future heatmaps
            end_on_collision: Whether to end episode on first collision
            camera_dropout_prob: Probability of camera failure per step
            gps_drift_sigma: Standard deviation of GPS position noise
            pedestrian_rush_prob: Probability of pedestrian rush event
            sudden_obstacle_prob: Probability of sudden obstacle appearing
            action_delay_prob: Probability of action being delayed/ignored
            
        Returns:
            Initial RobotCityObservation
        """
        # Set up episode
        if episode_id is None:
            episode_id = str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)
        
        # Clamp parameters
        num_robots = max(1, min(4, num_robots))
        num_peds = max(0, min(10, num_peds))
        
        # Initialize simulation state
        self._sim = get_scenario(scenario, num_robots=num_robots, num_peds=num_peds, seed=seed)
        
        # Configure
        self._max_steps = max_steps
        self._compute_overlays = compute_overlays
        self._end_on_collision = end_on_collision
        self._episode_reward = 0.0
        self._episode_events = []
        
        # Set up failure config
        self._failure_config = FailureConfig(
            camera_dropout_prob=camera_dropout_prob,
            gps_drift_sigma=gps_drift_sigma,
            pedestrian_rush_prob=pedestrian_rush_prob,
            sudden_obstacle_prob=sudden_obstacle_prob,
            action_delay_prob=action_delay_prob,
        )
        
        # Build initial observation
        return self._build_observation(events=["reset"], reward=0.0, done=False)
    
    def step(
        self,
        action: RobotCityAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> RobotCityObservation:
        """
        Execute one step in the environment.
        
        Args:
            action: RobotCityAction specifying robot_id, move, and speed
            timeout_s: Optional timeout (not used)
            
        Returns:
            RobotCityObservation with new state
        """
        if self._sim is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        events: List[str] = []
        
        # Apply failure injection
        self._sim, failure_events = apply_failures(
            self._sim, self._failure_config, self._sim.rng
        )
        events.extend(failure_events)
        
        # Check for action delay
        if "action_delay" in events:
            # Action is ignored this step
            move = "noop"
            speed = 0.0
        else:
            move = action.move
            speed = action.speed
        
        # Get robot's forward progress before step (for reward)
        robot_id = action.robot_id
        if robot_id < len(self._sim.robots):
            progress_before = compute_forward_progress(self._sim.robots[robot_id])
        else:
            progress_before = 0.0
        
        # Update simulation
        update_state(self._sim, robot_id, move, speed)
        self._state.step_count += 1
        
        # Check collisions
        collisions = check_collisions(self._sim)
        
        # Compute reward
        reward = 0.0
        
        # Forward progress reward
        if robot_id < len(self._sim.robots):
            progress_after = compute_forward_progress(self._sim.robots[robot_id])
            reward += 0.1 * max(0, progress_after)
        
        # Collision penalties
        for rid, pid in collisions.robot_ped_collisions:
            reward -= 5.0
            events.append(f"collision_ped_{rid}_{pid}")
        
        for rid, oid in collisions.robot_obs_collisions:
            reward -= 1.0
            events.append(f"collision_obs_{rid}_{oid}")
        
        for r1, r2 in collisions.robot_robot_collisions:
            reward -= 2.0
            events.append(f"collision_robot_{r1}_{r2}")
        
        # Near miss penalties
        for rid, pid, dist in collisions.near_misses:
            reward -= 0.2
            events.append(f"near_miss_{rid}_{pid}")
        
        # Action cost (encourage efficiency)
        if move != "noop":
            reward -= 0.01
        
        self._episode_reward += reward
        self._episode_events.extend(events)
        
        # Check done conditions
        done = False
        
        # Max steps
        if self._state.step_count >= self._max_steps:
            done = True
            events.append("max_steps_reached")
        
        # Collision end
        if self._end_on_collision and (
            collisions.robot_ped_collisions or 
            collisions.robot_obs_collisions or
            collisions.robot_robot_collisions
        ):
            done = True
            events.append("episode_end_collision")
        
        # All robots at goal
        all_at_goal = all(robot_at_goal(r) for r in self._sim.robots)
        if all_at_goal:
            done = True
            reward += 10.0  # Goal bonus
            events.append("all_goals_reached")
        
        return self._build_observation(events=events, reward=reward, done=done)
    
    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state
    
    def _build_observation(
        self,
        events: List[str],
        reward: float,
        done: bool,
    ) -> RobotCityObservation:
        """Build observation from current simulation state."""
        if self._sim is None:
            raise RuntimeError("Simulation not initialized")
        
        # Check for camera dropout
        if "camera_dropout" in events:
            dropout_img = create_dropout_image(self._render_size)
            rgb = base64.b64encode(dropout_img).decode("utf-8")
        else:
            rgb = render_global_view(self._sim, size=self._render_size)
        
        # Ego view (optional)
        ego_rgb = render_ego_view(self._sim, robot_id=0, size=64)
        
        # Overlays (optional, computationally expensive)
        uncertainty_overlay = None
        future_heatmap = None
        
        if self._compute_overlays:
            try:
                import numpy as np
                
                # Uncertainty heatmap
                uncertainty = compute_uncertainty_heatmap(self._sim, robot_id=0)
                uncertainty_overlay = render_heatmap(uncertainty, size=self._render_size, colormap="hot")
                
                # Future pedestrian heatmap
                future = compute_future_heatmap(self._sim)
                future_heatmap = render_heatmap(future, size=self._render_size, colormap="cool")
            except Exception:
                pass  # Skip overlays on error
        
        # Build state dict (with optional GPS drift)
        state_dict = self._sim.to_dict()
        
        # Apply GPS drift to reported positions
        if self._failure_config.gps_drift_sigma > 0:
            for robot_dict in state_dict["robots"]:
                noisy_x, noisy_y = apply_gps_drift(
                    robot_dict["x"], robot_dict["y"],
                    self._failure_config.gps_drift_sigma,
                    self._sim.rng
                )
                robot_dict["reported_x"] = noisy_x
                robot_dict["reported_y"] = noisy_y
        
        state_dict["step_count"] = self._state.step_count
        state_dict["episode_id"] = self._state.episode_id
        
        return RobotCityObservation(
            rgb=rgb,
            ego_rgb=ego_rgb,
            uncertainty_overlay=uncertainty_overlay,
            future_heatmap=future_heatmap,
            state=state_dict,
            events=events,
            reward=reward,
            done=done,
            metadata={
                "episode_reward": self._episode_reward,
                "episode_events": self._episode_events.copy(),
            },
        )
    
    # =========================================================================
    # Counterfactual support (snapshot/restore)
    # =========================================================================
    
    def snapshot(self) -> Dict[str, Any]:
        """Capture current state for counterfactual reasoning.
        
        Returns:
            Snapshot dictionary that can be passed to restore()
        """
        if self._sim is None:
            raise RuntimeError("Environment not initialized")
        
        return {
            "sim_state": self._sim.copy(),
            "env_state": State(
                episode_id=self._state.episode_id,
                step_count=self._state.step_count,
            ),
            "episode_reward": self._episode_reward,
            "episode_events": self._episode_events.copy(),
            "failure_config": FailureConfig(
                camera_dropout_prob=self._failure_config.camera_dropout_prob,
                gps_drift_sigma=self._failure_config.gps_drift_sigma,
                pedestrian_rush_prob=self._failure_config.pedestrian_rush_prob,
                sudden_obstacle_prob=self._failure_config.sudden_obstacle_prob,
                action_delay_prob=self._failure_config.action_delay_prob,
            ),
        }
    
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot.
        
        Args:
            snapshot: Snapshot from snapshot() method
        """
        self._sim = snapshot["sim_state"].copy()
        self._state = State(
            episode_id=snapshot["env_state"].episode_id,
            step_count=snapshot["env_state"].step_count,
        )
        self._episode_reward = snapshot["episode_reward"]
        self._episode_events = snapshot["episode_events"].copy()
        self._failure_config = snapshot["failure_config"]
