# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Robot City Isaac Environment Implementation.

A realistic visual robot simulation environment using NVIDIA Isaac Sim.
Supports multi-agent scenarios, failure injection, and counterfactual planning.
"""

import base64
import io
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from ..models import RobotCityIsaacAction, RobotCityIsaacObs
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from models import RobotCityIsaacAction, RobotCityIsaacObs

from .isaac_bridge import IsaacBridge, IsaacConfig
from .scenarios import get_scenario_config, get_failure_defaults
from .renderer_overlays import (
    draw_all_overlays,
    apply_motion_blur,
    create_black_frame,
)


def _encode_frame(frame: np.ndarray) -> str:
    """Encode numpy frame to base64 PNG."""
    try:
        from PIL import Image
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except ImportError:
        # Fallback: encode raw bytes (less efficient)
        return base64.b64encode(frame.tobytes()).decode("utf-8")


class RobotCityIsaacEnvironment(Environment):
    """
    Robot City Isaac Environment - Realistic robot simulation with Isaac Sim.
    
    Features:
    - Photorealistic RGB observations from Isaac Sim
    - Multi-agent support with pedestrians
    - Failure injection (camera dropout, action delay, motion blur)
    - Visual overlays for debugging
    - Counterfactual planning (snapshot/restore)
    
    Example:
        >>> env = RobotCityIsaacEnvironment()
        >>> obs = env.reset(scenario="sidewalk_delivery", num_agents=6)
        >>> obs = env.step(RobotCityIsaacAction(linear_vel=0.5, angular_vel=0.1))
    """
    
    SUPPORTS_CONCURRENT_SESSIONS = False
    
    def __init__(self, headless: bool = True, resolution: int = 512):
        """Initialize the environment.
        
        Args:
            headless: Run Isaac Sim in headless mode
            resolution: Image resolution (square)
        """
        self._config = IsaacConfig(
            headless=headless,
            resolution=(resolution, resolution),
        )
        self._bridge: Optional[IsaacBridge] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()
        
        # Episode state
        self._scenario: str = ""
        self._max_steps: int = 300
        self._episode_reward: float = 0.0
        self._episode_events: List[str] = []
        
        # Failure injection settings
        self._camera_dropout_prob: float = 0.0
        self._action_delay_prob: float = 0.0
        self._motion_blur_prob: float = 0.0
        
        # Delayed action buffer
        self._prev_action: Optional[RobotCityIsaacAction] = None
        
        # Overlay settings
        self._show_overlays: bool = True
        
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario: str = "sidewalk_delivery",
        num_agents: int = 6,
        resolution: Optional[int] = None,
        # Failure injection
        camera_dropout: Optional[float] = None,
        action_delay: Optional[float] = None,
        motion_blur: Optional[float] = None,
        # Overlay settings
        show_overlays: bool = True,
        **kwargs,
    ) -> RobotCityIsaacObs:
        """Reset environment to a new episode.
        
        Args:
            seed: Random seed
            episode_id: Optional episode identifier
            scenario: Scenario name
            num_agents: Number of pedestrian agents
            resolution: Image resolution
            camera_dropout: Camera dropout probability (0-1)
            action_delay: Action delay probability (0-1)
            motion_blur: Motion blur probability (0-1)
            show_overlays: Whether to draw visual overlays
            
        Returns:
            Initial observation
        """
        # Set up episode
        if episode_id is None:
            episode_id = str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)
        
        if seed is not None:
            self._rng = random.Random(seed)
        
        # Get scenario config
        scenario_config = get_scenario_config(scenario)
        self._scenario = scenario
        self._max_steps = scenario_config.get("max_steps", 300)
        self._episode_reward = 0.0
        self._episode_events = []
        
        # Set failure injection rates
        failure_defaults = get_failure_defaults(scenario)
        self._camera_dropout_prob = camera_dropout if camera_dropout is not None else failure_defaults["camera_dropout"]
        self._action_delay_prob = action_delay if action_delay is not None else failure_defaults["action_delay"]
        self._motion_blur_prob = motion_blur if motion_blur is not None else failure_defaults["motion_blur"]
        
        self._show_overlays = show_overlays
        self._prev_action = None
        
        # Initialize Isaac bridge if needed
        if self._bridge is None:
            # Only update resolution if explicitly passed
            if resolution is not None:
                self._config.resolution = (resolution, resolution)
            self._config.num_agents = num_agents
            self._bridge = IsaacBridge(self._config)
            self._bridge.initialize()
        
        # Reset simulation
        frame, metadata = self._bridge.reset(
            scenario=scenario,
            seed=seed,
            num_agents=num_agents,
        )
        
        return self._build_observation(
            frame=frame,
            events={"reset": True},
            reward=0.0,
            done=False,
        )
    
    def step(
        self,
        action: RobotCityIsaacAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> RobotCityIsaacObs:
        """Execute one environment step.
        
        Args:
            action: Robot action (linear_vel, angular_vel)
            timeout_s: Optional timeout (not used)
            
        Returns:
            Observation with new state
        """
        if self._bridge is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        events: Dict[str, Any] = {}
        
        # Check for action delay
        actual_action = action
        if self._rng.random() < self._action_delay_prob:
            events["delay_applied"] = True
            if self._prev_action is not None:
                actual_action = self._prev_action
            else:
                actual_action = RobotCityIsaacAction(linear_vel=0, angular_vel=0)
        
        self._prev_action = action
        
        # Step simulation
        frame, reward, done, step_info = self._bridge.step(
            linear_vel=actual_action.linear_vel,
            angular_vel=actual_action.angular_vel,
        )
        
        # Update state
        self._state.step_count += 1
        events.update(step_info)
        
        # Apply failure injection to frame
        frame, frame_events = self._apply_frame_failures(frame)
        events.update(frame_events)
        
        # Track episode stats
        self._episode_reward += reward
        if events.get("collision"):
            self._episode_events.append("collision")
        if events.get("near_miss"):
            self._episode_events.append("near_miss")
        if events.get("goal_reached"):
            self._episode_events.append("goal_reached")
        
        # Check max steps
        if self._state.step_count >= self._max_steps:
            done = True
            events["max_steps_reached"] = True
        
        return self._build_observation(
            frame=frame,
            events=events,
            reward=reward,
            done=done,
        )
    
    def _apply_frame_failures(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply failure injection to frame."""
        events = {}
        
        # Camera dropout
        if self._rng.random() < self._camera_dropout_prob:
            frame = create_black_frame(frame.shape)
            events["dropout_applied"] = True
        
        # Motion blur
        elif self._rng.random() < self._motion_blur_prob:
            frame = apply_motion_blur(frame, strength=0.5)
            events["blur_applied"] = True
        
        return frame, events
    
    def _build_observation(
        self,
        frame: np.ndarray,
        events: Dict[str, Any],
        reward: float,
        done: bool,
    ) -> RobotCityIsaacObs:
        """Build observation from current state."""
        if self._bridge is None:
            raise RuntimeError("Bridge not initialized")
        
        # Get state info
        ego_pose = self._bridge.get_ego_pose()
        agents = self._bridge.get_agent_states()
        goal_pose = tuple(self._bridge._goal_position)
        goal_distance = events.get("goal_distance", 0.0)
        
        # Encode base frame
        rgb_b64 = _encode_frame(frame)
        
        # Draw overlays if enabled
        rgb_overlay_b64 = None
        if self._show_overlays and not events.get("dropout_applied"):
            overlay_frame = draw_all_overlays(
                frame=frame,
                ego_pose=ego_pose,
                goal_pose=goal_pose,
                agents=agents,
                events=events,
                show_legend=True,
                show_heading=True,
                show_trajectories=True,
                show_risk_cone=True,
                show_uncertainty=events.get("dropout_applied") or events.get("blur_applied"),
            )
            rgb_overlay_b64 = _encode_frame(overlay_frame)
        
        return RobotCityIsaacObs(
            rgb=rgb_b64,
            rgb_with_overlay=rgb_overlay_b64,
            step_idx=self._state.step_count,
            ego_pose=ego_pose,
            goal_pose=goal_pose,
            goal_distance=goal_distance,
            agents=agents,
            events=events,
            reward=reward,
            done=done,
            metadata={
                "episode_id": self._state.episode_id,
                "scenario": self._scenario,
                "episode_reward": self._episode_reward,
                "episode_events": self._episode_events.copy(),
            },
        )
    
    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state
    
    def snapshot(self) -> Dict[str, Any]:
        """Capture state for counterfactual planning."""
        if self._bridge is None:
            raise RuntimeError("Environment not initialized")
        
        bridge_snapshot = self._bridge.snapshot()
        
        return {
            "bridge": bridge_snapshot,
            "state": {
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
            },
            "rng_state": self._rng.getstate(),
            "episode_reward": self._episode_reward,
            "episode_events": self._episode_events.copy(),
            "prev_action": {
                "linear_vel": self._prev_action.linear_vel,
                "angular_vel": self._prev_action.angular_vel,
            } if self._prev_action else None,
        }
    
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        if self._bridge is None:
            raise RuntimeError("Environment not initialized")
        
        self._bridge.restore(snapshot["bridge"])
        
        state_data = snapshot["state"]
        self._state = State(
            episode_id=state_data["episode_id"],
            step_count=state_data["step_count"],
        )
        
        self._rng.setstate(snapshot["rng_state"])
        self._episode_reward = snapshot["episode_reward"]
        self._episode_events = snapshot["episode_events"].copy()
        
        prev_action_data = snapshot.get("prev_action")
        if prev_action_data:
            self._prev_action = RobotCityIsaacAction(
                linear_vel=prev_action_data["linear_vel"],
                angular_vel=prev_action_data["angular_vel"],
            )
        else:
            self._prev_action = None
    
    def close(self) -> None:
        """Clean up resources."""
        if self._bridge:
            self._bridge.close()
            self._bridge = None
