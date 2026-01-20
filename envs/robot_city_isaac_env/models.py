# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Robot City Isaac Environment.

Defines Action and Observation types for realistic Isaac Sim-based simulation.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import Field

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class RobotCityIsaacAction(Action):
    """Action for controlling the robot in Isaac Sim.
    
    Uses continuous velocity control for smoother motion.
    
    Attributes:
        linear_vel: Forward/backward velocity in m/s (-1.0 to 1.0)
        angular_vel: Rotational velocity in rad/s (-1.0 to 1.0)
        metadata: Optional action metadata
    """
    linear_vel: float = Field(default=0.0, ge=-1.0, le=1.0, description="Linear velocity")
    angular_vel: float = Field(default=0.0, ge=-1.0, le=1.0, description="Angular velocity")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class AgentState(Action):
    """State of a single agent (pedestrian/other robot)."""
    agent_id: int = Field(default=0, description="Agent identifier")
    x: float = Field(default=0.0, description="X position in meters")
    y: float = Field(default=0.0, description="Y position in meters")
    z: float = Field(default=0.0, description="Z position in meters")
    heading: float = Field(default=0.0, description="Heading in radians")
    vx: float = Field(default=0.0, description="X velocity")
    vy: float = Field(default=0.0, description="Y velocity")
    agent_type: str = Field(default="pedestrian", description="Type: pedestrian, vehicle, robot")


class RobotCityIsaacObs(Observation):
    """Observation from Robot City Isaac environment.
    
    Contains realistic RGB frame from Isaac Sim plus state information.
    
    Attributes:
        rgb: Base64 encoded PNG of camera view (front-facing robot camera)
        rgb_with_overlay: Optional base64 PNG with visual overlays drawn
        depth: Optional base64 PNG depth image
        step_idx: Current step index in episode
        ego_pose: Ego robot pose as (x, y, z, roll, pitch, yaw)
        goal_pose: Goal position as (x, y, z)
        goal_distance: Distance to goal in meters
        agents: List of agent states (pedestrians, other robots)
        events: Dict of events this step
    """
    rgb: str = Field(..., description="Base64 encoded PNG from robot camera")
    rgb_with_overlay: Optional[str] = Field(default=None, description="RGB with overlays")
    depth: Optional[str] = Field(default=None, description="Base64 depth image")
    
    step_idx: int = Field(default=0, description="Current step index")
    ego_pose: Tuple[float, float, float, float, float, float] = Field(
        default=(0, 0, 0, 0, 0, 0),
        description="Ego pose (x, y, z, roll, pitch, yaw)"
    )
    goal_pose: Tuple[float, float, float] = Field(
        default=(0, 0, 0),
        description="Goal position (x, y, z)"
    )
    goal_distance: float = Field(default=0.0, description="Distance to goal")
    
    agents: List[Dict[str, Any]] = Field(default_factory=list, description="Agent states")
    
    events: Dict[str, Any] = Field(
        default_factory=dict,
        description="Events: collision, near_miss, goal_reached, dropout_applied, delay_applied, blur_applied"
    )


class RobotCityIsaacState:
    """Internal state for Isaac environment (not sent to client).
    
    Used for snapshot/restore functionality.
    """
    def __init__(self):
        self.rng_state: Any = None
        self.ego_transform: Tuple[float, ...] = (0, 0, 0, 0, 0, 0)
        self.ego_velocity: Tuple[float, float] = (0, 0)
        self.agent_transforms: List[Tuple[float, ...]] = []
        self.agent_velocities: List[Tuple[float, float]] = []
        self.step_idx: int = 0
        self.episode_id: str = ""
        self.scenario: str = ""
        self.goal_pose: Tuple[float, float, float] = (0, 0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng_state,
            "ego_transform": self.ego_transform,
            "ego_velocity": self.ego_velocity,
            "agent_transforms": self.agent_transforms,
            "agent_velocities": self.agent_velocities,
            "step_idx": self.step_idx,
            "episode_id": self.episode_id,
            "scenario": self.scenario,
            "goal_pose": self.goal_pose,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotCityIsaacState":
        state = cls()
        state.rng_state = data.get("rng_state")
        state.ego_transform = tuple(data.get("ego_transform", (0, 0, 0, 0, 0, 0)))
        state.ego_velocity = tuple(data.get("ego_velocity", (0, 0)))
        state.agent_transforms = [tuple(t) for t in data.get("agent_transforms", [])]
        state.agent_velocities = [tuple(v) for v in data.get("agent_velocities", [])]
        state.step_idx = data.get("step_idx", 0)
        state.episode_id = data.get("episode_id", "")
        state.scenario = data.get("scenario", "")
        state.goal_pose = tuple(data.get("goal_pose", (0, 0, 0)))
        return state
