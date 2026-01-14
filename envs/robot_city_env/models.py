# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Robot City Environment.

Defines Action and Observation types for multi-robot visual simulation.
"""

from typing import Dict, List, Literal, Optional, Any
from pydantic import Field

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


# Move types for robot control
MoveType = Literal["noop", "forward", "backward", "left", "right", "turn_left", "turn_right"]


class RobotCityAction(Action):
    """Action for controlling a robot in Robot City.
    
    Attributes:
        robot_id: Which robot to control (0-indexed)
        move: Movement command
        speed: Speed multiplier (0.0 to 2.0)
        metadata: Optional action metadata
    """
    robot_id: int = Field(default=0, ge=0, description="Robot index to control")
    move: MoveType = Field(default="noop", description="Movement command")
    speed: float = Field(default=1.0, ge=0.0, le=2.0, description="Speed multiplier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class RobotCityObservation(Observation):
    """Observation from Robot City environment.
    
    Contains visual outputs (RGB frames, overlays) and state information.
    
    Attributes:
        rgb: Base64 encoded PNG of top-down view
        ego_rgb: Optional base64 PNG of ego-centric view
        uncertainty_overlay: Optional base64 PNG heatmap of collision risk
        future_heatmap: Optional base64 PNG of predicted pedestrian occupancy
        state: Dictionary with robot poses, pedestrian poses, step count
        events: List of events that occurred (collision, near_miss, etc.)
    """
    rgb: str = Field(..., description="Base64 encoded PNG of global top-down view")
    ego_rgb: Optional[str] = Field(default=None, description="Base64 encoded PNG of ego view")
    uncertainty_overlay: Optional[str] = Field(default=None, description="Base64 PNG collision risk heatmap")
    future_heatmap: Optional[str] = Field(default=None, description="Base64 PNG pedestrian prediction")
    state: Dict[str, Any] = Field(default_factory=dict, description="World state dict")
    events: List[str] = Field(default_factory=list, description="Events this step")
