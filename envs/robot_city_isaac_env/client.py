# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Robot City Isaac Environment Client.

Provides HTTP/WebSocket-based client for connecting to Robot City Isaac server.
Can run without Isaac Sim installed (only server needs Isaac Sim).
"""

import base64
import io
from typing import Any, Dict, Optional

import numpy as np

# Support both in-repo and standalone imports
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from .models import RobotCityIsaacAction, RobotCityIsaacObs
except ImportError:
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from models import RobotCityIsaacAction, RobotCityIsaacObs


def decode_frame(b64_png: str) -> np.ndarray:
    """Decode base64 PNG to numpy array.
    
    Args:
        b64_png: Base64 encoded PNG string
        
    Returns:
        RGB numpy array (HxWx3 uint8)
    """
    try:
        from PIL import Image
        img_bytes = base64.b64decode(b64_png)
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)
    except ImportError:
        # Return empty array if PIL not available
        return np.zeros((512, 512, 3), dtype=np.uint8)


class RobotCityIsaacEnv(EnvClient[RobotCityIsaacAction, RobotCityIsaacObs, State]):
    """
    Client for Robot City Isaac Environment.
    
    Connects to a remote Isaac Sim server and provides a simple interface
    for running episodes and collecting observations.
    
    Example:
        >>> with RobotCityIsaacEnv(base_url="http://localhost:8001") as client:
        ...     result = client.reset(scenario="sidewalk_delivery", num_agents=6)
        ...     frame = client.decode_frame(result.observation.rgb)
        ...     result = client.step(RobotCityIsaacAction(linear_vel=0.5, angular_vel=0.0))
    """
    
    def _step_payload(self, action: RobotCityIsaacAction) -> Dict:
        """Convert action to JSON payload."""
        return {
            "linear_vel": action.linear_vel,
            "angular_vel": action.angular_vel,
            "metadata": action.metadata,
        }
    
    def _parse_result(self, payload: Dict) -> StepResult[RobotCityIsaacObs]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        
        observation = RobotCityIsaacObs(
            rgb=obs_data.get("rgb", ""),
            rgb_with_overlay=obs_data.get("rgb_with_overlay"),
            depth=obs_data.get("depth"),
            step_idx=obs_data.get("step_idx", 0),
            ego_pose=tuple(obs_data.get("ego_pose", (0, 0, 0, 0, 0, 0))),
            goal_pose=tuple(obs_data.get("goal_pose", (0, 0, 0))),
            goal_distance=obs_data.get("goal_distance", 0.0),
            agents=obs_data.get("agents", []),
            events=obs_data.get("events", {}),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
    
    @staticmethod
    def decode_frame(b64_png: str) -> np.ndarray:
        """Decode base64 PNG observation to numpy array.
        
        Args:
            b64_png: Base64 encoded PNG from observation
            
        Returns:
            RGB numpy array (HxWx3 uint8)
        """
        return decode_frame(b64_png)
