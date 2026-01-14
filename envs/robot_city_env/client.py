# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Robot City Environment Client.

Provides WebSocket-based client for connecting to Robot City server.
"""

from typing import Any, Dict, Optional

# Support both in-repo and standalone imports
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from .models import RobotCityAction, RobotCityObservation
except ImportError:
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from models import RobotCityAction, RobotCityObservation


class RobotCityEnv(EnvClient[RobotCityAction, RobotCityObservation, State]):
    """
    Client for the Robot City Environment.
    
    Example:
        >>> with RobotCityEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(scenario="intersection_crosswalk")
        ...     result = client.step(RobotCityAction(robot_id=0, move="forward"))
        ...     print(result.observation.events)
    """
    
    def _step_payload(self, action: RobotCityAction) -> Dict:
        """Convert action to JSON payload."""
        return {
            "robot_id": action.robot_id,
            "move": action.move,
            "speed": action.speed,
            "metadata": action.metadata,
        }
    
    def _parse_result(self, payload: Dict) -> StepResult[RobotCityObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        
        observation = RobotCityObservation(
            rgb=obs_data.get("rgb", ""),
            ego_rgb=obs_data.get("ego_rgb"),
            uncertainty_overlay=obs_data.get("uncertainty_overlay"),
            future_heatmap=obs_data.get("future_heatmap"),
            state=obs_data.get("state", {}),
            events=obs_data.get("events", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
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
