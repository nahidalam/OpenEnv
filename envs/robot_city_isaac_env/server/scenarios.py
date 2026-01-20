# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario configurations for Robot City Isaac Environment.

Each scenario defines:
- Start and goal positions
- Agent spawn regions
- Failure injection defaults
- Scene parameters
"""

from typing import Any, Dict, List, Tuple


# Scenario configurations
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "crosswalk_occlusion": {
        "name": "Crosswalk Occlusion",
        "description": "Robot navigates a crosswalk with occluded pedestrians",
        
        # Positions in meters
        "start_position": [0.0, -5.0, 0.0],
        "start_heading": 1.57,  # Facing +Y (forward)
        "goal_position": [0.0, 10.0, 0.0],
        
        # Agent spawn region (xmin, xmax, ymin, ymax)
        "agent_spawn_region": (-5.0, 5.0, -2.0, 8.0),
        "default_num_agents": 6,
        
        # Obstacle configuration
        "obstacles": [
            {"type": "wall", "position": [-3.0, 2.0, 0.0], "size": [0.5, 2.0, 1.5]},
            {"type": "wall", "position": [3.0, 2.0, 0.0], "size": [0.5, 2.0, 1.5]},
            {"type": "vehicle", "position": [-2.0, 0.0, 0.0], "size": [2.0, 4.0, 1.5]},
        ],
        
        # Failure injection defaults
        "default_camera_dropout": 0.0,
        "default_action_delay": 0.0,
        "default_motion_blur": 0.0,
        
        # Scene settings
        "scene_type": "outdoor",
        "lighting": "daylight",
        "max_steps": 300,
    },
    
    "sidewalk_delivery": {
        "name": "Sidewalk Delivery",
        "description": "Delivery robot navigates a busy sidewalk",
        
        "start_position": [-8.0, 0.0, 0.0],
        "start_heading": 0.0,  # Facing +X
        "goal_position": [8.0, 0.0, 0.0],
        
        "agent_spawn_region": (-6.0, 6.0, -2.0, 2.0),
        "default_num_agents": 8,
        
        "obstacles": [
            {"type": "bench", "position": [-4.0, 1.5, 0.0], "size": [1.5, 0.5, 0.5]},
            {"type": "bench", "position": [0.0, 1.5, 0.0], "size": [1.5, 0.5, 0.5]},
            {"type": "bench", "position": [4.0, 1.5, 0.0], "size": [1.5, 0.5, 0.5]},
            {"type": "planter", "position": [-2.0, -1.5, 0.0], "size": [0.8, 0.8, 0.6]},
            {"type": "planter", "position": [2.0, -1.5, 0.0], "size": [0.8, 0.8, 0.6]},
        ],
        
        "default_camera_dropout": 0.0,
        "default_action_delay": 0.0,
        "default_motion_blur": 0.0,
        
        "scene_type": "outdoor",
        "lighting": "daylight",
        "max_steps": 400,
    },
    
    "warehouse_aisles": {
        "name": "Warehouse Aisles",
        "description": "Robot navigates narrow warehouse aisles with workers",
        
        "start_position": [0.0, -8.0, 0.0],
        "start_heading": 1.57,  # Facing +Y
        "goal_position": [0.0, 8.0, 0.0],
        
        "agent_spawn_region": (-4.0, 4.0, -6.0, 6.0),
        "default_num_agents": 4,
        
        "obstacles": [
            # Shelf rows
            {"type": "shelf", "position": [-3.0, -4.0, 0.0], "size": [1.0, 6.0, 2.5]},
            {"type": "shelf", "position": [-3.0, 4.0, 0.0], "size": [1.0, 6.0, 2.5]},
            {"type": "shelf", "position": [3.0, -4.0, 0.0], "size": [1.0, 6.0, 2.5]},
            {"type": "shelf", "position": [3.0, 4.0, 0.0], "size": [1.0, 6.0, 2.5]},
        ],
        
        "default_camera_dropout": 0.05,  # More dropouts in warehouse lighting
        "default_action_delay": 0.02,
        "default_motion_blur": 0.0,
        
        "scene_type": "indoor",
        "lighting": "warehouse",
        "max_steps": 350,
    },
}


def get_scenario_config(scenario_name: str) -> Dict[str, Any]:
    """Get configuration for a scenario.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        Scenario configuration dictionary
        
    Raises:
        ValueError: If scenario not found
    """
    if scenario_name not in SCENARIOS:
        available = list(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {available}")
    
    return SCENARIOS[scenario_name].copy()


def list_scenarios() -> List[str]:
    """List available scenario names."""
    return list(SCENARIOS.keys())


def get_scenario_metadata(scenario_name: str) -> Dict[str, str]:
    """Get human-readable metadata for a scenario."""
    config = get_scenario_config(scenario_name)
    return {
        "name": config["name"],
        "description": config["description"],
        "scene_type": config["scene_type"],
    }


def get_failure_defaults(scenario_name: str) -> Dict[str, float]:
    """Get default failure injection parameters for a scenario."""
    config = get_scenario_config(scenario_name)
    return {
        "camera_dropout": config.get("default_camera_dropout", 0.0),
        "action_delay": config.get("default_action_delay", 0.0),
        "motion_blur": config.get("default_motion_blur", 0.0),
    }
