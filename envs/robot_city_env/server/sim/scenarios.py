# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Built-in scenarios for Robot City.

Each scenario defines initial robot/pedestrian placements, obstacles, and goals.
"""

import random
from typing import Dict, Any, List
from .dynamics import SimState, Robot, Pedestrian, Obstacle


def _create_intersection_crosswalk(num_robots: int = 1, num_peds: int = 4,
                                    rng: random.Random = None) -> SimState:
    """Intersection with crosswalks - robots navigate through pedestrian crossing.
    
    Layout:
    - Central intersection area
    - Pedestrians crossing in both directions
    - Robot starts at one corner, goal at opposite corner
    """
    if rng is None:
        rng = random.Random()
    
    state = SimState(rng=rng)
    
    # Obstacles: buildings at corners
    corners = [
        (0.15, 0.15), (0.85, 0.15),
        (0.15, 0.85), (0.85, 0.85),
    ]
    for cx, cy in corners:
        state.obstacles.append(Obstacle(x=cx, y=cy, width=0.2, height=0.2))
    
    # Robots: start bottom-left area, goal top-right
    for i in range(num_robots):
        offset = i * 0.05
        state.robots.append(Robot(
            x=0.1 + offset, y=0.3 + offset, theta=1.57,  # Facing up
            robot_id=i,
            goal_x=0.9 - offset, goal_y=0.7 - offset,
        ))
    
    # Pedestrians: crossing paths
    for i in range(num_peds):
        if i % 2 == 0:
            # Horizontal crossers
            state.pedestrians.append(Pedestrian(
                x=0.3 + rng.random() * 0.1,
                y=0.4 + rng.random() * 0.2,
                vx=0.01 + rng.random() * 0.005,
                vy=rng.gauss(0, 0.002),
                ped_id=i,
            ))
        else:
            # Vertical crossers
            state.pedestrians.append(Pedestrian(
                x=0.4 + rng.random() * 0.2,
                y=0.3 + rng.random() * 0.1,
                vx=rng.gauss(0, 0.002),
                vy=0.01 + rng.random() * 0.005,
                ped_id=i,
            ))
    
    return state


def _create_warehouse_aisles(num_robots: int = 1, num_peds: int = 3,
                              rng: random.Random = None) -> SimState:
    """Warehouse with shelf aisles - robots navigate narrow corridors.
    
    Layout:
    - Parallel shelf rows creating aisles
    - Workers (pedestrians) moving along aisles
    - Robot must navigate to pickup/dropoff points
    """
    if rng is None:
        rng = random.Random()
    
    state = SimState(rng=rng)
    
    # Shelf obstacles (horizontal bars)
    shelf_ys = [0.25, 0.5, 0.75]
    for sy in shelf_ys:
        # Left shelf segment
        state.obstacles.append(Obstacle(x=0.25, y=sy, width=0.35, height=0.08))
        # Right shelf segment  
        state.obstacles.append(Obstacle(x=0.75, y=sy, width=0.35, height=0.08))
    
    # Robots: start at entrance, goal at back
    for i in range(num_robots):
        state.robots.append(Robot(
            x=0.5, y=0.1 + i * 0.05, theta=1.57,
            robot_id=i,
            goal_x=0.5, goal_y=0.9,
        ))
    
    # Workers moving in aisles
    aisle_ys = [0.375, 0.625, 0.875]
    for i in range(num_peds):
        ay = aisle_ys[i % len(aisle_ys)]
        state.pedestrians.append(Pedestrian(
            x=0.2 + rng.random() * 0.6,
            y=ay + rng.gauss(0, 0.02),
            vx=rng.choice([-1, 1]) * (0.008 + rng.random() * 0.004),
            vy=rng.gauss(0, 0.001),
            ped_id=i,
        ))
    
    return state


def _create_sidewalk_delivery(num_robots: int = 1, num_peds: int = 5,
                               rng: random.Random = None) -> SimState:
    """Sidewalk delivery - robot navigates busy sidewalk.
    
    Layout:
    - Linear sidewalk path
    - Building wall on one side, curb on other
    - Pedestrians walking both directions
    """
    if rng is None:
        rng = random.Random()
    
    state = SimState(rng=rng)
    
    # Building wall (top)
    state.obstacles.append(Obstacle(x=0.5, y=0.9, width=1.0, height=0.15))
    
    # Curb/street boundary (bottom) - small obstacles
    for x in [0.15, 0.35, 0.55, 0.75]:
        state.obstacles.append(Obstacle(x=x, y=0.12, width=0.08, height=0.04))
    
    # Planter boxes as obstacles
    state.obstacles.append(Obstacle(x=0.3, y=0.5, width=0.08, height=0.08))
    state.obstacles.append(Obstacle(x=0.7, y=0.5, width=0.08, height=0.08))
    
    # Robot: delivery from left to right
    for i in range(num_robots):
        state.robots.append(Robot(
            x=0.1, y=0.4 + i * 0.1, theta=0.0,  # Facing right
            robot_id=i,
            goal_x=0.9, goal_y=0.5,
        ))
    
    # Pedestrians walking on sidewalk
    for i in range(num_peds):
        direction = rng.choice([-1, 1])
        state.pedestrians.append(Pedestrian(
            x=rng.random() * 0.8 + 0.1,
            y=0.35 + rng.random() * 0.35,
            vx=direction * (0.006 + rng.random() * 0.006),
            vy=rng.gauss(0, 0.002),
            ped_id=i,
        ))
    
    return state


# Scenario registry
SCENARIOS: Dict[str, Any] = {
    "intersection_crosswalk": _create_intersection_crosswalk,
    "warehouse_aisles": _create_warehouse_aisles,
    "sidewalk_delivery": _create_sidewalk_delivery,
}


def get_scenario(name: str, num_robots: int = 1, num_peds: int = 4,
                 seed: int = None) -> SimState:
    """Get a scenario by name.
    
    Args:
        name: Scenario name (or "random" for random selection)
        num_robots: Number of robots
        num_peds: Number of pedestrians
        seed: Random seed for reproducibility
        
    Returns:
        Initialized SimState
    """
    rng = random.Random(seed)
    
    if name == "random":
        name = rng.choice(list(SCENARIOS.keys()))
    
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    
    return SCENARIOS[name](num_robots=num_robots, num_peds=num_peds, rng=rng)
