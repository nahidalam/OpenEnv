# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Physics dynamics for Robot City simulation.

Implements entity movement, collision detection, and state updates.
All positions are in normalized coordinates [0, 1].
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class Robot:
    """Robot entity with pose and velocity."""
    x: float
    y: float
    theta: float  # Heading in radians
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.03
    goal_x: float = 0.5
    goal_y: float = 0.5
    robot_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "theta": self.theta,
            "vx": self.vx, "vy": self.vy, "radius": self.radius,
            "goal_x": self.goal_x, "goal_y": self.goal_y,
            "robot_id": self.robot_id,
        }


@dataclass
class Pedestrian:
    """Pedestrian entity with stochastic motion."""
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.02
    noise_sigma: float = 0.01
    ped_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
            "radius": self.radius, "ped_id": self.ped_id,
        }


@dataclass
class Obstacle:
    """Static rectangular obstacle."""
    x: float  # Center x
    y: float  # Center y
    width: float
    height: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class SimState:
    """Complete simulation state."""
    robots: List[Robot] = field(default_factory=list)
    pedestrians: List[Pedestrian] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    step_count: int = 0
    dt: float = 0.05  # Time step
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # xmin, ymin, xmax, ymax
    rng: random.Random = field(default_factory=random.Random)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "robots": [r.to_dict() for r in self.robots],
            "pedestrians": [p.to_dict() for p in self.pedestrians],
            "obstacles": [o.to_dict() for o in self.obstacles],
            "step_count": self.step_count,
        }
    
    def copy(self) -> "SimState":
        """Create a deep copy of the state."""
        new_state = SimState(
            robots=[Robot(**r.to_dict()) for r in self.robots],
            pedestrians=[Pedestrian(**p.to_dict()) for p in self.pedestrians],
            obstacles=[Obstacle(**o.to_dict()) for o in self.obstacles],
            step_count=self.step_count,
            dt=self.dt,
            bounds=self.bounds,
            rng=random.Random(),
        )
        new_state.rng.setstate(self.rng.getstate())
        return new_state


# Movement parameters
MOVE_SPEED = 0.02  # Base movement speed
TURN_RATE = 0.3  # Radians per step


def apply_action(robot: Robot, move: str, speed: float = 1.0) -> None:
    """Apply discrete action to robot, updating velocity."""
    base_speed = MOVE_SPEED * speed
    
    if move == "noop":
        robot.vx = 0.0
        robot.vy = 0.0
    elif move == "forward":
        robot.vx = base_speed * math.cos(robot.theta)
        robot.vy = base_speed * math.sin(robot.theta)
    elif move == "backward":
        robot.vx = -base_speed * math.cos(robot.theta)
        robot.vy = -base_speed * math.sin(robot.theta)
    elif move == "left":
        # Strafe left (perpendicular to heading)
        robot.vx = base_speed * math.cos(robot.theta + math.pi/2)
        robot.vy = base_speed * math.sin(robot.theta + math.pi/2)
    elif move == "right":
        # Strafe right
        robot.vx = base_speed * math.cos(robot.theta - math.pi/2)
        robot.vy = base_speed * math.sin(robot.theta - math.pi/2)
    elif move == "turn_left":
        robot.theta += TURN_RATE * speed
        robot.vx = 0.0
        robot.vy = 0.0
    elif move == "turn_right":
        robot.theta -= TURN_RATE * speed
        robot.vx = 0.0
        robot.vy = 0.0


def update_robot(robot: Robot, bounds: Tuple[float, float, float, float]) -> None:
    """Update robot position and clamp to bounds."""
    robot.x += robot.vx
    robot.y += robot.vy
    
    # Clamp to bounds
    xmin, ymin, xmax, ymax = bounds
    robot.x = max(xmin + robot.radius, min(xmax - robot.radius, robot.x))
    robot.y = max(ymin + robot.radius, min(ymax - robot.radius, robot.y))


def update_pedestrian(ped: Pedestrian, bounds: Tuple[float, float, float, float], 
                      rng: random.Random) -> None:
    """Update pedestrian with stochastic motion and boundary bounce."""
    # Add noise to velocity
    ped.vx += rng.gauss(0, ped.noise_sigma)
    ped.vy += rng.gauss(0, ped.noise_sigma)
    
    # Clamp velocity
    max_vel = 0.02
    speed = math.sqrt(ped.vx**2 + ped.vy**2)
    if speed > max_vel:
        ped.vx = ped.vx / speed * max_vel
        ped.vy = ped.vy / speed * max_vel
    
    # Update position
    ped.x += ped.vx
    ped.y += ped.vy
    
    # Bounce off bounds
    xmin, ymin, xmax, ymax = bounds
    if ped.x <= xmin + ped.radius or ped.x >= xmax - ped.radius:
        ped.vx *= -1
        ped.x = max(xmin + ped.radius, min(xmax - ped.radius, ped.x))
    if ped.y <= ymin + ped.radius or ped.y >= ymax - ped.radius:
        ped.vy *= -1
        ped.y = max(ymin + ped.radius, min(ymax - ped.radius, ped.y))


def update_state(state: SimState, robot_id: int, move: str, speed: float = 1.0) -> None:
    """Update simulation state for one time step."""
    # Apply action to specified robot
    if 0 <= robot_id < len(state.robots):
        apply_action(state.robots[robot_id], move, speed)
    
    # Update all robots
    for robot in state.robots:
        update_robot(robot, state.bounds)
    
    # Update all pedestrians
    for ped in state.pedestrians:
        update_pedestrian(ped, state.bounds, state.rng)
    
    state.step_count += 1


def circle_circle_collision(x1: float, y1: float, r1: float,
                            x2: float, y2: float, r2: float) -> bool:
    """Check collision between two circles."""
    dx = x2 - x1
    dy = y2 - y1
    dist_sq = dx*dx + dy*dy
    min_dist = r1 + r2
    return dist_sq < min_dist * min_dist


def circle_rect_collision(cx: float, cy: float, cr: float,
                          rx: float, ry: float, rw: float, rh: float) -> bool:
    """Check collision between circle and axis-aligned rectangle."""
    # Find closest point on rectangle to circle center
    half_w = rw / 2
    half_h = rh / 2
    closest_x = max(rx - half_w, min(rx + half_w, cx))
    closest_y = max(ry - half_h, min(ry + half_h, cy))
    
    # Check if closest point is inside circle
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx*dx + dy*dy) < cr*cr


@dataclass
class CollisionResult:
    """Result of collision detection."""
    robot_ped_collisions: List[Tuple[int, int]] = field(default_factory=list)
    robot_obs_collisions: List[Tuple[int, int]] = field(default_factory=list)
    robot_robot_collisions: List[Tuple[int, int]] = field(default_factory=list)
    near_misses: List[Tuple[int, int, float]] = field(default_factory=list)  # robot_id, ped_id, distance


def check_collisions(state: SimState, near_miss_threshold: float = 0.05) -> CollisionResult:
    """Check all collisions in the simulation."""
    result = CollisionResult()
    
    # Robot-pedestrian collisions and near misses
    for robot in state.robots:
        for ped in state.pedestrians:
            dx = robot.x - ped.x
            dy = robot.y - ped.y
            dist = math.sqrt(dx*dx + dy*dy)
            min_dist = robot.radius + ped.radius
            
            if dist < min_dist:
                result.robot_ped_collisions.append((robot.robot_id, ped.ped_id))
            elif dist < min_dist + near_miss_threshold:
                result.near_misses.append((robot.robot_id, ped.ped_id, dist))
    
    # Robot-obstacle collisions
    for robot in state.robots:
        for i, obs in enumerate(state.obstacles):
            if circle_rect_collision(robot.x, robot.y, robot.radius,
                                    obs.x, obs.y, obs.width, obs.height):
                result.robot_obs_collisions.append((robot.robot_id, i))
    
    # Robot-robot collisions
    for i, r1 in enumerate(state.robots):
        for j, r2 in enumerate(state.robots):
            if i < j:
                if circle_circle_collision(r1.x, r1.y, r1.radius,
                                          r2.x, r2.y, r2.radius):
                    result.robot_robot_collisions.append((r1.robot_id, r2.robot_id))
    
    return result


def compute_forward_progress(robot: Robot) -> float:
    """Compute progress toward goal (positive = closer, negative = farther)."""
    dx = robot.goal_x - robot.x
    dy = robot.goal_y - robot.y
    dist = math.sqrt(dx*dx + dy*dy)
    
    # Project velocity onto goal direction
    if dist > 1e-6:
        goal_dir_x = dx / dist
        goal_dir_y = dy / dist
        progress = robot.vx * goal_dir_x + robot.vy * goal_dir_y
        return progress
    return 0.0


def robot_at_goal(robot: Robot, threshold: float = 0.05) -> bool:
    """Check if robot has reached its goal."""
    dx = robot.goal_x - robot.x
    dy = robot.goal_y - robot.y
    return (dx*dx + dy*dy) < threshold * threshold
