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
    # Trail history: list of (x, y) positions per robot, most recent last
    robot_trails: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    trail_max_length: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "robots": [r.to_dict() for r in self.robots],
            "pedestrians": [p.to_dict() for p in self.pedestrians],
            "obstacles": [o.to_dict() for o in self.obstacles],
            "step_count": self.step_count,
        }
    
    def record_robot_positions(self) -> None:
        """Record current robot positions for trail rendering."""
        for robot in self.robots:
            rid = robot.robot_id
            if rid not in self.robot_trails:
                self.robot_trails[rid] = []
            self.robot_trails[rid].append((robot.x, robot.y))
            # Keep only last N positions
            if len(self.robot_trails[rid]) > self.trail_max_length:
                self.robot_trails[rid] = self.robot_trails[rid][-self.trail_max_length:]
    
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
            robot_trails={k: list(v) for k, v in self.robot_trails.items()},
            trail_max_length=self.trail_max_length,
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


def push_circle_out_of_rect(cx: float, cy: float, cr: float,
                            rx: float, ry: float, rw: float, rh: float) -> Tuple[float, float]:
    """Push a circle out of a rectangle if overlapping.
    
    Returns the new (cx, cy) position outside the rectangle.
    """
    half_w = rw / 2
    half_h = rh / 2
    
    # Rectangle bounds
    left = rx - half_w
    right = rx + half_w
    top = ry + half_h
    bottom = ry - half_h
    
    # Find closest point on rectangle to circle center
    closest_x = max(left, min(right, cx))
    closest_y = max(bottom, min(top, cy))
    
    # Distance from circle center to closest point
    dx = cx - closest_x
    dy = cy - closest_y
    dist_sq = dx * dx + dy * dy
    
    # If not colliding, return original position
    if dist_sq >= cr * cr:
        return cx, cy
    
    # Handle case where circle center is inside rectangle
    if dist_sq < 1e-10:
        # Circle center is inside rectangle, push out to nearest edge
        dist_to_left = cx - left
        dist_to_right = right - cx
        dist_to_top = top - cy
        dist_to_bottom = cy - bottom
        
        min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        if min_dist == dist_to_left:
            return left - cr, cy
        elif min_dist == dist_to_right:
            return right + cr, cy
        elif min_dist == dist_to_top:
            return cx, top + cr
        else:
            return cx, bottom - cr
    
    # Push circle out along the collision normal
    dist = math.sqrt(dist_sq)
    nx = dx / dist  # Normal direction
    ny = dy / dist
    
    # Move circle center so it's exactly touching the rectangle
    penetration = cr - dist
    new_cx = cx + nx * (penetration + 0.001)  # Small epsilon to ensure separation
    new_cy = cy + ny * (penetration + 0.001)
    
    return new_cx, new_cy


def update_robot(robot: Robot, bounds: Tuple[float, float, float, float], 
                 obstacles: List[Obstacle] = None) -> None:
    """Update robot position, clamp to bounds, and prevent obstacle penetration."""
    # Store old position
    old_x, old_y = robot.x, robot.y
    
    # Apply velocity
    robot.x += robot.vx
    robot.y += robot.vy
    
    # Clamp to world bounds
    xmin, ymin, xmax, ymax = bounds
    robot.x = max(xmin + robot.radius, min(xmax - robot.radius, robot.x))
    robot.y = max(ymin + robot.radius, min(ymax - robot.radius, robot.y))
    
    # Check and resolve obstacle collisions
    if obstacles:
        for obs in obstacles:
            if circle_rect_collision(robot.x, robot.y, robot.radius,
                                    obs.x, obs.y, obs.width, obs.height):
                # Push robot out of obstacle
                robot.x, robot.y = push_circle_out_of_rect(
                    robot.x, robot.y, robot.radius,
                    obs.x, obs.y, obs.width, obs.height
                )
        
        # Final bounds clamp after obstacle resolution
        robot.x = max(xmin + robot.radius, min(xmax - robot.radius, robot.x))
        robot.y = max(ymin + robot.radius, min(ymax - robot.radius, robot.y))


def update_pedestrian(ped: Pedestrian, bounds: Tuple[float, float, float, float], 
                      rng: random.Random, obstacles: List[Obstacle] = None) -> None:
    """Update pedestrian with stochastic motion, boundary bounce, and obstacle avoidance."""
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
    
    # Bounce off obstacles
    if obstacles:
        for obs in obstacles:
            if circle_rect_collision(ped.x, ped.y, ped.radius,
                                    obs.x, obs.y, obs.width, obs.height):
                # Push pedestrian out and reverse velocity
                new_x, new_y = push_circle_out_of_rect(
                    ped.x, ped.y, ped.radius,
                    obs.x, obs.y, obs.width, obs.height
                )
                # Reverse velocity component based on push direction
                if abs(new_x - ped.x) > abs(new_y - ped.y):
                    ped.vx *= -1
                else:
                    ped.vy *= -1
                ped.x, ped.y = new_x, new_y


def update_state(state: SimState, robot_id: int, move: str, speed: float = 1.0) -> None:
    """Update simulation state for one time step."""
    # Apply action to specified robot
    if 0 <= robot_id < len(state.robots):
        apply_action(state.robots[robot_id], move, speed)
    
    # Update all robots (with obstacle collision prevention)
    for robot in state.robots:
        update_robot(robot, state.bounds, state.obstacles)
    
    # Update all pedestrians (with obstacle collision prevention)
    for ped in state.pedestrians:
        update_pedestrian(ped, state.bounds, state.rng, state.obstacles)
    
    # Record robot positions for trail rendering
    state.record_robot_positions()
    
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
