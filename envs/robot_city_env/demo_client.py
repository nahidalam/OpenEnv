#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Demo client for Robot City Environment.

Runs a greedy heuristic policy (move toward goal), decodes frames, and saves a GIF.

Usage:
    python -m envs.robot_city_env.demo_client --local
    
    # Or with server:
    python -m envs.robot_city_env.demo_client --url http://localhost:8000
    
    # Different scenario:
    python -m envs.robot_city_env.demo_client --local --scenario warehouse_aisles
"""

import argparse
import base64
import io
import math
import os
from typing import Dict, List, Any

try:
    from PIL import Image
except ImportError:
    print("Pillow required: pip install Pillow")
    exit(1)

# Try imports
try:
    from .client import RobotCityEnv
    from .models import RobotCityAction
except ImportError:
    from client import RobotCityEnv
    from models import RobotCityAction


def decode_frame(b64_png: str) -> Image.Image:
    """Decode base64 PNG to PIL Image."""
    img_bytes = base64.b64decode(b64_png)
    return Image.open(io.BytesIO(img_bytes))


def greedy_policy(state: Dict[str, Any], robot_id: int = 0) -> str:
    """Simple greedy policy - move toward goal.
    
    Args:
        state: Environment state dict with robot poses and goals
        robot_id: Which robot to control
        
    Returns:
        Move action string
    """
    robots = state.get("robots", [])
    if robot_id >= len(robots):
        return "noop"
    
    robot = robots[robot_id]
    
    # Compute direction to goal
    dx = robot["goal_x"] - robot["x"]
    dy = robot["goal_y"] - robot["y"]
    dist = math.sqrt(dx * dx + dy * dy)
    
    if dist < 0.05:
        return "noop"  # Close enough to goal
    
    # Normalize
    goal_dx = dx / dist
    goal_dy = dy / dist
    
    # Current heading
    theta = robot.get("theta", 0.0)
    heading_x = math.cos(theta)
    heading_y = math.sin(theta)
    
    # Dot product (alignment with goal direction)
    alignment = heading_x * goal_dx + heading_y * goal_dy
    
    # Cross product (turn direction)
    cross = heading_x * goal_dy - heading_y * goal_dx
    
    # If well-aligned, go forward
    if alignment > 0.7:
        return "forward"
    
    # Otherwise, turn toward goal
    if cross > 0:
        return "turn_left"
    else:
        return "turn_right"


def run_demo(
    base_url: str = "http://localhost:8000",
    num_steps: int = 100,
    output_path: str = "runs/demo_robotcity.gif",
    scenario: str = "crosswalk_occlusion",
    camera_dropout: float = 0.0,
    action_delay: float = 0.0,
    seed: int = 42,
):
    """Run demo episode with greedy policy and save GIF.
    
    Args:
        base_url: Server URL
        num_steps: Maximum steps to run
        output_path: Output GIF path
        scenario: Scenario to use
        camera_dropout: Probability of camera dropout per step
        action_delay: Probability of action delay per step
        seed: Random seed
    """
    frames: List[Image.Image] = []
    total_reward = 0.0
    collisions = 0
    near_misses = 0
    camera_dropouts = 0
    action_delays = 0
    goal_reached = False
    
    print(f"=" * 60)
    print(f"ROBOT CITY DEMO")
    print(f"=" * 60)
    print(f"Connecting to {base_url}...")
    
    with RobotCityEnv(base_url=base_url) as client:
        # Reset with scenario and failure injection
        print(f"Scenario: {scenario}")
        print(f"Policy: greedy (move toward goal)")
        if camera_dropout > 0 or action_delay > 0:
            print(f"Failures: camera_dropout={camera_dropout}, action_delay={action_delay}")
        result = client.reset(
            scenario=scenario,
            num_robots=1,
            num_peds=4,
            seed=seed,
            camera_dropout_prob=camera_dropout,
            action_delay_prob=action_delay,
        )
        
        # Decode and save first frame
        frames.append(decode_frame(result.observation.rgb))
        print(f"\nStep 0: initialized")
        
        for step in range(num_steps):
            # Greedy policy toward goal
            move = greedy_policy(result.observation.state, robot_id=0)
            action = RobotCityAction(robot_id=0, move=move, speed=1.0)
            
            result = client.step(action)
            
            # Decode frame
            frames.append(decode_frame(result.observation.rgb))
            
            # Track stats
            total_reward += result.reward or 0.0
            
            for event in result.observation.events:
                if "collision" in event:
                    collisions += 1
                if "near_miss" in event:
                    near_misses += 1
                if "camera_dropout" in event:
                    camera_dropouts += 1
                if "action_delay" in event:
                    action_delays += 1
                if "goal" in event.lower():
                    goal_reached = True
            
            if (step + 1) % 20 == 0:
                robot = result.observation.state.get("robots", [{}])[0]
                goal_dist = math.sqrt(
                    (robot.get("goal_x", 0) - robot.get("x", 0))**2 +
                    (robot.get("goal_y", 0) - robot.get("y", 0))**2
                )
                print(f"Step {step + 1:3d}: reward={total_reward:+.2f}, goal_dist={goal_dist:.3f}, move={move}")
            
            if result.done:
                print(f"\nEpisode ended at step {step + 1}")
                if "all_goals_reached" in result.observation.events:
                    goal_reached = True
                break
        
        # Get final state
        state = client.state()
        
    print(f"\n{'=' * 60}")
    print(f"EPISODE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Steps: {state.step_count}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Goal Reached: {goal_reached}")
    print(f"  Collisions: {collisions}")
    print(f"  Near Misses: {near_misses}")
    if camera_dropouts > 0 or action_delays > 0:
        print(f"  Camera Dropouts: {camera_dropouts}")
        print(f"  Action Delays: {action_delays}")
    
    # Save GIF
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    if frames:
        print(f"\nSaving {len(frames)} frames to {output_path}...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 100ms per frame
            loop=0,
        )
        print(f"GIF saved to: {output_path}")
    
    print(f"{'=' * 60}")
    
    return total_reward, collisions, near_misses, camera_dropouts, action_delays


def run_local_demo(
    num_steps: int = 100,
    output_path: str = "runs/demo_robotcity.gif",
    scenario: str = "crosswalk_occlusion",
    camera_dropout: float = 0.0,
    action_delay: float = 0.0,
    seed: int = 42,
):
    """Run demo using local environment (no server).
    
    Useful for quick testing without starting server.
    
    Args:
        num_steps: Maximum steps to run
        output_path: Output GIF path  
        scenario: Scenario name
        camera_dropout: Probability of camera dropout per step
        action_delay: Probability of action delay per step
        seed: Random seed
    """
    try:
        from .server.robot_city_environment import RobotCityEnvironment
    except ImportError:
        from server.robot_city_environment import RobotCityEnvironment
    
    frames: List[Image.Image] = []
    total_reward = 0.0
    collisions = 0
    near_misses = 0
    camera_dropouts = 0
    action_delays = 0
    goal_reached = False
    
    print(f"=" * 60)
    print(f"ROBOT CITY DEMO (Local)")
    print(f"=" * 60)
    print(f"Scenario: {scenario}")
    print(f"Policy: greedy (move toward goal)")
    print(f"Max steps: {num_steps}")
    if camera_dropout > 0 or action_delay > 0:
        print(f"Failures: camera_dropout={camera_dropout}, action_delay={action_delay}")
    print(f"=" * 60)
    
    env = RobotCityEnvironment()
    obs = env.reset(
        scenario=scenario,
        num_robots=1,
        num_peds=4,
        seed=seed,
        camera_dropout_prob=camera_dropout,
        action_delay_prob=action_delay,
    )
    
    frames.append(decode_frame(obs.rgb))
    print(f"\nStep 0: initialized")
    
    for step in range(num_steps):
        # Greedy policy toward goal
        move = greedy_policy(obs.state, robot_id=0)
        action = RobotCityAction(robot_id=0, move=move, speed=1.0)
        
        obs = env.step(action)
        frames.append(decode_frame(obs.rgb))
        
        total_reward += obs.reward or 0.0
        
        for event in obs.events:
            if "collision" in event:
                collisions += 1
            if "near_miss" in event:
                near_misses += 1
            if "camera_dropout" in event:
                camera_dropouts += 1
            if "action_delay" in event:
                action_delays += 1
            if "goal" in event.lower():
                goal_reached = True
        
        if (step + 1) % 20 == 0:
            robot = obs.state.get("robots", [{}])[0]
            goal_dist = math.sqrt(
                (robot.get("goal_x", 0) - robot.get("x", 0))**2 +
                (robot.get("goal_y", 0) - robot.get("y", 0))**2
            )
            print(f"Step {step + 1:3d}: reward={total_reward:+.2f}, goal_dist={goal_dist:.3f}, move={move}")
        
        if obs.done:
            print(f"\nEpisode ended at step {step + 1}")
            if "all_goals_reached" in obs.events:
                goal_reached = True
            break
    
    print(f"\n{'=' * 60}")
    print(f"EPISODE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Steps: {env.state.step_count}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Goal Reached: {goal_reached}")
    print(f"  Collisions: {collisions}")
    print(f"  Near Misses: {near_misses}")
    if camera_dropouts > 0 or action_delays > 0:
        print(f"  Camera Dropouts: {camera_dropouts}")
        print(f"  Action Delays: {action_delays}")
    
    # Save GIF
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    if frames:
        print(f"\nSaving {len(frames)} frames to {output_path}...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"GIF saved to: {output_path}")
    
    print(f"{'=' * 60}")
    
    return total_reward, collisions, near_misses, camera_dropouts, action_delays, goal_reached


def main():
    parser = argparse.ArgumentParser(
        description="Robot City Demo Client - Run greedy policy and save GIF"
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--steps", type=int, default=100, help="Max steps (default: 100)")
    parser.add_argument("--output", default="runs/demo_robotcity.gif", help="Output GIF path")
    parser.add_argument(
        "--scenario",
        default="crosswalk_occlusion",
        choices=["crosswalk_occlusion", "sidewalk_delivery", "warehouse_aisles"],
        help="Scenario name (default: crosswalk_occlusion)"
    )
    parser.add_argument("--local", action="store_true", help="Run locally without server")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    # Failure injection options
    parser.add_argument(
        "--camera-dropout", type=float, default=0.0,
        help="Camera dropout probability per step (default: 0.0)"
    )
    parser.add_argument(
        "--action-delay", type=float, default=0.0,
        help="Action delay probability per step (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    if args.local:
        run_local_demo(
            num_steps=args.steps,
            output_path=args.output,
            scenario=args.scenario,
            camera_dropout=args.camera_dropout,
            action_delay=args.action_delay,
            seed=args.seed,
        )
    else:
        run_demo(
            base_url=args.url,
            num_steps=args.steps,
            output_path=args.output,
            scenario=args.scenario,
            camera_dropout=args.camera_dropout,
            action_delay=args.action_delay,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
