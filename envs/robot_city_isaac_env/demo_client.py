#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Demo client for Robot City Isaac Environment.

Runs a greedy policy, records frames, and exports video (MP4/GIF) + JSONL logs.

Usage:
    python -m envs.robot_city_isaac_env.demo_client --url http://localhost:8001
    
    # With failure injection
    python -m envs.robot_city_isaac_env.demo_client \
        --url http://localhost:8001 \
        --scenario sidewalk_delivery \
        --camera-dropout 0.2 \
        --action-delay 0.1 \
        --output runs/isaac_demo.mp4
"""

import argparse
import base64
import io
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Pillow required: pip install Pillow")
    sys.exit(1)

# Try imports
try:
    from .client import RobotCityIsaacEnv, decode_frame
    from .models import RobotCityIsaacAction
except ImportError:
    from client import RobotCityIsaacEnv, decode_frame
    from models import RobotCityIsaacAction


def greedy_policy(
    ego_pose: Tuple[float, ...],
    goal_pose: Tuple[float, float, float],
) -> Tuple[float, float]:
    """Simple greedy policy - move toward goal.
    
    Args:
        ego_pose: Ego pose (x, y, z, roll, pitch, yaw)
        goal_pose: Goal position (x, y, z)
        
    Returns:
        (linear_vel, angular_vel) tuple
    """
    ego_x, ego_y = ego_pose[0], ego_pose[1]
    ego_yaw = ego_pose[5] if len(ego_pose) > 5 else 0
    goal_x, goal_y = goal_pose[0], goal_pose[1]
    
    # Compute direction to goal
    dx = goal_x - ego_x
    dy = goal_y - ego_y
    dist = math.sqrt(dx * dx + dy * dy)
    
    if dist < 0.5:
        return 0.0, 0.0  # Close enough to goal
    
    # Target heading
    target_heading = math.atan2(dy, dx)
    
    # Heading error
    heading_error = target_heading - ego_yaw
    
    # Normalize to [-pi, pi]
    while heading_error > math.pi:
        heading_error -= 2 * math.pi
    while heading_error < -math.pi:
        heading_error += 2 * math.pi
    
    # Compute velocities
    if abs(heading_error) > 0.3:
        # Need to turn
        linear_vel = 0.2
        angular_vel = 0.5 * (1 if heading_error > 0 else -1)
    else:
        # Mostly aligned, go forward
        linear_vel = min(0.8, 0.3 + 0.5 * (1 - abs(heading_error) / math.pi))
        angular_vel = 0.3 * heading_error
    
    return linear_vel, angular_vel


def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 20,
) -> None:
    """Save frames as video (MP4 or GIF).
    
    Args:
        frames: List of RGB frames
        output_path: Output path (.mp4 or .gif)
        fps: Frames per second
    """
    if not frames:
        print("No frames to save")
        return
    
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == ".mp4":
        # Try imageio first
        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            return
        except ImportError:
            pass
        
        # Try OpenCV
        try:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return
        except ImportError:
            print("Warning: Neither imageio nor OpenCV available for MP4. Saving as GIF.")
            output_path = output_path.replace(".mp4", ".gif")
    
    # GIF fallback
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def save_jsonl(
    logs: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Save logs as JSONL file.
    
    Args:
        logs: List of log dictionaries
        output_path: Output path
    """
    with open(output_path, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")


def run_demo(
    base_url: str = "http://localhost:8001",
    scenario: str = "sidewalk_delivery",
    num_steps: int = 200,
    num_agents: int = 6,
    seed: int = 42,
    resolution: int = 512,
    camera_dropout: float = 0.0,
    action_delay: float = 0.0,
    motion_blur: float = 0.0,
    show_overlay: bool = True,
    output_path: str = "runs/isaac_demo.mp4",
):
    """Run demo episode and save video + logs.
    
    Args:
        base_url: Server URL
        scenario: Scenario name
        num_steps: Maximum steps
        num_agents: Number of pedestrian agents
        seed: Random seed
        resolution: Image resolution
        camera_dropout: Camera dropout probability
        action_delay: Action delay probability
        motion_blur: Motion blur probability
        show_overlay: Whether to use overlay frames
        output_path: Output video path
    """
    frames: List[np.ndarray] = []
    logs: List[Dict[str, Any]] = []
    total_reward = 0.0
    collisions = 0
    near_misses = 0
    dropouts = 0
    delays = 0
    goal_reached = False
    
    print("=" * 60)
    print("ROBOT CITY ISAAC DEMO")
    print("=" * 60)
    print(f"Server: {base_url}")
    print(f"Scenario: {scenario}")
    print(f"Agents: {num_agents}")
    print(f"Max steps: {num_steps}")
    if camera_dropout > 0 or action_delay > 0 or motion_blur > 0:
        print(f"Failures: dropout={camera_dropout}, delay={action_delay}, blur={motion_blur}")
    print("=" * 60)
    
    with RobotCityIsaacEnv(base_url=base_url) as client:
        # Reset
        print("\nInitializing episode...")
        result = client.reset(
            scenario=scenario,
            num_agents=num_agents,
            seed=seed,
            resolution=resolution,
            camera_dropout=camera_dropout,
            action_delay=action_delay,
            motion_blur=motion_blur,
            show_overlays=show_overlay,
        )
        
        # Get initial frame
        rgb_key = "rgb_with_overlay" if show_overlay and result.observation.rgb_with_overlay else "rgb"
        rgb_b64 = getattr(result.observation, rgb_key) or result.observation.rgb
        frame = decode_frame(rgb_b64)
        frames.append(frame)
        
        print(f"Step 0: initialized, goal_dist={result.observation.goal_distance:.2f}m")
        
        for step in range(num_steps):
            # Greedy policy
            linear_vel, angular_vel = greedy_policy(
                result.observation.ego_pose,
                result.observation.goal_pose,
            )
            
            action = RobotCityIsaacAction(
                linear_vel=linear_vel,
                angular_vel=angular_vel,
            )
            
            # Step
            result = client.step(action)
            
            # Get frame
            rgb_key = "rgb_with_overlay" if show_overlay and result.observation.rgb_with_overlay else "rgb"
            rgb_b64 = getattr(result.observation, rgb_key) or result.observation.rgb
            frame = decode_frame(rgb_b64)
            frames.append(frame)
            
            # Track stats
            total_reward += result.reward or 0.0
            events = result.observation.events
            
            if events.get("collision"):
                collisions += 1
            if events.get("near_miss"):
                near_misses += 1
            if events.get("dropout_applied"):
                dropouts += 1
            if events.get("delay_applied"):
                delays += 1
            if events.get("goal_reached"):
                goal_reached = True
            
            # Log
            logs.append({
                "step_idx": step + 1,
                "action": {"linear_vel": linear_vel, "angular_vel": angular_vel},
                "reward": result.reward,
                "done": result.done,
                "goal_distance": result.observation.goal_distance,
                "events": events,
            })
            
            # Print progress
            if (step + 1) % 20 == 0:
                print(f"Step {step + 1:3d}: reward={total_reward:+.2f}, "
                      f"goal_dist={result.observation.goal_distance:.2f}m")
            
            if result.done:
                print(f"\nEpisode ended at step {step + 1}")
                break
    
    # Summary
    print("\n" + "=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"  Steps: {len(frames) - 1}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Goal Reached: {goal_reached}")
    print(f"  Collisions: {collisions}")
    print(f"  Near Misses: {near_misses}")
    if dropouts > 0:
        print(f"  Camera Dropouts: {dropouts}")
    if delays > 0:
        print(f"  Action Delays: {delays}")
    
    # Save outputs
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    print(f"\nSaving {len(frames)} frames to {output_path}...")
    save_video(frames, output_path, fps=20)
    print(f"Video saved to: {output_path}")
    
    # Save logs
    log_path = output_path.rsplit(".", 1)[0] + ".jsonl"
    save_jsonl(logs, log_path)
    print(f"Logs saved to: {log_path}")
    
    print("=" * 60)
    
    return total_reward, goal_reached


def run_local_demo(
    scenario: str = "sidewalk_delivery",
    num_steps: int = 100,
    num_agents: int = 6,
    seed: int = 42,
    resolution: int = 512,
    camera_dropout: float = 0.0,
    action_delay: float = 0.0,
    motion_blur: float = 0.0,
    show_overlay: bool = True,
    output_path: str = "runs/isaac_local_demo.mp4",
):
    """Run demo locally without server.
    
    Uses fallback rendering if Isaac Sim is not available.
    """
    try:
        from .server.robot_city_isaac_env import RobotCityIsaacEnvironment
    except ImportError:
        from server.robot_city_isaac_env import RobotCityIsaacEnvironment
    
    frames: List[np.ndarray] = []
    logs: List[Dict[str, Any]] = []
    total_reward = 0.0
    goal_reached = False
    
    print("=" * 60)
    print("ROBOT CITY ISAAC DEMO (Local)")
    print("=" * 60)
    print(f"Scenario: {scenario}")
    print(f"Agents: {num_agents}")
    print(f"Max steps: {num_steps}")
    print("=" * 60)
    
    env = RobotCityIsaacEnvironment(headless=True, resolution=resolution)
    
    obs = env.reset(
        scenario=scenario,
        num_agents=num_agents,
        seed=seed,
        resolution=resolution,
        camera_dropout=camera_dropout,
        action_delay=action_delay,
        motion_blur=motion_blur,
        show_overlays=show_overlay,
    )
    
    # Get initial frame
    rgb_b64 = obs.rgb_with_overlay or obs.rgb
    frame = decode_frame(rgb_b64)
    frames.append(frame)
    
    print(f"\nStep 0: initialized")
    
    for step in range(num_steps):
        # Greedy policy
        linear_vel, angular_vel = greedy_policy(obs.ego_pose, obs.goal_pose)
        
        action = RobotCityIsaacAction(
            linear_vel=linear_vel,
            angular_vel=angular_vel,
        )
        
        obs = env.step(action)
        
        # Get frame
        rgb_b64 = obs.rgb_with_overlay or obs.rgb
        frame = decode_frame(rgb_b64)
        frames.append(frame)
        
        total_reward += obs.reward or 0.0
        
        if obs.events.get("goal_reached"):
            goal_reached = True
        
        logs.append({
            "step_idx": step + 1,
            "action": {"linear_vel": linear_vel, "angular_vel": angular_vel},
            "reward": obs.reward,
            "done": obs.done,
            "goal_distance": obs.goal_distance,
            "events": obs.events,
        })
        
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1:3d}: reward={total_reward:+.2f}, "
                  f"goal_dist={obs.goal_distance:.2f}m")
        
        if obs.done:
            print(f"\nEpisode ended at step {step + 1}")
            break
    
    env.close()
    
    print("\n" + "=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"  Steps: {len(frames) - 1}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Goal Reached: {goal_reached}")
    
    # Save outputs
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    print(f"\nSaving {len(frames)} frames to {output_path}...")
    save_video(frames, output_path, fps=20)
    print(f"Video saved to: {output_path}")
    
    log_path = output_path.rsplit(".", 1)[0] + ".jsonl"
    save_jsonl(logs, log_path)
    print(f"Logs saved to: {log_path}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Robot City Isaac Demo - Generate realistic robot simulation videos"
    )
    parser.add_argument("--url", default="http://localhost:8001", help="Server URL")
    parser.add_argument("--local", action="store_true", help="Run locally without server")
    parser.add_argument(
        "--scenario",
        default="sidewalk_delivery",
        choices=["crosswalk_occlusion", "sidewalk_delivery", "warehouse_aisles"],
        help="Scenario name"
    )
    parser.add_argument("--steps", type=int, default=200, help="Max steps")
    parser.add_argument("--num-agents", type=int, default=6, help="Number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--output", default="runs/isaac_demo.mp4", help="Output video path")
    
    # Failure injection
    parser.add_argument("--camera-dropout", type=float, default=0.0,
                       help="Camera dropout probability")
    parser.add_argument("--action-delay", type=float, default=0.0,
                       help="Action delay probability")
    parser.add_argument("--motion-blur", type=float, default=0.0,
                       help="Motion blur probability")
    
    # Overlay
    parser.add_argument("--overlay", action="store_true", default=True,
                       help="Show visual overlays")
    parser.add_argument("--no-overlay", action="store_false", dest="overlay",
                       help="Disable visual overlays")
    
    args = parser.parse_args()
    
    if args.local:
        run_local_demo(
            scenario=args.scenario,
            num_steps=args.steps,
            num_agents=args.num_agents,
            seed=args.seed,
            resolution=args.resolution,
            camera_dropout=args.camera_dropout,
            action_delay=args.action_delay,
            motion_blur=args.motion_blur,
            show_overlay=args.overlay,
            output_path=args.output,
        )
    else:
        run_demo(
            base_url=args.url,
            scenario=args.scenario,
            num_steps=args.steps,
            num_agents=args.num_agents,
            seed=args.seed,
            resolution=args.resolution,
            camera_dropout=args.camera_dropout,
            action_delay=args.action_delay,
            motion_blur=args.motion_blur,
            show_overlay=args.overlay,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
