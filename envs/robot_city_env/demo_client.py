#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Demo client for Robot City Environment.

Runs a random policy, decodes frames, and saves a GIF.

Usage:
    python -m envs.robot_city_env.demo_client
    
    # Or with custom server URL:
    python -m envs.robot_city_env.demo_client --url http://localhost:8000
"""

import argparse
import base64
import io
import os
import random
from typing import List

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


def run_demo(
    base_url: str = "http://localhost:8000",
    num_steps: int = 50,
    output_path: str = "runs/demo_robotcity.gif",
    scenario: str = "intersection_crosswalk",
):
    """Run demo episode and save GIF.
    
    Args:
        base_url: Server URL
        num_steps: Maximum steps to run
        output_path: Output GIF path
        scenario: Scenario to use
    """
    moves = ["forward", "forward", "forward", "turn_left", "turn_right", "noop"]
    
    frames: List[Image.Image] = []
    total_reward = 0.0
    collisions = 0
    near_misses = 0
    
    print(f"Connecting to {base_url}...")
    
    with RobotCityEnv(base_url=base_url) as client:
        # Reset with scenario
        print(f"Starting episode: scenario={scenario}")
        result = client.reset(scenario=scenario, num_robots=1, num_peds=4)
        
        # Decode and save first frame
        frames.append(decode_frame(result.observation.rgb))
        print(f"Step 0: events={result.observation.events}")
        
        for step in range(num_steps):
            # Random policy with bias toward forward
            move = random.choice(moves)
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
            
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}: reward={total_reward:.2f}, events={result.observation.events}")
            
            if result.done:
                print(f"Episode ended at step {step + 1}")
                break
        
        # Get final state
        state = client.state()
        print(f"\n=== Episode Summary ===")
        print(f"Steps: {state.step_count}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Collisions: {collisions}")
        print(f"Near Misses: {near_misses}")
    
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
        print(f"GIF saved to {output_path}")
    
    return total_reward, collisions, near_misses


def run_local_demo(num_steps: int = 30, output_path: str = "runs/demo_robotcity.gif"):
    """Run demo using local environment (no server).
    
    Useful for quick testing without starting server.
    """
    try:
        from .server.robot_city_environment import RobotCityEnvironment
    except ImportError:
        from server.robot_city_environment import RobotCityEnvironment
    
    moves = ["forward", "forward", "forward", "turn_left", "turn_right", "noop"]
    
    frames: List[Image.Image] = []
    total_reward = 0.0
    collisions = 0
    
    print("Running local demo (no server)...")
    
    env = RobotCityEnvironment()
    obs = env.reset(scenario="intersection_crosswalk", num_robots=1, num_peds=4)
    
    frames.append(decode_frame(obs.rgb))
    
    for step in range(num_steps):
        move = random.choice(moves)
        action = RobotCityAction(robot_id=0, move=move, speed=1.0)
        
        obs = env.step(action)
        frames.append(decode_frame(obs.rgb))
        
        total_reward += obs.reward or 0.0
        
        for event in obs.events:
            if "collision" in event:
                collisions += 1
        
        if obs.done:
            print(f"Episode ended at step {step + 1}")
            break
    
    print(f"\n=== Episode Summary ===")
    print(f"Steps: {env.state.step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Collisions: {collisions}")
    
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
        print(f"GIF saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Robot City Demo Client")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--steps", type=int, default=50, help="Max steps")
    parser.add_argument("--output", default="runs/demo_robotcity.gif", help="Output GIF path")
    parser.add_argument("--scenario", default="intersection_crosswalk", help="Scenario name")
    parser.add_argument("--local", action="store_true", help="Run locally without server")
    
    args = parser.parse_args()
    
    if args.local:
        run_local_demo(num_steps=args.steps, output_path=args.output)
    else:
        run_demo(
            base_url=args.url,
            num_steps=args.steps,
            output_path=args.output,
            scenario=args.scenario,
        )


if __name__ == "__main__":
    main()
