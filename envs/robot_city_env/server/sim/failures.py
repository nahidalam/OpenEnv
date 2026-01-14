# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Failure injection modes for robustness testing.

Supports various failure modes that affect observations and dynamics.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .dynamics import SimState, Pedestrian, Obstacle


@dataclass
class FailureConfig:
    """Configuration for failure injection modes.
    
    All probabilities are per-step probabilities.
    """
    # Camera dropout: observation rgb is blank/noisy
    camera_dropout_prob: float = 0.0
    
    # GPS drift: robot position in state is noisy
    gps_drift_sigma: float = 0.0
    
    # Pedestrian rush: pedestrians suddenly speed up
    pedestrian_rush_prob: float = 0.0
    pedestrian_rush_speed_boost: float = 2.0
    
    # Sudden obstacle: new obstacle appears
    sudden_obstacle_prob: float = 0.0
    
    # Action delay: action is ignored this step
    action_delay_prob: float = 0.0
    
    # Active failure events this step
    active_events: List[str] = field(default_factory=list)


def apply_failures(state: SimState, config: FailureConfig, 
                   rng: random.Random = None) -> Tuple[SimState, List[str]]:
    """Apply failure injection to state.
    
    Args:
        state: Current simulation state
        config: Failure configuration
        rng: Random number generator
        
    Returns:
        Tuple of (modified state, list of triggered events)
    """
    if rng is None:
        rng = state.rng
    
    events: List[str] = []
    
    # Camera dropout - just record event, actual handling in observation
    if config.camera_dropout_prob > 0 and rng.random() < config.camera_dropout_prob:
        events.append("camera_dropout")
    
    # GPS drift - add noise to reported robot positions
    if config.gps_drift_sigma > 0:
        for robot in state.robots:
            # Note: this affects the observation, not actual position
            # Actual drift is applied when building observation
            pass
        if config.gps_drift_sigma > 0.01:
            events.append("gps_drift")
    
    # Pedestrian rush
    if config.pedestrian_rush_prob > 0 and rng.random() < config.pedestrian_rush_prob:
        for ped in state.pedestrians:
            ped.vx *= config.pedestrian_rush_speed_boost
            ped.vy *= config.pedestrian_rush_speed_boost
        events.append("pedestrian_rush")
    
    # Sudden obstacle
    if config.sudden_obstacle_prob > 0 and rng.random() < config.sudden_obstacle_prob:
        # Add obstacle in a random location (avoiding robots)
        for _ in range(10):  # Try up to 10 times to find valid spot
            ox = rng.random() * 0.6 + 0.2
            oy = rng.random() * 0.6 + 0.2
            
            # Check not too close to any robot
            valid = True
            for robot in state.robots:
                dx = ox - robot.x
                dy = oy - robot.y
                if dx*dx + dy*dy < 0.1*0.1:
                    valid = False
                    break
            
            if valid:
                state.obstacles.append(Obstacle(x=ox, y=oy, width=0.05, height=0.05))
                events.append("sudden_obstacle")
                break
    
    # Action delay is handled at action application time
    if config.action_delay_prob > 0 and rng.random() < config.action_delay_prob:
        events.append("action_delay")
    
    return state, events


def apply_gps_drift(x: float, y: float, sigma: float, rng: random.Random) -> Tuple[float, float]:
    """Apply GPS drift noise to coordinates.
    
    Args:
        x, y: True coordinates
        sigma: Noise standard deviation
        rng: Random number generator
        
    Returns:
        Noisy coordinates (clamped to [0, 1])
    """
    noisy_x = x + rng.gauss(0, sigma)
    noisy_y = y + rng.gauss(0, sigma)
    noisy_x = max(0, min(1, noisy_x))
    noisy_y = max(0, min(1, noisy_y))
    return noisy_x, noisy_y


def create_dropout_image(size: int = 128) -> bytes:
    """Create a camera dropout image (static noise or black).
    
    Args:
        size: Image size in pixels
        
    Returns:
        PNG bytes
    """
    try:
        from PIL import Image
        import io
        import numpy as np
        
        # Create noise image
        noise = np.random.randint(0, 50, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        return b""
