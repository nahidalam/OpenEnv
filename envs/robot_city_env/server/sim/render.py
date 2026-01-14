# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rendering utilities for Robot City using Pillow.

Generates RGB images and heatmap overlays.
"""

import base64
import io
import math
from typing import List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise ImportError("Pillow is required for rendering. Install with: pip install Pillow")

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for rendering. Install with: pip install numpy")

from .dynamics import SimState, Robot, Pedestrian, Obstacle


# Colors (RGB)
COLOR_BG = (240, 240, 240)
COLOR_ROBOT = (0, 100, 200)  # Blue
COLOR_ROBOT_GOAL = (0, 200, 100)  # Green
COLOR_PEDESTRIAN = (200, 100, 0)  # Orange
COLOR_OBSTACLE = (80, 80, 80)  # Dark gray
COLOR_LANE = (200, 200, 200)  # Light gray


def _to_pixel(x: float, y: float, size: int) -> Tuple[int, int]:
    """Convert normalized coords to pixel coords."""
    px = int(x * size)
    py = int((1.0 - y) * size)  # Flip y for image coords
    return px, py


def _draw_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float,
                 size: int, color: Tuple[int, int, int], outline: Optional[Tuple[int, int, int]] = None) -> None:
    """Draw a filled circle."""
    px, py = _to_pixel(cx, cy, size)
    r_px = int(radius * size)
    draw.ellipse([px - r_px, py - r_px, px + r_px, py + r_px], fill=color, outline=outline)


def _draw_robot(draw: ImageDraw.ImageDraw, robot: Robot, size: int) -> None:
    """Draw a robot with heading indicator."""
    px, py = _to_pixel(robot.x, robot.y, size)
    r_px = int(robot.radius * size)
    
    # Body
    draw.ellipse([px - r_px, py - r_px, px + r_px, py + r_px], 
                 fill=COLOR_ROBOT, outline=(0, 50, 150))
    
    # Heading indicator
    head_len = r_px * 1.5
    hx = px + int(head_len * math.cos(-robot.theta + math.pi/2))
    hy = py + int(head_len * math.sin(-robot.theta + math.pi/2))
    draw.line([px, py, hx, hy], fill=(255, 255, 255), width=2)
    
    # Goal marker (small)
    gx, gy = _to_pixel(robot.goal_x, robot.goal_y, size)
    gr = 4
    draw.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=COLOR_ROBOT_GOAL)


def _draw_pedestrian(draw: ImageDraw.ImageDraw, ped: Pedestrian, size: int) -> None:
    """Draw a pedestrian."""
    _draw_circle(draw, ped.x, ped.y, ped.radius, size, COLOR_PEDESTRIAN)


def _draw_obstacle(draw: ImageDraw.ImageDraw, obs: Obstacle, size: int) -> None:
    """Draw a rectangular obstacle."""
    half_w = obs.width / 2
    half_h = obs.height / 2
    
    x1, y1 = _to_pixel(obs.x - half_w, obs.y + half_h, size)
    x2, y2 = _to_pixel(obs.x + half_w, obs.y - half_h, size)
    
    draw.rectangle([x1, y1, x2, y2], fill=COLOR_OBSTACLE)


def render_global_view(state: SimState, size: int = 128) -> str:
    """Render top-down global view as base64 PNG.
    
    Args:
        state: Simulation state
        size: Image size in pixels (square)
        
    Returns:
        Base64 encoded PNG string
    """
    img = Image.new("RGB", (size, size), COLOR_BG)
    draw = ImageDraw.Draw(img)
    
    # Draw obstacles first (background)
    for obs in state.obstacles:
        _draw_obstacle(draw, obs, size)
    
    # Draw pedestrians
    for ped in state.pedestrians:
        _draw_pedestrian(draw, ped, size)
    
    # Draw robots on top
    for robot in state.robots:
        _draw_robot(draw, robot, size)
    
    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_ego_view(state: SimState, robot_id: int = 0, size: int = 64, 
                    view_radius: float = 0.15) -> Optional[str]:
    """Render ego-centric view around a robot.
    
    Args:
        state: Simulation state
        robot_id: Which robot's view to render
        size: Image size in pixels
        view_radius: View extent in normalized coords
        
    Returns:
        Base64 encoded PNG or None if robot_id invalid
    """
    if robot_id >= len(state.robots):
        return None
    
    robot = state.robots[robot_id]
    
    img = Image.new("RGB", (size, size), COLOR_BG)
    draw = ImageDraw.Draw(img)
    
    # Coordinate transform: center on robot
    def to_ego_pixel(x: float, y: float) -> Tuple[int, int]:
        # Relative to robot
        dx = x - robot.x
        dy = y - robot.y
        # Scale to view
        px = int((dx / view_radius + 1.0) * size / 2)
        py = int((1.0 - dy / view_radius) * size / 2)
        return px, py
    
    # Draw nearby obstacles
    for obs in state.obstacles:
        if abs(obs.x - robot.x) < view_radius * 2 and abs(obs.y - robot.y) < view_radius * 2:
            half_w = obs.width / 2
            half_h = obs.height / 2
            x1, y1 = to_ego_pixel(obs.x - half_w, obs.y + half_h)
            x2, y2 = to_ego_pixel(obs.x + half_w, obs.y - half_h)
            draw.rectangle([x1, y1, x2, y2], fill=COLOR_OBSTACLE)
    
    # Draw nearby pedestrians
    for ped in state.pedestrians:
        if abs(ped.x - robot.x) < view_radius and abs(ped.y - robot.y) < view_radius:
            px, py = to_ego_pixel(ped.x, ped.y)
            r = int(ped.radius / view_radius * size / 2)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=COLOR_PEDESTRIAN)
    
    # Draw robot at center
    cx, cy = size // 2, size // 2
    r = int(robot.radius / view_radius * size / 2)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLOR_ROBOT)
    
    # Encode
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_heatmap(values: np.ndarray, size: int = 128, 
                   colormap: str = "hot") -> str:
    """Render a 2D array as a heatmap PNG.
    
    Args:
        values: 2D numpy array of values in [0, 1]
        size: Output image size
        colormap: Color scheme ("hot" or "cool")
        
    Returns:
        Base64 encoded PNG
    """
    # Resize to output size
    h, w = values.shape
    img_array = np.zeros((size, size, 4), dtype=np.uint8)
    
    # Simple bilinear-ish upscale
    for py in range(size):
        for px in range(size):
            vy = int(py / size * h)
            vx = int(px / size * w)
            vy = min(vy, h - 1)
            vx = min(vx, w - 1)
            val = values[vy, vx]
            
            # Convert to color
            if colormap == "hot":
                # Red-yellow gradient
                r = int(min(255, val * 2 * 255))
                g = int(max(0, (val - 0.5) * 2 * 255))
                b = 0
            else:  # cool
                # Blue-cyan gradient
                r = 0
                g = int(max(0, (val - 0.5) * 2 * 255))
                b = int(min(255, val * 2 * 255))
            
            # Alpha based on value (transparent when low)
            a = int(val * 180)
            
            img_array[py, px] = [r, g, b, a]
    
    img = Image.fromarray(img_array, mode="RGBA")
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def compute_uncertainty_heatmap(state: SimState, robot_id: int = 0,
                                 grid_size: int = 16, num_samples: int = 8,
                                 horizon: int = 8) -> np.ndarray:
    """Compute collision risk heatmap via stochastic rollouts.
    
    Args:
        state: Current simulation state
        robot_id: Robot to compute risk for
        grid_size: Resolution of output heatmap
        num_samples: Number of stochastic rollouts
        horizon: Steps to simulate forward
        
    Returns:
        2D numpy array of collision probabilities
    """
    from .dynamics import update_pedestrian, circle_circle_collision
    
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    if robot_id >= len(state.robots):
        return heatmap
    
    robot = state.robots[robot_id]
    
    for _ in range(num_samples):
        # Copy pedestrian states
        ped_states = [(p.x, p.y, p.vx, p.vy) for p in state.pedestrians]
        
        # Simulate forward
        for t in range(horizon):
            for i, (px, py, pvx, pvy) in enumerate(ped_states):
                # Add noise and update
                noise_x = state.rng.gauss(0, 0.005)
                noise_y = state.rng.gauss(0, 0.005)
                new_vx = pvx + noise_x
                new_vy = pvy + noise_y
                new_x = px + new_vx
                new_y = py + new_vy
                
                # Bounce
                if new_x < 0 or new_x > 1:
                    new_vx *= -1
                    new_x = max(0, min(1, new_x))
                if new_y < 0 or new_y > 1:
                    new_vy *= -1
                    new_y = max(0, min(1, new_y))
                
                ped_states[i] = (new_x, new_y, new_vx, new_vy)
                
                # Mark grid cell as risky if collision possible
                gx = int(new_x * grid_size)
                gy = int(new_y * grid_size)
                gx = min(gx, grid_size - 1)
                gy = min(gy, grid_size - 1)
                
                # Check if robot could be here
                dist = math.sqrt((new_x - robot.x)**2 + (new_y - robot.y)**2)
                if dist < 0.2:  # Within robot's potential reach
                    heatmap[grid_size - 1 - gy, gx] += 1.0 / (num_samples * horizon)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def compute_future_heatmap(state: SimState, horizon: int = 10,
                           grid_size: int = 16) -> np.ndarray:
    """Predict pedestrian occupancy heatmap at t+horizon.
    
    Simple linear extrapolation of pedestrian motion.
    
    Args:
        state: Current simulation state
        horizon: Steps to predict ahead
        grid_size: Resolution of output heatmap
        
    Returns:
        2D numpy array of predicted occupancy
    """
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for ped in state.pedestrians:
        # Extrapolate position
        future_x = ped.x + ped.vx * horizon
        future_y = ped.y + ped.vy * horizon
        
        # Clamp to bounds
        future_x = max(0, min(1, future_x))
        future_y = max(0, min(1, future_y))
        
        # Mark on grid with Gaussian spread
        gx = int(future_x * grid_size)
        gy = int(future_y * grid_size)
        gx = min(gx, grid_size - 1)
        gy = min(gy, grid_size - 1)
        
        # Add to heatmap with spread
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    dist = math.sqrt(dx*dx + dy*dy)
                    weight = math.exp(-dist * dist / 2)
                    heatmap[grid_size - 1 - ny, nx] += weight
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap
