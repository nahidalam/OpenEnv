# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Renderer overlays for Robot City Isaac Environment.

Draws visual overlays on RGB frames:
- Legend (robot, pedestrians, goal)
- Ego heading arrow
- Future heatmap (risk cone)
- Uncertainty overlay
- Predicted agent trajectories
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Try to import OpenCV, fall back gracefully
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None

# Try PIL as fallback
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    pass


# Colors (BGR for OpenCV, RGB for PIL)
COLORS_BGR = {
    "robot": (200, 100, 0),      # Blue
    "pedestrian": (0, 150, 255),  # Orange
    "goal": (0, 255, 100),        # Green
    "trajectory": (255, 200, 0),  # Cyan
    "risk_high": (0, 0, 255),     # Red
    "risk_low": (0, 255, 255),    # Yellow
    "uncertainty": (128, 0, 128), # Purple
    "text": (255, 255, 255),      # White
}

COLORS_RGB = {
    "robot": (0, 100, 200),
    "pedestrian": (255, 150, 0),
    "goal": (100, 255, 0),
    "trajectory": (0, 200, 255),
    "risk_high": (255, 0, 0),
    "risk_low": (255, 255, 0),
    "uncertainty": (128, 0, 128),
    "text": (255, 255, 255),
}


def draw_all_overlays(
    frame: np.ndarray,
    ego_pose: Tuple[float, ...],
    goal_pose: Tuple[float, float, float],
    agents: List[Dict[str, Any]],
    events: Optional[Dict[str, Any]] = None,
    show_legend: bool = True,
    show_heading: bool = True,
    show_trajectories: bool = True,
    show_risk_cone: bool = True,
    show_uncertainty: bool = False,
) -> np.ndarray:
    """Draw all enabled overlays on frame.
    
    Args:
        frame: RGB frame (HxWx3 uint8)
        ego_pose: Ego pose (x, y, z, roll, pitch, yaw)
        goal_pose: Goal position (x, y, z)
        agents: List of agent state dicts
        events: Optional events dict (for uncertainty overlay)
        show_legend: Draw legend
        show_heading: Draw ego heading arrow
        show_trajectories: Draw predicted agent trajectories
        show_risk_cone: Draw risk/future heatmap cone
        show_uncertainty: Draw uncertainty overlay
        
    Returns:
        Frame with overlays drawn
    """
    if not CV2_AVAILABLE and not PIL_AVAILABLE:
        return frame
    
    # Work on a copy
    result = frame.copy()
    
    if show_uncertainty and events:
        result = draw_uncertainty_overlay(result, events)
    
    if show_risk_cone:
        result = draw_future_heatmap(result, ego_pose, agents)
    
    if show_trajectories:
        result = draw_predicted_trajectories(result, agents, ego_pose)
    
    if show_heading:
        result = draw_ego_heading_arrow(result, ego_pose)
    
    if show_legend:
        result = draw_legend(result)
    
    return result


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draw legend in top-left corner."""
    h, w = frame.shape[:2]
    
    if CV2_AVAILABLE:
        result = frame.copy()
        
        # Semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, (5, 5), (120, 75), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Legend entries
        entries = [
            ("Robot", COLORS_BGR["robot"]),
            ("Pedestrian", COLORS_BGR["pedestrian"]),
            ("Goal", COLORS_BGR["goal"]),
        ]
        
        for i, (label, color) in enumerate(entries):
            y = 20 + i * 20
            cv2.circle(result, (15, y), 6, color, -1)
            cv2.putText(result, label, (28, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, COLORS_BGR["text"], 1)
        
        return result
        
    elif PIL_AVAILABLE:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Background
        draw.rectangle([5, 5, 120, 75], fill=(40, 40, 40, 180))
        
        entries = [
            ("Robot", COLORS_RGB["robot"]),
            ("Pedestrian", COLORS_RGB["pedestrian"]),
            ("Goal", COLORS_RGB["goal"]),
        ]
        
        for i, (label, color) in enumerate(entries):
            y = 15 + i * 20
            draw.ellipse([9, y - 6, 21, y + 6], fill=color)
            draw.text((28, y - 6), label, fill=COLORS_RGB["text"])
        
        return np.array(img)
    
    return frame


def draw_ego_heading_arrow(
    frame: np.ndarray,
    ego_pose: Tuple[float, ...],
) -> np.ndarray:
    """Draw arrow showing robot heading direction."""
    h, w = frame.shape[:2]
    
    # Arrow starts from bottom center (robot position in camera view)
    start_x = w // 2
    start_y = h - 50
    
    # Get yaw from pose
    yaw = ego_pose[5] if len(ego_pose) > 5 else 0
    
    # Arrow length and direction (pointing forward/up in image)
    length = 40
    # In image coords, up is -y, and yaw=0 means forward
    end_x = start_x + int(length * math.sin(yaw) * 0.5)
    end_y = start_y - int(length * math.cos(yaw) * 0.5) - 20
    
    if CV2_AVAILABLE:
        result = frame.copy()
        cv2.arrowedLine(result, (start_x, start_y), (end_x, end_y),
                       COLORS_BGR["robot"], 3, tipLength=0.3)
        # Add robot icon
        cv2.circle(result, (start_x, start_y), 15, COLORS_BGR["robot"], -1)
        cv2.circle(result, (start_x, start_y), 15, (255, 255, 255), 2)
        return result
        
    elif PIL_AVAILABLE:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.line([start_x, start_y, end_x, end_y], fill=COLORS_RGB["robot"], width=3)
        draw.ellipse([start_x - 15, start_y - 15, start_x + 15, start_y + 15],
                    fill=COLORS_RGB["robot"], outline=(255, 255, 255))
        return np.array(img)
    
    return frame


def draw_future_heatmap(
    frame: np.ndarray,
    ego_pose: Tuple[float, ...],
    agents: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw risk/future heatmap as a cone in front of robot."""
    h, w = frame.shape[:2]
    
    if CV2_AVAILABLE:
        result = frame.copy()
        overlay = result.copy()
        
        # Draw risk cone from bottom center
        center_x = w // 2
        center_y = h - 30
        
        # Cone parameters
        cone_length = h // 2
        cone_angle = 45  # degrees
        
        # Calculate cone points
        left_angle = math.radians(-cone_angle)
        right_angle = math.radians(cone_angle)
        
        left_x = center_x + int(cone_length * math.sin(left_angle))
        left_y = center_y - int(cone_length * math.cos(left_angle))
        right_x = center_x + int(cone_length * math.sin(right_angle))
        right_y = center_y - int(cone_length * math.cos(right_angle))
        
        # Draw gradient cone (simplified as solid with transparency)
        pts = np.array([[center_x, center_y], [left_x, left_y], [right_x, right_y]])
        cv2.fillPoly(overlay, [pts], (0, 100, 255))  # Orange tint
        
        # Blend
        cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
        
        return result
        
    elif PIL_AVAILABLE:
        img = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        center_x = w // 2
        center_y = h - 30
        cone_length = h // 2
        cone_angle = 45
        
        left_angle = math.radians(-cone_angle)
        right_angle = math.radians(cone_angle)
        
        left_x = center_x + int(cone_length * math.sin(left_angle))
        left_y = center_y - int(cone_length * math.cos(left_angle))
        right_x = center_x + int(cone_length * math.sin(right_angle))
        right_y = center_y - int(cone_length * math.cos(right_angle))
        
        draw.polygon([(center_x, center_y), (left_x, left_y), (right_x, right_y)],
                    fill=(255, 100, 0, 50))
        
        result = Image.alpha_composite(img, overlay)
        return np.array(result.convert("RGB"))
    
    return frame


def draw_uncertainty_overlay(
    frame: np.ndarray,
    events: Dict[str, Any],
) -> np.ndarray:
    """Draw uncertainty overlay when failures are active."""
    h, w = frame.shape[:2]
    
    # Check if any uncertainty-inducing events
    dropout = events.get("dropout_applied", False)
    blur = events.get("blur_applied", False)
    
    if not dropout and not blur:
        return frame
    
    if CV2_AVAILABLE:
        result = frame.copy()
        overlay = result.copy()
        
        # Add purple tint for uncertainty
        cv2.rectangle(overlay, (0, 0), (w, h), COLORS_BGR["uncertainty"], -1)
        alpha = 0.2 if blur else 0.4
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        
        # Add "UNCERTAIN" text
        text = "CAMERA DROPOUT" if dropout else "MOTION BLUR"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(result, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)
        
        return result
        
    elif PIL_AVAILABLE:
        img = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (*COLORS_RGB["uncertainty"], 50 if blur else 100))
        result = Image.alpha_composite(img, overlay)
        
        draw = ImageDraw.Draw(result)
        text = "CAMERA DROPOUT" if dropout else "MOTION BLUR"
        draw.text((w // 2 - 60, 15), text, fill=(255, 255, 255))
        
        return np.array(result.convert("RGB"))
    
    return frame


def draw_predicted_trajectories(
    frame: np.ndarray,
    agents: List[Dict[str, Any]],
    ego_pose: Tuple[float, ...],
) -> np.ndarray:
    """Draw predicted trajectory lines for agents."""
    h, w = frame.shape[:2]
    
    if not agents:
        return frame
    
    ego_x, ego_y = ego_pose[0], ego_pose[1]
    ego_yaw = ego_pose[5] if len(ego_pose) > 5 else 0
    
    if CV2_AVAILABLE:
        result = frame.copy()
        
        for agent in agents:
            ax, ay = agent.get("x", 0), agent.get("y", 0)
            vx, vy = agent.get("vx", 0), agent.get("vy", 0)
            
            # Transform to ego-relative coordinates
            rel_x = ax - ego_x
            rel_y = ay - ego_y
            
            # Rotate by ego heading
            cos_h = math.cos(-ego_yaw)
            sin_h = math.sin(-ego_yaw)
            local_x = rel_x * cos_h - rel_y * sin_h
            local_y = rel_x * sin_h + rel_y * cos_h
            
            # Only draw if in front
            if local_y > 0.5:
                # Project to screen
                screen_x = w // 2 + int(local_x / local_y * w * 0.3)
                screen_y = h // 2 + int((1.0 / local_y) * h * 0.2)
                
                if 0 < screen_x < w and 0 < screen_y < h:
                    # Draw trajectory line
                    traj_len = 30
                    speed = math.sqrt(vx * vx + vy * vy)
                    if speed > 0.01:
                        dir_x = vx / speed
                        dir_y = vy / speed
                        
                        # Rotate velocity direction
                        local_vx = dir_x * cos_h - dir_y * sin_h
                        local_vy = dir_x * sin_h + dir_y * cos_h
                        
                        end_x = screen_x + int(local_vx * traj_len)
                        end_y = screen_y - int(local_vy * traj_len * 0.5)
                        
                        cv2.arrowedLine(result, (screen_x, screen_y), (end_x, end_y),
                                       COLORS_BGR["trajectory"], 2, tipLength=0.3)
        
        return result
        
    elif PIL_AVAILABLE:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        for agent in agents:
            ax, ay = agent.get("x", 0), agent.get("y", 0)
            vx, vy = agent.get("vx", 0), agent.get("vy", 0)
            
            rel_x = ax - ego_x
            rel_y = ay - ego_y
            
            cos_h = math.cos(-ego_yaw)
            sin_h = math.sin(-ego_yaw)
            local_x = rel_x * cos_h - rel_y * sin_h
            local_y = rel_x * sin_h + rel_y * cos_h
            
            if local_y > 0.5:
                screen_x = w // 2 + int(local_x / local_y * w * 0.3)
                screen_y = h // 2 + int((1.0 / local_y) * h * 0.2)
                
                if 0 < screen_x < w and 0 < screen_y < h:
                    speed = math.sqrt(vx * vx + vy * vy)
                    if speed > 0.01:
                        dir_x = vx / speed
                        dir_y = vy / speed
                        
                        local_vx = dir_x * cos_h - dir_y * sin_h
                        local_vy = dir_x * sin_h + dir_y * cos_h
                        
                        end_x = screen_x + int(local_vx * 30)
                        end_y = screen_y - int(local_vy * 15)
                        
                        draw.line([screen_x, screen_y, end_x, end_y],
                                 fill=COLORS_RGB["trajectory"], width=2)
        
        return np.array(img)
    
    return frame


def apply_motion_blur(frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply motion blur effect to frame."""
    if not CV2_AVAILABLE:
        return frame
    
    # Create motion blur kernel
    kernel_size = int(15 * strength)
    if kernel_size < 3:
        return frame
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    
    return cv2.filter2D(frame, -1, kernel)


def create_black_frame(shape: Tuple[int, int, int]) -> np.ndarray:
    """Create a black frame for camera dropout."""
    return np.zeros(shape, dtype=np.uint8)
