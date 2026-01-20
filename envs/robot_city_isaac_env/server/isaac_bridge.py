# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Isaac Sim Bridge - Wrapper for NVIDIA Isaac Sim simulation.

This module handles:
- Isaac Sim initialization and scene loading
- Robot and agent spawning
- Camera setup and RGB frame capture
- Simulation stepping
- Snapshot/restore for counterfactual planning

Requires NVIDIA Isaac Sim to be installed.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Isaac Sim imports - these will fail gracefully if not installed
ISAAC_AVAILABLE = False
try:
    from omni.isaac.kit import SimulationApp
    ISAAC_AVAILABLE = True
except ImportError:
    SimulationApp = None

# Lazy imports for Isaac Sim modules (loaded after SimulationApp starts)
_isaac_modules = {}


def _lazy_import_isaac():
    """Lazy import Isaac Sim modules after SimulationApp is started."""
    global _isaac_modules
    if _isaac_modules:
        return _isaac_modules
    
    try:
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core import World
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.prims import XFormPrim, RigidPrim
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.sensor import Camera
        import omni.replicator.core as rep
        
        _isaac_modules = {
            "prim_utils": prim_utils,
            "World": World,
            "Robot": Robot,
            "XFormPrim": XFormPrim,
            "RigidPrim": RigidPrim,
            "add_reference_to_stage": add_reference_to_stage,
            "Camera": Camera,
            "rep": rep,
        }
    except ImportError as e:
        print(f"Warning: Could not import Isaac Sim modules: {e}")
        _isaac_modules = {}
    
    return _isaac_modules


@dataclass
class IsaacConfig:
    """Configuration for Isaac Sim environment."""
    headless: bool = True
    resolution: Tuple[int, int] = (512, 512)
    physics_dt: float = 1.0 / 60.0  # Physics timestep
    rendering_dt: float = 1.0 / 20.0  # Render/control timestep
    num_agents: int = 4
    scene_path: Optional[str] = None  # Path to USD scene file
    robot_usd: Optional[str] = None  # Path to robot USD
    

@dataclass
class AgentData:
    """Data for a single agent in the simulation."""
    agent_id: int
    prim_path: str
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))  # quaternion
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    agent_type: str = "pedestrian"
    target_position: Optional[np.ndarray] = None


class IsaacBridge:
    """
    Bridge to NVIDIA Isaac Sim for realistic robot simulation.
    
    Handles:
    - Scene loading and management
    - Robot control and observation
    - Agent (pedestrian) spawning and motion
    - Camera capture
    - Snapshot/restore for counterfactuals
    
    Example:
        >>> bridge = IsaacBridge(headless=True)
        >>> bridge.initialize()
        >>> frame, metadata = bridge.reset(scenario="sidewalk_delivery", seed=42)
        >>> frame, reward, done, info = bridge.step(linear_vel=0.5, angular_vel=0.1)
        >>> bridge.close()
    """
    
    def __init__(self, config: Optional[IsaacConfig] = None):
        """Initialize the Isaac bridge.
        
        Args:
            config: Isaac configuration. If None, uses defaults.
        """
        self.config = config or IsaacConfig()
        self._app: Optional[Any] = None
        self._world: Optional[Any] = None
        self._robot: Optional[Any] = None
        self._camera: Optional[Any] = None
        self._agents: List[AgentData] = []
        self._goal_position: np.ndarray = np.array([0.0, 0.0, 0.0])
        self._step_idx: int = 0
        self._rng = random.Random()
        self._initialized = False
        
        # Fallback mode when Isaac Sim is not available
        self._fallback_mode = not ISAAC_AVAILABLE
        
    def initialize(self) -> bool:
        """Initialize Isaac Sim application and world.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True
            
        if self._fallback_mode:
            print("Isaac Sim not available, using fallback rendering mode")
            self._initialized = True
            return True
        
        try:
            # Start Isaac Sim application
            self._app = SimulationApp({
                "headless": self.config.headless,
                "width": self.config.resolution[0],
                "height": self.config.resolution[1],
            })
            
            # Import Isaac modules after app starts
            modules = _lazy_import_isaac()
            if not modules:
                print("Failed to import Isaac modules, using fallback mode")
                self._fallback_mode = True
                self._initialized = True
                return True
            
            # Create world
            World = modules["World"]
            self._world = World(
                physics_dt=self.config.physics_dt,
                rendering_dt=self.config.rendering_dt,
                stage_units_in_meters=1.0,
            )
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize Isaac Sim: {e}")
            print("Using fallback rendering mode")
            self._fallback_mode = True
            self._initialized = True
            return True
    
    def _load_scene(self, scenario: str) -> None:
        """Load a scene based on scenario name."""
        if self._fallback_mode:
            return
            
        modules = _lazy_import_isaac()
        if not modules:
            return
            
        # Try to load built-in Isaac assets
        # These paths may vary based on Isaac Sim installation
        scene_paths = {
            "sidewalk_delivery": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
            "crosswalk_occlusion": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
            "warehouse_aisles": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
        }
        
        scene_path = self.config.scene_path or scene_paths.get(scenario)
        
        if scene_path:
            try:
                add_ref = modules["add_reference_to_stage"]
                add_ref(usd_path=scene_path, prim_path="/World/Scene")
            except Exception as e:
                print(f"Could not load scene {scene_path}: {e}")
                # Create a simple ground plane instead
                self._create_simple_scene()
        else:
            self._create_simple_scene()
    
    def _create_simple_scene(self) -> None:
        """Create a simple scene when USD files are not available."""
        if self._fallback_mode or not self._world:
            return
            
        try:
            from omni.isaac.core.objects import GroundPlane
            GroundPlane(prim_path="/World/GroundPlane", size=50, color=np.array([0.5, 0.5, 0.5]))
        except Exception as e:
            print(f"Could not create ground plane: {e}")
    
    def _spawn_robot(self, position: np.ndarray, heading: float) -> None:
        """Spawn the ego robot at given position."""
        if self._fallback_mode:
            return
            
        modules = _lazy_import_isaac()
        if not modules or not self._world:
            return
        
        try:
            # Try to use a built-in robot or simple prim
            XFormPrim = modules["XFormPrim"]
            
            # Create a simple robot representation
            self._robot = XFormPrim(
                prim_path="/World/Robot",
                name="ego_robot",
                position=position,
                orientation=self._euler_to_quat(0, 0, heading),
            )
            
            # Add to world
            self._world.scene.add(self._robot)
            
        except Exception as e:
            print(f"Could not spawn robot: {e}")
    
    def _setup_camera(self) -> None:
        """Set up the robot-mounted camera."""
        if self._fallback_mode:
            return
            
        modules = _lazy_import_isaac()
        if not modules or not self._world:
            return
            
        try:
            Camera = modules["Camera"]
            
            self._camera = Camera(
                prim_path="/World/Robot/Camera",
                resolution=self.config.resolution,
                frequency=20,
            )
            self._camera.initialize()
            
        except Exception as e:
            print(f"Could not setup camera: {e}")
    
    def _spawn_agents(self, num_agents: int, spawn_region: Tuple[float, float, float, float]) -> None:
        """Spawn pedestrian/agent prims."""
        if self._fallback_mode:
            # Create fallback agent data
            self._agents = []
            xmin, xmax, ymin, ymax = spawn_region
            for i in range(num_agents):
                pos = np.array([
                    self._rng.uniform(xmin, xmax),
                    self._rng.uniform(ymin, ymax),
                    0.0
                ])
                vel = np.array([
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.5, 0.5),
                ])
                self._agents.append(AgentData(
                    agent_id=i,
                    prim_path=f"/World/Agent_{i}",
                    position=pos,
                    velocity=vel,
                ))
            return
            
        modules = _lazy_import_isaac()
        if not modules or not self._world:
            return
            
        try:
            XFormPrim = modules["XFormPrim"]
            xmin, xmax, ymin, ymax = spawn_region
            
            self._agents = []
            for i in range(num_agents):
                pos = np.array([
                    self._rng.uniform(xmin, xmax),
                    self._rng.uniform(ymin, ymax),
                    0.0
                ])
                
                agent_prim = XFormPrim(
                    prim_path=f"/World/Agent_{i}",
                    name=f"agent_{i}",
                    position=pos,
                )
                self._world.scene.add(agent_prim)
                
                self._agents.append(AgentData(
                    agent_id=i,
                    prim_path=f"/World/Agent_{i}",
                    position=pos,
                    velocity=np.array([
                        self._rng.uniform(-0.5, 0.5),
                        self._rng.uniform(-0.5, 0.5),
                    ]),
                ))
                
        except Exception as e:
            print(f"Could not spawn agents: {e}")
    
    def reset(
        self,
        scenario: str = "sidewalk_delivery",
        seed: Optional[int] = None,
        num_agents: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the simulation to initial state.
        
        Args:
            scenario: Scenario name
            seed: Random seed
            num_agents: Number of agents to spawn
            
        Returns:
            Tuple of (rgb_frame, metadata)
        """
        if not self._initialized:
            self.initialize()
        
        # Set seed
        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)
        
        self._step_idx = 0
        n_agents = num_agents if num_agents is not None else self.config.num_agents
        
        if self._fallback_mode:
            return self._fallback_reset(scenario, n_agents)
        
        # Clear previous scene
        if self._world:
            self._world.reset()
        
        # Load scene and spawn entities
        self._load_scene(scenario)
        
        # Spawn robot at scenario-specific start position
        from .scenarios import get_scenario_config
        config = get_scenario_config(scenario)
        
        start_pos = np.array(config["start_position"])
        start_heading = config["start_heading"]
        self._goal_position = np.array(config["goal_position"])
        
        self._spawn_robot(start_pos, start_heading)
        self._setup_camera()
        self._spawn_agents(n_agents, config["agent_spawn_region"])
        
        # Step simulation once to settle
        if self._world:
            self._world.step(render=True)
        
        # Capture initial frame
        frame = self._capture_frame()
        
        metadata = {
            "scenario": scenario,
            "num_agents": n_agents,
            "ego_pose": tuple(start_pos) + (0, 0, start_heading),
            "goal_pose": tuple(self._goal_position),
            "step_idx": 0,
        }
        
        return frame, metadata
    
    def _fallback_reset(self, scenario: str, num_agents: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fallback reset when Isaac Sim is not available."""
        from .scenarios import get_scenario_config
        config = get_scenario_config(scenario)
        
        # Initialize positions
        self._ego_position = np.array(config["start_position"])
        self._ego_heading = config["start_heading"]
        self._goal_position = np.array(config["goal_position"])
        
        # Spawn agents
        self._spawn_agents(num_agents, config["agent_spawn_region"])
        
        # Generate fallback frame
        frame = self._render_fallback_frame()
        
        metadata = {
            "scenario": scenario,
            "num_agents": num_agents,
            "ego_pose": tuple(self._ego_position) + (0, 0, self._ego_heading),
            "goal_pose": tuple(self._goal_position),
            "step_idx": 0,
        }
        
        return frame, metadata
    
    def step(
        self,
        linear_vel: float,
        angular_vel: float,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one simulation step.
        
        Args:
            linear_vel: Linear velocity (-1 to 1)
            angular_vel: Angular velocity (-1 to 1)
            
        Returns:
            Tuple of (rgb_frame, reward, done, info)
        """
        self._step_idx += 1
        
        if self._fallback_mode:
            return self._fallback_step(linear_vel, angular_vel)
        
        # Apply velocities to robot
        self._apply_robot_control(linear_vel, angular_vel)
        
        # Update agent positions
        self._update_agents()
        
        # Step physics
        if self._world:
            self._world.step(render=True)
        
        # Capture frame
        frame = self._capture_frame()
        
        # Compute reward and check done
        reward, done, info = self._compute_reward_done()
        
        return frame, reward, done, info
    
    def _fallback_step(
        self,
        linear_vel: float,
        angular_vel: float,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Fallback step when Isaac Sim is not available."""
        dt = self.config.rendering_dt
        
        # Update ego position (simple kinematics)
        self._ego_heading += angular_vel * dt * 2.0
        dx = math.cos(self._ego_heading) * linear_vel * dt * 2.0
        dy = math.sin(self._ego_heading) * linear_vel * dt * 2.0
        self._ego_position[0] += dx
        self._ego_position[1] += dy
        
        # Update agents
        for agent in self._agents:
            # Simple random walk
            agent.velocity[0] += self._rng.gauss(0, 0.1)
            agent.velocity[1] += self._rng.gauss(0, 0.1)
            # Clamp velocity
            speed = np.linalg.norm(agent.velocity)
            if speed > 1.0:
                agent.velocity = agent.velocity / speed
            
            agent.position[0] += agent.velocity[0] * dt
            agent.position[1] += agent.velocity[1] * dt
            
            # Bounce off bounds
            for i in range(2):
                if agent.position[i] < -10 or agent.position[i] > 10:
                    agent.velocity[i] *= -1
                    agent.position[i] = np.clip(agent.position[i], -10, 10)
        
        # Render fallback frame
        frame = self._render_fallback_frame()
        
        # Compute reward
        reward, done, info = self._compute_reward_done()
        
        return frame, reward, done, info
    
    def _render_fallback_frame(self) -> np.ndarray:
        """Render a synthetic frame when Isaac Sim is not available.
        
        Creates a simple 3D-like visualization using numpy/pillow.
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            # Return random noise if PIL not available
            return np.random.randint(0, 255, (self.config.resolution[0], self.config.resolution[1], 3), dtype=np.uint8)
        
        h, w = self.config.resolution  # (height, width)
        
        # Create base image with gradient sky
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        
        # Sky gradient (top half)
        for y in range(h // 2):
            ratio = y / (h // 2)
            r = int(135 + ratio * 70)  # Light blue to white
            g = int(206 + ratio * 49)
            b = int(235 + ratio * 20)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        
        # Ground (bottom half) - perspective grid
        horizon = h // 2
        for y in range(horizon, h):
            depth = (y - horizon) / (h - horizon)
            gray = int(100 + depth * 50)
            draw.line([(0, y), (w, y)], fill=(gray, gray + 10, gray))
        
        # Draw grid lines for perspective
        for i in range(-5, 6):
            # Vanishing point at center horizon
            x_far = w // 2 + i * 20
            x_near = w // 2 + i * 100
            draw.line([(x_far, horizon), (x_near, h)], fill=(80, 80, 80), width=1)
        
        # Draw agents as colored circles
        for agent in self._agents:
            # Project 3D position to 2D (simple orthographic)
            rel_x = agent.position[0] - self._ego_position[0]
            rel_y = agent.position[1] - self._ego_position[1]
            
            # Rotate by ego heading
            cos_h = math.cos(-self._ego_heading)
            sin_h = math.sin(-self._ego_heading)
            local_x = rel_x * cos_h - rel_y * sin_h
            local_y = rel_x * sin_h + rel_y * cos_h
            
            # Only draw if in front
            if local_y > 0.5:
                # Perspective projection
                screen_x = w // 2 + int(local_x / local_y * w * 0.3)
                screen_y = horizon + int((1.0 / local_y) * h * 0.3)
                size = max(3, int(20 / local_y))
                
                if 0 < screen_x < w and horizon < screen_y < h:
                    # Orange for pedestrians
                    draw.ellipse([screen_x - size, screen_y - size, 
                                 screen_x + size, screen_y + size],
                                fill=(255, 150, 50))
        
        # Draw goal marker
        rel_x = self._goal_position[0] - self._ego_position[0]
        rel_y = self._goal_position[1] - self._ego_position[1]
        cos_h = math.cos(-self._ego_heading)
        sin_h = math.sin(-self._ego_heading)
        local_x = rel_x * cos_h - rel_y * sin_h
        local_y = rel_x * sin_h + rel_y * cos_h
        
        if local_y > 0.5:
            screen_x = w // 2 + int(local_x / local_y * w * 0.3)
            screen_y = horizon + int((1.0 / local_y) * h * 0.3)
            size = max(5, int(30 / local_y))
            
            if 0 < screen_x < w and horizon < screen_y < h:
                # Green goal marker
                draw.ellipse([screen_x - size, screen_y - size,
                             screen_x + size, screen_y + size],
                            fill=(50, 255, 50), outline=(0, 200, 0))
        
        # Add some text overlay
        try:
            draw.text((10, 10), f"Step: {self._step_idx}", fill=(255, 255, 255))
            goal_dist = np.linalg.norm(self._goal_position[:2] - self._ego_position[:2])
            draw.text((10, 30), f"Goal dist: {goal_dist:.1f}m", fill=(255, 255, 255))
        except:
            pass
        
        return np.array(img)
    
    def _capture_frame(self) -> np.ndarray:
        """Capture RGB frame from camera."""
        if self._fallback_mode:
            return self._render_fallback_frame()
            
        if self._camera:
            try:
                frame = self._camera.get_rgba()
                if frame is not None:
                    return frame[:, :, :3]  # RGB only
            except Exception as e:
                print(f"Camera capture failed: {e}")
        
        # Return placeholder frame
        return np.zeros((*self.config.resolution, 3), dtype=np.uint8)
    
    def _apply_robot_control(self, linear_vel: float, angular_vel: float) -> None:
        """Apply velocity commands to robot."""
        if self._fallback_mode or not self._robot:
            return
            
        # Scale velocities
        lin_speed = linear_vel * 2.0  # m/s
        ang_speed = angular_vel * 1.0  # rad/s
        
        try:
            # Get current pose
            pos, quat = self._robot.get_world_pose()
            
            # Update heading
            _, _, yaw = self._quat_to_euler(quat)
            yaw += ang_speed * self.config.rendering_dt
            
            # Update position
            dx = math.cos(yaw) * lin_speed * self.config.rendering_dt
            dy = math.sin(yaw) * lin_speed * self.config.rendering_dt
            pos[0] += dx
            pos[1] += dy
            
            # Set new pose
            new_quat = self._euler_to_quat(0, 0, yaw)
            self._robot.set_world_pose(pos, new_quat)
            
        except Exception as e:
            print(f"Robot control failed: {e}")
    
    def _update_agents(self) -> None:
        """Update agent positions with simple motion."""
        if self._fallback_mode:
            return
            
        modules = _lazy_import_isaac()
        if not modules:
            return
            
        dt = self.config.rendering_dt
        
        for agent in self._agents:
            # Update velocity with noise
            agent.velocity[0] += self._rng.gauss(0, 0.1)
            agent.velocity[1] += self._rng.gauss(0, 0.1)
            
            # Clamp velocity
            speed = np.linalg.norm(agent.velocity)
            if speed > 1.0:
                agent.velocity = agent.velocity / speed
            
            # Update position
            agent.position[0] += agent.velocity[0] * dt
            agent.position[1] += agent.velocity[1] * dt
            
            # Try to update prim position
            try:
                XFormPrim = modules["XFormPrim"]
                prim = self._world.scene.get_object(f"agent_{agent.agent_id}")
                if prim:
                    prim.set_world_pose(agent.position)
            except:
                pass
    
    def _compute_reward_done(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Compute reward and check termination conditions."""
        info = {
            "step_idx": self._step_idx,
            "collision": False,
            "near_miss": False,
            "goal_reached": False,
        }
        
        # Get ego position
        if self._fallback_mode:
            ego_pos = self._ego_position[:2]
        elif self._robot:
            try:
                pos, _ = self._robot.get_world_pose()
                ego_pos = pos[:2]
            except:
                ego_pos = np.array([0, 0])
        else:
            ego_pos = np.array([0, 0])
        
        goal_pos = self._goal_position[:2]
        
        # Distance to goal
        goal_dist = np.linalg.norm(goal_pos - ego_pos)
        info["goal_distance"] = float(goal_dist)
        
        # Check goal reached
        if goal_dist < 1.0:
            info["goal_reached"] = True
            return 10.0, True, info
        
        # Check collisions with agents
        for agent in self._agents:
            agent_pos = agent.position[:2]
            dist = np.linalg.norm(agent_pos - ego_pos)
            
            if dist < 0.5:  # Collision threshold
                info["collision"] = True
                return -5.0, True, info
            elif dist < 1.5:  # Near miss threshold
                info["near_miss"] = True
        
        # Progress reward
        reward = -0.01  # Time penalty
        reward += 0.1 * max(0, 10 - goal_dist) / 10  # Progress reward
        
        if info["near_miss"]:
            reward -= 0.1
        
        # Check max steps
        done = self._step_idx >= 500
        
        return reward, done, info
    
    def snapshot(self) -> Dict[str, Any]:
        """Capture current state for counterfactual planning."""
        state = {
            "step_idx": self._step_idx,
            "rng_state": self._rng.getstate(),
            "goal_position": self._goal_position.tolist(),
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "position": a.position.tolist(),
                    "velocity": a.velocity.tolist(),
                }
                for a in self._agents
            ],
        }
        
        if self._fallback_mode:
            state["ego_position"] = self._ego_position.tolist()
            state["ego_heading"] = self._ego_heading
        elif self._robot:
            try:
                pos, quat = self._robot.get_world_pose()
                state["ego_position"] = pos.tolist()
                state["ego_quat"] = quat.tolist()
            except:
                pass
        
        return state
    
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        self._step_idx = snapshot.get("step_idx", 0)
        self._rng.setstate(snapshot.get("rng_state", self._rng.getstate()))
        self._goal_position = np.array(snapshot.get("goal_position", [0, 0, 0]))
        
        # Restore agents
        agent_data = snapshot.get("agents", [])
        for i, data in enumerate(agent_data):
            if i < len(self._agents):
                self._agents[i].position = np.array(data["position"])
                self._agents[i].velocity = np.array(data["velocity"])
        
        if self._fallback_mode:
            self._ego_position = np.array(snapshot.get("ego_position", [0, 0, 0]))
            self._ego_heading = snapshot.get("ego_heading", 0)
        elif self._robot:
            try:
                pos = np.array(snapshot.get("ego_position", [0, 0, 0]))
                quat = np.array(snapshot.get("ego_quat", [0, 0, 0, 1]))
                self._robot.set_world_pose(pos, quat)
            except:
                pass
    
    def get_agent_states(self) -> List[Dict[str, Any]]:
        """Get current state of all agents."""
        return [
            {
                "agent_id": a.agent_id,
                "x": float(a.position[0]),
                "y": float(a.position[1]),
                "z": float(a.position[2]) if len(a.position) > 2 else 0.0,
                "vx": float(a.velocity[0]),
                "vy": float(a.velocity[1]),
                "agent_type": a.agent_type,
            }
            for a in self._agents
        ]
    
    def get_ego_pose(self) -> Tuple[float, ...]:
        """Get ego robot pose."""
        if self._fallback_mode:
            return tuple(self._ego_position) + (0, 0, self._ego_heading)
        elif self._robot:
            try:
                pos, quat = self._robot.get_world_pose()
                roll, pitch, yaw = self._quat_to_euler(quat)
                return tuple(pos) + (roll, pitch, yaw)
            except:
                pass
        return (0, 0, 0, 0, 0, 0)
    
    def close(self) -> None:
        """Clean up Isaac Sim resources."""
        if self._world:
            self._world.stop()
        if self._app:
            self._app.close()
        self._initialized = False
    
    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        return np.array([
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ])
    
    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles."""
        x, y, z, w = quat
        
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
