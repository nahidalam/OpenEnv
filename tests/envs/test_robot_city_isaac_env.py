# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Robot City Isaac Environment.

Tests skip if Isaac Sim is not available.
Also tests fallback mode which works without Isaac Sim.
"""

import base64
import pytest

# Check dependencies
NUMPY_AVAILABLE = False
PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass


# Skip all tests if numpy or PIL not available
pytestmark = pytest.mark.skipif(
    not NUMPY_AVAILABLE or not PIL_AVAILABLE,
    reason="numpy and PIL required for robot_city_isaac_env"
)


class TestIsaacBridgeFallback:
    """Test Isaac bridge in fallback mode (no Isaac Sim required)."""
    
    def test_bridge_initializes_fallback(self):
        """Bridge should initialize in fallback mode."""
        from envs.robot_city_isaac_env.server.isaac_bridge import IsaacBridge, IsaacConfig
        
        config = IsaacConfig(headless=True, resolution=(256, 256))
        bridge = IsaacBridge(config)
        
        assert bridge.initialize()
        assert bridge._fallback_mode  # Should be in fallback mode if Isaac not available
        
        bridge.close()
    
    def test_bridge_reset_returns_frame(self):
        """Reset should return valid frame and metadata."""
        from envs.robot_city_isaac_env.server.isaac_bridge import IsaacBridge, IsaacConfig
        
        config = IsaacConfig(headless=True, resolution=(256, 256))
        bridge = IsaacBridge(config)
        bridge.initialize()
        
        frame, metadata = bridge.reset(scenario="sidewalk_delivery", seed=42, num_agents=4)
        
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8
        assert "scenario" in metadata
        assert "ego_pose" in metadata
        
        bridge.close()
    
    def test_bridge_step_returns_results(self):
        """Step should return frame, reward, done, info."""
        from envs.robot_city_isaac_env.server.isaac_bridge import IsaacBridge, IsaacConfig
        
        config = IsaacConfig(headless=True, resolution=(256, 256))
        bridge = IsaacBridge(config)
        bridge.initialize()
        bridge.reset(scenario="sidewalk_delivery", seed=42)
        
        frame, reward, done, info = bridge.step(linear_vel=0.5, angular_vel=0.0)
        
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (256, 256, 3)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        bridge.close()
    
    def test_bridge_snapshot_restore(self):
        """Snapshot/restore should preserve state."""
        from envs.robot_city_isaac_env.server.isaac_bridge import IsaacBridge, IsaacConfig
        
        config = IsaacConfig(headless=True, resolution=(128, 128))
        bridge = IsaacBridge(config)
        bridge.initialize()
        bridge.reset(scenario="sidewalk_delivery", seed=42)
        
        # Take snapshot
        snapshot = bridge.snapshot()
        original_step = bridge._step_idx
        
        # Step a few times
        for _ in range(5):
            bridge.step(linear_vel=0.5, angular_vel=0.1)
        
        assert bridge._step_idx != original_step
        
        # Restore
        bridge.restore(snapshot)
        
        assert bridge._step_idx == original_step
        
        bridge.close()


class TestRobotCityIsaacEnvironment:
    """Test the main environment class."""
    
    def test_env_reset_returns_observation(self):
        """Reset should return valid observation."""
        from envs.robot_city_isaac_env.server.robot_city_isaac_env import RobotCityIsaacEnvironment
        from envs.robot_city_isaac_env.models import RobotCityIsaacObs
        
        env = RobotCityIsaacEnvironment(headless=True, resolution=128)
        obs = env.reset(scenario="sidewalk_delivery", num_agents=3, seed=42)
        
        assert isinstance(obs, RobotCityIsaacObs)
        assert obs.rgb is not None
        assert len(obs.rgb) > 0
        assert obs.step_idx == 0
        
        env.close()
    
    def test_env_step_returns_observation(self):
        """Step should return valid observation."""
        from envs.robot_city_isaac_env.server.robot_city_isaac_env import RobotCityIsaacEnvironment
        from envs.robot_city_isaac_env.models import RobotCityIsaacAction, RobotCityIsaacObs
        
        env = RobotCityIsaacEnvironment(headless=True, resolution=128)
        env.reset(scenario="sidewalk_delivery", seed=42)
        
        action = RobotCityIsaacAction(linear_vel=0.5, angular_vel=0.0)
        obs = env.step(action)
        
        assert isinstance(obs, RobotCityIsaacObs)
        assert obs.rgb is not None
        assert isinstance(obs.reward, float)
        assert isinstance(obs.done, bool)
        assert obs.step_idx == 1
        
        env.close()
    
    def test_env_rgb_is_valid_base64_png(self):
        """RGB observation should be valid base64 PNG."""
        from envs.robot_city_isaac_env.server.robot_city_isaac_env import RobotCityIsaacEnvironment
        
        env = RobotCityIsaacEnvironment(headless=True, resolution=128)
        obs = env.reset(scenario="sidewalk_delivery", seed=42)
        
        # Decode base64
        img_bytes = base64.b64decode(obs.rgb)
        
        # Check PNG header
        assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n', "Should be valid PNG"
        
        # Load with PIL
        import io
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size == (128, 128)
        
        env.close()
    
    def test_camera_dropout_produces_black_frame(self):
        """Camera dropout should produce black frames."""
        from envs.robot_city_isaac_env.server.robot_city_isaac_env import RobotCityIsaacEnvironment
        from envs.robot_city_isaac_env.models import RobotCityIsaacAction
        
        env = RobotCityIsaacEnvironment(headless=True, resolution=64)
        env.reset(scenario="sidewalk_delivery", seed=42, camera_dropout=1.0)  # Always dropout
        
        action = RobotCityIsaacAction(linear_vel=0.5, angular_vel=0.0)
        obs = env.step(action)
        
        # Should have dropout event
        assert obs.events.get("dropout_applied", False), "Dropout should be applied"
        
        # Decode frame and check if mostly black
        import io
        img_bytes = base64.b64decode(obs.rgb)
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        
        # Black frame should have very low mean value
        assert arr.mean() < 10, "Dropout frame should be mostly black"
        
        env.close()
    
    def test_action_delay_repeats_previous(self):
        """Action delay should apply previous action."""
        from envs.robot_city_isaac_env.server.robot_city_isaac_env import RobotCityIsaacEnvironment
        from envs.robot_city_isaac_env.models import RobotCityIsaacAction
        
        env = RobotCityIsaacEnvironment(headless=True, resolution=64)
        env.reset(scenario="sidewalk_delivery", seed=42, action_delay=1.0)  # Always delay
        
        # First action
        action1 = RobotCityIsaacAction(linear_vel=0.5, angular_vel=0.0)
        obs1 = env.step(action1)
        
        # Second action (should be delayed)
        action2 = RobotCityIsaacAction(linear_vel=0.0, angular_vel=1.0)
        obs2 = env.step(action2)
        
        # Should have delay event
        assert obs2.events.get("delay_applied", False), "Delay should be applied"
        
        env.close()


class TestScenarios:
    """Test scenario configurations."""
    
    def test_all_scenarios_have_required_fields(self):
        """All scenarios should have required configuration fields."""
        from envs.robot_city_isaac_env.server.scenarios import SCENARIOS, get_scenario_config
        
        required_fields = [
            "start_position",
            "goal_position",
            "agent_spawn_region",
            "max_steps",
        ]
        
        for scenario_name in SCENARIOS:
            config = get_scenario_config(scenario_name)
            for field in required_fields:
                assert field in config, f"Scenario {scenario_name} missing field: {field}"
    
    def test_list_scenarios(self):
        """Should list available scenarios."""
        from envs.robot_city_isaac_env.server.scenarios import list_scenarios
        
        scenarios = list_scenarios()
        assert len(scenarios) >= 3
        assert "sidewalk_delivery" in scenarios
        assert "crosswalk_occlusion" in scenarios
        assert "warehouse_aisles" in scenarios


class TestRendererOverlays:
    """Test overlay rendering functions."""
    
    def test_draw_legend(self):
        """Legend drawing should not crash."""
        from envs.robot_city_isaac_env.server.renderer_overlays import draw_legend
        
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        result = draw_legend(frame)
        
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
    
    def test_draw_all_overlays(self):
        """Full overlay drawing should not crash."""
        from envs.robot_city_isaac_env.server.renderer_overlays import draw_all_overlays
        
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        ego_pose = (0, 0, 0, 0, 0, 0)
        goal_pose = (5, 5, 0)
        agents = [
            {"agent_id": 0, "x": 2, "y": 3, "vx": 0.5, "vy": 0.0},
            {"agent_id": 1, "x": -1, "y": 2, "vx": -0.3, "vy": 0.2},
        ]
        
        result = draw_all_overlays(
            frame=frame,
            ego_pose=ego_pose,
            goal_pose=goal_pose,
            agents=agents,
            events={},
        )
        
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


class TestDeterminism:
    """Test deterministic behavior."""
    
    def test_same_seed_same_observations(self):
        """Same seed should produce same observations."""
        from envs.robot_city_isaac_env.server.robot_city_isaac_env import RobotCityIsaacEnvironment
        from envs.robot_city_isaac_env.models import RobotCityIsaacAction
        import hashlib
        
        def run_episode(seed: int, steps: int = 3):
            env = RobotCityIsaacEnvironment(headless=True, resolution=64)
            obs = env.reset(scenario="sidewalk_delivery", seed=seed)
            hashes = [hashlib.md5(obs.rgb.encode()).hexdigest()]
            
            for _ in range(steps):
                action = RobotCityIsaacAction(linear_vel=0.5, angular_vel=0.0)
                obs = env.step(action)
                hashes.append(hashlib.md5(obs.rgb.encode()).hexdigest())
            
            env.close()
            return hashes
        
        hashes1 = run_episode(seed=42)
        hashes2 = run_episode(seed=42)
        
        assert hashes1 == hashes2, "Same seed should produce same observations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
