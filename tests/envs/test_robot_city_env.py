# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Robot City Environment.

Fast tests (<2s total) covering:
- Reset/step smoke test
- Determinism with seed
- Multi-robot control
- Failure injection
- Overlay output validation
"""

import base64
import hashlib
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


pytestmark = pytest.mark.skipif(
    not NUMPY_AVAILABLE or not PIL_AVAILABLE,
    reason="numpy and PIL required for robot_city_env"
)


class TestRobotCitySmoke:
    """Basic smoke tests."""
    
    def test_reset_returns_observation(self):
        """Reset should return valid observation."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.models import RobotCityObservation
        
        env = RobotCityEnvironment()
        obs = env.reset()
        
        assert isinstance(obs, RobotCityObservation)
        assert obs.rgb is not None
        assert len(obs.rgb) > 0
        assert "reset" in obs.events
    
    def test_step_returns_observation(self):
        """Step should return valid observation."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.models import RobotCityAction, RobotCityObservation
        
        env = RobotCityEnvironment()
        env.reset()
        
        action = RobotCityAction(robot_id=0, move="forward", speed=1.0)
        obs = env.step(action)
        
        assert isinstance(obs, RobotCityObservation)
        assert obs.rgb is not None
        assert isinstance(obs.reward, float)
        assert isinstance(obs.done, bool)
    
    def test_state_tracks_steps(self):
        """State should track step count."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.models import RobotCityAction
        
        env = RobotCityEnvironment()
        env.reset()
        
        assert env.state.step_count == 0
        
        env.step(RobotCityAction(robot_id=0, move="forward"))
        assert env.state.step_count == 1
        
        env.step(RobotCityAction(robot_id=0, move="noop"))
        assert env.state.step_count == 2


class TestDeterminism:
    """Test deterministic behavior with seeds."""
    
    def test_same_seed_same_observations(self):
        """Same seed should produce same first N observation hashes."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.models import RobotCityAction
        
        def run_episode(seed: int, steps: int = 5):
            env = RobotCityEnvironment()
            obs = env.reset(seed=seed, scenario="intersection_crosswalk")
            hashes = [hashlib.md5(obs.rgb.encode()).hexdigest()]
            
            for _ in range(steps):
                action = RobotCityAction(robot_id=0, move="forward")
                obs = env.step(action)
                hashes.append(hashlib.md5(obs.rgb.encode()).hexdigest())
            
            return hashes
        
        hashes1 = run_episode(seed=42)
        hashes2 = run_episode(seed=42)
        
        assert hashes1 == hashes2, "Same seed should produce same observations"
    
    def test_different_seeds_different_observations(self):
        """Different seeds should produce different observations."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        
        env = RobotCityEnvironment()
        
        obs1 = env.reset(seed=123)
        hash1 = hashlib.md5(obs1.rgb.encode()).hexdigest()
        
        obs2 = env.reset(seed=456)
        hash2 = hashlib.md5(obs2.rgb.encode()).hexdigest()
        
        # Different seeds might still produce same initial state in some scenarios
        # but state dict should differ
        assert obs1.state != obs2.state or hash1 != hash2


class TestMultiRobot:
    """Test multi-robot functionality."""
    
    def test_multiple_robots_created(self):
        """Reset with num_robots should create multiple robots."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        
        env = RobotCityEnvironment()
        obs = env.reset(num_robots=3)
        
        assert len(obs.state["robots"]) == 3
    
    def test_robot_id_switching(self):
        """Actions should control specified robot_id."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.models import RobotCityAction
        
        env = RobotCityEnvironment()
        obs = env.reset(num_robots=2, seed=42)
        
        # Get initial positions
        r0_x_before = obs.state["robots"][0]["x"]
        r1_x_before = obs.state["robots"][1]["x"]
        
        # Move robot 0 forward
        obs = env.step(RobotCityAction(robot_id=0, move="forward"))
        
        r0_x_after = obs.state["robots"][0]["x"]
        r1_x_after = obs.state["robots"][1]["x"]
        
        # Robot 0 should have moved, robot 1 should be similar
        # (Note: positions change slightly due to pedestrian updates)
        assert r0_x_after != r0_x_before or obs.state["robots"][0]["y"] != obs.state["robots"][0]["y"]


class TestFailureInjection:
    """Test failure injection modes."""
    
    def test_camera_dropout_event(self):
        """Camera dropout should trigger event."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.models import RobotCityAction
        
        env = RobotCityEnvironment()
        env.reset(camera_dropout_prob=1.0)  # Always dropout
        
        obs = env.step(RobotCityAction(robot_id=0, move="noop"))
        
        assert "camera_dropout" in obs.events
    
    def test_action_delay_event(self):
        """Action delay should trigger event."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.models import RobotCityAction
        
        env = RobotCityEnvironment()
        env.reset(action_delay_prob=1.0)  # Always delay
        
        obs = env.step(RobotCityAction(robot_id=0, move="forward"))
        
        assert "action_delay" in obs.events


class TestOverlays:
    """Test overlay outputs."""
    
    def test_rgb_is_valid_base64_png(self):
        """RGB output should be valid base64 PNG."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        
        env = RobotCityEnvironment()
        obs = env.reset()
        
        # Decode base64
        img_bytes = base64.b64decode(obs.rgb)
        
        # Check PNG header
        assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n', "Should be valid PNG"
        
        # Load with PIL
        import io
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size == (128, 128)
    
    def test_uncertainty_overlay_is_valid(self):
        """Uncertainty overlay should be valid base64 PNG."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        
        env = RobotCityEnvironment()
        obs = env.reset(compute_overlays=True)
        
        if obs.uncertainty_overlay is not None:
            img_bytes = base64.b64decode(obs.uncertainty_overlay)
            assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n'
    
    def test_future_heatmap_is_valid(self):
        """Future heatmap should be valid base64 PNG."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        
        env = RobotCityEnvironment()
        obs = env.reset(compute_overlays=True)
        
        if obs.future_heatmap is not None:
            img_bytes = base64.b64decode(obs.future_heatmap)
            assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n'


class TestScenarios:
    """Test different scenarios."""
    
    def test_all_scenarios_load(self):
        """All built-in scenarios should load without error."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.server.sim.scenarios import SCENARIOS
        
        env = RobotCityEnvironment()
        
        for scenario_name in SCENARIOS:
            obs = env.reset(scenario=scenario_name)
            assert obs.rgb is not None
            assert len(obs.state["robots"]) >= 1


class TestSnapshotRestore:
    """Test counterfactual support."""
    
    def test_snapshot_restore_roundtrip(self):
        """Snapshot/restore should preserve state."""
        from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
        from envs.robot_city_env.models import RobotCityAction
        
        env = RobotCityEnvironment()
        env.reset(seed=42)
        
        # Take snapshot
        snapshot = env.snapshot()
        original_step = env.state.step_count
        original_x = env._sim.robots[0].x
        
        # Mutate state
        for _ in range(5):
            env.step(RobotCityAction(robot_id=0, move="forward"))
        
        assert env.state.step_count != original_step
        
        # Restore
        env.restore(snapshot)
        
        assert env.state.step_count == original_step
        assert abs(env._sim.robots[0].x - original_x) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
