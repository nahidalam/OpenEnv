"""Tests for openenv.counterfactual.serialize module.

Tests:
- summary with fake rgb numpy frame serializes to base64
- non-image arrays summarized safely
- json.dumps() works on outputs
"""

import json
import pytest

# Check if numpy is available
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
class TestSummaryToJson:
    """Test summary_to_json function."""
    
    def test_basic_summary_fields(self):
        """Should include basic fields: steps, horizon, done, total_reward, info."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.serialize import summary_to_json
        
        summary = TrajectorySummary(
            horizon=10,
            final_observation={"x": 1},
            steps=5,
            total_reward=12.5,
            done=True,
            info={"custom": "value"},
        )
        
        result = summary_to_json(summary)
        
        assert result["steps"] == 5
        assert result["horizon"] == 10
        assert result["done"] is True
        assert result["total_reward"] == 12.5
        assert result["info"] == {"custom": "value"}
        assert "final_observation" in result
    
    def test_rgb_frame_serializes_to_base64(self):
        """Summary with fake rgb numpy frame should serialize to base64."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.serialize import summary_to_json
        
        # Create fake RGB frame
        rgb_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb_frame[20:40, 20:40] = [255, 0, 0]  # Red square
        
        observations = [
            {"rgb": rgb_frame, "reward": 1.0},
            {"rgb": rgb_frame, "reward": 2.0},
        ]
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation=observations[-1],
            steps=2,
            observations=observations,
            total_reward=3.0,
            done=False,
        )
        
        result = summary_to_json(summary, include_observations=True)
        
        # Check observations were included
        assert "observations" in result
        assert len(result["observations"]) == 2
        
        # Check rgb was encoded to base64 string
        for obs in result["observations"]:
            assert isinstance(obs["rgb"], str)
            # Base64 strings contain only these characters
            assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" 
                      for c in obs["rgb"])
            # reward should be preserved
            assert isinstance(obs["reward"], float)
    
    def test_image_key_serializes_to_base64(self):
        """Frame under 'image' key should also serialize to base64."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.serialize import summary_to_json
        
        image_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"image": image_frame},
            steps=1,
            observations=[{"image": image_frame}],
            total_reward=1.0,
        )
        
        result = summary_to_json(summary, include_observations=True)
        
        # Check image was encoded
        assert isinstance(result["observations"][0]["image"], str)
        assert isinstance(result["final_observation"]["image"], str)
    
    def test_frame_key_serializes_to_base64(self):
        """Frame under 'frame' key should also serialize to base64."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.serialize import summary_to_json
        
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"frame": frame},
            steps=1,
            observations=[{"frame": frame}],
            total_reward=1.0,
        )
        
        result = summary_to_json(summary, include_observations=True)
        
        assert isinstance(result["observations"][0]["frame"], str)
    
    def test_non_image_arrays_summarized_safely(self):
        """Non-image numpy arrays should be summarized safely."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.serialize import summary_to_json
        
        # Large non-image array (1D, > 2048 elements)
        large_array = np.zeros(5000, dtype=np.float32)
        
        # Small array
        small_array = np.array([1, 2, 3, 4, 5])
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"large": large_array, "small": small_array},
            steps=1,
            total_reward=1.0,
        )
        
        result = summary_to_json(summary)
        
        # Large array should become summary dict
        final_obs = result["final_observation"]
        assert isinstance(final_obs["large"], dict)
        assert "shape" in final_obs["large"]
        assert "dtype" in final_obs["large"]
        
        # Small array should become list
        assert isinstance(final_obs["small"], list)
        assert final_obs["small"] == [1, 2, 3, 4, 5]
    
    def test_without_observations(self):
        """By default, observations should not be included."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.serialize import summary_to_json
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"x": 1},
            steps=3,
            observations=[{"x": 1}, {"x": 2}, {"x": 3}],
            total_reward=6.0,
        )
        
        result = summary_to_json(summary, include_observations=False)
        
        assert "observations" not in result
    
    def test_json_dumps_works(self):
        """json.dumps() should work on output."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.serialize import summary_to_json
        
        rgb_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"rgb": rgb_frame, "value": 42},
            steps=2,
            observations=[{"rgb": rgb_frame}],
            total_reward=10.0,
            done=True,
            info={"step_count": 2},
        )
        
        result = summary_to_json(summary, include_observations=True)
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["total_reward"] == 10.0


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
class TestCandidateResultsToJson:
    """Test candidate_results_to_json function."""
    
    def test_basic_conversion(self):
        """Should convert list of CandidateResult to JSON dicts."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.compare import CandidateResult
        from openenv.counterfactual.serialize import candidate_results_to_json
        
        summary1 = TrajectorySummary(
            horizon=5, final_observation={"x": 1}, steps=3, total_reward=10.0
        )
        summary2 = TrajectorySummary(
            horizon=5, final_observation={"x": 2}, steps=3, total_reward=5.0
        )
        
        results = [
            CandidateResult(
                candidate_id="c1",
                summary=summary1,
                score=10.0,
                action_seq=[{"delta": 1}],
            ),
            CandidateResult(
                candidate_id="c2",
                summary=summary2,
                score=5.0,
                action_seq=[{"delta": 2}],
            ),
        ]
        
        json_results = candidate_results_to_json(results)
        
        assert len(json_results) == 2
        assert json_results[0]["candidate_id"] == "c1"
        assert json_results[0]["score"] == 10.0
        assert "summary" in json_results[0]
        assert json_results[0]["summary"]["total_reward"] == 10.0
    
    def test_includes_action_representation(self):
        """Should include action/action_seq representation."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.compare import CandidateResult
        from openenv.counterfactual.serialize import candidate_results_to_json
        from dataclasses import dataclass
        
        @dataclass
        class TestAction:
            delta: int
        
        summary = TrajectorySummary(
            horizon=5, final_observation={"x": 1}, steps=2, total_reward=5.0
        )
        
        results = [
            CandidateResult(
                candidate_id="c1",
                summary=summary,
                score=5.0,
                action=TestAction(delta=3),
            ),
        ]
        
        json_results = candidate_results_to_json(results)
        
        assert "action" in json_results[0]
        assert json_results[0]["action"]["delta"] == 3
    
    def test_summary_excludes_observations(self):
        """Summary in candidate results should not include observations."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.compare import CandidateResult
        from openenv.counterfactual.serialize import candidate_results_to_json
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"x": 1},
            steps=3,
            observations=[{"x": 1}, {"x": 2}, {"x": 3}],
            total_reward=6.0,
        )
        
        results = [
            CandidateResult(
                candidate_id="c1",
                summary=summary,
                score=6.0,
                action_seq=[1, 2, 3],
            ),
        ]
        
        json_results = candidate_results_to_json(results)
        
        # Summary should not have observations
        assert "observations" not in json_results[0]["summary"]
    
    def test_json_dumps_works_on_candidate_results(self):
        """json.dumps() should work on candidate results output."""
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.compare import CandidateResult
        from openenv.counterfactual.serialize import candidate_results_to_json
        
        rgb_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"rgb": rgb_frame},
            steps=2,
            total_reward=10.0,
        )
        
        results = [
            CandidateResult(
                candidate_id="c1",
                summary=summary,
                score=10.0,
                action_seq=[{"move": "left"}],
            ),
        ]
        
        json_results = candidate_results_to_json(results)
        
        # Should not raise
        json_str = json.dumps(json_results)
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert len(parsed) == 1


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
def test_import_works():
    """Verify imports work."""
    from openenv.counterfactual import summary_to_json, candidate_results_to_json
    
    assert callable(summary_to_json)
    assert callable(candidate_results_to_json)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
