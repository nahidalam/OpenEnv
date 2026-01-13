"""Tests for openenv.counterfactual.logging module.

Tests:
- make_real_step_row with fake observation containing rgb frame
- make_counterfactual_row with CandidateResult
- rows_from_tree conversion
- Ensure all rows json.dumps() successfully
"""

import json
import pytest
from dataclasses import dataclass

# Check if numpy is available
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
class TestMakeRealStepRow:
    """Test make_real_step_row function."""
    
    def test_basic_row_fields(self):
        """Should include all basic fields."""
        from openenv.counterfactual.logging import make_real_step_row
        
        row = make_real_step_row(
            step_idx=5,
            observation={"x": 1, "y": 2},
            action={"move": "left"},
            reward=1.5,
            done=False,
        )
        
        assert row["type"] == "real_step"
        assert row["step_idx"] == 5
        assert row["reward"] == 1.5
        assert row["done"] is False
        assert "timestamp" in row
        assert "observation" in row
        assert "action" in row
    
    def test_rgb_frame_without_include_frame(self):
        """RGB frame should be summarized when include_frame=False."""
        from openenv.counterfactual.logging import make_real_step_row
        
        rgb_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        
        row = make_real_step_row(
            step_idx=0,
            observation={"rgb": rgb_frame, "score": 10},
            action="noop",
            reward=0.0,
            done=False,
            include_frame=False,
        )
        
        # RGB should be converted (either to base64 by safe_jsonify or summary)
        # The key is that it should be JSON-serializable
        json_str = json.dumps(row)
        assert isinstance(json_str, str)
    
    def test_rgb_frame_with_include_frame(self):
        """RGB frame should be base64 encoded when include_frame=True."""
        from openenv.counterfactual.logging import make_real_step_row
        
        rgb_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb_frame[20:40, 20:40] = [255, 0, 0]  # Red square
        
        row = make_real_step_row(
            step_idx=0,
            observation={"rgb": rgb_frame, "score": 10},
            action="noop",
            reward=0.0,
            done=False,
            include_frame=True,
        )
        
        # RGB should be base64 encoded
        assert isinstance(row["observation"]["rgb"], str)
        # Base64 check
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                  for c in row["observation"]["rgb"])
        
        # Should be JSON serializable
        json_str = json.dumps(row)
        assert isinstance(json_str, str)
    
    def test_dataclass_action(self):
        """Dataclass actions should be serialized."""
        from openenv.counterfactual.logging import make_real_step_row
        
        @dataclass
        class MoveAction:
            direction: str
            speed: float
        
        row = make_real_step_row(
            step_idx=0,
            observation={"x": 1},
            action=MoveAction(direction="north", speed=1.5),
            reward=1.0,
            done=False,
        )
        
        assert row["action"]["direction"] == "north"
        assert row["action"]["speed"] == 1.5
        
        # Should be JSON serializable
        json_str = json.dumps(row)
        assert isinstance(json_str, str)
    
    def test_none_reward_and_done(self):
        """Should handle None reward and done."""
        from openenv.counterfactual.logging import make_real_step_row
        
        row = make_real_step_row(
            step_idx=0,
            observation={"x": 1},
            action="noop",
            reward=None,
            done=None,
        )
        
        assert row["reward"] is None
        assert row["done"] is None
        
        # Should be JSON serializable
        json_str = json.dumps(row)
        assert isinstance(json_str, str)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
class TestMakeCounterfactualRow:
    """Test make_counterfactual_row function."""
    
    def test_basic_row_fields(self):
        """Should include all basic fields."""
        from openenv.counterfactual.logging import make_counterfactual_row
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.compare import CandidateResult
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"x": 10},
            steps=3,
            total_reward=15.0,
            done=True,
        )
        
        result = CandidateResult(
            candidate_id="c1",
            summary=summary,
            score=15.0,
            action_seq=[{"delta": 1}, {"delta": 2}],
        )
        
        row = make_counterfactual_row(
            step_idx=2,
            branch_id="branch_001",
            candidate=[{"delta": 1}, {"delta": 2}],
            result=result,
        )
        
        assert row["type"] == "counterfactual"
        assert row["step_idx"] == 2
        assert row["branch_id"] == "branch_001"
        assert row["candidate_id"] == "c1"
        assert row["score"] == 15.0
        assert "timestamp" in row
        assert "candidate" in row
        assert "summary" in row
    
    def test_with_rgb_frame_observation(self):
        """Should handle observation with RGB frame."""
        from openenv.counterfactual.logging import make_counterfactual_row
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.compare import CandidateResult
        
        rgb_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"rgb": rgb_frame, "value": 42},
            steps=2,
            total_reward=10.0,
        )
        
        result = CandidateResult(
            candidate_id="c1",
            summary=summary,
            score=10.0,
            action={"move": "up"},
        )
        
        row = make_counterfactual_row(
            step_idx=1,
            branch_id="b1",
            candidate={"move": "up"},
            result=result,
            include_frame=True,
        )
        
        # Should have final_observation with encoded frame
        assert "final_observation" in row
        assert isinstance(row["final_observation"]["rgb"], str)
        
        # Should be JSON serializable
        json_str = json.dumps(row)
        assert isinstance(json_str, str)
    
    def test_without_include_frame(self):
        """Should not include final_observation when include_frame=False."""
        from openenv.counterfactual.logging import make_counterfactual_row
        from openenv.counterfactual.simulate import TrajectorySummary
        from openenv.counterfactual.compare import CandidateResult
        
        summary = TrajectorySummary(
            horizon=5,
            final_observation={"x": 10},
            steps=2,
            total_reward=5.0,
        )
        
        result = CandidateResult(
            candidate_id="c1",
            summary=summary,
            score=5.0,
            action=1,
        )
        
        row = make_counterfactual_row(
            step_idx=1,
            branch_id="b1",
            candidate=1,
            result=result,
            include_frame=False,
        )
        
        # final_observation should not be present
        assert "final_observation" not in row
        
        # Should be JSON serializable
        json_str = json.dumps(row)
        assert isinstance(json_str, str)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
class TestRowsFromTree:
    """Test rows_from_tree function."""
    
    def test_basic_tree_conversion(self):
        """Should convert tree to list of row dicts."""
        from openenv.counterfactual.logging import rows_from_tree
        from openenv.counterfactual.tree import TrajectoryTree
        from openenv.counterfactual.simulate import TrajectorySummary
        
        tree = TrajectoryTree()
        
        # Add root
        root_summary = TrajectorySummary(
            horizon=5, final_observation={"x": 0}, steps=1, total_reward=0.0
        )
        root_id = tree.add_root(root_summary, snapshot_id="snap_0", kind="real")
        
        # Add children
        for i in range(2):
            child_summary = TrajectorySummary(
                horizon=5, final_observation={"x": i + 1}, steps=2, total_reward=float(i)
            )
            tree.add_child(
                parent_id=root_id,
                action=f"action_{i}",
                summary=child_summary,
                kind="counterfactual",
            )
        
        rows = rows_from_tree(tree)
        
        assert len(rows) == 3  # 1 root + 2 children
        
        # All rows should have required fields
        for row in rows:
            assert row["type"] == "tree_node"
            assert "node_id" in row
            assert "parent_id" in row
            assert "kind" in row
            assert "depth" in row
            assert "timestamp" in row
    
    def test_rows_are_json_serializable(self):
        """All rows should be JSON serializable."""
        from openenv.counterfactual.logging import rows_from_tree
        from openenv.counterfactual.tree import TrajectoryTree
        from openenv.counterfactual.simulate import TrajectorySummary
        
        tree = TrajectoryTree()
        
        summary = TrajectorySummary(
            horizon=5, final_observation={"x": 0}, steps=1, total_reward=0.0
        )
        tree.add_root(summary, kind="real")
        
        rows = rows_from_tree(tree)
        
        # Should not raise
        json_str = json.dumps(rows)
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert len(parsed) == 1


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
def test_all_rows_json_dumps_successfully():
    """Comprehensive test that all row types can be json.dumps()."""
    from openenv.counterfactual.logging import (
        make_real_step_row,
        make_counterfactual_row,
        rows_from_tree,
    )
    from openenv.counterfactual.tree import TrajectoryTree
    from openenv.counterfactual.simulate import TrajectorySummary
    from openenv.counterfactual.compare import CandidateResult
    
    # Create complex observation with RGB frame
    rgb_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Test make_real_step_row
    real_row = make_real_step_row(
        step_idx=0,
        observation={"rgb": rgb_frame, "position": [1.0, 2.0, 3.0]},
        action={"type": "move", "direction": [0.5, 0.5]},
        reward=1.5,
        done=False,
        include_frame=True,
    )
    json.dumps(real_row)  # Should not raise
    
    # Test make_counterfactual_row
    summary = TrajectorySummary(
        horizon=10,
        final_observation={"rgb": rgb_frame, "score": 100},
        steps=5,
        total_reward=25.0,
        done=False,
    )
    result = CandidateResult(
        candidate_id="candidate_1",
        summary=summary,
        score=25.0,
        action_seq=[{"move": "up"}, {"move": "down"}],
    )
    cf_row = make_counterfactual_row(
        step_idx=3,
        branch_id="branch_xyz",
        candidate=[{"move": "up"}, {"move": "down"}],
        result=result,
        include_frame=True,
    )
    json.dumps(cf_row)  # Should not raise
    
    # Test rows_from_tree
    tree = TrajectoryTree()
    tree.add_root(summary, snapshot_id="snap", kind="real")
    tree_rows = rows_from_tree(tree)
    json.dumps(tree_rows)  # Should not raise


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")
def test_import_works():
    """Verify imports work."""
    from openenv.counterfactual import (
        make_real_step_row,
        make_counterfactual_row,
        rows_from_tree,
    )
    
    assert callable(make_real_step_row)
    assert callable(make_counterfactual_row)
    assert callable(rows_from_tree)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
