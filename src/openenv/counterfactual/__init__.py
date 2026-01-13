# openenv/counterfactual/__init__.py
from .snapshot import Snapshot
from .protocols import Snapshotable, Stepable
from .simulate import simulate, TrajectorySummary
from .compare import compare, CandidateResult, default_scorer
from .tree import TrajectoryTree, TrajectoryNode
from .logging import (
    TrajectoryLogEntry,
    summary_to_log_entries,
    tree_to_log_entries,
    write_jsonl,
    read_jsonl,
    make_vision_frame_processor,
    make_real_step_row,
    make_counterfactual_row,
    rows_from_tree,
)
from .serialize import (
    summary_to_json,
    candidate_results_to_json,
)

