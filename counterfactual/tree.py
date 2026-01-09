"""Trajectory tree data structure for counterfactual reasoning.

Provides a tree structure to represent branching counterfactual trajectories
in a JSON-friendly format suitable for logging and evaluation.
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
from .simulate import TrajectorySummary


def _summary_to_dict(summary: TrajectorySummary) -> Dict[str, Any]:
    """Convert TrajectorySummary to JSON-safe dictionary.
    
    Args:
        summary: TrajectorySummary to convert
        
    Returns:
        JSON-safe dictionary representation
    """
    # Convert summary to dict, handling non-serializable types
    result = {
        "horizon": summary.horizon,
        "steps": summary.steps,
        "total_reward": summary.total_reward,
        "done": summary.done,
        "info": summary.info,
    }
    
    # Handle observations - only include if they're JSON-serializable
    # For efficiency, we typically don't store all observations
    if summary.observations:
        # Try to serialize observations, skip if not possible
        try:
            json.dumps(summary.observations)
            result["observations"] = summary.observations
        except (TypeError, ValueError):
            # Observations not JSON-serializable, skip them
            result["observations"] = None
    
    # Handle final_observation
    try:
        json.dumps(summary.final_observation)
        result["final_observation"] = summary.final_observation
    except (TypeError, ValueError):
        # Final observation not JSON-serializable, convert to string representation
        result["final_observation"] = str(summary.final_observation)
    
    return result


def _action_to_repr(action: Any) -> str:
    """Convert action to string representation for JSON safety.
    
    Args:
        action: Action to stringify
        
    Returns:
        String representation of the action
    """
    if action is None:
        return "None"
    try:
        # Try JSON serialization first (for simple types)
        json.dumps(action)
        return json.dumps(action)
    except (TypeError, ValueError):
        # Fall back to string representation
        return str(action)


@dataclass
class TrajectoryNode:
    """A node in a trajectory tree (JSON-safe).
    
    All fields are JSON-serializable to enable easy logging and storage.
    """
    node_id: str
    parent_id: Optional[str] = None
    kind: Literal["real", "counterfactual"] = "real"
    snapshot_id: Optional[str] = None
    action_repr: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary (JSON-safe).
        
        Returns:
            Dictionary representation of the node
        """
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "snapshot_id": self.snapshot_id,
            "action_repr": self.action_repr,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "depth": self.depth,
        }


class TrajectoryTree:
    """A tree structure representing counterfactual trajectories (JSON-friendly).
    
    Designed for logging and evaluation, with methods to export to JSON/JSONL formats.
    """
    
    def __init__(self, root_id: Optional[str] = None):
        """Initialize a trajectory tree.
        
        Args:
            root_id: Optional root node ID. If None, will be generated when root is added.
        """
        self._nodes: Dict[str, TrajectoryNode] = {}
        self._root_id: Optional[str] = root_id
        self._next_node_id: int = 0
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID.
        
        Returns:
            Unique node ID string
        """
        node_id = f"node_{self._next_node_id:06d}"
        self._next_node_id += 1
        return node_id
    
    def add_root(
        self,
        summary: TrajectorySummary,
        snapshot_id: Optional[str] = None,
        kind: Literal["real", "counterfactual"] = "real",
        node_id: Optional[str] = None,
    ) -> str:
        """Add a root node to the tree.
        
        Args:
            summary: TrajectorySummary for the root node
            snapshot_id: Optional snapshot ID associated with this node
            kind: Type of trajectory ("real" or "counterfactual")
            node_id: Optional node ID. If None, will be generated.
            
        Returns:
            The node_id of the created root node
        """
        if node_id is None:
            node_id = self._root_id if self._root_id is not None else self._generate_node_id()
        
        if node_id in self._nodes:
            raise ValueError(f"Node with id {node_id} already exists")
        
        root_node = TrajectoryNode(
            node_id=node_id,
            parent_id=None,
            kind=kind,
            snapshot_id=snapshot_id,
            action_repr="",  # Root has no action
            summary=_summary_to_dict(summary),
            timestamp=time.time(),
            depth=0,
        )
        
        self._nodes[node_id] = root_node
        self._root_id = node_id
        
        return node_id
    
    def add_child(
        self,
        parent_id: str,
        action: Any,
        summary: TrajectorySummary,
        snapshot_id: Optional[str] = None,
        kind: Literal["real", "counterfactual"] = "counterfactual",
        node_id: Optional[str] = None,
    ) -> str:
        """Add a child node to the tree.
        
        Args:
            parent_id: ID of the parent node
            action: Action that led to this node (will be stringified)
            summary: TrajectorySummary for this node
            snapshot_id: Optional snapshot ID associated with this node
            kind: Type of trajectory ("real" or "counterfactual")
            node_id: Optional node ID. If None, will be generated.
            
        Returns:
            The node_id of the created child node
            
        Raises:
            ValueError: If parent_id doesn't exist
        """
        if parent_id not in self._nodes:
            raise ValueError(f"Parent node with id {parent_id} does not exist")
        
        parent_node = self._nodes[parent_id]
        
        if node_id is None:
            node_id = self._generate_node_id()
        
        if node_id in self._nodes:
            raise ValueError(f"Node with id {node_id} already exists")
        
        child_node = TrajectoryNode(
            node_id=node_id,
            parent_id=parent_id,
            kind=kind,
            snapshot_id=snapshot_id,
            action_repr=_action_to_repr(action),
            summary=_summary_to_dict(summary),
            timestamp=time.time(),
            depth=parent_node.depth + 1,
        )
        
        self._nodes[node_id] = child_node
        
        return node_id
    
    def get_node(self, node_id: str) -> Optional[TrajectoryNode]:
        """Get a node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            TrajectoryNode if found, None otherwise
        """
        return self._nodes.get(node_id)
    
    def get_root(self) -> Optional[TrajectoryNode]:
        """Get the root node.
        
        Returns:
            Root TrajectoryNode if exists, None otherwise
        """
        if self._root_id is None:
            return None
        return self._nodes.get(self._root_id)
    
    def get_children(self, node_id: str) -> List[TrajectoryNode]:
        """Get all children of a node.
        
        Args:
            node_id: ID of the parent node
            
        Returns:
            List of child TrajectoryNodes
        """
        return [node for node in self._nodes.values() if node.parent_id == node_id]
    
    def to_json(self) -> Dict[str, Any]:
        """Convert tree to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the entire tree
        """
        return {
            "root_id": self._root_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
        }
    
    def to_jsonl_rows(self) -> List[Dict[str, Any]]:
        """Convert tree to JSONL format (one row per node).
        
        This is the format that evaluation stacks can store as JSONL.
        Each row is a complete node representation.
        
        Returns:
            List of dictionaries, one per node, suitable for JSONL output
        """
        # Sort nodes by depth and then by node_id for consistent ordering
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: (n.depth, n.node_id)
        )
        return [node.to_dict() for node in sorted_nodes]
    
    def __repr__(self) -> str:
        num_nodes = len(self._nodes)
        root_str = f"root={self._root_id!r}" if self._root_id else "no root"
        return f"TrajectoryTree({root_str}, nodes={num_nodes})"
