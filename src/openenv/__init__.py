"""
Unified OpenEnv package bundling the CLI and core runtime.

Modules:
    core: Core environment client/server runtime
    cli: Command-line interface
    counterfactual: Counterfactual reasoning utilities (snapshot, simulate, compare)
    vision: Vision-friendly utilities for image encoding and trajectory logging
"""

from importlib import metadata

__all__ = ["core", "cli", "counterfactual", "vision"]

try:
    __version__ = metadata.version("openenv")  # type: ignore[arg-type]
except metadata.PackageNotFoundError:  # pragma: no cover - local dev
    __version__ = "0.0.0"
