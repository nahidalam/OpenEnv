# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Robot City Isaac Environment - Realistic RGB simulation using NVIDIA Isaac Sim.

This environment provides:
- Photorealistic RGB observations from Isaac Sim
- Multi-agent support (ego robot + pedestrians)
- Failure injection modes (camera dropout, action delay, motion blur)
- Visual overlays for debugging/visualization
- Counterfactual planning support (snapshot/restore)

Requires NVIDIA Isaac Sim to be installed for server-side rendering.
Client can run without Isaac Sim.
"""

from .client import RobotCityIsaacEnv
from .models import RobotCityIsaacAction, RobotCityIsaacObs

__all__ = ["RobotCityIsaacEnv", "RobotCityIsaacAction", "RobotCityIsaacObs"]
