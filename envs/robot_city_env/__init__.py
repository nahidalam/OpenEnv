# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Robot City Environment - A visual multi-robot simulation environment.

Supports:
- Multi-robot control with discrete actions
- Visual RGB observations with uncertainty/future overlays
- Pedestrian dynamics and collision detection
- Failure injection modes for robustness testing
- Counterfactual planning support (snapshot/restore)
"""

from .client import RobotCityEnv
from .models import RobotCityAction, RobotCityObservation

__all__ = ["RobotCityEnv", "RobotCityAction", "RobotCityObservation"]
