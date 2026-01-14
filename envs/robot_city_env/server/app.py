# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Robot City Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from ..models import RobotCityAction, RobotCityObservation
    from .robot_city_environment import RobotCityEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import RobotCityAction, RobotCityObservation
    from server.robot_city_environment import RobotCityEnvironment

# Create the app
app = create_app(
    RobotCityEnvironment,
    RobotCityAction,
    RobotCityObservation,
    env_name="robot_city_env",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
