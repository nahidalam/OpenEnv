# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for Robot City Isaac Environment.

Usage:
    uvicorn envs.robot_city_isaac_env.server.app:app --host 0.0.0.0 --port 8001
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from ..models import RobotCityIsaacAction, RobotCityIsaacObs
    from .robot_city_isaac_env import RobotCityIsaacEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import RobotCityIsaacAction, RobotCityIsaacObs
    from server.robot_city_isaac_env import RobotCityIsaacEnvironment


# Create the FastAPI app using OpenEnv's standard pattern
app = create_app(
    RobotCityIsaacEnvironment,
    RobotCityIsaacAction,
    RobotCityIsaacObs,
    env_name="robot_city_isaac_env",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
