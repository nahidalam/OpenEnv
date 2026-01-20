# Robot City Isaac Environment

Realistic visual robot simulation environment powered by NVIDIA Isaac Sim.

## Overview

This environment provides photorealistic RGB observations of a robot navigating urban scenarios with pedestrians. It integrates with OpenEnv's server/client architecture and supports:

- **Realistic rendering** via Isaac Sim (or fallback rendering without Isaac)
- **Multi-agent support**: ego robot + configurable pedestrians
- **Failure injection**: camera dropout, action delay, motion blur
- **Visual overlays**: heading arrow, risk cone, trajectory predictions
- **Counterfactual planning**: snapshot/restore for branching simulations
- **Video export**: MP4 and JSONL logging

## Scenarios

| Scenario | Description |
|----------|-------------|
| `crosswalk_occlusion` | Navigate a crosswalk with occluded pedestrians |
| `sidewalk_delivery` | Delivery robot on a busy sidewalk |
| `warehouse_aisles` | Navigate narrow warehouse aisles |

## Installation

```bash
# Core dependencies (required)
pip install numpy pillow pydantic

# Video export (optional)
pip install imageio imageio-ffmpeg

# Visual overlays (optional, recommended)
pip install opencv-python
```

**For realistic rendering**: Install [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim). Without Isaac Sim, the environment runs in fallback mode with synthetic rendering.

## Quickstart

### Local Demo (No Server)

```bash
# Basic demo
python -m envs.robot_city_isaac_env.demo_client --local --steps 100 --output runs/demo.gif

# With failure injection
python -m envs.robot_city_isaac_env.demo_client --local \
  --scenario crosswalk_occlusion \
  --camera-dropout 0.2 \
  --action-delay 0.1 \
  --steps 150 \
  --output runs/failure_demo.gif
```

### Server Mode

```bash
# Terminal 1: Start server
uvicorn envs.robot_city_isaac_env.server.app:app --port 8001

# Terminal 2: Run client
python -m envs.robot_city_isaac_env.demo_client \
  --url http://127.0.0.1:8001 \
  --scenario sidewalk_delivery \
  --num-agents 6 \
  --steps 200 \
  --overlay \
  --output runs/isaac_sidewalk.mp4
```

## CLI Reference

```
usage: demo_client.py [-h] [--url URL] [--local] [--scenario SCENARIO]
                      [--steps STEPS] [--num-agents NUM_AGENTS] [--seed SEED]
                      [--resolution RESOLUTION] [--output OUTPUT]
                      [--camera-dropout P] [--action-delay P] [--motion-blur P]
                      [--overlay] [--no-overlay]

Arguments:
  --url URL              Server URL (default: http://localhost:8001)
  --local                Run locally without server
  --scenario             Scenario: crosswalk_occlusion, sidewalk_delivery, warehouse_aisles
  --steps                Max episode steps (default: 200)
  --num-agents           Number of pedestrian agents (default: 6)
  --seed                 Random seed (default: 42)
  --resolution           Image resolution (default: 512)
  --output               Output path (.mp4 or .gif)
  --camera-dropout       Camera dropout probability 0-1
  --action-delay         Action delay probability 0-1
  --motion-blur          Motion blur probability 0-1
  --overlay / --no-overlay   Enable/disable visual overlays
```

## API Usage

### Python Client

```python
from envs.robot_city_isaac_env import RobotCityIsaacEnv, RobotCityIsaacAction

with RobotCityIsaacEnv(base_url="http://localhost:8001") as client:
    # Reset to a scenario
    result = client.reset(
        scenario="sidewalk_delivery",
        num_agents=6,
        seed=42,
        camera_dropout=0.1,
    )
    
    # Run episode
    for step in range(200):
        action = RobotCityIsaacAction(
            linear_vel=0.5,  # -1 to 1
            angular_vel=0.0, # -1 to 1
        )
        result = client.step(action)
        
        # Access observation
        frame = client.decode_frame(result.observation.rgb)
        goal_dist = result.observation.goal_distance
        events = result.observation.events
        
        if result.done:
            break
```

### Direct Environment (No Server)

```python
from envs.robot_city_isaac_env.server import RobotCityIsaacEnvironment
from envs.robot_city_isaac_env.models import RobotCityIsaacAction

env = RobotCityIsaacEnvironment(headless=True, resolution=512)

obs = env.reset(scenario="sidewalk_delivery", num_agents=6)

for step in range(200):
    action = RobotCityIsaacAction(linear_vel=0.5, angular_vel=0.0)
    obs = env.step(action)
    
    if obs.done:
        break

env.close()
```

## Observation Format

| Field | Type | Description |
|-------|------|-------------|
| `rgb` | str | Base64 PNG of robot camera view |
| `rgb_with_overlay` | str | RGB with visual overlays drawn |
| `step_idx` | int | Current step index |
| `ego_pose` | tuple | Robot pose (x, y, z, roll, pitch, yaw) |
| `goal_pose` | tuple | Goal position (x, y, z) |
| `goal_distance` | float | Distance to goal in meters |
| `agents` | list | List of agent state dicts |
| `events` | dict | Events: collision, near_miss, goal_reached, dropout_applied |
| `reward` | float | Step reward |
| `done` | bool | Episode termination flag |

## Failure Injection

| Mode | Effect |
|------|--------|
| `camera_dropout` | Returns black frame |
| `action_delay` | Applies previous action instead |
| `motion_blur` | Blurs the RGB frame |

## Visual Legend

In overlay mode:
- **Blue circle**: Robot (ego)
- **Orange dots**: Pedestrians
- **Green marker**: Goal
- **Yellow cone**: Risk/attention zone
- **Cyan lines**: Predicted agent trajectories

## Testing

```bash
# Run all tests (no Isaac Sim required)
pytest tests/envs/test_robot_city_isaac_env.py -v
```

## Architecture

```
envs/robot_city_isaac_env/
├── __init__.py           # Package exports
├── models.py             # Action/Observation dataclasses
├── client.py             # HTTP client for server
├── demo_client.py        # CLI demo script
├── README.md             # This file
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI server
    ├── isaac_bridge.py   # Isaac Sim wrapper
    ├── robot_city_isaac_env.py  # Main environment
    ├── scenarios.py      # Scenario configs
    └── renderer_overlays.py  # Overlay drawing
```

## Notes

- **Fallback mode**: Without Isaac Sim, the environment renders synthetic first-person views. This is useful for development and testing.
- **GPU recommended**: Isaac Sim requires an NVIDIA GPU for realistic rendering.
- **Deterministic**: Same seed produces same observations for reproducibility.
