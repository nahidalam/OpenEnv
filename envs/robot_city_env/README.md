# Robot City Environment

A visual multi-robot simulation environment for OpenEnv. Robots navigate through dynamic environments with pedestrians, obstacles, and optional failure injection modes.

## Features

- **Visual Output**: RGB top-down view (128×128) every step
- **Multi-Robot**: Control 1-4 robots with discrete actions
- **Dynamic Pedestrians**: Stochastic pedestrian motion with collision detection
- **Uncertainty Overlays**: Collision risk heatmaps from stochastic rollouts
- **Future Prediction**: Pedestrian occupancy prediction heatmaps
- **Failure Injection**: Camera dropout, GPS drift, pedestrian rush, sudden obstacles
- **Counterfactual Support**: Snapshot/restore for imagination-based planning
- **Green Agent**: Compute-aware action selection helper

## Quick Start

### Local Testing (No Server)

```python
from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
from envs.robot_city_env.models import RobotCityAction

env = RobotCityEnvironment()
obs = env.reset(scenario="intersection_crosswalk", num_robots=1)

for step in range(50):
    action = RobotCityAction(robot_id=0, move="forward")
    obs = env.step(action)
    
    if obs.done:
        break

print(f"Episode reward: {obs.metadata['episode_reward']}")
```

### With Server

```bash
# Start server
cd envs/robot_city_env
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run demo
python -m envs.robot_city_env.demo_client --local
```

### Run Demo (Saves GIF)

```bash
# Local demo (no server needed)
python -m envs.robot_city_env.demo_client --local --output runs/demo.gif

# With server
python -m envs.robot_city_env.demo_client --url http://localhost:8000
```

## Action Space

```python
RobotCityAction(
    robot_id: int = 0,      # Which robot to control (0-indexed)
    move: str = "noop",     # "noop", "forward", "backward", "left", "right", "turn_left", "turn_right"
    speed: float = 1.0,     # Speed multiplier (0.0 to 2.0)
)
```

## Observation Space

```python
RobotCityObservation(
    rgb: str,                    # Base64 PNG (128×128 top-down view)
    ego_rgb: str | None,         # Base64 PNG (64×64 ego-centric view)
    uncertainty_overlay: str | None,  # Base64 PNG collision risk heatmap
    future_heatmap: str | None,  # Base64 PNG predicted pedestrian occupancy
    state: dict,                 # Robot poses, pedestrian poses, step count
    events: list[str],           # Events: "collision_ped_*", "near_miss_*", etc.
    reward: float,
    done: bool,
)
```

## Scenarios

| Scenario | Description |
|----------|-------------|
| `intersection_crosswalk` | Intersection with pedestrian crossings |
| `warehouse_aisles` | Warehouse with shelf aisles and workers |
| `sidewalk_delivery` | Sidewalk delivery with busy pedestrian traffic |

## Reward Structure

| Event | Reward |
|-------|--------|
| Forward progress toward goal | +0.1 × progress |
| Collision with pedestrian | -5.0 |
| Collision with obstacle | -1.0 |
| Near miss | -0.2 |
| Action cost (non-noop) | -0.01 |
| All robots reach goal | +10.0 |

## Failure Injection

Enable at reset:

```python
obs = env.reset(
    camera_dropout_prob=0.1,    # 10% chance of camera failure
    gps_drift_sigma=0.02,       # GPS noise standard deviation
    pedestrian_rush_prob=0.05,  # 5% chance of pedestrian speed boost
    sudden_obstacle_prob=0.02,  # 2% chance of new obstacle
    action_delay_prob=0.05,     # 5% chance of action ignored
)
```

## Green Agent (Compute-Aware)

```python
from envs.robot_city_env.green_agent import GreenAgent, select_action_green

# Simple function
action = select_action_green(obs.state, obs, robot_id=0, env=env)

# Or use wrapper class
agent = GreenAgent(risk_threshold=0.3)
action = agent.act(obs, env=env)  # Uses counterfactual only when risky
```

## Counterfactual Planning

```python
# Snapshot current state
snapshot = env.snapshot()

# Simulate counterfactual
for action in candidate_actions:
    env.restore(snapshot)
    obs = env.step(action)
    # ... evaluate outcome

# Restore to real state
env.restore(snapshot)
```

## Tests

```bash
pytest tests/envs/test_robot_city_env.py -v
```

## Dependencies

- numpy
- pillow
- pydantic
- fastapi
- uvicorn
