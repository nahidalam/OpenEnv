# Robot City Environment

A visual multi-robot simulation environment for OpenEnv. Robots navigate through dynamic urban environments with pedestrians, obstacles, and optional failure injection modes for robustness testing.

![Demo GIF](../../runs/demo_robotcity.gif)

---

## Visual Legend

| Symbol | Color | Meaning |
|--------|-------|---------|
| 🔵 | **Blue circle** | Ego robot (the one being controlled/evaluated) |
| ➡️ | **White arrow on robot** | Robot heading direction (points where robot will move) |
| 🔵 | **Light blue trail dots** | Last 20 robot positions (fades with age) |
| 🟠 | **Orange circles** | Dynamic agents (pedestrians, other moving actors) |
| 🟢 | **Green dot** | Goal / target location |
| ⬛ | **Gray rectangles** | Static obstacles / walls / blocked regions |

### On-Screen Legend
A small legend appears in the top-left corner of each frame:
- **Rob** = Robot (blue)
- **Ped** = Pedestrian (orange)  
- **Goal** = Target (green)

---

## Architecture Overview

```
robot_city_env/
├── models.py                    # Action & Observation data types
├── client.py                    # WebSocket client for remote server
├── demo_client.py               # Demo script with greedy policy
├── green_agent.py               # Compute-aware action selection
└── server/
    ├── app.py                   # FastAPI server (OpenEnv pattern)
    ├── robot_city_environment.py  # Main environment logic
    └── sim/
        ├── dynamics.py          # Physics, collision detection
        ├── scenarios.py         # Scenario definitions
        ├── render.py            # Pillow-based RGB rendering
        └── failures.py          # Failure injection modes
```

### Main Blocks

#### 1. **Models** (`models.py`)
Defines the Action and Observation types using Pydantic:

- **`RobotCityAction`**: What the agent sends each step
  - `robot_id`: Which robot to control (0-indexed)
  - `move`: One of `"noop"`, `"forward"`, `"backward"`, `"left"`, `"right"`, `"turn_left"`, `"turn_right"`
  - `speed`: Speed multiplier (0.0 to 2.0)

- **`RobotCityObservation`**: What the environment returns
  - `rgb`: Base64 PNG of top-down view (128×128)
  - `ego_rgb`: Base64 PNG of ego-centric view (64×64)
  - `state`: Dict with robot poses, pedestrian poses, goals
  - `events`: List of events (`"collision_ped_0"`, `"near_miss"`, `"camera_dropout"`, etc.)
  - `reward`: Float reward for this step
  - `done`: Whether episode ended

#### 2. **Simulation Core** (`server/sim/`)

- **`dynamics.py`**: Physics engine
  - Robot movement with discrete actions
  - Pedestrian stochastic motion (random walk with drift)
  - Circle-circle and circle-rectangle collision detection
  - Near-miss detection (close calls)

- **`scenarios.py`**: Pre-built scenario layouts
  - `crosswalk_occlusion`: Intersection with pedestrian crossings
  - `sidewalk_delivery`: Busy sidewalk with obstacles
  - `warehouse_aisles`: Narrow corridors with workers

- **`render.py`**: Visual output generation
  - Top-down RGB view using Pillow (no GPU needed)
  - Ego-centric cropped view
  - Optional heatmaps (uncertainty, future prediction)

- **`failures.py`**: Robustness testing
  - Camera dropout (blank/noisy frames)
  - Action delay (commands ignored)
  - GPS drift (noisy position reports)
  - Pedestrian rush (sudden speed increase)

#### 3. **Environment** (`server/robot_city_environment.py`)
The main `RobotCityEnvironment` class implementing OpenEnv's `Environment` interface:

- `reset()`: Initialize new episode with scenario and parameters
- `step()`: Execute action, update physics, return observation
- `snapshot()` / `restore()`: Counterfactual planning support

#### 4. **Server** (`server/app.py`)
FastAPI application using OpenEnv's `create_app()` helper:
- HTTP endpoints for reset/step
- Health check at `/health`
- Runs with uvicorn

#### 5. **Client** (`client.py`)
WebSocket-based client extending `EnvClient`:
- Connects to remote server
- Handles serialization/deserialization
- Context manager support (`with RobotCityEnv(...) as client:`)

#### 6. **Green Agent** (`green_agent.py`)
Compute-aware action selection:
- Default: Greedy toward goal (cheap)
- High uncertainty: Counterfactual rollouts (expensive)
- Useful for efficient inference

---

## Quick Start

### Local Demo (No Server)

```bash
cd /path/to/OpenEnv

# Basic demo
python -m envs.robot_city_env.demo_client --local

# With failure injection
python -m envs.robot_city_env.demo_client --local \
  --scenario sidewalk_delivery \
  --camera-dropout 0.1 \
  --action-delay 0.1 \
  --output runs/failure_test.gif
```

### With Server

```bash
# Terminal 1: Start server
uvicorn envs.robot_city_env.server.app:app --port 8001

# Terminal 2: Run demo
python -m envs.robot_city_env.demo_client \
  --url http://localhost:8001 \
  --scenario crosswalk_occlusion \
  --steps 100 \
  --output runs/demo.gif
```

### Expected Output

```
============================================================
ROBOT CITY DEMO (Local)
============================================================
Scenario: crosswalk_occlusion
Policy: greedy (move toward goal)
Max steps: 100
============================================================

Step 0: initialized
Step  20: reward=+1.85, goal_dist=0.542, move=forward
Step  40: reward=+3.92, goal_dist=0.341, move=forward
Step  60: reward=+5.78, goal_dist=0.142, move=forward

Episode ended at step 72

============================================================
EPISODE SUMMARY
============================================================
  Steps: 72
  Total Reward: 16.45
  Goal Reached: True
  Collisions: 0
  Near Misses: 2

Saving 73 frames to runs/demo_robotcity.gif...
GIF saved to: runs/demo_robotcity.gif
============================================================
```

---

## Scenarios

### `crosswalk_occlusion`
Robot navigates an intersection with pedestrian crossings. Pedestrians cross horizontally and vertically, creating occlusion scenarios.

```
Layout:
┌─────┬─────────────┬─────┐
│ ███ │             │ ███ │  ███ = Building corners
│     │  ← peds →   │     │
│     │      ↑      │     │
│     │    robot    │     │
│ ███ │      ↓      │ ███ │
│     │   ★ goal    │     │
└─────┴─────────────┴─────┘
```

### `sidewalk_delivery`
Delivery robot on a busy sidewalk. Must navigate around pedestrians, benches, and planters.

```
Layout:
┌────────────────────────────────┐
│████████ BUILDING WALL █████████│
│                                │
│  🔵→    🟠  ▢  🟠    ▢    🟢  │  ▢ = Obstacles
│        🟠      🟠              │
│ ═══════════════════════════════│  ═ = Curb
└────────────────────────────────┘
```

### `warehouse_aisles`
Warehouse with narrow aisles between shelves. Workers move along aisles.

```
Layout:
┌────────────────────────────────┐
│  ████████     ████████        │  █ = Shelves
│          aisle                 │
│  ████████     ████████    🟢  │
│          aisle      🟠         │
│  ████████     ████████        │
│     🔵                         │
└────────────────────────────────┘
```

---

## Reward Structure

| Event | Reward | Description |
|-------|--------|-------------|
| Forward progress | +0.1 × progress | Moving toward goal |
| Collision (pedestrian) | -5.0 | Hit a pedestrian |
| Collision (obstacle) | -1.0 | Hit a wall/obstacle |
| Collision (robot) | -2.0 | Hit another robot |
| Near miss | -0.2 | Close call with pedestrian |
| Action cost | -0.01 | Any non-noop action |
| Goal reached | +10.0 | All robots at goal |

---

## Failure Injection

Test robustness by injecting failures at reset:

```python
obs = env.reset(
    scenario="crosswalk_occlusion",
    camera_dropout_prob=0.1,   # 10% blank frames
    action_delay_prob=0.05,    # 5% commands ignored
    gps_drift_sigma=0.02,      # Position noise
    pedestrian_rush_prob=0.02, # Sudden speed boost
)
```

### Failure Events
- `camera_dropout`: RGB frame is noisy/blank this step
- `action_delay`: Action was ignored (noop executed instead)
- `pedestrian_rush`: Pedestrians suddenly sped up
- `gps_drift`: Reported position has noise (check `reported_x`, `reported_y`)

---

## Programmatic Usage

```python
from envs.robot_city_env.server.robot_city_environment import RobotCityEnvironment
from envs.robot_city_env.models import RobotCityAction

# Create environment
env = RobotCityEnvironment()

# Reset with scenario
obs = env.reset(
    scenario="crosswalk_occlusion",
    num_robots=1,
    num_peds=4,
    seed=42,
)

# Run episode
total_reward = 0
for step in range(200):
    # Your policy here
    action = RobotCityAction(robot_id=0, move="forward", speed=1.0)
    
    obs = env.step(action)
    total_reward += obs.reward
    
    # Check events
    if "collision" in str(obs.events):
        print(f"Collision at step {step}!")
    
    if obs.done:
        break

print(f"Episode finished: reward={total_reward:.2f}")
```

### Counterfactual Planning

```python
# Snapshot current state
snapshot = env.snapshot()

# Try different actions
for move in ["forward", "turn_left", "turn_right"]:
    env.restore(snapshot)
    obs = env.step(RobotCityAction(move=move))
    print(f"{move}: reward={obs.reward}")

# Restore and execute best
env.restore(snapshot)
```

---

## CLI Reference

```bash
python -m envs.robot_city_env.demo_client [OPTIONS]

Options:
  --url URL              Server URL (default: http://localhost:8000)
  --local                Run locally without server
  --scenario SCENARIO    crosswalk_occlusion | sidewalk_delivery | warehouse_aisles
  --steps N              Max steps (default: 100)
  --output PATH          Output GIF path (default: runs/demo_robotcity.gif)
  --seed N               Random seed (default: 42)
  --camera-dropout P     Camera dropout probability (default: 0.0)
  --action-delay P       Action delay probability (default: 0.0)
```

---

## Verified Commands

All commands below have been tested and work. Run from the OpenEnv root directory.

### 1. Functional Tests

```bash
python -m pytest tests/envs/test_robot_city_env.py -v
```

Expected: 14 tests pass in ~1 second.

### 2. Start the Server

```bash
uvicorn envs.robot_city_env.server.app:app --port 8001
```

Server runs at `http://127.0.0.1:8001`. Keep this terminal open.

### 3. Scenario Differentiation Test

Run all three scenarios to compare:

```bash
# Crosswalk with pedestrian crossings
python -m envs.robot_city_env.demo_client \
  --url http://127.0.0.1:8001 \
  --scenario crosswalk_occlusion \
  --steps 80 \
  --output runs/crosswalk.gif

# Busy sidewalk delivery
python -m envs.robot_city_env.demo_client \
  --url http://127.0.0.1:8001 \
  --scenario sidewalk_delivery \
  --steps 80 \
  --output runs/sidewalk.gif

# Warehouse aisles
python -m envs.robot_city_env.demo_client \
  --url http://127.0.0.1:8001 \
  --scenario warehouse_aisles \
  --steps 80 \
  --output runs/warehouse.gif
```

### 4. Extended Motion Test

Longer episode to observe full navigation:

```bash
python -m envs.robot_city_env.demo_client \
  --url http://127.0.0.1:8001 \
  --scenario crosswalk_occlusion \
  --steps 120 \
  --output runs/crosswalk.gif
```

### 5. Failure Injection Test

Test robustness with camera dropouts and action delays:

```bash
python -m envs.robot_city_env.demo_client \
  --url http://127.0.0.1:8001 \
  --scenario crosswalk_occlusion \
  --steps 120 \
  --camera-dropout 0.2 \
  --action-delay 0.2 \
  --output runs/failure_demo.gif
```

Expected output will show failure events:
```
============================================================
EPISODE SUMMARY
============================================================
  Steps: 120
  Total Reward: -15.32
  Goal Reached: False
  Collisions: 3
  Near Misses: 8
  Camera Dropouts: 24
  Action Delays: 19
============================================================
```

---

## Unit Tests

```bash
# Run all tests (from OpenEnv root)
python -m pytest tests/envs/test_robot_city_env.py -v

# Quick smoke test only
pytest tests/envs/test_robot_city_env.py::TestRobotCitySmoke -v

# Determinism test
pytest tests/envs/test_robot_city_env.py::TestDeterminism -v

# Failure injection test
pytest tests/envs/test_robot_city_env.py::TestFailureInjection -v
```

All 14 tests should pass in ~1 second.

---

## Dependencies

Minimal - no PyTorch required:

```
numpy
pillow
pydantic
fastapi    # server only
uvicorn    # server only
```

Install:
```bash
pip install numpy pillow pydantic fastapi uvicorn
```

---

## Integration with World Model Platform

This environment is designed to work with the [World Model Platform](https://github.com/yourname/world_model_platform):

```python
# In world_model_platform
from wmp.clients import OpenEnvRobotCityClient

client = OpenEnvRobotCityClient()
client.connect("http://localhost:8001")

obs = client.reset(scenario_id="crosswalk_occlusion", seed=42)
result = client.step(RobotCityAction(forward=0.8, turn=0.0))
```

The WMP client decodes base64 frames to PIL Images and provides a simpler API for running evaluation episodes.
