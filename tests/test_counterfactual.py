"""Tests for counterfactual module.

Tests core functionality:
- Snapshot round-trip
- Compare restores real env
- Deterministic replay (checks RNG outcomes)
"""

import random
from dataclasses import dataclass
import pytest

from openenv.counterfactual.snapshot import Snapshot, make_snapshot_id, capture_rng_state, restore_rng_state
from openenv.counterfactual.simulate import simulate
from openenv.counterfactual.compare import compare


@dataclass
class ToyAction:
    delta: int = 1


@dataclass
class ToyObs:
    x: int
    reward: float
    done: bool
    info: dict


class ToyCounterfactualEnv:
    """
    Minimal env with:
      - integer state x
      - RNG usage to test determinism
      - snapshot/restore protocol
      - step(action) -> obs with reward/done
    """
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)  # env-local RNG
        self.x = 0
        self.t = 0

    def reset(self):
        self.rng = random.Random(self.seed)
        self.x = 0
        self.t = 0
        return ToyObs(x=self.x, reward=0.0, done=False, info={})

    def step(self, action: ToyAction):
        if action is None:
            action = ToyAction(delta=0)  # noop
        self.t += 1
        noise = self.rng.randint(0, 2)  # uses RNG
        self.x += action.delta + noise
        reward = float(self.x)
        done = self.t >= 5
        return ToyObs(x=self.x, reward=reward, done=done, info={"noise": noise, "t": self.t})

    # --- Counterfactual protocol ---
    def snapshot(self) -> Snapshot:
        return Snapshot(
            snapshot_id=make_snapshot_id(),
            payload={"x": self.x, "t": self.t, "seed": self.seed},
            rng=capture_rng_state(env=self),
            meta={"note": "toy"},
        )

    def restore(self, snapshot: Snapshot) -> None:
        self.x = snapshot.payload["x"]
        self.t = snapshot.payload["t"]
        self.seed = snapshot.payload["seed"]
        if snapshot.rng is not None:
            restore_rng_state(snapshot.rng, env=self)


def test_snapshot_restore_roundtrip():
    env = ToyCounterfactualEnv(seed=123)
    env.reset()
    env.step(ToyAction(delta=2))
    env.step(ToyAction(delta=2))

    snap = env.snapshot()
    x_before, t_before = env.x, env.t

    # mutate
    env.step(ToyAction(delta=100))
    assert env.x != x_before

    # restore
    env.restore(snap)
    assert env.x == x_before
    assert env.t == t_before


def test_simulate_does_not_mutate_real_env():
    env = ToyCounterfactualEnv(seed=123)
    env.reset()
    env.step(ToyAction(delta=1))  # real state changes

    real = env.snapshot()
    x_real, t_real = env.x, env.t

    # simulate from real snapshot
    summary = simulate(
        env,
        snapshot=real,
        action_seq=[ToyAction(delta=1), ToyAction(delta=1)],
        horizon=2,
        record_observations=True,
        restore_to=real,  # IMPORTANT: restore to real at end
    )

    assert summary.steps in (1, 2)  # depending on stop_on_done
    # ensure env was restored
    assert env.x == x_real
    assert env.t == t_real


def test_compare_restores_real_env_and_ranks():
    env = ToyCounterfactualEnv(seed=7)
    env.reset()
    env.step(ToyAction(delta=1))
    env.step(ToyAction(delta=1))

    real = env.snapshot()
    x_real, t_real = env.x, env.t

    base = real  # compare candidates from current snapshot

    # candidate actions: lower delta vs higher delta
    candidates = [
        [ToyAction(delta=1)],
        [ToyAction(delta=5)],
    ]

    results = compare(env, base_snapshot=base, candidates=candidates, horizon=3)

    # env must be restored to real after compare
    assert env.x == x_real
    assert env.t == t_real

    # best candidate should have higher score (likely higher reward trajectory)
    assert results[0].score >= results[1].score


def test_deterministic_replay_from_snapshot():
    env = ToyCounterfactualEnv(seed=42)
    env.reset()
    env.step(ToyAction(delta=1))  # move to non-trivial state
    snap = env.snapshot()

    action_seq = [ToyAction(delta=1), ToyAction(delta=1), ToyAction(delta=1)]

    s1 = simulate(env, snapshot=snap, action_seq=action_seq, horizon=3, restore_to=snap, record_observations=True)
    s2 = simulate(env, snapshot=snap, action_seq=action_seq, horizon=3, restore_to=snap, record_observations=True)

    # deterministic: final obs should match
    assert getattr(s1.final_observation, "x", None) == getattr(s2.final_observation, "x", None)
    # if you recorded obs, each step should match too
    if s1.observations and s2.observations:
        assert [o.x for o in s1.observations] == [o.x for o in s2.observations]
        # Upgrade A: Explicitly check RNG determinism via noise sequence
        assert [o.info["noise"] for o in s1.observations] == [o.info["noise"] for o in s2.observations]


def test_compare_restores_real_env_on_exception():
    """
    MUST NEVER REGRESS: Exception safety test.
    
    This test proves compare() uses try/finally restoration even on failures.
    If this test fails, compare() will leave the real environment mutated
    when candidate simulation raises an exception - a critical bug.
    """
    env = ToyCounterfactualEnv(seed=7)
    env.reset()
    env.step(ToyAction(delta=1))
    real = env.snapshot()
    x_real, t_real = env.x, env.t

    candidates = [
        [ToyAction(delta=1)],
        ["not an action"],  # will raise inside env.step
    ]

    with pytest.raises(Exception):
        compare(env, base_snapshot=real, candidates=candidates, horizon=2)

    assert env.x == x_real
    assert env.t == t_real