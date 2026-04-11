---
title: EV Charging Scheduler OpenEnv
emoji: ⚡
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
---

# EV Charging Scheduler — OpenEnv RL Benchmark

An RL benchmark for allocating charging slots across electric vehicles to **minimise energy cost and lateness** while **respecting grid limits**. Built for the [OpenEnv](https://huggingface.co/openenv) challenge.

## Why EV Charging?

EV charging is a timely, real-world optimisation problem with clean mathematical structure. The agent must balance competing objectives (cost vs. timeliness) under hard constraints (grid capacity, charger availability), making it ideal for RL training and evaluation.

---

## Core Loop

```
state → action → reward (every step)
```

```python
from environment import ChargingEnvironment
from models import Action

env = ChargingEnvironment()
obs = env.reset("medium", seed=42)

done = False
while not done:
    mask = obs.action_mask          # N×4 boolean mask
    actions = your_agent(obs.state, mask)
    obs, reward, done, info = env.step(Action(actions=actions))
    print(f"reward={reward.value:+.3f}  cost={reward.breakdown['energy_cost']:.3f}")

grade = env.grade()
print(f"Final score: {grade['score']:.4f}")
```

---

## Observation Space

Fixed-size `float32` vector, fully numeric. For `N` charger ports:

| Index Range | Feature | Range |
|---|---|---|
| `[0..N-1]` | Current SoC per port (0 if empty) | `[0, 1]` |
| `[N..2N-1]` | Target SoC per port | `[0, 1]` |
| `[2N..3N-1]` | Time remaining until departure (normalised) | `[0, 1]` |
| `[3N..4N-1]` | Port occupied (binary) | `{0, 1}` |
| `[4N..5N-1]` | Port operational (binary) | `{0, 1}` |
| `[5N]` | Current electricity tariff | `[0, 1]` |
| `[5N+1]` | Grid load ratio | `[0, 1]` |
| `[5N+2]` | Time of day (normalised) | `[0, 1]` |
| `[5N+3]` | Steps remaining (normalised) | `[0, 1]` |

**Total size**: `5N + 4` (e.g., 24 for easy, 34 for medium, 44 for hard)

---

## Action Space

**MultiDiscrete([4] × N)** — for each of `N` ports, select a charging level:

| Level | Power | Description |
|---|---|---|
| 0 | 0 kW | OFF |
| 1 | 3.6 kW | LOW |
| 2 | 7 kW | MEDIUM |
| 3 | 11 kW | HIGH |

### Action Masking

Every observation includes an `action_mask: bool[N][4]`:

| Condition | Masked Actions |
|---|---|
| Port empty | Only OFF allowed |
| Port failed (hard mode) | Only OFF allowed |
| Vehicle fully charged | Only OFF allowed |
| Grid overload risk | HIGH masked on excess ports |

---

## Reward Design — Dense, Per-Step

| Signal | Value | Purpose |
|---|---|---|
| Energy cost | `−Σ(power × tariff)` (normalised) | Push toward cheap charging |
| Charging progress | `+Σ(ΔSoC)` | Reward making progress |
| Lateness penalty | `−0.5 × shortfall` per vehicle | Punish failing departure targets |
| Grid violation | `−0.5` | Hard constraint signal |
| Idle penalty | `−0.01` per idle occupied port | Discourage inaction |
| Completion bonus | `+0.3` per satisfied vehicle | Milestone reward |
| Episode end | `+1.0 × (satisfied/total)` | Overall success signal |

Every step returns a meaningful scalar — **no deferred grading**.

---

## Tasks & Difficulty Curriculum

| Task | Ports | Vehicles | Tariff | Grid Limit | Failures | Urgent | Departures |
|---|---|---|---|---|---|---|---|
| **easy** | 4 | 3 | Flat (0.30) | 100 kW | ✗ | ✗ | Relaxed |
| **medium** | 6 | 5 | Time-of-Use | 55 kW | ✗ | ✗ | Tighter |
| **hard** | 8 | 8 | Time-of-Use | 45 kW | ✓ | ✓ | Mixed |

Each difficulty adds **strictly more constraints** — same core objective, tighter bounds. This is an algorithmically progressive curriculum, not scenario-driven.

---

## Grader (Score 0.0–1.0)

| Component | Weight | Description |
|---|---|---|
| Vehicle satisfaction | 0.40 | Fraction of vehicles reaching target SoC |
| Energy efficiency | 0.25 | Lower total cost = higher score |
| Grid compliance | 0.20 | Fewer grid violations = higher score |
| Timeliness | 0.15 | How close all vehicles are to their targets |

---

## Baseline Scores

Three built-in heuristic baselines (no LLM needed):

| Strategy | Easy | Medium | Hard | Average |
|---|---|---|---|---|
| Random | ~0.90 | ~0.63 | ~0.47 | ~0.67 |
| Greedy-Cheapest | ~0.90 | ~0.92 | ~0.80 | ~0.87 |
| Urgency-Aware | ~0.90 | ~0.92 | ~0.80 | ~0.87 |

Run baselines:
```bash
python heuristic.py
```

---

## Setup & Usage

### Local Development
```bash
pip install -r requirements.txt
uvicorn server.app:app --reload
```

### Run Tests
```bash
pip install pytest
python -m pytest test_environment.py test_api.py -v
```

### Run Inference (Self-Contained)
```bash
# Uses in-process environment — no HTTP dependency
export HF_TOKEN="your-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

### Docker
```bash
docker build -t openenv-ev-charging .
docker run -p 8000:8000 openenv-ev-charging
```

---

## PPO Training Example (Conceptual)

```python
import gymnasium as gym
from stable_baselines3 import PPO
from environment import ChargingEnvironment
from models import Action

# Wrap ChargingEnvironment in a Gym-compatible wrapper
class EVChargingGymEnv(gym.Env):
    def __init__(self, task_id="medium"):
        super().__init__()
        self.env = ChargingEnvironment()
        self.task_id = task_id
        N = self.env.config["num_ports"]
        self.observation_space = gym.spaces.Box(0, 1, shape=(5*N+4,))
        self.action_space = gym.spaces.MultiDiscrete([4] * N)

    def reset(self, seed=None, **kwargs):
        obs = self.env.reset(self.task_id, seed=seed)
        return obs.state, {"action_mask": obs.action_mask}

    def step(self, action):
        obs, reward, done, info = self.env.step(Action(actions=action.tolist()))
        return obs.state, reward.value, done, False, info.metrics

env = EVChargingGymEnv("medium")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## Project Structure

```
selene/
├── main.py              # FastAPI server (step/reset/state/tasks/grader/baseline)
├── server/app.py        # OpenEnv ASGI entrypoint
├── environment.py       # Core environment: ChargingEnvironment
├── models.py            # Pydantic models (Action, Observation, Reward, Info)
├── inference.py         # Self-contained inference (in-process env, no HTTP)
├── baseline.py          # Multi-strategy baseline runner
├── heuristic.py         # Standalone heuristic agents (random/greedy/urgency)
├── test_environment.py  # Unit tests for environment
├── test_api.py          # HTTP integration tests
├── openenv.yaml         # OpenEnv spec metadata
├── pyproject.toml       # Packaging metadata
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
└── .dockerignore        # Docker build exclusions
```

## Verification

- `python -m pytest test_environment.py test_api.py -v` → 40+ tests
- `python heuristic.py` → baseline scores
- `openenv validate` → passed
