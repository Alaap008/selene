from __future__ import annotations

"""
heuristic.py — Standalone heuristic agent for the EV Charging Scheduler.

Three strategies:
  1. Random agent     — uniform random valid actions (respects mask)
  2. Greedy-cheapest  — charge HIGH during off-peak, LOW during peak
  3. Urgency-aware    — prioritises vehicles close to departure

Runnable as: python heuristic.py
Importable:  from heuristic import run_all_tasks
"""

import json
import random as stdlib_random
from typing import Callable

from environment import ChargingEnvironment, POWER_LEVELS, MAX_STEPS
from models import Action


# ---------------------------------------------------------------------------
# Strategy functions: obs_dict → list[int]
# ---------------------------------------------------------------------------

def random_agent(obs: dict, rng: stdlib_random.Random | None = None) -> list[int]:
    """Pick a uniformly random valid action for each port."""
    if rng is None:
        rng = stdlib_random.Random()
    num_ports = obs["num_ports"]
    mask = obs["action_mask"]
    actions = []
    for i in range(num_ports):
        valid = [j for j in range(4) if mask[i][j]]
        actions.append(rng.choice(valid) if valid else 0)
    return actions


def greedy_cheapest(obs: dict, **_kwargs) -> list[int]:
    """
    Charge at the highest valid level during cheap periods,
    lowest during expensive periods.
    """
    num_ports = obs["num_ports"]
    mask = obs["action_mask"]
    state = obs["state"]

    # Tariff is at index 5*N
    tariff = state[5 * num_ports] if len(state) > 5 * num_ports else 0.3

    actions = []
    for i in range(num_ports):
        port_mask = mask[i]
        if tariff > 0.6:
            # Peak — use LOW if available
            for level in [1, 2, 0]:
                if port_mask[level]:
                    actions.append(level)
                    break
            else:
                actions.append(0)
        elif tariff > 0.3:
            # Mid — use MEDIUM
            for level in [2, 3, 1]:
                if port_mask[level]:
                    actions.append(level)
                    break
            else:
                actions.append(0)
        else:
            # Off-peak — charge at HIGH
            for level in [3, 2, 1]:
                if port_mask[level]:
                    actions.append(level)
                    break
            else:
                actions.append(0)
    return actions


def urgency_aware(obs: dict, **_kwargs) -> list[int]:
    """
    Prioritise vehicles with least time remaining.
    Always charge urgent vehicles at HIGH, others based on tariff.
    """
    num_ports = obs["num_ports"]
    mask = obs["action_mask"]
    state = obs["state"]
    N = num_ports

    tariff = state[5 * N] if len(state) > 5 * N else 0.3

    actions = []
    for i in range(N):
        port_mask = mask[i]

        # Check if port has a vehicle that needs charging
        occupied = state[3 * N + i] > 0.5 if len(state) > 3 * N + i else False
        time_remaining = state[2 * N + i] if len(state) > 2 * N + i else 1.0
        current_soc = state[i] if len(state) > i else 0.0
        target_soc = state[N + i] if len(state) > N + i else 0.0

        if not occupied or current_soc >= target_soc:
            actions.append(0)
            continue

        # Urgency: time_remaining < 0.2 means departing soon
        if time_remaining < 0.2:
            # Urgent — charge at maximum
            for level in [3, 2, 1]:
                if port_mask[level]:
                    actions.append(level)
                    break
            else:
                actions.append(0)
        elif tariff > 0.6:
            # Expensive — charge at LOW
            for level in [1, 2, 0]:
                if port_mask[level]:
                    actions.append(level)
                    break
            else:
                actions.append(0)
        else:
            # Normal — charge at HIGH
            for level in [3, 2, 1]:
                if port_mask[level]:
                    actions.append(level)
                    break
            else:
                actions.append(0)

    return actions


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, Callable] = {
    "random": random_agent,
    "greedy_cheapest": greedy_cheapest,
    "urgency_aware": urgency_aware,
}


def run_episode(
    strategy_fn: Callable,
    task_id: str,
    seed: int = 42,
) -> dict:
    """Run a single episode with the given strategy."""
    env = ChargingEnvironment()
    obs_model = env.reset(task_id=task_id, seed=seed)
    obs = obs_model.model_dump()

    total_reward = 0.0
    steps = 0
    rng = stdlib_random.Random(seed)

    for step_num in range(1, MAX_STEPS + 1):
        if env.is_done:
            break

        if strategy_fn == random_agent:
            actions = strategy_fn(obs, rng=rng)
        else:
            actions = strategy_fn(obs)

        action = Action(actions=actions)
        obs_model, reward, done, info = env.step(action)
        obs = obs_model.model_dump()
        total_reward += reward.value
        steps = step_num

        if done:
            break

    grade = env.grade()
    return {
        "task": task_id,
        "score": grade["score"],
        "breakdown": grade.get("breakdown", {}),
        "total_reward": round(total_reward, 4),
        "steps": steps,
    }


def run_all_tasks(
    strategy: str = "urgency_aware",
    seed: int = 42,
) -> dict:
    """Run all three tasks with a strategy and return aggregate results."""
    strategy_fn = STRATEGIES.get(strategy, urgency_aware)
    results = {}
    total_score = 0.0

    for task_id in ("easy", "medium", "hard"):
        result = run_episode(strategy_fn, task_id, seed=seed)
        results[task_id] = result
        total_score += result["score"]

    avg_score = round(total_score / 3.0, 4)
    return {
        "strategy": strategy,
        "seed": seed,
        "results": results,
        "total_score": round(total_score, 4),
        "average_score": avg_score,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Run all strategies and print results."""
    print("=" * 70)
    print("  EV Charging Scheduler — Heuristic Baselines")
    print("=" * 70)

    for strategy_name in STRATEGIES:
        result = run_all_tasks(strategy=strategy_name)
        print(f"\n{'─' * 50}")
        print(f"  Strategy: {strategy_name}")
        print(f"  Average Score: {result['average_score']:.4f}")
        print(f"{'─' * 50}")
        for task_id, task_result in result["results"].items():
            print(f"  {task_id:8s}  score={task_result['score']:.4f}  "
                  f"reward={task_result['total_reward']:+.3f}  "
                  f"steps={task_result['steps']}")
            if task_result.get("breakdown"):
                for k, v in task_result["breakdown"].items():
                    print(f"             {k}: {v:.4f}")

    print(f"\n{'=' * 70}")
    print("  JSON output")
    print("=" * 70)
    all_results = {}
    for strategy_name in STRATEGIES:
        all_results[strategy_name] = run_all_tasks(strategy=strategy_name)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
