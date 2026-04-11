from __future__ import annotations

"""
Baseline agents for the EV Charging Scheduler.
Includes random, greedy, and LLM-based agents.

Runnable as: python baseline.py
The /baseline endpoint uses the greedy heuristic (no LLM needed).
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

from environment import ChargingEnvironment, MAX_STEPS
from models import Action
from heuristic import (
    random_agent,
    greedy_cheapest,
    urgency_aware,
    run_episode,
    run_all_tasks,
    STRATEGIES,
)

load_dotenv(Path(__file__).with_name(".env.local"), override=False)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_SEED = int(os.getenv("OPENAI_BASELINE_SEED", "42"))
DEFAULT_TASKS = ("easy", "medium", "hard")


def create_openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


class LocalEnvironmentClient:
    """In-process environment client for baselines."""

    def __init__(self, env: ChargingEnvironment | None = None):
        self.env = env or ChargingEnvironment()

    def reset(self, task_id: str, seed: int) -> dict:
        return self.env.reset(task_id=task_id, seed=seed).model_dump()

    def step(self, actions: list[int]) -> dict:
        action = Action(actions=actions)
        obs, reward, done, info = self.env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info.model_dump(),
        }

    def grade(self) -> dict:
        return self.env.grade()


def run_llm_task(
    llm_client,
    task_id: str,
    model: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run a single task using the LLM agent with greedy fallback."""
    from inference import SYSTEM_PROMPT, get_action_from_llm, greedy_action

    env_client = LocalEnvironmentClient()
    obs = env_client.reset(task_id=task_id, seed=seed)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    total_reward = 0.0
    step_count = 0
    last_reward = None

    print(f"\n  Task: {task_id}  (LLM agent)")

    while not done and step_count < MAX_STEPS:
        step_count += 1
        try:
            actions = get_action_from_llm(llm_client, obs, messages, last_reward)
        except Exception:
            actions = greedy_action(obs)

        step_data = env_client.step(actions)
        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        total_reward += reward["value"]
        last_reward = reward

        print(f"  Step {step_count}: actions={actions[:4]}  "
              f"reward={reward['value']:+.3f}")

    grade_data = env_client.grade()
    print(f"  Final Grade: {grade_data.get('score', 0.0):.4f}")

    grade_data["total_reward"] = round(total_reward, 4)
    return grade_data


def run_baseline(
    llm_client=None,
    model: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    fail_on_error: bool = True,
) -> dict:
    """Run all baseline strategies including LLM if available."""
    results = {}
    total_score = 0.0

    # First run heuristic baselines
    for strategy_name in STRATEGIES:
        strategy_result = run_all_tasks(strategy=strategy_name, seed=seed)
        results[f"heuristic_{strategy_name}"] = strategy_result
        print(f"\n  Heuristic ({strategy_name}): avg_score={strategy_result['average_score']:.4f}")

    # Then run LLM baseline if client is available
    if llm_client is not None:
        llm_results = {}
        llm_total = 0.0
        failed_tasks = []

        for task_id in DEFAULT_TASKS:
            try:
                grade = run_llm_task(llm_client, task_id, model=model, seed=seed)
                llm_results[task_id] = grade
                llm_total += grade.get("score", 0.0)
            except Exception as e:
                print(f"\n  FAILED LLM task {task_id}: {e}", file=sys.stderr)
                llm_results[task_id] = {"score": 0.0, "error": str(e)}
                failed_tasks.append(task_id)

        llm_avg = round(llm_total / 3.0, 4)
        results["llm"] = {
            "model": model,
            "results": llm_results,
            "average_score": llm_avg,
        }

        if failed_tasks and fail_on_error:
            raise RuntimeError(f"LLM baseline failed for: {', '.join(failed_tasks)}")
    else:
        # Use urgency-aware as the primary baseline score
        total_score = results.get("heuristic_urgency_aware", {}).get("total_score", 0.0)

    # Compute best average
    best_avg = max(
        r.get("average_score", 0.0) for r in results.values()
    )

    final = {
        "model": model if llm_client else "heuristic",
        "seed": seed,
        "baseline_results": results,
        "average_score": best_avg,
    }

    print(f"\n{'=' * 60}")
    print(f"  BASELINE COMPLETE — Best Average Score: {best_avg:.4f}")
    print(f"{'=' * 60}")
    print("\n--- BASELINE_JSON ---")
    print(json.dumps(final, indent=2))

    return final


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    llm_client = None
    if api_key:
        llm_client = create_openai_client(api_key)
    else:
        print("  OPENAI_API_KEY not set — running heuristic baselines only.")

    try:
        run_baseline(
            llm_client=llm_client,
            model=DEFAULT_MODEL,
            seed=DEFAULT_SEED,
            fail_on_error=False,
        )
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
