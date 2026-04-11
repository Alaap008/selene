from __future__ import annotations

"""
inference.py — OpenEnv EV Charging Scheduler
=============================================
Mandatory entrypoint for the OpenEnv evaluation platform.

CRITICAL: This runs the environment IN-PROCESS — no HTTP dependency.
This was the #1 evaluation failure risk in the old version.

Environment variables (all required at runtime):
    API_BASE_URL       LLM API endpoint  (default: https://api.openai.com/v1)
    MODEL_NAME         Model identifier  (default: gpt-4o-mini)
    HF_TOKEN           Hugging Face / OpenAI API key
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment import ChargingEnvironment, POWER_LEVELS
from models import Action

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "ev_charging_scheduler"
DEFAULT_TASKS = ("easy", "medium", "hard")
DEFAULT_SEED = 42
MAX_STEPS = 48
TEMPERATURE = 0.0

SYSTEM_PROMPT = """\
You are an RL agent controlling an EV Charging Station.

Each step, you receive a JSON observation with:
- `state`: a numeric vector describing per-port SoC, targets, time remaining, \
port status, tariff, grid load, time of day, and steps remaining.
- `action_mask`: an N×4 boolean mask — action_mask[i][j] = true means level j \
is valid for port i.
- `num_ports`: number of charger ports (N).

You must output a JSON object with:
  {"actions": [level_0, level_1, ..., level_{N-1}]}

Where each level is an integer 0–3:
  0 = OFF    (0 kW)
  1 = LOW    (3.6 kW)
  2 = MEDIUM (7 kW)
  3 = HIGH   (11 kW)

## Objective
Minimise energy cost and lateness while respecting grid limits.
- Charge vehicles to their target SoC before their departure time.
- Prefer cheaper tariff periods (lower tariff in the state vector).
- Don't exceed grid capacity (the mask will help, but be efficient).
- Every vehicle that departs at target SoC earns a completion bonus.

## Strategy Tips
- If tariff > 0.7, prefer LOW or OFF unless a vehicle is urgently departing.
- If a vehicle's time_remaining is low, charge at HIGH regardless of tariff.
- Respect the action mask — masked actions will be forced to OFF.
- Output ONLY the JSON action object. No explanations.
"""


# ---------------------------------------------------------------------------
# Structured stdout logging ([START], [STEP], [END])
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# In-process environment client (NO HTTP dependency)
# ---------------------------------------------------------------------------

class LocalEnvironmentClient:
    """Runs the environment in-process for self-contained inference."""

    def __init__(self):
        self.env = ChargingEnvironment()

    def reset(self, task_id: str, seed: int) -> dict:
        obs = self.env.reset(task_id=task_id, seed=seed)
        return obs.model_dump()

    def step(self, action_list: list[int]) -> dict:
        action = Action(actions=action_list)
        obs, reward, done, info = self.env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info.model_dump(),
        }

    def grade(self) -> dict:
        return self.env.grade()


# ---------------------------------------------------------------------------
# Greedy fallback heuristic
# ---------------------------------------------------------------------------

def greedy_action(obs: dict) -> list[int]:
    """Simple heuristic: charge at highest valid level, prefer LOW when tariff is high."""
    num_ports = obs["num_ports"]
    mask = obs["action_mask"]
    state = obs["state"]

    # Extract tariff (at index 5*N)
    tariff = state[5 * num_ports] if len(state) > 5 * num_ports else 0.3

    actions = []
    for i in range(num_ports):
        port_mask = mask[i]
        if tariff > 0.7:
            # Prefer lower power during peak
            for level in [1, 2, 3]:
                if port_mask[level]:
                    actions.append(level)
                    break
            else:
                actions.append(0)
        else:
            # Prefer highest power during off-peak
            for level in [3, 2, 1]:
                if port_mask[level]:
                    actions.append(level)
                    break
            else:
                actions.append(0)
    return actions


# ---------------------------------------------------------------------------
# LLM action generation
# ---------------------------------------------------------------------------

def get_action_from_llm(
    client: OpenAI,
    obs: dict,
    messages: list,
    last_reward: Optional[dict] = None,
) -> list[int]:
    """Ask the LLM for charging actions given the current observation."""
    num_ports = obs["num_ports"]

    # Compact observation for the LLM
    compact_obs = {
        "num_ports": num_ports,
        "state": [round(x, 3) for x in obs["state"]],
        "action_mask": obs["action_mask"],
        "step": obs["step_count"],
        "max_steps": obs["max_steps"],
    }

    feedback = ""
    if last_reward is not None:
        feedback = (
            f"Previous reward: {last_reward.get('value', 0):+.3f} "
            f"({last_reward.get('reason', '')})\n"
        )

    messages.append({
        "role": "user",
        "content": (
            f"{feedback}"
            f"Observation:\n```json\n{json.dumps(compact_obs)}\n```\n"
            f"Output ONLY the JSON action object."
        ),
    })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            top_p=1.0,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        parsed = json.loads(content)
        actions = parsed.get("actions", [0] * num_ports)
        return [int(a) for a in actions]
    except Exception:
        # Fall back to greedy heuristic
        return greedy_action(obs)


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(
    env_client: LocalEnvironmentClient,
    llm_client: Optional[OpenAI],
    task_id: str,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Run one episode of the given task."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_client.reset(task_id=task_id, seed=seed)
        messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False
        last_reward: Optional[dict] = None

        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            if llm_client is not None:
                actions = get_action_from_llm(llm_client, obs, messages, last_reward)
            else:
                actions = greedy_action(obs)

            step_data = env_client.step(actions)
            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]

            reward_value = reward["value"]
            rewards.append(reward_value)
            steps_taken = step_num
            last_reward = reward

            action_str = f"[{','.join(str(a) for a in actions)}]"
            log_step(
                step=step_num,
                action=action_str,
                reward=reward_value,
                done=done,
                error=None,
            )

            if done:
                break

        grade_data = env_client.grade()
        score = grade_data.get("score", 0.001)
        success = score > 0.001

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)

    finally:
        score = round(min(max(score, 0.001), 0.999), 4)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    llm_client: Optional[OpenAI] = None

    if HF_TOKEN:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    else:
        print("[DEBUG] HF_TOKEN not set — using greedy heuristic fallback.", flush=True)

    env_client = LocalEnvironmentClient()
    results: Dict[str, Any] = {}
    total_score = 0.0

    for task_id in DEFAULT_TASKS:
        result = run_task(env_client, llm_client, task_id, seed=DEFAULT_SEED)
        results[task_id] = result
        total_score += result["score"]

    avg_score = total_score / len(DEFAULT_TASKS) if DEFAULT_TASKS else 0.0
    print(f"[DEBUG] All tasks complete — average score: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
