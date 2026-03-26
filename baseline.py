import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import requests
from dotenv import load_dotenv

from environment import SupportEnvironment

load_dotenv(Path(__file__).with_name(".env.local"), override=False)

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_SEED = int(os.getenv("OPENAI_BASELINE_SEED", "42"))
DEFAULT_TASKS = ("easy", "medium", "hard")

SYSTEM_PROMPT = """\
You are an AI Customer Service Agent. You resolve customer support tickets by:
1. Querying internal systems (orders, customers, policies, knowledge base)
2. Communicating with the customer via messages
3. Taking actions (refunds, escalations)
4. Closing the ticket with a resolution

You MUST output ONLY a valid JSON object (no markdown, no explanation) matching the action schema.

## Available Actions

### call_api — Query or mutate internal systems
- GET /orders/{order_id} — Retrieve order details
- GET /customers/{customer_id} — Retrieve customer profile (check for fraud flags)
- GET /policies — Retrieve company refund and fraud policies
- GET /knowledge_base?q={query} — Search the knowledge base for procedures
- POST /refunds — Issue a refund. Payload: {"order_id": "...", "amount": X.YZ, "reason": "..."}
- POST /escalate — Escalate to supervisor

Example: {"action_type": "call_api", "method": "GET", "endpoint": "/orders/O-100"}
Example: {"action_type": "call_api", "method": "POST", "endpoint": "/refunds", "payload": {"order_id": "O-200", "amount": 120.0, "reason": "Standard return within policy"}}

### send_message — Reply to the customer
Example: {"action_type": "send_message", "message": "I'm looking into this for you right now."}

### close_ticket — Close the ticket with a final resolution
resolution_code must be one of: "resolved", "refunded", "escalated", "denied", "info_provided"
Example: {"action_type": "close_ticket", "resolution": "Order O-100 is shipped with tracking TRK999.", "resolution_code": "info_provided"}

## Guidelines
- Always retrieve order and customer info before making decisions.
- Always check /policies and /knowledge_base for relevant procedures.
- Always send at least one message to the customer before closing.
- If a customer is fraud-flagged, DENY the refund and explain politely.
- Use the correct resolution_code when closing.
- Be efficient — avoid unnecessary steps.
"""


def create_openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


class HttpEnvironmentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_id: str, seed: int) -> dict:
        response = self.session.post(
            f"{self.base_url}/reset", json={"task_id": task_id, "seed": seed}, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: dict) -> dict:
        response = self.session.post(f"{self.base_url}/step", json=action, timeout=30)
        response.raise_for_status()
        return response.json()

    def grade(self) -> dict:
        response = self.session.get(f"{self.base_url}/grader", timeout=30)
        response.raise_for_status()
        return response.json()


class LocalEnvironmentClient:
    def __init__(self, env: SupportEnvironment | None = None):
        self.env = env or SupportEnvironment()

    def reset(self, task_id: str, seed: int) -> dict:
        return self.env.reset(task_id=task_id, seed=seed).model_dump()

    def step(self, action: dict) -> dict:
        obs, reward, done, info = self.env.step(action=self._coerce_action(action))
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info.model_dump(),
        }

    def grade(self) -> dict:
        return self.env.grade()

    @staticmethod
    def _coerce_action(action: dict):
        from models import Action

        return Action(**action)


def get_action_from_llm(client, obs: dict, messages: list, model: str, seed: int) -> dict:
    """Ask the LLM for the next action given the current observation."""
    obs_str = json.dumps(obs, indent=2)
    messages.append({"role": "user", "content": f"Current Observation:\n```json\n{obs_str}\n```\nWhat is your next action? Output ONLY the JSON action object."})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            top_p=1.0,
            seed=seed,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"LLM action generation failed: {e}") from e


def run_task(env_client, llm_client, task_id: str, model: str, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Run a single task episode and return the grader score."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    obs = env_client.reset(task_id=task_id, seed=seed)
    print(f"  Ticket: {obs['ticket_id']} — {obs['customer_request'][:80]}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    done = False
    step_count = 0
    max_steps = 15  # leave some buffer before env's 20-step hard limit
    cumulative_reward = 0.0

    while not done and step_count < max_steps:
        step_count += 1
        action_dict = get_action_from_llm(llm_client, obs, messages, model=model, seed=seed)
        print(f"  Step {step_count}: {action_dict.get('action_type', '?')} ", end="")

        try:
            step_data = env_client.step(action_dict)
            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            cumulative_reward += reward["value"]
            print(f"— reward: {reward['value']:+.4f} ({reward['reason']})")

        except Exception as e:
            raise RuntimeError(f"Environment step failed: {e}") from e

    grade_data = env_client.grade()
    print(f"\n  Final Grade: {grade_data.get('score', 0.0):.4f}")
    if "breakdown" in grade_data:
        for k, v in grade_data["breakdown"].items():
            print(f"    {k}: {v}")
    print(f"  Cumulative Reward: {cumulative_reward:+.4f}")

    grade_data["cumulative_reward"] = round(cumulative_reward, 4)
    return grade_data


def run_baseline(
    llm_client,
    env_client,
    model: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    fail_on_error: bool = True,
) -> dict:
    results = {}
    total_score = 0.0
    failed_tasks = []

    for task_id in DEFAULT_TASKS:
        try:
            grade = run_task(env_client, llm_client, task_id, model=model, seed=seed)
            results[task_id] = grade
            total_score += grade.get("score", 0.0)
        except Exception as e:
            print(f"\n  FAILED task {task_id}: {e}", file=sys.stderr)
            results[task_id] = {"score": 0.0, "error": str(e)}
            failed_tasks.append(task_id)

    avg_score = round(total_score / 3.0, 4)
    print(f"\n{'='*60}")
    print(f"  BASELINE COMPLETE — Average Score: {avg_score}")
    print(f"{'='*60}")

    final = {
        "model": model,
        "seed": seed,
        "baseline_results": results,
        "total_score": round(total_score, 4),
        "average_score": avg_score,
    }
    print("\n--- BASELINE_JSON ---")
    print(json.dumps(final, indent=2))
    if failed_tasks and fail_on_error:
        raise RuntimeError(f"Baseline failed for tasks: {', '.join(failed_tasks)}")
    return final


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(json.dumps({"error": "OPENAI_API_KEY not set"}))
        sys.exit(1)

    llm_client = create_openai_client(api_key)
    env_client = HttpEnvironmentClient(BASE_URL)
    try:
        run_baseline(
            llm_client=llm_client,
            env_client=env_client,
            model=DEFAULT_MODEL,
            seed=DEFAULT_SEED,
            fail_on_error=True,
        )
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
