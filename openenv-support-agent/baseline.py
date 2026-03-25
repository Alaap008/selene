"""
Baseline inference script for the Customer Service Agent OpenEnv environment.
Uses the OpenAI API (gpt-4o) to run a model agent against all 3 tasks.
Reads OPENAI_API_KEY from environment variables.
Produces a reproducible baseline score.
"""

import os
import sys
import requests
import json
import time
from typing import Dict, Any

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")

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


def get_action_from_llm(client, obs: dict, messages: list) -> dict:
    """Ask the LLM for the next action given the current observation."""
    obs_str = json.dumps(obs, indent=2)
    messages.append({"role": "user", "content": f"Current Observation:\n```json\n{obs_str}\n```\nWhat is your next action? Output ONLY the JSON action object."})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,  # deterministic for reproducibility
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return json.loads(content)
    except Exception as e:
        print(f"  LLM Error: {e}", file=sys.stderr)
        return {"action_type": "close_ticket", "resolution": f"LLM error: {e}", "resolution_code": "resolved"}


def run_task(client, task_id: str, seed: int = 42) -> Dict[str, Any]:
    """Run a single task episode and return the grader score."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    # Reset
    res = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": seed})
    res.raise_for_status()
    obs = res.json()
    print(f"  Ticket: {obs['ticket_id']} — {obs['customer_request'][:80]}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    done = False
    step_count = 0
    max_steps = 15  # leave some buffer before env's 20-step hard limit
    cumulative_reward = 0.0

    while not done and step_count < max_steps:
        step_count += 1
        action_dict = get_action_from_llm(client, obs, messages)
        print(f"  Step {step_count}: {action_dict.get('action_type', '?')} ", end="")

        try:
            step_res = requests.post(f"{BASE_URL}/step", json=action_dict)
            if step_res.status_code != 200:
                print(f"— ERROR {step_res.status_code}")
                obs["last_api_response"] = f"Server error: {step_res.text}"
                continue

            step_data = step_res.json()
            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            cumulative_reward += reward["value"]
            print(f"— reward: {reward['value']:+.4f} ({reward['reason']})")

        except Exception as e:
            print(f"— EXCEPTION: {e}")
            break

    # Get final grade
    time.sleep(0.3)
    grade_res = requests.get(f"{BASE_URL}/grader")
    grade_data = grade_res.json()
    print(f"\n  Final Grade: {grade_data.get('score', 0.0):.4f}")
    if "breakdown" in grade_data:
        for k, v in grade_data["breakdown"].items():
            print(f"    {k}: {v}")
    print(f"  Cumulative Reward: {cumulative_reward:+.4f}")

    grade_data["cumulative_reward"] = round(cumulative_reward, 4)
    return grade_data


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(json.dumps({"error": "OPENAI_API_KEY not set"}))
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    results = {}
    total_score = 0.0

    for task_id in ["easy", "medium", "hard"]:
        try:
            grade = run_task(client, task_id, seed=42)
            results[task_id] = grade
            total_score += grade.get("score", 0.0)
        except Exception as e:
            print(f"\n  FAILED task {task_id}: {e}", file=sys.stderr)
            results[task_id] = {"score": 0.0, "error": str(e)}

    avg_score = round(total_score / 3.0, 4)
    print(f"\n{'='*60}")
    print(f"  BASELINE COMPLETE — Average Score: {avg_score}")
    print(f"{'='*60}")

    final = {"baseline_results": results, "total_score": round(total_score, 4), "average_score": avg_score}
    print("\n--- BASELINE_JSON ---")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
