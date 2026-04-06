"""
inference.py — OpenEnv Customer Service Agent
==============================================
Mandatory entrypoint for the OpenEnv evaluation platform.

Environment variables (all required at runtime):
    API_BASE_URL       LLM API endpoint  (default: https://api.openai.com/v1)
    MODEL_NAME         Model identifier  (default: gpt-4o-mini)
    HF_TOKEN           Hugging Face / OpenAI API key
    OPENENV_BASE_URL   Environment server URL (default: http://localhost:8000)
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")

BENCHMARK = "customer_service"
DEFAULT_TASKS = ("easy", "medium", "hard")
DEFAULT_SEED = 42
MAX_STEPS = 15
TEMPERATURE = 0.0

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
- Use reward feedback from the previous step to improve your next action.
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
# Environment HTTP client
# ---------------------------------------------------------------------------

class HttpEnvironmentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_id: str, seed: int) -> dict:
        resp = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        resp = self.session.post(
            f"{self.base_url}/step", json=action, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> dict:
        resp = self.session.get(f"{self.base_url}/grader", timeout=30)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Progress tracking (mirrors baseline scoring patterns)
# ---------------------------------------------------------------------------

def _extract_progress_signals(task_id: str, obs: dict) -> dict:
    action_history = obs.get("action_history", [])
    messages_sent = obs.get("messages_sent", [])

    checked_order = any("GET /orders/" in e for e in action_history)
    checked_customer = any("/customers/" in e for e in action_history)
    checked_policy = any("/policies" in e for e in action_history)
    checked_kb = any("/knowledge_base" in e for e in action_history)
    issued_refund = any("POST /refunds" in e for e in action_history)
    escalated = any("POST /escalate" in e for e in action_history)
    communicated = len(messages_sent) > 0

    progress = {
        "communicated": int(communicated),
        "order_check": int(checked_order),
        "customer_check": int(checked_customer),
        "policy_check": int(checked_policy),
        "kb_check": int(checked_kb),
        "refund_attempted": int(issued_refund),
        "escalated": int(escalated),
    }

    if task_id == "hard":
        progress["hard_research_gate_unlocked"] = int(checked_policy and checked_kb)

    return progress


def _missing_checklist(task_id: str, progress: dict) -> list[str]:
    if task_id == "easy":
        required = ["order_check", "communicated"]
    elif task_id == "medium":
        required = ["order_check", "policy_check", "refund_attempted", "communicated"]
    else:
        required = ["order_check", "customer_check", "policy_check", "kb_check", "communicated"]
    return [key for key in required if not progress.get(key)]


# ---------------------------------------------------------------------------
# LLM action generation
# ---------------------------------------------------------------------------

def get_action_from_llm(
    client: OpenAI,
    obs: dict,
    messages: list,
    task_id: str,
    last_reward: Optional[dict] = None,
) -> dict:
    obs_str = json.dumps(obs, indent=2)
    feedback_block = ""
    if last_reward is not None:
        reward_value = last_reward.get("value", 0.0)
        reward_reason = last_reward.get("reason", "")
        feedback_block = (
            "Feedback from your previous action:\n"
            f"- reward: {reward_value:+.4f}\n"
            f"- reason: {reward_reason}\n\n"
        )

    progress = _extract_progress_signals(task_id=task_id, obs=obs)
    missing = _missing_checklist(task_id=task_id, progress=progress)
    scoring_block = (
        "Scoring Pattern Progress (0/1 checks):\n"
        f"{json.dumps(progress, separators=(',', ':'))}\n"
        f"Missing high-impact checks: {', '.join(missing) if missing else 'none'}\n"
    )
    if task_id == "hard":
        scoring_block += (
            "Hard-task gate: core decision credit is locked until BOTH "
            "/policies and /knowledge_base are checked.\n"
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"{feedback_block}"
                f"{scoring_block}\n"
                f"Current Observation:\n```json\n{obs_str}\n```\n"
                "What is your next action? Output ONLY the JSON action object."
            ),
        }
    )

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
        return json.loads(content)
    except Exception as exc:
        raise RuntimeError(f"LLM action generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Action formatting for [STEP] logs
# ---------------------------------------------------------------------------

def _format_action(action_dict: dict) -> str:
    atype = action_dict.get("action_type", "unknown")
    if atype == "call_api":
        method = action_dict.get("method", "?")
        endpoint = action_dict.get("endpoint", "?")
        return f"call_api('{method} {endpoint}')"
    if atype == "send_message":
        msg = (action_dict.get("message") or "")[:50]
        return f"send_message('{msg}')"
    if atype == "close_ticket":
        code = action_dict.get("resolution_code", "resolved")
        return f"close_ticket('{code}')"
    return f"{atype}()"


# ---------------------------------------------------------------------------
# Single-task runner (one [START] … [STEP]* … [END] episode)
# ---------------------------------------------------------------------------

def run_task(
    env_client: HttpEnvironmentClient,
    llm_client: OpenAI,
    task_id: str,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_client.reset(task_id=task_id, seed=seed)
        messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False
        last_reward: Optional[dict] = None

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_action_from_llm(
                llm_client, obs, messages, task_id, last_reward
            )

            # Catch 422 validation errors so a bad action logs a -0.10 penalty
            # and the episode continues rather than aborting the task.
            try:
                step_data = env_client.step(action_dict)
            except requests.HTTPError as http_err:
                error_msg = f"invalid_action:{http_err.response.status_code}"
                log_step(
                    step=step,
                    action=_format_action(action_dict),
                    reward=-0.10,
                    done=False,
                    error=error_msg,
                )
                penalty_reward = {"value": -0.10, "reason": error_msg}
                rewards.append(-0.10)
                steps_taken = step
                last_reward = penalty_reward
                # Feed the error back to the LLM so it can self-correct
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your last action was rejected by the environment "
                        f"(HTTP {http_err.response.status_code}). "
                        "Make sure action_type, method, endpoint, resolution_code "
                        "and all required fields are valid. Try again."
                    ),
                })
                continue

            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]

            reward_value = reward["value"]
            error = None

            rewards.append(reward_value)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=_format_action(action_dict),
                reward=reward_value,
                done=done,
                error=error,
            )

            if done:
                break

        grade_data = env_client.grade()
        score = grade_data.get("score", 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("[DEBUG] HF_TOKEN / API_KEY not set — aborting.", flush=True)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env_client = HttpEnvironmentClient(OPENENV_BASE_URL)

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
