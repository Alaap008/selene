"""
FastAPI server for the Customer Service Agent OpenEnv environment.
Exposes: /step, /reset, /state, /tasks, /grader, /baseline, /
"""

import subprocess
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from models import Action, Observation, Reward, Info
from environment import SupportEnvironment

app = FastAPI(
    title="Customer Service Agent OpenEnv",
    description="A real-world OpenEnv environment simulating customer service ticket resolution.",
    version="1.0.0",
)

# Global environment instance (single-session, suitable for hackathon evaluation)
session_env = SupportEnvironment()


# ---- Response / Request models ----

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Info


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


# ---- Core OpenEnv endpoints ----

@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    """Execute one agent action and return (observation, reward, done, info)."""
    obs, reward, done, info = session_env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.post("/reset", response_model=Observation)
async def reset(request: Optional[ResetRequest] = None):
    """Reset environment to a fresh episode for the given task."""
    task_id = request.task_id if request else "easy"
    seed = request.seed if request else None
    return session_env.reset(task_id=task_id, seed=seed)


@app.get("/state", response_model=Observation)
async def state():
    """Return the current observation without advancing the episode."""
    return session_env.get_state()


# ---- Challenge-required endpoints ----

@app.get("/tasks")
async def get_tasks():
    """Return list of tasks and the action schema."""
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Look up an order status and provide tracking info to the customer.",
                "difficulty": "easy",
            },
            {
                "id": "medium",
                "description": "Process a standard refund after verifying policy compliance.",
                "difficulty": "medium",
            },
            {
                "id": "hard",
                "description": "Handle complex cases: fraud-flagged accounts, partial refunds past the return window, or escalation.",
                "difficulty": "hard",
            },
        ],
        "action_schema": Action.model_json_schema(),
        "observation_schema": Observation.model_json_schema(),
    }


@app.get("/grader")
async def get_grader():
    """Return grader score for the current (completed) episode."""
    return session_env.grade()


@app.post("/baseline")
async def run_baseline():
    """Trigger the baseline inference script and return scores."""
    env = os.environ.copy()
    env["OPENENV_BASE_URL"] = "http://localhost:8000"
    result = subprocess.run(
        ["python", "baseline.py"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "environment": "CustomerServiceAgent", "version": "1.0.0"}
