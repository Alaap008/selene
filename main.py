import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from baseline import (
    LocalEnvironmentClient,
    create_openai_client,
    run_baseline as execute_baseline,
)
from environment import SupportEnvironment
from models import Action, Observation, Reward, Info

load_dotenv(Path(__file__).with_name(".env.local"), override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("openenv")

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("OpenEnv CustomerServiceAgent v1.0.0 started")
    logger.info("Active sessions: %d  OPENAI_API_KEY set: %s", len(_session_envs), bool(os.environ.get("OPENAI_API_KEY")))
    yield

app = FastAPI(
    title="Customer Service Agent OpenEnv",
    description="A real-world OpenEnv environment simulating customer service ticket resolution.",
    version="1.0.0",
    lifespan=lifespan,
)

SESSION_COOKIE_NAME = "openenv_session_id"
_DEFAULT_SESSION_ID = "default"
_session_lock = threading.Lock()
_session_envs: dict[str, SupportEnvironment] = {_DEFAULT_SESSION_ID: SupportEnvironment()}


# ---- Response / Request models ----

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Info


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


def _get_or_create_session_id(request: Request) -> tuple[str, bool]:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        return session_id, False
    return str(uuid.uuid4()), True


def _get_env_for_session(session_id: str) -> SupportEnvironment:
    with _session_lock:
        env = _session_envs.get(session_id)
        if env is None:
            env = SupportEnvironment()
            _session_envs[session_id] = env
        return env


# ---- Core OpenEnv endpoints ----

@app.post("/step", response_model=StepResponse)
async def step(action: Action, request: Request):
    """Execute one agent action and return (observation, reward, done, info)."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME, _DEFAULT_SESSION_ID)
    env = _get_env_for_session(session_id)
    obs, reward, done, info = env.step(action)
    detail = ""
    if action.action_type == "call_api":
        detail = f" {action.method} {action.endpoint}"
    elif action.action_type == "send_message":
        detail = f" \"{(action.message or '')[:60]}...\""
    elif action.action_type == "close_ticket":
        detail = f" code={action.resolution_code}"
    logger.info(
        "STEP  session=%s  ticket=%s  action=%s%s  reward=%+.2f  done=%s",
        session_id[:8], env.ticket.get("id", "?"), action.action_type, detail,
        reward.value, done,
    )
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.post("/reset", response_model=Observation)
async def reset(
    request: Request,
    response: Response,
    request_body: Optional[ResetRequest] = None,
):
    """Reset environment to a fresh episode for the given task."""
    task_id = request_body.task_id if request_body else "easy"
    seed = request_body.seed if request_body else None
    session_id, is_new = _get_or_create_session_id(request)
    if is_new:
        response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True)
        logger.info("SESSION created session=%s", session_id[:8])
    env = _get_env_for_session(session_id)
    obs = env.reset(task_id=task_id, seed=seed)
    logger.info(
        "RESET session=%s  task=%s  seed=%s  ticket=%s  variant=%s  customer=%s",
        session_id[:8], task_id, seed, obs.ticket_id,
        env.variant.get("id", "?"), obs.customer_name,
    )
    return obs


@app.get("/state", response_model=Observation)
async def state(request: Request):
    """Return the current observation without advancing the episode."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME, _DEFAULT_SESSION_ID)
    env = _get_env_for_session(session_id)
    return env.get_state()


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


@app.get("/health")
async def health():
    """OpenEnv-compatible health check."""
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata():
    """OpenEnv-compatible metadata endpoint."""
    return {
        "name": "CustomerServiceAgent",
        "description": "A real-world customer support environment with order lookup, refunds, and fraud handling.",
    }


@app.get("/schema")
async def schema():
    """Return action, observation, and state schemas."""
    observation_schema = Observation.model_json_schema()
    return {
        "action": Action.model_json_schema(),
        "observation": observation_schema,
        "state": observation_schema,
    }


@app.post("/mcp")
async def mcp():
    """Minimal JSON-RPC-compatible MCP endpoint for validator reachability."""
    return {"jsonrpc": "2.0", "id": None, "result": {"status": "ok"}}


@app.get("/grader")
async def get_grader(request: Request):
    """Return grader score for the current (completed) episode."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME, _DEFAULT_SESSION_ID)
    env = _get_env_for_session(session_id)
    result = env.grade()
    logger.info(
        "GRADE session=%s  task=%s  score=%.4f  breakdown=%s",
        session_id[:8], result.get("task", "?"), result.get("score", 0.0),
        result.get("breakdown", {}),
    )
    return result


@app.post("/baseline")
async def run_baseline_endpoint():
    """Run the baseline agent against an isolated in-process environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    seed = int(os.getenv("OPENAI_BASELINE_SEED", "42"))
    logger.info("BASELINE started  model=%s  seed=%d", model, seed)
    client = create_openai_client(api_key)
    baseline_env = LocalEnvironmentClient(SupportEnvironment())
    try:
        result = execute_baseline(
            llm_client=client,
            env_client=baseline_env,
            model=model,
            seed=seed,
            fail_on_error=True,
        )
        logger.info("BASELINE complete  avg_score=%.4f  results=%s", result["average_score"], {
            t: r.get("score", 0.0) for t, r in result.get("baseline_results", {}).items()
        })
        return result
    except RuntimeError as exc:
        logger.error("BASELINE failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "environment": "CustomerServiceAgent", "version": "1.0.0"}
