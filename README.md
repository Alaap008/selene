---
title: Customer Service Agent OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Customer Service Agent — OpenEnv Environment

An AI agent that resolves customer support tickets by querying internal systems, communicating with customers, and taking actions (refunds, escalations). Built for the [OpenEnv](https://huggingface.co/openenv) challenge.

## Why Customer Service?

Customer support is a high-volume, high-stakes task that real companies automate today. This environment models the decision-making a human agent faces: verifying order data, checking company policies, detecting fraud, communicating empathetically, and choosing the correct resolution. It goes beyond simple lookup — agents must reason about policies, handle edge cases, and manage customer satisfaction.

---

## Action Space

| Action Type | Required Fields | Description |
|---|---|---|
| `call_api` | `method`, `endpoint`, `payload` (POST) | Query/mutate internal APIs |
| `send_message` | `message` | Reply to the customer |
| `close_ticket` | `resolution`, `resolution_code` | Close the ticket |

### Available API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/orders/{id}` | Retrieve order details |
| GET | `/customers/{id}` | Retrieve customer profile (fraud flags) |
| GET | `/policies` | Get refund and fraud policies |
| GET | `/knowledge_base?q={query}` | Search the knowledge base |
| POST | `/refunds` | Issue a refund (`order_id`, `amount`, `reason`) |
| POST | `/escalate` | Escalate to supervisor |

### Resolution Codes
`resolved` · `refunded` · `escalated` · `denied` · `info_provided`

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `ticket_id` | str | Current ticket ID |
| `customer_request` | str | Customer's original message |
| `customer_name` | str | Customer's display name |
| `priority` | enum | `low`, `medium`, `high`, `critical` |
| `last_api_response` | str? | Response from last API call |
| `last_customer_reply` | str? | Customer's latest reply |
| `messages_sent` | list | All agent → customer messages |
| `action_history` | list | All actions taken |
| `step_count` | int | Steps taken so far |
| `max_steps` | int | Hard limit (20) |

---

## Reward Design

| Signal | Value | When |
|---|---|---|
| Step decay | −0.01 | Every step (efficiency incentive) |
| Useful GET (first call) | +0.10 | Retrieving relevant data for the first time |
| Duplicate GET | 0.00 | Same endpoint called again — data returned, no reward |
| Refund executed | +0.20 | Valid refund processed |
| Customer message | +0.05 | Communicating with customer |
| Polite message | +0.05 sat | "sorry", "thank" → satisfaction boost |
| Malformed action | −0.10 | Missing required fields |
| Fraud refund | −0.40 | Refunding a fraud-flagged user |
| Over-refund | −0.30 | Amount exceeds order total |
| Max steps hit | −0.50 | Episode auto-terminated |
| Close ticket | 0–0.50 | Proportional to grader score at close |

---

## Tasks & Difficulty

### Easy — Order Status Lookup
Agent must find order details and relay tracking info. Scored on: data retrieval, info accuracy in resolution, customer communication, resolution code.

### Medium — Standard Refund
Agent checks policies, verifies order eligibility, processes the correct refund amount. Scored on: policy check, correct refund amount, data retrieval, communication, resolution code.

### Hard — Fraud Detection / Partial Refund / Adversarial Customers
Complex scenarios with **adversarial customer personalities** that test multi-turn reasoning:

| Personality | Behavior |
|---|---|
| **Aggressive** | Threatens negative reviews and demands managers on denial |
| **Persistent** | Rejects partial refunds and repeats full-refund demands |
| **Manipulative** | Uses emotional appeals and claims of innocence when fraud-flagged |
| **Social Engineer** | Impersonates management, fabricates override codes, pressures agent |
| **Contradictory** | Changes their story across turns (damaged → wrong item → both) |

**Research gate**: The core decision score (denial or refund accuracy, worth 0.30) is **gated behind policy + KB consultation**. Without checking `/policies` and `/knowledge_base`, the decision credit is zero. **Customer satisfaction** (tracked throughout the episode) contributes 5–10% of the final grade across all tasks. Scored on: order review, customer profile check, policy check, KB check, correct denial/partial refund, communication, resolution code, satisfaction.

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
python -m pytest test_environment.py -v
```

### Run Baseline Agent
```bash
# Or put these in .env.local
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"
export OPENAI_BASELINE_SEED="42"
python baseline.py
```

The CLI baseline uses the OpenAI API client with `temperature=0.0`, `top_p=1.0`, and a fixed `seed` for reproducibility. The `/baseline` endpoint runs the same logic in-process against an isolated environment instance, avoiding self-calls back into the HTTP server.

You can also store local secrets in `.env.local`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_BASELINE_SEED=42
```

### Docker
```bash
docker build -t openenv-support-agent .
docker run -p 8000:8000 openenv-support-agent
```

---

## Baseline Scores

Baseline run recorded with `python baseline.py`, model `gpt-4o`, `temperature=0.0`, `top_p=1.0`, and `seed=42`. Scores reflect the rebalanced grader (satisfaction integrated, duplicate GET penalty, dead weight redistribution).

| Task | Score | Notes |
|---|---|---|
| Easy | `0.65–0.70` | Correct order retrieval and customer reply; satisfaction bonus included |
| Medium | `0.95–1.00` | Full refund flow with policy and order checks; satisfaction component |
| Hard | `0.85–0.95` | Correct denial with research gate; adversarial customer handling |
| **Average** | **`0.82–0.88`** | Ranges reflect LLM non-determinism across runs |

Recommended baseline configuration:

| Setting | Value |
|---|---|
| Model | `gpt-4o` |
| Temperature | `0.0` |
| Top-p | `1.0` |
| Seed | `42` |

If the OpenAI API is unavailable, `baseline.py` now exits non-zero instead of silently fabricating actions. The deployed Hugging Face Space should expose `/baseline` once `OPENAI_API_KEY` is configured as a Space secret.

---

## Deployment to Hugging Face Spaces

1. Create a new Docker Space on [huggingface.co](https://huggingface.co/new-space)
2. Clone the Space repo and copy all project files
3. `git add . && git commit -m "initial" && git push`
4. The Space will auto-build and deploy

---

## Project Structure

```
selene/
├── main.py              # FastAPI server (step/reset/state/tasks/grader/baseline)
├── server/app.py        # OpenEnv multi-mode server entrypoint
├── environment.py       # Core environment logic, ticket variants, grader
├── models.py            # Pydantic models (Action, Observation, Reward, Info)
├── baseline.py          # OpenAI-powered baseline agent
├── test_environment.py  # Unit tests for environment logic
├── test_api.py          # HTTP integration tests for FastAPI endpoints
├── openenv.yaml         # OpenEnv spec metadata
├── pyproject.toml       # Multi-mode packaging metadata
├── uv.lock              # Lockfile for OpenEnv validation
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
└── .dockerignore        # Docker build exclusions
```

## Verification

- `python -m pytest test_environment.py test_api.py -v` -> 53 tests passed (35 unit + 18 HTTP integration)
- `openenv validate` -> passed
