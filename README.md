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
| Useful GET | +0.10 | Retrieving relevant data |
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

### Hard — Fraud Detection / Partial Refund
Complex scenarios: fraud-flagged customers (deny refund), partial refunds past the return window, or escalation needs. Scored on: customer profile check, correct denial/partial refund, policy & KB usage, communication, resolution code.

---

## Setup & Usage

### Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Run Tests
```bash
pip install pytest
python -m pytest test_environment.py -v
```

### Run Baseline Agent
```bash
export OPENAI_API_KEY="sk-..."
python baseline.py
```

### Docker
```bash
docker build -t openenv-support-agent .
docker run -p 8000:8000 openenv-support-agent
```

---

## Baseline Scores

| Task | Score | Notes |
|---|---|---|
| Easy | ~0.85–1.0 | GPT-4o reliably retrieves and reports order info |
| Medium | ~0.70–0.90 | Usually processes refund correctly; occasionally skips policy check |
| Hard | ~0.50–0.75 | Fraud detection is inconsistent; partial refund logic varies |
| **Average** | **~0.70** | |

*(Scores from `baseline.py` with `seed=42`, `temperature=0.0`, model `gpt-4o`)*

---

## Deployment to Hugging Face Spaces

1. Create a new Docker Space on [huggingface.co](https://huggingface.co/new-space)
2. Clone the Space repo and copy all project files
3. `git add . && git commit -m "initial" && git push`
4. The Space will auto-build and deploy

---

## Project Structure

```
openenv-support-agent/
├── main.py              # FastAPI server (step/reset/state/tasks/grader/baseline)
├── environment.py       # Core environment logic, ticket variants, grader
├── models.py            # Pydantic models (Action, Observation, Reward, Info)
├── baseline.py          # OpenAI-powered baseline agent
├── test_environment.py  # pytest test suite
├── openenv.yaml         # OpenEnv spec metadata
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
└── .dockerignore        # Docker build exclusions
```
