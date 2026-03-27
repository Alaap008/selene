"""
HTTP integration tests for the FastAPI endpoints.
Run with: python -m pytest test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture()
def client():
    return TestClient(app)


class TestHealthAndMetadata:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_metadata(self, client):
        r = client.get("/metadata")
        assert r.status_code == 200
        body = r.json()
        assert "name" in body
        assert "description" in body

    def test_schema_returns_all_keys(self, client):
        r = client.get("/schema")
        assert r.status_code == 200
        body = r.json()
        for key in ("action", "observation", "state"):
            assert key in body
            assert "properties" in body[key]

    def test_mcp(self, client):
        r = client.post("/mcp")
        assert r.status_code == 200
        assert r.json()["result"]["status"] == "ok"


class TestTasks:
    def test_returns_three_tasks(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        body = r.json()
        assert len(body["tasks"]) == 3
        assert "action_schema" in body
        assert "observation_schema" in body

    def test_tasks_have_required_fields(self, client):
        tasks = client.get("/tasks").json()["tasks"]
        for task in tasks:
            assert {"id", "description", "difficulty"} <= set(task.keys())

    def test_difficulty_levels(self, client):
        tasks = client.get("/tasks").json()["tasks"]
        difficulties = {t["difficulty"] for t in tasks}
        assert difficulties == {"easy", "medium", "hard"}


class TestResetStepState:
    def test_reset_returns_observation(self, client):
        r = client.post("/reset", json={"task_id": "easy", "seed": 42})
        assert r.status_code == 200
        body = r.json()
        assert body["ticket_id"] != ""
        assert body["step_count"] == 0
        assert body["max_steps"] == 20

    def test_reset_defaults_to_easy(self, client):
        r = client.post("/reset")
        assert r.status_code == 200
        assert r.json()["ticket_id"] != ""

    def test_state_reflects_reset(self, client):
        client.post("/reset", json={"task_id": "medium", "seed": 42})
        r = client.get("/state")
        assert r.status_code == 200
        body = r.json()
        assert "refund" in body["customer_request"].lower() or "return" in body["customer_request"].lower()

    def test_step_returns_full_response(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={
            "action_type": "call_api",
            "method": "GET",
            "endpoint": "/policies",
        })
        assert r.status_code == 200
        body = r.json()
        for key in ("observation", "reward", "done", "info"):
            assert key in body
        assert body["done"] is False

    def test_step_rejects_invalid_action_type(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={"action_type": "fly_to_moon"})
        assert r.status_code == 422


class TestGraderEndpoint:
    def test_grader_before_done(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.get("/grader")
        assert r.status_code == 200
        assert r.json()["score"] == 0.0

    def test_grader_after_close(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        client.post("/step", json={
            "action_type": "close_ticket",
            "resolution": "Done.",
            "resolution_code": "resolved",
        })
        r = client.get("/grader")
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["score"] <= 1.0
        assert "breakdown" in body


class TestFullEpisodeFlow:
    def test_easy_episode(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})

        r = client.post("/step", json={
            "action_type": "call_api", "method": "GET", "endpoint": "/policies",
        })
        assert r.json()["done"] is False

        r = client.post("/step", json={
            "action_type": "send_message",
            "message": "Let me check your order for you.",
        })
        assert r.json()["observation"]["last_customer_reply"] is not None

        r = client.post("/step", json={
            "action_type": "close_ticket",
            "resolution": "Resolved.",
            "resolution_code": "info_provided",
        })
        assert r.json()["done"] is True

        grade = client.get("/grader").json()
        assert 0.0 < grade["score"] <= 1.0

    def test_hard_episode_with_research(self, client):
        """Full hard episode with policy + KB produces high score."""
        client.post("/reset", json={"task_id": "hard", "seed": 7})

        for action in [
            {"action_type": "call_api", "method": "GET", "endpoint": "/orders/O-302"},
            {"action_type": "call_api", "method": "GET", "endpoint": "/customers/C-140"},
            {"action_type": "call_api", "method": "GET", "endpoint": "/policies"},
            {"action_type": "call_api", "method": "GET", "endpoint": "/knowledge_base?q=fraud"},
            {"action_type": "send_message", "message": "Your refund has been denied per policy."},
        ]:
            r = client.post("/step", json=action)
            assert r.status_code == 200

        r = client.post("/step", json={
            "action_type": "close_ticket",
            "resolution": "Denied — fraud-flagged per policy.",
            "resolution_code": "denied",
        })
        assert r.json()["done"] is True

        grade = client.get("/grader").json()
        assert grade["score"] == 1.0

    def test_hard_episode_without_research_is_capped(self, client):
        """Hard episode without policy/KB consultation is capped below 0.50."""
        client.post("/reset", json={"task_id": "hard", "seed": 7})

        client.post("/step", json={
            "action_type": "call_api", "method": "GET", "endpoint": "/orders/O-302",
        })
        client.post("/step", json={
            "action_type": "send_message", "message": "Denied.",
        })
        client.post("/step", json={
            "action_type": "close_ticket",
            "resolution": "Denied.",
            "resolution_code": "denied",
        })

        grade = client.get("/grader").json()
        assert grade["score"] < 0.50


class TestSessionIsolation:
    def test_two_clients_have_independent_state(self):
        c1 = TestClient(app)
        c2 = TestClient(app)

        c1.post("/reset", json={"task_id": "easy", "seed": 42})
        c2.post("/reset", json={"task_id": "hard", "seed": 7})

        s1 = c1.get("/state").json()
        s2 = c2.get("/state").json()

        assert s1["ticket_id"] != s2["ticket_id"]
        assert s1["priority"] != s2["priority"] or s1["customer_request"] != s2["customer_request"]
