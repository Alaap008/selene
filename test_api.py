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
        assert data["environment"] == "EVChargingScheduler"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_metadata(self, client):
        r = client.get("/metadata")
        assert r.status_code == 200
        body = r.json()
        assert "name" in body
        assert body["name"] == "EVChargingScheduler"
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
        assert body["step_count"] == 0
        assert body["max_steps"] == 48
        assert body["num_ports"] == 4
        assert len(body["state"]) == 5 * 4 + 4  # 5*N + 4

    def test_reset_defaults_to_easy(self, client):
        r = client.post("/reset")
        assert r.status_code == 200
        assert r.json()["num_ports"] == 4

    def test_state_reflects_reset(self, client):
        client.post("/reset", json={"task_id": "medium", "seed": 42})
        r = client.get("/state")
        assert r.status_code == 200
        body = r.json()
        assert body["num_ports"] == 6

    def test_step_returns_full_response(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={"actions": [0, 0, 0, 0]})
        assert r.status_code == 200
        body = r.json()
        for key in ("observation", "reward", "done", "info"):
            assert key in body
        assert body["done"] is False

    def test_step_with_charging(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={"actions": [3, 3, 3, 3]})
        assert r.status_code == 200
        body = r.json()
        assert body["observation"]["step_count"] == 1
        assert "breakdown" in body["reward"]

    def test_step_action_validation(self, client):
        """Actions with out-of-range values should be clamped, not rejected."""
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={"actions": [5, -1, 3, 0]})
        assert r.status_code == 200  # Clamped, not rejected


class TestGraderEndpoint:
    def test_grader_before_done(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.get("/grader")
        assert r.status_code == 200
        assert r.json()["score"] == 0.001

    def test_grader_after_episode(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        # Step until done
        for _ in range(48):
            r = client.post("/step", json={"actions": [3, 3, 3, 3]})
            if r.json()["done"]:
                break
        r = client.get("/grader")
        assert r.status_code == 200
        body = r.json()
        assert 0.0 < body["score"] <= 1.0
        assert "breakdown" in body


class TestFullEpisodeFlow:
    def test_easy_episode(self, client):
        """Run a full easy episode with aggressive charging."""
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        done = False
        steps = 0
        while not done and steps < 48:
            r = client.post("/step", json={"actions": [3, 3, 3, 3]})
            body = r.json()
            done = body["done"]
            steps += 1
            assert r.status_code == 200

        grade = client.get("/grader").json()
        assert grade["score"] > 0.3

    def test_medium_episode_with_varied_actions(self, client):
        """Run a medium episode with varied charging levels."""
        client.post("/reset", json={"task_id": "medium", "seed": 42})
        done = False
        steps = 0
        while not done and steps < 48:
            # Alternate between HIGH and LOW
            level = 3 if steps % 2 == 0 else 1
            r = client.post("/step", json={"actions": [level] * 6})
            done = r.json()["done"]
            steps += 1

        grade = client.get("/grader").json()
        assert 0.0 < grade["score"] <= 1.0

    def test_hard_episode(self, client):
        """Run a hard episode."""
        client.post("/reset", json={"task_id": "hard", "seed": 42})
        done = False
        steps = 0
        while not done and steps < 48:
            r = client.post("/step", json={"actions": [2] * 8})
            done = r.json()["done"]
            steps += 1

        grade = client.get("/grader").json()
        assert 0.0 < grade["score"] <= 1.0


class TestSessionIsolation:
    def test_two_clients_have_independent_state(self):
        c1 = TestClient(app)
        c2 = TestClient(app)

        c1.post("/reset", json={"task_id": "easy", "seed": 42})
        c2.post("/reset", json={"task_id": "hard", "seed": 7})

        s1 = c1.get("/state").json()
        s2 = c2.get("/state").json()

        assert s1["num_ports"] != s2["num_ports"]


class TestActionMaskInResponse:
    def test_reset_includes_action_mask(self, client):
        r = client.post("/reset", json={"task_id": "easy", "seed": 42})
        body = r.json()
        assert "action_mask" in body
        assert len(body["action_mask"]) == 4  # 4 ports

    def test_step_includes_updated_mask(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 42})
        r = client.post("/step", json={"actions": [3, 3, 3, 3]})
        body = r.json()
        assert "action_mask" in body["observation"]
