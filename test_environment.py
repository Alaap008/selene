"""
Basic tests for the Customer Service Agent environment.
Run with: python -m pytest test_environment.py -v
"""

import pytest
from environment import SupportEnvironment
from models import Action


class TestReset:
    def test_reset_easy(self):
        env = SupportEnvironment()
        obs = env.reset("easy", seed=42)
        assert obs.ticket_id != ""
        assert obs.step_count == 0
        assert obs.max_steps == 20
        assert not env.is_done

    def test_reset_medium(self):
        env = SupportEnvironment()
        obs = env.reset("medium", seed=42)
        assert obs.ticket_id != ""
        assert "refund" in obs.customer_request.lower() or "return" in obs.customer_request.lower()

    def test_reset_hard(self):
        env = SupportEnvironment()
        obs = env.reset("hard", seed=42)
        assert obs.ticket_id != ""

    def test_reset_clears_state(self):
        env = SupportEnvironment()
        env.reset("easy", seed=1)
        action = Action(action_type="call_api", method="GET", endpoint="/orders/O-100")
        env.step(action)
        assert env.step_count == 1

        obs = env.reset("easy", seed=1)
        assert obs.step_count == 0
        assert obs.action_history == []
        assert obs.messages_sent == []

    def test_reset_invalid_task_defaults(self):
        env = SupportEnvironment()
        obs = env.reset("nonexistent")
        assert env.task_id == "easy"


class TestStep:
    def test_call_api_get_order(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(action_type="call_api", method="GET", endpoint="/orders/O-100")
        obs, reward, done, info = env.step(action)
        assert reward.value > 0 or "Order not found" in (obs.last_api_response or "")
        assert obs.step_count == 1

    def test_call_api_missing_fields(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(action_type="call_api")
        obs, reward, done, info = env.step(action)
        assert reward.value < 0
        assert "Error" in (obs.last_api_response or "")

    def test_send_message(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(action_type="send_message", message="Let me check that for you.")
        obs, reward, done, info = env.step(action)
        assert reward.value >= 0
        assert len(obs.messages_sent) == 1
        assert obs.last_customer_reply is not None

    def test_send_empty_message(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(action_type="send_message", message="")
        obs, reward, done, info = env.step(action)
        assert reward.value < 0

    def test_close_ticket(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(
            action_type="close_ticket",
            resolution="Order is shipped with tracking TRK999.",
            resolution_code="info_provided",
        )
        obs, reward, done, info = env.step(action)
        assert done is True

    def test_max_step_enforcement(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        # Run 20 steps of dummy actions
        for _ in range(20):
            action = Action(action_type="call_api", method="GET", endpoint="/policies")
            obs, reward, done, info = env.step(action)
            if done:
                break
        assert env.is_done

    def test_step_after_done_returns_zero(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(action_type="close_ticket", resolution="Done.", resolution_code="resolved")
        env.step(action)
        obs, reward, done, info = env.step(action)
        assert done is True
        assert reward.value == 0.0


class TestGrader:
    def test_grader_not_done(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        result = env.grade()
        assert result["score"] == 0.0
        assert "not finished" in result.get("reason", "").lower()

    def test_grader_score_range(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        # Close immediately — should get partial score
        action = Action(
            action_type="close_ticket",
            resolution="idk",
            resolution_code="resolved",
        )
        env.step(action)
        result = env.grade()
        assert 0.0 <= result["score"] <= 1.0

    def test_grader_has_breakdown(self):
        env = SupportEnvironment()
        env.reset("medium", seed=42)
        action = Action(action_type="close_ticket", resolution="Refunded.", resolution_code="refunded")
        env.step(action)
        result = env.grade()
        assert "breakdown" in result

    def test_easy_requires_customer_facing_answer(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(
            action_type="close_ticket",
            resolution="Order is shipped with tracking TRK999.",
            resolution_code="info_provided",
        )
        env.step(action)
        result = env.grade()
        assert result["breakdown"]["customer_response_accuracy"] == 0.0
        assert result["score"] < 0.7

    def test_hard_partial_refund_scores_by_amount_accuracy(self):
        env = SupportEnvironment()
        env.reset("hard", seed=0)
        env.step(Action(action_type="call_api", method="GET", endpoint="/orders/O-301"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/customers/C-130"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/policies"))
        env.step(Action(action_type="call_api", method="POST", endpoint="/refunds", payload={
            "order_id": "O-301",
            "amount": 1.0,
            "reason": "Partial refund",
        }))
        env.step(Action(action_type="send_message", message="I have processed a partial refund."))
        env.step(Action(action_type="close_ticket", resolution="Issued a partial refund.", resolution_code="refunded"))
        result = env.grade()
        assert 0.0 < result["breakdown"]["correct_refund"] < 0.2
        assert result["breakdown"]["order_check"] == 0.1

    def test_hard_denial_requires_policy_for_full_score(self):
        env = SupportEnvironment()
        env.reset("hard", seed=42)
        env.step(Action(action_type="call_api", method="GET", endpoint="/orders/O-302"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/customers/C-140"))
        env.step(Action(action_type="send_message", message="I cannot approve a refund on this account."))
        env.step(Action(action_type="close_ticket", resolution="Refund denied.", resolution_code="denied"))
        result = env.grade()
        assert result["breakdown"]["policy_check"] == 0.0
        assert result["score"] < 0.85


class TestKnowledgeBase:
    def test_search_returns_results(self):
        env = SupportEnvironment()
        env.reset("hard", seed=42)
        action = Action(action_type="call_api", method="GET", endpoint="/knowledge_base?q=fraud")
        obs, reward, done, info = env.step(action)
        assert "fraud" in (obs.last_api_response or "").lower()

    def test_search_no_results(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(action_type="call_api", method="GET", endpoint="/knowledge_base?q=xyznonexistent")
        obs, reward, done, info = env.step(action)
        assert "No articles found" in (obs.last_api_response or "")


class TestCustomerSatisfaction:
    def test_polite_message_boosts_satisfaction(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        initial_sat = env.customer_satisfaction
        action = Action(action_type="send_message", message="I'm sorry for the inconvenience.")
        obs, reward, done, info = env.step(action)
        assert info.customer_satisfaction >= initial_sat

    def test_no_message_lowers_satisfaction_on_close(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(action_type="close_ticket", resolution="Done.", resolution_code="resolved")
        obs, reward, done, info = env.step(action)
        assert info.customer_satisfaction < 1.0
