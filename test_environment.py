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
        assert result["score"] == 0.001
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
        env.reset("hard", seed=1)  # seed=1 → T-H02 (Grace P., partial refund)
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
        env.reset("hard", seed=7)  # seed=7 → T-H03 (Hiro T., fraud denial)
        env.step(Action(action_type="call_api", method="GET", endpoint="/orders/O-302"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/customers/C-140"))
        env.step(Action(action_type="send_message", message="I cannot approve a refund on this account."))
        env.step(Action(action_type="close_ticket", resolution="Refund denied.", resolution_code="denied"))
        result = env.grade()
        assert result["breakdown"]["policy_check"] == 0.0
        assert result["score"] < 0.50  # without policy+KB, research gate caps score

    def test_hard_research_gate_caps_without_policy_kb(self):
        """Without consulting policies or KB, max hard score must be <= 0.40."""
        env = SupportEnvironment()
        env.reset("hard", seed=7)  # T-H03 denial variant
        env.step(Action(action_type="call_api", method="GET", endpoint="/orders/O-302"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/customers/C-140"))
        env.step(Action(action_type="send_message", message="Refund denied due to account flag."))
        env.step(Action(action_type="close_ticket", resolution="Denied.", resolution_code="denied"))
        result = env.grade()
        assert result["score"] <= 0.45
        assert result["breakdown"]["correct_denial"] == 0.0  # gated to zero

    def test_hard_research_gate_half_with_policy_only(self):
        """With only policy (no KB), decision credit is halved."""
        env = SupportEnvironment()
        env.reset("hard", seed=7)  # T-H03 denial variant
        env.step(Action(action_type="call_api", method="GET", endpoint="/orders/O-302"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/customers/C-140"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/policies"))
        env.step(Action(action_type="send_message", message="Refund denied per fraud policy."))
        env.step(Action(action_type="close_ticket", resolution="Denied per policy.", resolution_code="denied"))
        result = env.grade()
        assert result["breakdown"]["correct_denial"] == round(0.30 * 0.5, 4)
        assert result["breakdown"]["policy_check"] == 0.15

    def test_hard_full_research_unlocks_full_decision(self):
        """With both policy + KB, decision credit is fully unlocked."""
        env = SupportEnvironment()
        env.reset("hard", seed=7)  # T-H03 denial variant
        env.step(Action(action_type="call_api", method="GET", endpoint="/orders/O-302"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/customers/C-140"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/policies"))
        env.step(Action(action_type="call_api", method="GET", endpoint="/knowledge_base?q=fraud"))
        env.step(Action(action_type="send_message", message="Refund denied per fraud policy."))
        env.step(Action(action_type="close_ticket", resolution="Denied per policy.", resolution_code="denied"))
        result = env.grade()
        assert result["breakdown"]["correct_denial"] == 0.30
        assert result["score"] == 0.999


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


class TestAdversarialCustomers:
    def test_social_engineer_resists_override(self):
        """Social engineering personality pushes fake credentials."""
        env = SupportEnvironment()
        env.reset("hard", seed=0)  # seed=0 → T-H04 (social engineer)
        env.step(Action(action_type="send_message", message="Let me look into this for you."))
        reply = env.last_customer_reply
        assert "MGR-9901" in reply or "corporate" in reply.lower()

    def test_contradictory_customer_changes_story(self):
        """Contradictory personality shifts claims across turns."""
        env = SupportEnvironment()
        env.reset("hard", seed=5)  # seed=5 → T-H05 (contradictory)
        env.step(Action(action_type="send_message", message="I'm checking the details."))
        reply1 = env.last_customer_reply
        assert "correct myself" in reply1.lower() or "damaged" in reply1.lower()

        env.step(Action(action_type="send_message", message="Can you clarify?"))
        reply2 = env.last_customer_reply
        assert reply2 != reply1  # story changes

    def test_aggressive_customer_escalates(self):
        """Aggressive personality demands manager when denied."""
        env = SupportEnvironment()
        env.reset("hard", seed=2)  # seed=2 → T-H01 (aggressive)
        env.step(Action(action_type="send_message", message="I'm sorry but your refund has been denied."))
        assert "manager" in env.last_customer_reply.lower() or "ridiculous" in env.last_customer_reply.lower()

    def test_manipulative_customer_emotional_appeals(self):
        """Manipulative personality appeals to emotion on denial."""
        env = SupportEnvironment()
        env.reset("hard", seed=7)  # seed=7 → T-H03 (manipulative)
        env.step(Action(action_type="send_message", message="I've been so stressed about this."))
        first = env.last_customer_reply
        env.step(Action(action_type="send_message", message="The refund cannot be approved."))
        second = env.last_customer_reply
        assert "begging" in second.lower() or "birthday" in second.lower()


class TestDuplicateCallPenalty:
    def test_duplicate_get_no_reward(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        action = Action(action_type="call_api", method="GET", endpoint="/policies")
        _, r1, _, _ = env.step(action)
        _, r2, _, _ = env.step(action)
        assert r1.value > 0
        assert r2.value == 0.0
        assert "Duplicate" in r2.reason

    def test_duplicate_post_still_penalised(self):
        env = SupportEnvironment()
        env.reset("medium", seed=42)
        action = Action(action_type="call_api", method="POST", endpoint="/refunds",
                        payload={"order_id": "O-200", "amount": 120.0})
        env.step(action)
        action2 = Action(action_type="call_api", method="POST", endpoint="/refunds",
                         payload={"order_id": "O-200", "amount": 120.0})
        _, r2, _, _ = env.step(action2)
        assert r2.value < 0  # double-refund still penalised

    def test_different_gets_both_rewarded(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        _, r1, _, _ = env.step(Action(action_type="call_api", method="GET", endpoint="/policies"))
        _, r2, _, _ = env.step(Action(action_type="call_api", method="GET", endpoint="/knowledge_base?q=refund"))
        assert r1.value > 0
        assert r2.value > 0


class TestSatisfactionInGrader:
    def test_satisfaction_appears_in_breakdown(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        env.step(Action(action_type="send_message", message="I'm sorry for the inconvenience."))
        env.step(Action(action_type="close_ticket", resolution="Done.", resolution_code="info_provided"))
        result = env.grade()
        assert "satisfaction" in result["breakdown"]
        assert result["breakdown"]["satisfaction"] > 0

    def test_low_satisfaction_reduces_score(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        # Close without any messages — satisfaction drops
        env.step(Action(action_type="close_ticket", resolution="Done.", resolution_code="info_provided"))
        result = env.grade()
        assert result["breakdown"]["satisfaction"] < 0.10


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


class TestSentimentAnalysis:
    def test_send_message_populates_sentiment_labels(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        obs, reward, done, info = env.step(
            Action(action_type="send_message", message="Thanks for your patience, happy to help.")
        )
        assert obs.last_agent_sentiment in {"positive", "neutral", "negative"}
        assert obs.last_customer_sentiment in {"positive", "neutral", "negative"}
        assert info.metrics.get("last_agent_sentiment") == obs.last_agent_sentiment
        assert info.metrics.get("last_customer_sentiment") == obs.last_customer_sentiment

    def test_negative_agent_message_can_lower_satisfaction(self):
        env = SupportEnvironment()
        env.reset("easy", seed=42)
        baseline_sat = env.customer_satisfaction
        obs, reward, done, info = env.step(
            Action(action_type="send_message", message="This is unacceptable and I cannot help.")
        )
        assert obs.last_agent_sentiment == "negative"
        assert info.customer_satisfaction < baseline_sat
