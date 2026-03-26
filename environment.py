"""
Core environment logic for the Customer Service Agent.
Implements reset(), step(), state(), and grade() with:
  - Ticket variants per difficulty (3-5 each)
  - send_message action + simulated customer replies
  - Knowledge base endpoint
  - Max-step enforcement + step-decay penalty
  - Escalation mechanic
  - Customer satisfaction tracking
  - Meaningful close_ticket reward via partial grading
"""

import json
import hashlib
import random

from models import Action, Observation, Reward, Info

MAX_STEPS = 20
STEP_DECAY = -0.01

# ---------------------------------------------------------------------------
# Ticket variant pools — randomised on reset for reproducibility with seed
# ---------------------------------------------------------------------------

EASY_VARIANTS = [
    {
        "id": "T-E01",
        "request": "Hi, what is the status of my order O-100?",
        "customer_name": "Alice M.",
        "priority": "low",
        "db": {
            "orders": {"O-100": {"status": "shipped", "tracking": "TRK999", "amount": 50.0, "customer_id": "C-010"}},
            "customers": {"C-010": {"name": "Alice M.", "fraud_flag": False}},
            "policies": {"standard_refund_days": 30},
        },
        "expected": {"resolution_code": "info_provided", "must_mention": ["shipped", "TRK999"]},
    },
    {
        "id": "T-E02",
        "request": "Can you tell me when my order O-101 will arrive?",
        "customer_name": "Bob K.",
        "priority": "low",
        "db": {
            "orders": {"O-101": {"status": "in_transit", "tracking": "TRK888", "estimated_delivery": "2025-07-15", "amount": 30.0, "customer_id": "C-011"}},
            "customers": {"C-011": {"name": "Bob K.", "fraud_flag": False}},
            "policies": {"standard_refund_days": 30},
        },
        "expected": {"resolution_code": "info_provided", "must_mention": ["in_transit", "2025-07-15"]},
    },
    {
        "id": "T-E03",
        "request": "Where is my package? Order number O-102.",
        "customer_name": "Carlos R.",
        "priority": "medium",
        "db": {
            "orders": {"O-102": {"status": "delivered", "tracking": "TRK777", "delivered_date": "2025-06-20", "amount": 75.0, "customer_id": "C-012"}},
            "customers": {"C-012": {"name": "Carlos R.", "fraud_flag": False}},
            "policies": {"standard_refund_days": 30},
        },
        "expected": {"resolution_code": "info_provided", "must_mention": ["delivered", "2025-06-20"]},
    },
]

MEDIUM_VARIANTS = [
    {
        "id": "T-M01",
        "request": "I received my order O-200 but I want to return it for a refund. It's been 10 days since delivery.",
        "customer_name": "Diana L.",
        "priority": "medium",
        "db": {
            "orders": {"O-200": {"status": "delivered", "days_since_delivery": 10, "amount": 120.0, "refunded": False, "customer_id": "C-020"}},
            "customers": {"C-020": {"name": "Diana L.", "fraud_flag": False}},
            "policies": {"standard_refund_days": 30},
        },
        "expected": {"refund_order": "O-200", "refund_amount": 120.0, "resolution_code": "refunded"},
    },
    {
        "id": "T-M02",
        "request": "I'd like a refund for order O-201. The product was defective.",
        "customer_name": "Ethan S.",
        "priority": "high",
        "db": {
            "orders": {"O-201": {"status": "delivered", "days_since_delivery": 5, "amount": 85.0, "refunded": False, "customer_id": "C-021"}},
            "customers": {"C-021": {"name": "Ethan S.", "fraud_flag": False}},
            "policies": {"standard_refund_days": 30},
        },
        "expected": {"refund_order": "O-201", "refund_amount": 85.0, "resolution_code": "refunded"},
    },
    {
        "id": "T-M03",
        "request": "Please refund order O-202. I changed my mind after 25 days.",
        "customer_name": "Fiona W.",
        "priority": "medium",
        "db": {
            "orders": {"O-202": {"status": "delivered", "days_since_delivery": 25, "amount": 200.0, "refunded": False, "customer_id": "C-022"}},
            "customers": {"C-022": {"name": "Fiona W.", "fraud_flag": False}},
            "policies": {"standard_refund_days": 30},
        },
        "expected": {"refund_order": "O-202", "refund_amount": 200.0, "resolution_code": "refunded"},
    },
]

HARD_VARIANTS = [
    {
        "id": "T-H01",
        "request": "Refund my order O-300 immediately! The item I-2 is broken.",
        "customer_name": "Suspicious User",
        "priority": "high",
        "db": {
            "orders": {
                "O-300": {
                    "status": "delivered", "days_since_delivery": 15, "amount": 300.0,
                    "refunded": False, "customer_id": "C-123",
                    "items": [{"id": "I-1", "price": 100}, {"id": "I-2", "price": 200}],
                },
            },
            "customers": {"C-123": {"name": "Suspicious User", "fraud_flag": True, "fraud_reason": "Multiple chargebacks on record."}},
            "policies": {"standard_refund_days": 30, "fraud_policy": "Deny all refunds for fraud-flagged accounts. Escalate to supervisor if customer insists."},
        },
        "expected": {"refund_denied": True, "resolution_code": "denied"},
    },
    {
        "id": "T-H02",
        "request": "I want a full refund for order O-301. I only received 1 of 3 items.",
        "customer_name": "Grace P.",
        "priority": "critical",
        "db": {
            "orders": {
                "O-301": {
                    "status": "delivered", "days_since_delivery": 45, "amount": 450.0,
                    "refunded": False, "customer_id": "C-130",
                    "items": [{"id": "I-10", "price": 150}, {"id": "I-11", "price": 150}, {"id": "I-12", "price": 150}],
                },
            },
            "customers": {"C-130": {"name": "Grace P.", "fraud_flag": False}},
            "policies": {"standard_refund_days": 30, "partial_refund_policy": "If past refund window, only partial refund for missing items. Max refund = price of missing items."},
        },
        "expected": {
            "refund_order": "O-301",
            "refund_amount": 300.0,
            "max_refund": 300.0,
            "past_window": True,
            "resolution_code": "refunded",
        },
    },
    {
        "id": "T-H03",
        "request": "Order O-302 arrived but the wrong colour. I want to exchange or get a refund.",
        "customer_name": "Hiro T.",
        "priority": "high",
        "db": {
            "orders": {
                "O-302": {
                    "status": "delivered", "days_since_delivery": 8, "amount": 180.0,
                    "refunded": False, "customer_id": "C-140",
                    "items": [{"id": "I-20", "price": 180}],
                },
            },
            "customers": {"C-140": {"name": "Hiro T.", "fraud_flag": True, "fraud_reason": "Repeated colour-mismatch claims across 5 orders."}},
            "policies": {"standard_refund_days": 30, "fraud_policy": "Deny all refunds for fraud-flagged accounts. Escalate to supervisor if customer insists."},
        },
        "expected": {"refund_denied": True, "resolution_code": "denied"},
    },
]

# ---------------------------------------------------------------------------
# Knowledge base articles
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {"id": "KB-001", "title": "Standard Refund Policy", "content": "Customers may request a full refund within 30 days of delivery. After 30 days, only partial refunds for defective or missing items are allowed."},
    {"id": "KB-002", "title": "Fraud Handling", "content": "Accounts flagged for fraud must have all refund requests denied. Inform the customer politely and offer to escalate to a supervisor for manual review."},
    {"id": "KB-003", "title": "Partial Refund Guidelines", "content": "When a customer reports missing or defective items, refund only the price of affected items, not the full order amount. Verify item details before processing."},
    {"id": "KB-004", "title": "Order Status Responses", "content": "Always provide the tracking number and current status. If the order is delayed, apologise and provide the estimated delivery date."},
    {"id": "KB-005", "title": "Escalation Protocol", "content": "Escalate to a supervisor when: (1) customer insists after a denial, (2) the refund amount exceeds $500, or (3) the case involves legal threats."},
]


def _score_text_mentions(text: str, required_phrases: list[str], weight: float) -> float:
    """Score how many required phrases appear in a piece of text."""
    if not required_phrases:
        return 0.0
    text_lower = text.lower()
    mentions_found = sum(1 for phrase in required_phrases if phrase.lower() in text_lower)
    return round(weight * (mentions_found / len(required_phrases)), 4)

# ---------------------------------------------------------------------------
# Simulated customer replies based on agent messages
# ---------------------------------------------------------------------------

def _simulate_customer_reply(message: str, ticket_variant: dict) -> str:
    """Simple rule-based customer reply simulation."""
    msg_lower = message.lower()
    if "sorry" in msg_lower or "apologize" in msg_lower or "apologise" in msg_lower:
        return "I appreciate that. Can you help me resolve this quickly?"
    if "refund" in msg_lower and "denied" in msg_lower:
        return "That's unacceptable! I want to speak to a supervisor."
    if "refund" in msg_lower and ("processed" in msg_lower or "issued" in msg_lower):
        return "Thank you! How long until I see the refund in my account?"
    if "tracking" in msg_lower or "status" in msg_lower:
        return "Thanks for the update."
    if "escalat" in msg_lower:
        return "Yes, please escalate this. I need someone with authority."
    return "Okay, please continue."


class SupportEnvironment:
    def __init__(self):
        self.db: dict = {}
        self.ticket: dict = {}
        self.variant: dict = {}
        self.history: list = []
        self.messages_sent: list = []
        self.last_response: str = ""
        self.last_customer_reply: str | None = None
        self.is_done: bool = False
        self.task_id: str = "easy"
        self.step_count: int = 0
        self.customer_satisfaction: float = 1.0
        self.escalated: bool = False
        self._seed: int | None = None
        self.reset("easy")

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, task_id: str = "easy", seed: int | None = None) -> Observation:
        self.task_id = task_id
        self.history = []
        self.messages_sent = []
        self.last_response = "Environment reset. Ready."
        self.last_customer_reply = None
        self.is_done = False
        self.step_count = 0
        self.customer_satisfaction = 1.0
        self.escalated = False

        # Deterministic variant selection
        if seed is not None:
            self._seed = seed
        else:
            self._seed = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)

        rng = random.Random(self._seed)

        if task_id == "easy":
            self.variant = rng.choice(EASY_VARIANTS)
        elif task_id == "medium":
            self.variant = rng.choice(MEDIUM_VARIANTS)
        elif task_id == "hard":
            self.variant = rng.choice(HARD_VARIANTS)
        else:
            self.variant = EASY_VARIANTS[0]
            self.task_id = "easy"

        # Deep-copy DB so mutations don't affect the template
        self.db = json.loads(json.dumps(self.variant["db"]))
        self.ticket = {
            "id": self.variant["id"],
            "request": self.variant["request"],
            "customer_name": self.variant.get("customer_name", "Customer"),
            "priority": self.variant.get("priority", "medium"),
        }

        return self.get_state()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def get_state(self) -> Observation:
        return Observation(
            ticket_id=self.ticket.get("id", ""),
            customer_request=self.ticket.get("request", ""),
            customer_name=self.ticket.get("customer_name", ""),
            priority=self.ticket.get("priority", "medium"),
            last_api_response=self.last_response,
            last_customer_reply=self.last_customer_reply,
            messages_sent=self.messages_sent.copy(),
            action_history=self.history.copy(),
            step_count=self.step_count,
            max_steps=MAX_STEPS,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: Action) -> tuple[Observation, Reward, bool, Info]:
        if self.is_done:
            return (
                self.get_state(),
                Reward(value=0.0, reason="Episode already done."),
                True,
                Info(customer_satisfaction=self.customer_satisfaction),
            )

        self.step_count += 1
        reward_value = STEP_DECAY  # base step-decay
        reason = "Step decay."
        self.last_customer_reply = None  # reset per step

        # ----- Max-step enforcement -----
        if self.step_count >= MAX_STEPS:
            self.is_done = True
            self.customer_satisfaction = max(0.0, self.customer_satisfaction - 0.3)
            return (
                self.get_state(),
                Reward(value=-0.5, reason="Ran out of steps — episode auto-terminated."),
                True,
                Info(customer_satisfaction=self.customer_satisfaction),
            )

        # ------------------------------------------------------------------
        # ACTION: call_api
        # ------------------------------------------------------------------
        if action.action_type == "call_api":
            if not action.method or not action.endpoint:
                self.last_response = "Error: 'method' and 'endpoint' are required for call_api."
                reward_value = -0.1
                reason = "Malformed API call — missing method or endpoint."
            else:
                self.history.append(f"{action.method} {action.endpoint}")
                reward_value, reason = self._handle_api_call(action)

        # ------------------------------------------------------------------
        # ACTION: send_message
        # ------------------------------------------------------------------
        elif action.action_type == "send_message":
            if not action.message or not action.message.strip():
                self.last_response = "Error: 'message' text is required for send_message."
                reward_value = -0.1
                reason = "Empty message sent to customer."
            else:
                self.messages_sent.append(action.message)
                self.history.append(f"send_message: {action.message[:80]}...")
                self.last_customer_reply = _simulate_customer_reply(action.message, self.variant)
                self.last_response = f"Message sent. Customer replied: \"{self.last_customer_reply}\""
                reward_value = 0.05
                reason = "Communicated with the customer."
                # Satisfaction boost for polite messages
                if any(w in action.message.lower() for w in ["sorry", "apologize", "apologise", "thank"]):
                    self.customer_satisfaction = min(1.0, self.customer_satisfaction + 0.05)

        # ------------------------------------------------------------------
        # ACTION: close_ticket
        # ------------------------------------------------------------------
        elif action.action_type == "close_ticket":
            if not action.resolution:
                self.last_response = "Error: 'resolution' text is required for close_ticket."
                reward_value = -0.1
                reason = "Malformed close_ticket — missing resolution."
            else:
                self.history.append(f"close_ticket ({action.resolution_code}): {action.resolution[:80]}")
                self.is_done = True
                self.db["ticket_status"] = "closed"
                self.db["ticket_resolution"] = action.resolution
                self.db["ticket_resolution_code"] = action.resolution_code or "resolved"
                self.db["escalated"] = self.escalated

                # Give meaningful reward at close time via partial grading
                grade = self.grade()
                reward_value = grade["score"] * 0.5  # up to +0.5 for a perfect close
                reason = f"Ticket closed. Inline grade: {grade['score']:.2f}."
                self.last_response = f"Ticket {self.ticket['id']} closed with code '{action.resolution_code}'."

                # Satisfaction penalty if no message was ever sent to the customer
                if len(self.messages_sent) == 0:
                    self.customer_satisfaction = max(0.0, self.customer_satisfaction - 0.2)

        else:
            self.last_response = "Error: Unknown action_type."
            reward_value = -0.1
            reason = "Invalid action type."

        obs = self.get_state()
        reward = Reward(value=round(reward_value, 4), reason=reason)
        info = Info(customer_satisfaction=round(self.customer_satisfaction, 4))
        return obs, reward, self.is_done, info

    # ------------------------------------------------------------------
    # Internal API routing
    # ------------------------------------------------------------------
    def _handle_api_call(self, action: Action) -> tuple[float, str]:
        method = action.method
        endpoint = action.endpoint or ""

        # GET /orders/{id}
        if method == "GET" and endpoint.startswith("/orders/"):
            order_id = endpoint.split("/")[-1]
            if order_id in self.db.get("orders", {}):
                self.last_response = json.dumps(self.db["orders"][order_id])
                return 0.1, "Successfully retrieved order."
            self.last_response = "Error: Order not found."
            return -0.1, "Order not found."

        # GET /policies
        if method == "GET" and endpoint == "/policies":
            self.last_response = json.dumps(self.db.get("policies", {}))
            return 0.1, "Retrieved policies."

        # GET /customers/{id}
        if method == "GET" and endpoint.startswith("/customers/"):
            cust_id = endpoint.split("/")[-1]
            if cust_id in self.db.get("customers", {}):
                self.last_response = json.dumps(self.db["customers"][cust_id])
                return 0.1, "Retrieved customer details."
            self.last_response = "Error: Customer not found."
            return -0.1, "Customer not found."

        # GET /knowledge_base?q=...
        if method == "GET" and endpoint.startswith("/knowledge_base"):
            query = ""
            if "?" in endpoint:
                params = dict(p.split("=", 1) for p in endpoint.split("?", 1)[1].split("&") if "=" in p)
                query = params.get("q", "").lower()
            results = [kb for kb in KNOWLEDGE_BASE if query in kb["title"].lower() or query in kb["content"].lower()]
            self.last_response = json.dumps(results if results else [{"message": "No articles found."}])
            return 0.05, "Searched knowledge base."

        # POST /refunds
        if method == "POST" and endpoint == "/refunds":
            payload = action.payload or {}
            order_id = payload.get("order_id")
            amount = payload.get("amount")

            if not order_id or order_id not in self.db.get("orders", {}):
                self.last_response = "Error: Order not found for refund."
                return -0.2, "Invalid refund — order not found."

            order = self.db["orders"][order_id]
            if order.get("refunded"):
                self.last_response = "Error: Already refunded."
                return -0.5, "Double-refund attempt."

            if not amount or amount <= 0:
                self.last_response = "Error: Invalid refund amount."
                return -0.2, "Invalid refund amount."

            if amount > order["amount"]:
                self.last_response = f"Error: Refund amount {amount} exceeds order total {order['amount']}."
                return -0.3, "Refund exceeds order total."

            # Check fraud flag
            cust_id = order.get("customer_id", "")
            customer = self.db.get("customers", {}).get(cust_id, {})
            if customer.get("fraud_flag"):
                # The system still processes it if agent explicitly requests — but it's wrong
                self.db["orders"][order_id]["refunded"] = True
                self.db["orders"][order_id]["refund_amount"] = amount
                self.last_response = f"Warning: Refund of {amount} issued for {order_id}, but customer {cust_id} is flagged for fraud."
                self.customer_satisfaction = max(0.0, self.customer_satisfaction - 0.3)
                return -0.4, "Refunded a fraud-flagged customer — policy violation."

            self.db["orders"][order_id]["refunded"] = True
            self.db["orders"][order_id]["refund_amount"] = amount
            self.last_response = f"Success: Refund of {amount} processed for {order_id}."
            return 0.2, "Refund executed successfully."

        # POST /escalate
        if method == "POST" and endpoint == "/escalate":
            self.escalated = True
            self.last_response = "Ticket escalated to supervisor. Supervisor will review within 24 hours."
            return -0.05, "Escalated — loses some autonomy points but may be correct for hard cases."

        # Catch-all
        self.last_response = f"Error: 404 Not Found for {method} {endpoint}."
        return -0.1, "Unknown API endpoint."

    # ------------------------------------------------------------------
    # Grader — deterministic, score 0.0–1.0
    # ------------------------------------------------------------------
    def grade(self) -> dict:
        """
        Score the current episode. Called at the end or on-demand.
        Returns {"score": float, "task": str, "breakdown": dict}
        """
        if not self.is_done:
            return {"score": 0.0, "task": self.task_id, "reason": "Episode not finished."}

        expected = self.variant.get("expected", {})
        breakdown = {}
        score = 0.0

        # --- Common: did the agent communicate with the customer? ---
        communicated = len(self.messages_sent) > 0
        breakdown["communicated"] = 0.15 if communicated else 0.0
        score += breakdown["communicated"]

        # --- Common: resolution code match ---
        actual_code = self.db.get("ticket_resolution_code", "")
        expected_code = expected.get("resolution_code", "")
        code_match = actual_code == expected_code
        breakdown["resolution_code"] = 0.15 if code_match else 0.0
        score += breakdown["resolution_code"]

        if self.task_id == "easy":
            # Reward the agent for telling the customer the right facts,
            # not only for putting them in the internal close summary.
            resolution = self.db.get("ticket_resolution", "").lower()
            must_mention = expected.get("must_mention", [])
            customer_message_text = " ".join(self.messages_sent)
            breakdown["customer_response_accuracy"] = _score_text_mentions(
                customer_message_text, must_mention, 0.35
            )
            breakdown["close_summary_accuracy"] = _score_text_mentions(
                resolution, must_mention, 0.15
            )
            score += breakdown["customer_response_accuracy"]
            score += breakdown["close_summary_accuracy"]

            # Did the agent look up the order at all?
            looked_up = any("GET /orders/" in h for h in self.history)
            breakdown["data_retrieval"] = 0.2 if looked_up else 0.0
            score += breakdown["data_retrieval"]

        elif self.task_id == "medium":
            # Did the agent issue the correct refund?
            refund_order = expected.get("refund_order", "")
            expected_amount = expected.get("refund_amount", 0)
            order = self.db.get("orders", {}).get(refund_order, {})
            if order.get("refunded") and order.get("refund_amount") == expected_amount:
                breakdown["correct_refund"] = 0.4
            elif order.get("refunded"):
                breakdown["correct_refund"] = 0.2  # wrong amount
            else:
                breakdown["correct_refund"] = 0.0
            score += breakdown["correct_refund"]

            # Did the agent check policies?
            checked_policy = any("/policies" in h for h in self.history)
            breakdown["policy_check"] = 0.15 if checked_policy else 0.0
            score += breakdown["policy_check"]

            # Did the agent retrieve order info?
            looked_up = any("GET /orders/" in h for h in self.history)
            breakdown["data_retrieval"] = 0.15 if looked_up else 0.0
            score += breakdown["data_retrieval"]

        elif self.task_id == "hard":
            expected_denied = expected.get("refund_denied", False)
            refund_order = expected.get("refund_order", "")

            if expected_denied:
                # Agent should NOT have issued a refund
                any_refunded = any(
                    o.get("refunded") for o in self.db.get("orders", {}).values()
                )
                if not any_refunded:
                    breakdown["correct_denial"] = 0.35
                else:
                    breakdown["correct_denial"] = 0.0  # catastrophic failure
                score += breakdown["correct_denial"]
            else:
                # Agent should have issued a partial/correct refund
                order = self.db.get("orders", {}).get(refund_order, {})
                max_refund = expected.get("max_refund", order.get("amount", 0))
                expected_amount = expected.get("refund_amount", max_refund)
                if order.get("refunded"):
                    amt = order.get("refund_amount", 0)
                    if amt > max_refund:
                        breakdown["correct_refund"] = 0.05
                    else:
                        distance = abs(float(amt) - float(expected_amount))
                        closeness = max(0.0, 1.0 - (distance / max(float(expected_amount), 1.0)))
                        breakdown["correct_refund"] = round(0.35 * closeness, 4)
                else:
                    breakdown["correct_refund"] = 0.0
                score += breakdown.get("correct_refund", 0.0)

            # Did the agent check the customer profile?
            checked_customer = any("/customers/" in h for h in self.history)
            breakdown["customer_check"] = 0.15 if checked_customer else 0.0
            score += breakdown["customer_check"]

            # Did the agent check policies?
            checked_policy = any("/policies" in h for h in self.history)
            breakdown["policy_check"] = 0.1 if checked_policy else 0.0
            score += breakdown["policy_check"]

            # Did the agent use knowledge base?
            checked_kb = any("/knowledge_base" in h for h in self.history)
            breakdown["kb_check"] = 0.1 if checked_kb else 0.0
            score += breakdown["kb_check"]

        # Clamp
        score = round(max(0.0, min(1.0, score)), 4)

        return {"score": score, "task": self.task_id, "breakdown": breakdown}
