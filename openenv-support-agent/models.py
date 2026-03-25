"""
Typed Pydantic models for the Customer Service Agent OpenEnv environment.
Covers Action, Observation, Reward, and Info as required by the OpenEnv spec.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal


class Action(BaseModel):
    """
    The action the agent takes each step.
    Three action types:
      - call_api: interact with internal backend systems (orders, customers, policies, knowledge base, refunds)
      - send_message: reply to the customer
      - close_ticket: resolve and close the ticket
    """
    action_type: Literal["call_api", "send_message", "close_ticket"] = Field(
        ...,
        description=(
            "The type of action. "
            "'call_api' queries or mutates internal systems. "
            "'send_message' replies to the customer. "
            "'close_ticket' closes the ticket with a resolution."
        ),
    )
    # --- call_api fields ---
    endpoint: Optional[str] = Field(
        None,
        description="API endpoint to call, e.g. '/orders/O-100', '/policies', '/customers/C-123', '/knowledge_base?q=refund', '/refunds'. Required when action_type='call_api'.",
    )
    method: Optional[Literal["GET", "POST"]] = Field(
        None,
        description="HTTP method. Required when action_type='call_api'.",
    )
    payload: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON body for POST requests. Required for POST /refunds (keys: order_id, amount, reason).",
    )
    # --- send_message fields ---
    message: Optional[str] = Field(
        None,
        description="Message text to send to the customer. Required when action_type='send_message'.",
    )
    # --- close_ticket fields ---
    resolution: Optional[str] = Field(
        None,
        description="Resolution summary. Required when action_type='close_ticket'.",
    )
    resolution_code: Optional[Literal[
        "resolved", "refunded", "escalated", "denied", "info_provided"
    ]] = Field(
        None,
        description="Structured resolution code. Required when action_type='close_ticket'.",
    )


class Observation(BaseModel):
    """What the agent sees after each step."""
    ticket_id: str = Field(..., description="Current support ticket ID.")
    customer_request: str = Field(..., description="The customer's original request text.")
    customer_name: str = Field("", description="The customer's display name.")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        "medium", description="Ticket priority level."
    )
    last_api_response: Optional[str] = Field(
        None, description="JSON response from the last API call, or error message."
    )
    last_customer_reply: Optional[str] = Field(
        None, description="The customer's latest reply after agent sent a message."
    )
    messages_sent: List[str] = Field(
        default_factory=list, description="All messages the agent has sent to the customer."
    )
    action_history: List[str] = Field(
        default_factory=list, description="All actions taken so far."
    )
    step_count: int = Field(0, description="Number of steps taken so far.")
    max_steps: int = Field(20, description="Maximum steps before auto-termination.")


class Reward(BaseModel):
    """Reward signal returned each step."""
    value: float = Field(..., description="Reward value for the last action.")
    reason: str = Field(..., description="Human-readable explanation.")


class Info(BaseModel):
    """Additional episode metadata."""
    customer_satisfaction: float = Field(
        1.0, description="Customer satisfaction score (0.0–1.0), degrades with bad interactions."
    )
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Debug metrics.")
