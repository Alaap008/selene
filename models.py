"""
Typed Pydantic models for the EV Charging Scheduler OpenEnv environment.
Covers Action, Observation, Reward, and Info as required by the OpenEnv spec.

All observations are fixed-size numeric arrays — no free-form text.
Actions are discrete per-port charging levels with action masking.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class Action(BaseModel):
    """
    The action the agent takes each step.

    Each element corresponds to a charger port and selects a discrete
    charging power level:
        0 = OFF    (0 kW)
        1 = LOW    (3.6 kW)
        2 = MEDIUM (7 kW)
        3 = HIGH   (11 kW)
    """
    actions: List[int] = Field(
        ...,
        description=(
            "List of charging level selections, one per port. "
            "Length must equal the number of ports. "
            "Each value in {0, 1, 2, 3}."
        ),
    )


class Observation(BaseModel):
    """
    Fixed-size numeric observation vector plus action mask.

    The state vector layout for N ports:
        [0..N-1]     current_soc      — current state-of-charge per port (0 if empty)
        [N..2N-1]    target_soc       — required SoC at departure per port
        [2N..3N-1]   time_remaining   — normalised time until departure per port
        [3N..4N-1]   port_occupied    — 1.0 if a vehicle is present, 0.0 otherwise
        [4N..5N-1]   port_operational — 1.0 if the port is working, 0.0 if failed
        [5N]         tariff           — current electricity tariff (normalised)
        [5N+1]       grid_load        — current demand / max grid capacity
        [5N+2]       time_of_day      — normalised hour (0.0=midnight, 0.5=noon)
        [5N+3]       steps_remaining  — normalised steps left in the episode
    """
    state: List[float] = Field(
        ..., description="Fixed-size numeric observation vector.",
    )
    action_mask: List[List[bool]] = Field(
        ...,
        description=(
            "Boolean mask of shape (N, 4). action_mask[i][j] is True if "
            "charging level j is valid for port i at this step."
        ),
    )
    step_count: int = Field(0, description="Number of steps taken so far.")
    max_steps: int = Field(48, description="Maximum steps in this episode.")
    num_ports: int = Field(..., description="Number of charger ports (N).")


class Reward(BaseModel):
    """Dense reward signal returned every step."""
    value: float = Field(..., description="Scalar reward for the last action.")
    reason: str = Field(..., description="Human-readable explanation.")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown (energy_cost, progress, lateness, etc.).",
    )


class Info(BaseModel):
    """Episode metrics and debug information."""
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured metrics: energy_cost_total, vehicles_satisfied, "
            "vehicles_departed, grid_violations, total_energy_kwh, etc."
        ),
    )
