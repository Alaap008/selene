from __future__ import annotations

"""
Core environment logic for the EV Charging Scheduler.
Implements reset(), step(), get_state(), action_mask(), and grade().

Design principles (addressing hackathon feedback):
  - Fixed-size numeric observations (no text)
  - Bounded MultiDiscrete action space with action masking
  - Dense per-step rewards (energy cost, progress, lateness, grid violations)
  - Algorithmic curriculum: easy → medium → hard adds constraints, not scenarios
  - Deterministic with seed support
"""

import math
import random
from typing import Optional

from models import Action, Observation, Reward, Info

# ---------------------------------------------------------------------------
# Charging power levels (kW) for each discrete action
# ---------------------------------------------------------------------------
POWER_LEVELS = [0.0, 3.6, 7.0, 11.0]  # OFF, LOW, MEDIUM, HIGH
NUM_LEVELS = len(POWER_LEVELS)

# Time resolution
STEP_DURATION_HOURS = 0.5  # 30 minutes per step
MAX_STEPS = 48             # 24 hours

# Battery constants
BATTERY_CAPACITY_KWH = 60.0  # Standard EV battery
CHARGE_EFFICIENCY = 0.92     # Charging efficiency factor

# Tariff schedule — 48 half-hour slots (normalised 0–1 range)
# Pattern: cheap overnight, moderate morning/evening, expensive afternoon peak
_BASE_TARIFF_24H = [
    # 00:00–06:00 (12 slots) — off-peak
    0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25,
    # 06:00–12:00 (12 slots) — morning ramp
    0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    # 12:00–18:00 (12 slots) — peak
    0.95, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55,
    # 18:00–24:00 (12 slots) — evening ramp-down
    0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.15, 0.15, 0.15,
]

# ---------------------------------------------------------------------------
# Task configurations — algorithmic curriculum
# ---------------------------------------------------------------------------
EASY_CONFIG = {
    "num_ports": 4,
    "num_vehicles": 3,
    "grid_capacity_kw": 35.0,         # Can't charge all 3 at HIGH (3×11=33, tight)
    "flat_tariff": True,              # No tariff variation
    "flat_tariff_value": 0.30,        # Constant price
    "enable_failures": False,
    "enable_urgent": False,
    "departure_window": (24, 42),     # Depart between step 24–42
    "soc_range": (0.15, 0.35),        # Start with 15–35% charge
    "target_soc_range": (0.85, 0.95), # Need substantial charging
}

MEDIUM_CONFIG = {
    "num_ports": 6,
    "num_vehicles": 5,
    "grid_capacity_kw": 40.0,         # Only ~3 ports at HIGH (3×11=33 < 40)
    "flat_tariff": False,             # Time-of-use tariffs active
    "flat_tariff_value": 0.30,
    "enable_failures": False,
    "enable_urgent": False,
    "departure_window": (16, 38),     # Tighter window
    "soc_range": (0.10, 0.30),
    "target_soc_range": (0.85, 0.95),
}

HARD_CONFIG = {
    "num_ports": 8,
    "num_vehicles": 8,
    "grid_capacity_kw": 35.0,         # Very tight — only ~3 at HIGH simultaneously
    "flat_tariff": False,             # Time-of-use tariffs active
    "flat_tariff_value": 0.30,
    "enable_failures": True,          # Random charger failures mid-episode
    "enable_urgent": True,            # Some vehicles need to leave early
    "departure_window": (12, 38),     # Wide range, some very early
    "soc_range": (0.10, 0.25),        # Lower starting charge
    "target_soc_range": (0.90, 1.00), # Higher targets
}

TASK_CONFIGS = {
    "easy": EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard": HARD_CONFIG,
}

# ---------------------------------------------------------------------------
# Vehicle data structure
# ---------------------------------------------------------------------------
class Vehicle:
    """Represents an EV connected to a charging port."""
    __slots__ = (
        "soc", "target_soc", "departure_step", "departed", "satisfied",
        "energy_received_kwh",
    )

    def __init__(self, soc: float, target_soc: float, departure_step: int):
        self.soc = soc
        self.target_soc = target_soc
        self.departure_step = departure_step
        self.departed = False
        self.satisfied = False
        self.energy_received_kwh = 0.0


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class ChargingEnvironment:
    """
    OpenEnv-compatible EV Charging Scheduler.

    Observation: fixed-size float32 vector + action mask
    Action: list of N ints in {0,1,2,3}
    Reward: dense scalar every step
    """

    def __init__(self):
        self.config: dict = EASY_CONFIG
        self.task_id: str = "easy"
        self.num_ports: int = 4
        self.vehicles: list[Optional[Vehicle]] = []
        self.port_operational: list[bool] = []
        self.step_count: int = 0
        self.is_done: bool = False
        self._seed: int = 42
        self._rng: random.Random = random.Random(42)

        # Episode accumulators
        self._total_energy_cost: float = 0.0
        self._total_energy_kwh: float = 0.0
        self._grid_violations: int = 0
        self._vehicles_departed: int = 0
        self._vehicles_satisfied: int = 0
        self._total_vehicles: int = 0
        self._failure_steps: dict[int, int] = {}  # port → step when failure occurs

        self.reset("easy")

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> Observation:
        """Reset the environment to a fresh episode."""
        if task_id not in TASK_CONFIGS:
            task_id = "easy"
        self.task_id = task_id
        self.config = TASK_CONFIGS[task_id]
        self.num_ports = self.config["num_ports"]

        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)

        self.step_count = 0
        self.is_done = False
        self._total_energy_cost = 0.0
        self._total_energy_kwh = 0.0
        self._grid_violations = 0
        self._vehicles_departed = 0
        self._vehicles_satisfied = 0

        # Initialise ports
        self.port_operational = [True] * self.num_ports
        self.vehicles = [None] * self.num_ports

        # Place vehicles
        num_vehicles = min(self.config["num_vehicles"], self.num_ports)
        self._total_vehicles = num_vehicles

        # Assign vehicles to random ports
        ports = list(range(self.num_ports))
        self._rng.shuffle(ports)
        occupied_ports = ports[:num_vehicles]

        soc_lo, soc_hi = self.config["soc_range"]
        target_lo, target_hi = self.config["target_soc_range"]
        dep_lo, dep_hi = self.config["departure_window"]

        for i, port_idx in enumerate(occupied_ports):
            soc = round(self._rng.uniform(soc_lo, soc_hi), 2)
            target = round(self._rng.uniform(target_lo, target_hi), 2)
            departure = self._rng.randint(dep_lo, dep_hi)

            # Hard mode: some vehicles are urgent (depart early)
            if self.config["enable_urgent"] and i < 2:
                departure = self._rng.randint(8, 16)

            self.vehicles[port_idx] = Vehicle(soc, target, departure)

        # Pre-compute failure schedule for hard mode
        self._failure_steps = {}
        if self.config["enable_failures"]:
            # 1–2 ports fail at random times
            num_failures = self._rng.randint(1, 2)
            fail_ports = self._rng.sample(range(self.num_ports), min(num_failures, self.num_ports))
            for fp in fail_ports:
                self._failure_steps[fp] = self._rng.randint(10, 35)

        return self.get_state()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def get_state(self) -> Observation:
        """Build the fixed-size numeric observation vector."""
        N = self.num_ports
        state = []

        # Per-port features (5 features × N)
        current_soc = []
        target_soc = []
        time_remaining = []
        port_occupied = []
        port_operational = []

        for i in range(N):
            v = self.vehicles[i]
            if v is not None and not v.departed:
                current_soc.append(v.soc)
                target_soc.append(v.target_soc)
                steps_left = max(0, v.departure_step - self.step_count)
                time_remaining.append(steps_left / MAX_STEPS)
                port_occupied.append(1.0)
            else:
                current_soc.append(0.0)
                target_soc.append(0.0)
                time_remaining.append(0.0)
                port_occupied.append(0.0)
            port_operational.append(1.0 if self.port_operational[i] else 0.0)

        state.extend(current_soc)
        state.extend(target_soc)
        state.extend(time_remaining)
        state.extend(port_occupied)
        state.extend(port_operational)

        # Global features (4)
        tariff = self._get_tariff()
        state.append(tariff)

        total_demand = sum(
            POWER_LEVELS[0] for _ in range(N)
        )  # Placeholder — actual demand depends on action
        state.append(0.0)  # grid_load — updated properly in step, 0 at state query

        time_of_day = (self.step_count % MAX_STEPS) / MAX_STEPS
        state.append(time_of_day)

        steps_remaining = max(0, MAX_STEPS - self.step_count) / MAX_STEPS
        state.append(steps_remaining)

        # Round all values
        state = [round(float(x), 4) for x in state]

        return Observation(
            state=state,
            action_mask=self.action_mask(),
            step_count=self.step_count,
            max_steps=MAX_STEPS,
            num_ports=N,
        )

    # ------------------------------------------------------------------
    # action_mask
    # ------------------------------------------------------------------
    def action_mask(self) -> list[list[bool]]:
        """
        Return an N×4 boolean mask.
        mask[i][j] = True means action j is valid for port i.
        """
        N = self.num_ports
        mask = []

        # Compute current grid headroom (assume worst case: all at current level)
        grid_cap = self.config["grid_capacity_kw"]

        for i in range(N):
            port_mask = [False, False, False, False]
            v = self.vehicles[i]

            # OFF is always valid
            port_mask[0] = True

            if v is None or v.departed:
                # Empty port — only OFF
                mask.append(port_mask)
                continue

            if not self.port_operational[i]:
                # Failed port — only OFF
                mask.append(port_mask)
                continue

            if v.soc >= v.target_soc:
                # Fully charged — only OFF
                mask.append(port_mask)
                continue

            # Vehicle present, port operational, needs charging
            port_mask[1] = True  # LOW always ok if vehicle needs charge
            port_mask[2] = True  # MEDIUM
            port_mask[3] = True  # HIGH

            mask.append(port_mask)

        # Grid capacity constraint: if total possible demand at HIGH exceeds
        # grid capacity, mask HIGH on ports where it would cause violation.
        # We do a greedy pass: count how many ports can use HIGH.
        active_ports = []
        for i in range(N):
            if mask[i][3]:  # HIGH is currently allowed
                active_ports.append(i)

        if active_ports:
            max_high = int(grid_cap / POWER_LEVELS[3]) if POWER_LEVELS[3] > 0 else len(active_ports)
            if len(active_ports) > max_high:
                # Mask HIGH on the excess ports (last ones — deterministic)
                for idx in active_ports[max_high:]:
                    mask[idx][3] = False

        return mask

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: Action) -> tuple[Observation, Reward, bool, Info]:
        """Execute one timestep of charging decisions."""
        if self.is_done:
            return (
                self.get_state(),
                Reward(value=0.0, reason="Episode already done.", breakdown={}),
                True,
                Info(metrics=self._build_metrics()),
            )

        self.step_count += 1
        actions = action.actions
        N = self.num_ports

        # Pad or truncate actions to match port count
        if len(actions) < N:
            actions = actions + [0] * (N - len(actions))
        actions = actions[:N]

        # Clamp actions to valid range
        actions = [max(0, min(3, a)) for a in actions]

        # Apply action mask — force invalid actions to OFF
        mask = self.action_mask()
        for i in range(N):
            if not mask[i][actions[i]]:
                actions[i] = 0

        # --- Apply charger failures ---
        for port_idx, fail_step in self._failure_steps.items():
            if self.step_count >= fail_step and self.port_operational[port_idx]:
                self.port_operational[port_idx] = False

        # --- Compute power demand and update SoC ---
        tariff = self._get_tariff()
        total_demand_kw = 0.0
        energy_cost = 0.0
        charging_progress = 0.0
        idle_count = 0

        for i in range(N):
            power_kw = POWER_LEVELS[actions[i]]
            v = self.vehicles[i]

            if v is None or v.departed or not self.port_operational[i]:
                continue

            if power_kw > 0:
                # Charge the vehicle
                energy_kwh = power_kw * STEP_DURATION_HOURS * CHARGE_EFFICIENCY
                soc_increase = energy_kwh / BATTERY_CAPACITY_KWH
                old_soc = v.soc
                v.soc = min(v.target_soc, v.soc + soc_increase)
                actual_increase = v.soc - old_soc
                v.energy_received_kwh += actual_increase * BATTERY_CAPACITY_KWH

                total_demand_kw += power_kw
                step_cost = power_kw * STEP_DURATION_HOURS * tariff
                energy_cost += step_cost
                charging_progress += actual_increase
            else:
                # Vehicle is present but not charging
                if v.soc < v.target_soc:
                    idle_count += 1

        self._total_energy_cost += energy_cost
        self._total_energy_kwh += total_demand_kw * STEP_DURATION_HOURS

        # --- Grid violation check ---
        grid_violation = total_demand_kw > self.config["grid_capacity_kw"]
        if grid_violation:
            self._grid_violations += 1

        # --- Handle departures ---
        completion_bonus = 0.0
        lateness_penalty = 0.0

        for i in range(N):
            v = self.vehicles[i]
            if v is None or v.departed:
                continue
            if self.step_count >= v.departure_step:
                v.departed = True
                self._vehicles_departed += 1
                if v.soc >= v.target_soc - 0.01:  # Small tolerance
                    v.satisfied = True
                    self._vehicles_satisfied += 1
                    completion_bonus += 0.3
                else:
                    # Penalty proportional to how far from target
                    shortfall = v.target_soc - v.soc
                    lateness_penalty += 0.5 * shortfall

        # --- Build reward ---
        breakdown = {}

        # Energy cost penalty (normalised — divide by max possible cost per step)
        max_possible_cost = POWER_LEVELS[3] * N * STEP_DURATION_HOURS * 1.0
        normalised_cost = energy_cost / max(max_possible_cost, 0.01)
        breakdown["energy_cost"] = round(-normalised_cost * 0.3, 4)

        # Charging progress reward
        breakdown["charging_progress"] = round(charging_progress * 2.0, 4)

        # Lateness penalty
        breakdown["lateness_penalty"] = round(-lateness_penalty, 4)

        # Grid violation penalty
        breakdown["grid_violation"] = -0.5 if grid_violation else 0.0

        # Idle penalty
        breakdown["idle_penalty"] = round(-0.01 * idle_count, 4)

        # Completion bonus
        breakdown["completion_bonus"] = round(completion_bonus, 4)

        # Episode end bonus
        all_departed = all(
            v is None or v.departed for v in self.vehicles
        )
        episode_end = self.step_count >= MAX_STEPS or all_departed
        breakdown["episode_end_bonus"] = 0.0

        if episode_end:
            self.is_done = True
            # Force-depart all remaining vehicles
            for v in self.vehicles:
                if v is not None and not v.departed:
                    v.departed = True
                    self._vehicles_departed += 1
                    if v.soc >= v.target_soc - 0.01:
                        v.satisfied = True
                        self._vehicles_satisfied += 1

            if self._total_vehicles > 0:
                ratio = self._vehicles_satisfied / self._total_vehicles
                breakdown["episode_end_bonus"] = round(1.0 * ratio, 4)

        reward_value = sum(breakdown.values())
        reward_value = round(reward_value, 4)

        # Build reason string
        parts = []
        if breakdown["energy_cost"] != 0:
            parts.append(f"energy={breakdown['energy_cost']:+.3f}")
        if breakdown["charging_progress"] != 0:
            parts.append(f"progress={breakdown['charging_progress']:+.3f}")
        if breakdown["lateness_penalty"] != 0:
            parts.append(f"late={breakdown['lateness_penalty']:+.3f}")
        if breakdown["grid_violation"] != 0:
            parts.append(f"grid_viol={breakdown['grid_violation']:+.1f}")
        if breakdown["completion_bonus"] != 0:
            parts.append(f"complete={breakdown['completion_bonus']:+.3f}")
        if breakdown["episode_end_bonus"] != 0:
            parts.append(f"episode_end={breakdown['episode_end_bonus']:+.3f}")
        reason = "; ".join(parts) if parts else "No significant change."

        obs = self.get_state()
        reward = Reward(value=reward_value, reason=reason, breakdown=breakdown)
        info = Info(metrics=self._build_metrics())

        return obs, reward, self.is_done, info

    # ------------------------------------------------------------------
    # grade — deterministic score 0.0–1.0
    # ------------------------------------------------------------------
    def grade(self) -> dict:
        """
        Score the current episode. Called at the end or on-demand.
        Returns {"score": float, "task": str, "breakdown": dict}

        Grading rewards:
          - Charging all vehicles on time         (0.45)
          - Minimising cost (compared to naive)   (0.20)
          - Avoiding grid violations              (0.15)
          - Minimising wasted energy at peak       (0.10)
          - Charge precision (SoC closeness)       (0.10)
        """
        if not self.is_done:
            return {"score": 0.001, "task": self.task_id, "reason": "Episode not finished."}

        breakdown = {}

        # 1. Vehicle satisfaction ratio (0.45 weight) — THE key metric
        if self._total_vehicles > 0:
            sat_ratio = self._vehicles_satisfied / self._total_vehicles
        else:
            sat_ratio = 1.0
        breakdown["vehicle_satisfaction"] = round(0.45 * sat_ratio, 4)

        # 2. Cost efficiency (0.20 weight)
        # Compare against naive cost (all vehicles charged at HIGH with peak tariff)
        # Lower actual cost relative to naive = higher score
        naive_energy_needed = 0.0
        for v in self.vehicles:
            if v is not None:
                gap = max(0.0, v.target_soc - (v.soc - (v.energy_received_kwh / BATTERY_CAPACITY_KWH)))
                naive_energy_needed += gap * BATTERY_CAPACITY_KWH
        # Naive cost: all at peak tariff (1.0)
        naive_cost = naive_energy_needed * 1.0  # peak price per kWh
        if naive_cost > 0:
            cost_ratio = min(1.0, self._total_energy_cost / naive_cost)
            efficiency = 1.0 - cost_ratio
        else:
            efficiency = 0.5  # No energy needed → neutral
        breakdown["cost_efficiency"] = round(0.20 * max(0.0, efficiency), 4)

        # 3. Grid compliance (0.15 weight)
        if MAX_STEPS > 0:
            violation_ratio = min(1.0, self._grid_violations / MAX_STEPS)
            compliance = 1.0 - violation_ratio
        else:
            compliance = 1.0
        breakdown["grid_compliance"] = round(0.15 * compliance, 4)

        # 4. Waste penalty (0.10 weight)
        # Penalise energy spent during peak tariff periods (tariff > 0.7)
        # Smart agents shift load to cheap periods
        if self._total_energy_kwh > 0:
            # Estimate: what fraction of energy was "wasted" at high tariff
            # We approximate using the ratio of cost to energy — higher = more waste
            avg_tariff_paid = self._total_energy_cost / max(self._total_energy_kwh, 0.01)
            waste_score = 1.0 - min(1.0, avg_tariff_paid / 1.0)  # Lower avg tariff = less waste
        else:
            waste_score = 0.0  # No energy used at all — bad
        breakdown["tariff_awareness"] = round(0.10 * waste_score, 4)

        # 5. Charge precision (0.10 weight)
        # How close each vehicle got to its target
        total_precision = 0.0
        count = 0
        for v in self.vehicles:
            if v is not None:
                precision = min(1.0, v.soc / max(v.target_soc, 0.01))
                total_precision += precision
                count += 1
        avg_precision = total_precision / max(count, 1)
        breakdown["charge_precision"] = round(0.10 * avg_precision, 4)

        score = sum(breakdown.values())
        # Clamp to (0.001, 0.999)
        score = round(max(0.001, min(0.999, score)), 4)

        return {"score": score, "task": self.task_id, "breakdown": breakdown}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_tariff(self) -> float:
        """Return the current electricity tariff (normalised 0–1)."""
        if self.config["flat_tariff"]:
            return self.config["flat_tariff_value"]
        idx = self.step_count % len(_BASE_TARIFF_24H)
        return _BASE_TARIFF_24H[idx]

    def _build_metrics(self) -> dict:
        """Build metrics dict for Info."""
        return {
            "total_energy_cost": round(self._total_energy_cost, 4),
            "total_energy_kwh": round(self._total_energy_kwh, 4),
            "grid_violations": self._grid_violations,
            "vehicles_satisfied": self._vehicles_satisfied,
            "vehicles_departed": self._vehicles_departed,
            "total_vehicles": self._total_vehicles,
            "step": self.step_count,
            "tariff": self._get_tariff(),
        }
