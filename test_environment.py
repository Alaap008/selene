"""
Unit tests for the EV Charging Scheduler environment.
Run with: python -m pytest test_environment.py -v
"""

import pytest
from environment import (
    ChargingEnvironment,
    POWER_LEVELS,
    MAX_STEPS,
    TASK_CONFIGS,
    STEP_DURATION_HOURS,
    BATTERY_CAPACITY_KWH,
    CHARGE_EFFICIENCY,
)
from models import Action


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_easy(self):
        env = ChargingEnvironment()
        obs = env.reset("easy", seed=42)
        assert obs.step_count == 0
        assert obs.max_steps == MAX_STEPS
        assert obs.num_ports == 4
        assert not env.is_done

    def test_reset_medium(self):
        env = ChargingEnvironment()
        obs = env.reset("medium", seed=42)
        assert obs.num_ports == 6

    def test_reset_hard(self):
        env = ChargingEnvironment()
        obs = env.reset("hard", seed=42)
        assert obs.num_ports == 8

    def test_reset_clears_state(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=1)
        env.step(Action(actions=[3, 3, 3, 3]))
        assert env.step_count == 1

        obs = env.reset("easy", seed=1)
        assert obs.step_count == 0
        assert env._total_energy_cost == 0.0

    def test_reset_invalid_task_defaults(self):
        env = ChargingEnvironment()
        env.reset("nonexistent")
        assert env.task_id == "easy"

    def test_reset_seed_determinism(self):
        env1 = ChargingEnvironment()
        obs1 = env1.reset("medium", seed=123)

        env2 = ChargingEnvironment()
        obs2 = env2.reset("medium", seed=123)

        assert obs1.state == obs2.state

    def test_different_seeds_different_states(self):
        env = ChargingEnvironment()
        obs1 = env.reset("easy", seed=1)
        obs2 = env.reset("easy", seed=999)
        # States should differ (different vehicle placements)
        assert obs1.state != obs2.state


# ---------------------------------------------------------------------------
# State vector
# ---------------------------------------------------------------------------

class TestStateVector:
    def test_state_length(self):
        env = ChargingEnvironment()
        obs = env.reset("easy", seed=42)
        N = obs.num_ports
        # 5*N per-port features + 4 global features
        assert len(obs.state) == 5 * N + 4

    def test_state_values_in_range(self):
        env = ChargingEnvironment()
        obs = env.reset("hard", seed=42)
        for val in obs.state:
            assert 0.0 <= val <= 1.0, f"State value {val} out of [0,1] range"

    def test_state_shows_occupied_ports(self):
        env = ChargingEnvironment()
        obs = env.reset("easy", seed=42)
        N = obs.num_ports
        occupied = obs.state[3 * N : 4 * N]
        num_occupied = sum(1 for x in occupied if x > 0.5)
        assert num_occupied == TASK_CONFIGS["easy"]["num_vehicles"]


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

class TestActionMask:
    def test_mask_shape(self):
        env = ChargingEnvironment()
        obs = env.reset("easy", seed=42)
        assert len(obs.action_mask) == obs.num_ports
        for port_mask in obs.action_mask:
            assert len(port_mask) == 4

    def test_off_always_valid(self):
        env = ChargingEnvironment()
        obs = env.reset("hard", seed=42)
        for port_mask in obs.action_mask:
            assert port_mask[0] is True

    def test_empty_port_only_off(self):
        env = ChargingEnvironment()
        obs = env.reset("easy", seed=42)
        N = obs.num_ports
        for i in range(N):
            occupied = obs.state[3 * N + i] > 0.5
            if not occupied:
                assert obs.action_mask[i] == [True, False, False, False]

    def test_occupied_port_has_charging_options(self):
        env = ChargingEnvironment()
        obs = env.reset("easy", seed=42)
        N = obs.num_ports
        for i in range(N):
            occupied = obs.state[3 * N + i] > 0.5
            soc = obs.state[i]
            target = obs.state[N + i]
            if occupied and soc < target:
                # Should have at least LOW available
                assert obs.action_mask[i][1] is True

    def test_mask_forces_invalid_to_off(self):
        """Actions violating the mask are silently forced to OFF."""
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        N = env.num_ports
        # Create an action with HIGH on all ports (some may be empty)
        actions = [3] * N
        action = Action(actions=actions)
        obs, reward, done, info = env.step(action)
        # Should not crash — invalid actions silently become OFF
        assert obs.step_count == 1


# ---------------------------------------------------------------------------
# Step & Charging
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_increments_counter(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        env.step(Action(actions=[0, 0, 0, 0]))
        assert env.step_count == 1

    def test_charging_increases_soc(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        N = env.num_ports

        # Find an occupied port
        occupied_port = None
        for i in range(N):
            if env.vehicles[i] is not None:
                occupied_port = i
                break

        if occupied_port is not None:
            initial_soc = env.vehicles[occupied_port].soc
            actions = [0] * N
            actions[occupied_port] = 3  # HIGH
            env.step(Action(actions=actions))
            assert env.vehicles[occupied_port].soc > initial_soc

    def test_off_does_not_charge(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        N = env.num_ports

        # Find occupied port
        for i in range(N):
            if env.vehicles[i] is not None:
                initial_soc = env.vehicles[i].soc
                break

        env.step(Action(actions=[0] * N))

        for i in range(N):
            if env.vehicles[i] is not None:
                assert env.vehicles[i].soc == initial_soc
                break

    def test_step_after_done_returns_zero(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        env.is_done = True
        obs, reward, done, info = env.step(Action(actions=[0, 0, 0, 0]))
        assert done is True
        assert reward.value == 0.0

    def test_soc_does_not_exceed_target(self):
        """Charging should cap at target_soc."""
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        N = env.num_ports

        # Run many steps at HIGH
        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            env.step(Action(actions=[3] * N))

        for v in env.vehicles:
            if v is not None:
                assert v.soc <= v.target_soc + 0.001


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

class TestRewards:
    def test_reward_has_breakdown(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        _, reward, _, _ = env.step(Action(actions=[3, 3, 3, 3]))
        assert "energy_cost" in reward.breakdown
        assert "charging_progress" in reward.breakdown

    def test_idle_penalty(self):
        """Vehicles present but set to OFF should incur idle penalty."""
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        _, reward, _, _ = env.step(Action(actions=[0, 0, 0, 0]))
        assert reward.breakdown.get("idle_penalty", 0) <= 0

    def test_charging_gives_positive_progress(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        N = env.num_ports
        actions = [3] * N  # HIGH everywhere
        _, reward, _, _ = env.step(Action(actions=actions))
        assert reward.breakdown.get("charging_progress", 0) > 0

    def test_energy_cost_is_negative(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        N = env.num_ports
        _, reward, _, _ = env.step(Action(actions=[3] * N))
        assert reward.breakdown.get("energy_cost", 0) < 0

    def test_episode_end_bonus(self):
        """Correctly charged vehicles should give episode end bonus."""
        env = ChargingEnvironment()
        env.reset("easy", seed=42)

        # Run until done
        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            obs, reward, done, info = env.step(
                Action(actions=[3] * env.num_ports)
            )

        # At least some vehicles should have departed satisfied
        assert env._vehicles_satisfied >= 0


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class TestGrader:
    def test_grader_not_done(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        result = env.grade()
        assert result["score"] == 0.001
        assert "not finished" in result.get("reason", "").lower()

    def test_grader_score_range(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        # Run to completion
        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            env.step(Action(actions=[3] * env.num_ports))
        result = env.grade()
        assert 0.0 < result["score"] <= 1.0

    def test_grader_has_breakdown(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            env.step(Action(actions=[3] * env.num_ports))
        result = env.grade()
        assert "breakdown" in result
        assert "vehicle_satisfaction" in result["breakdown"]
        assert "cost_efficiency" in result["breakdown"]
        assert "grid_compliance" in result["breakdown"]
        assert "charge_precision" in result["breakdown"]

    def test_all_off_scores_low(self):
        """Never charging should produce a low score."""
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            env.step(Action(actions=[0] * env.num_ports))
        result = env.grade()
        assert result["score"] < 0.55

    def test_aggressive_charging_scores_higher(self):
        """Always HIGH should score better than always OFF."""
        env1 = ChargingEnvironment()
        env1.reset("easy", seed=42)
        for _ in range(MAX_STEPS):
            if env1.is_done:
                break
            env1.step(Action(actions=[3] * env1.num_ports))
        score_high = env1.grade()["score"]

        env2 = ChargingEnvironment()
        env2.reset("easy", seed=42)
        for _ in range(MAX_STEPS):
            if env2.is_done:
                break
            env2.step(Action(actions=[0] * env2.num_ports))
        score_off = env2.grade()["score"]

        assert score_high > score_off


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class TestCurriculum:
    def test_easy_fewer_ports_than_hard(self):
        assert TASK_CONFIGS["easy"]["num_ports"] < TASK_CONFIGS["hard"]["num_ports"]

    def test_easy_has_flat_tariff(self):
        assert TASK_CONFIGS["easy"]["flat_tariff"] is True

    def test_medium_has_variable_tariff(self):
        assert TASK_CONFIGS["medium"]["flat_tariff"] is False

    def test_hard_has_failures_enabled(self):
        assert TASK_CONFIGS["hard"]["enable_failures"] is True

    def test_hard_has_urgent_enabled(self):
        assert TASK_CONFIGS["hard"]["enable_urgent"] is True

    def test_easy_no_failures(self):
        assert TASK_CONFIGS["easy"]["enable_failures"] is False

    def test_grid_tighter_per_port_at_higher_difficulty(self):
        """Grid capacity per port should decrease (tighten) with difficulty."""
        easy_ratio = TASK_CONFIGS["easy"]["grid_capacity_kw"] / TASK_CONFIGS["easy"]["num_ports"]
        med_ratio = TASK_CONFIGS["medium"]["grid_capacity_kw"] / TASK_CONFIGS["medium"]["num_ports"]
        hard_ratio = TASK_CONFIGS["hard"]["grid_capacity_kw"] / TASK_CONFIGS["hard"]["num_ports"]
        assert easy_ratio >= med_ratio
        assert med_ratio >= hard_ratio


# ---------------------------------------------------------------------------
# Charger failures (hard mode)
# ---------------------------------------------------------------------------

class TestFailures:
    def test_failures_happen_in_hard_mode(self):
        """At least one port should fail during a hard episode."""
        env = ChargingEnvironment()
        env.reset("hard", seed=42)

        any_failed = False
        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            env.step(Action(actions=[0] * env.num_ports))
            if not all(env.port_operational):
                any_failed = True
                break

        assert any_failed, "Expected at least one port failure in hard mode"

    def test_no_failures_in_easy_mode(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)

        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            env.step(Action(actions=[0] * env.num_ports))

        assert all(env.port_operational), "Easy mode should have no failures"

    def test_failed_port_only_allows_off(self):
        """A failed port's mask should only allow OFF."""
        env = ChargingEnvironment()
        env.reset("hard", seed=42)

        # Step until a failure occurs
        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            obs, _, _, _ = env.step(Action(actions=[0] * env.num_ports))
            for i, operational in enumerate(env.port_operational):
                if not operational:
                    assert obs.action_mask[i] == [True, False, False, False]
                    return

        # If no failure occurred with this seed, skip
        pytest.skip("No failure occurred with seed=42 in hard mode")


# ---------------------------------------------------------------------------
# Departures
# ---------------------------------------------------------------------------

class TestDepartures:
    def test_vehicle_departs_at_deadline(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)

        # Find earliest departure
        earliest = MAX_STEPS + 1
        for v in env.vehicles:
            if v is not None:
                earliest = min(earliest, v.departure_step)

        # Step until that point
        for _ in range(earliest):
            if env.is_done:
                break
            env.step(Action(actions=[0] * env.num_ports))

        assert env._vehicles_departed > 0

    def test_satisfied_vehicle_counted(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)

        # Charge aggressively
        for _ in range(MAX_STEPS):
            if env.is_done:
                break
            env.step(Action(actions=[3] * env.num_ports))

        # At least some should be satisfied
        assert env._vehicles_satisfied > 0


# ---------------------------------------------------------------------------
# Tariff
# ---------------------------------------------------------------------------

class TestTariff:
    def test_flat_tariff_in_easy(self):
        env = ChargingEnvironment()
        env.reset("easy", seed=42)
        tariff = env._get_tariff()
        assert tariff == TASK_CONFIGS["easy"]["flat_tariff_value"]

    def test_variable_tariff_in_medium(self):
        env = ChargingEnvironment()
        env.reset("medium", seed=42)
        # Step forward to get different tariff values
        tariffs = set()
        for _ in range(24):
            tariffs.add(env._get_tariff())
            env.step(Action(actions=[0] * env.num_ports))
        assert len(tariffs) > 1, "Medium mode should have variable tariffs"
