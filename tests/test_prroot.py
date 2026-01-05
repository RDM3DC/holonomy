"""
PR-Root Test Suite v1.0.1
"""
import math
import cmath
import pytest

from prroot import PRConfig, prin, wrap_to_pi_a, ClosureClass
from prroot.state import PRState
from prroot.config import check_closure
from prroot.operations import pr_sqrt, principal_sqrt_value, pr_mul, pr_div, pr_pow, pr_unwrap_path


class TestPrin:
    """Tests for principal reduction operator."""

    def test_prin_default_interval_contains_endpoints(self):
        cfg = PRConfig()
        # interval is (pi-2pi, pi] = (-pi, pi]
        assert prin(0.0, cfg) == pytest.approx(0.0)
        assert prin(math.pi, cfg) == pytest.approx(math.pi)
        # lower boundary maps to upper endpoint
        assert prin(-math.pi, cfg) == pytest.approx(math.pi)

    def test_prin_periodicity(self):
        cfg = PRConfig()
        x = 0.123
        assert prin(x + 2 * math.pi, cfg) == pytest.approx(prin(x, cfg))
        assert prin(x - 2 * math.pi, cfg) == pytest.approx(prin(x, cfg))

    def test_prin_custom_theta_cut(self):
        cfg = PRConfig(theta_cut=0.0)
        # interval is (-2pi, 0]
        assert prin(0.0, cfg) == pytest.approx(0.0)
        assert prin(2 * math.pi, cfg) == pytest.approx(0.0)
        assert prin(-2 * math.pi, cfg) == pytest.approx(0.0)
        assert -2 * math.pi < prin(-0.1, cfg) <= 0.0


class TestWrapToPiA:
    """Tests for wrap_to_pi_a."""

    def test_wrap_to_pi_a_basic(self):
        cfg = PRConfig()
        assert wrap_to_pi_a(0.1, cfg) == pytest.approx(0.1)
        assert wrap_to_pi_a(2 * math.pi + 0.2, cfg) == pytest.approx(0.2)
        assert wrap_to_pi_a(-2 * math.pi - 0.2, cfg) == pytest.approx(-0.2)

    def test_wrap_to_pi_a_tie_minus_pi_to_plus_pi(self):
        cfg = PRConfig(eps_tie=1e-9)
        assert wrap_to_pi_a(-math.pi, cfg) == pytest.approx(math.pi)


class TestPRState:
    """Tests for PRState."""

    def test_state_from_complex_initializes_consistently(self):
        cfg = PRConfig()
        s = PRState.from_complex(1 + 0j, config=cfg)
        assert s.A == pytest.approx(1.0)
        assert s.theta == pytest.approx(0.0)
        assert s.theta_R == pytest.approx(0.0)
        assert s.w == 0
        assert s.b == 0

    def test_state_z_roundtrip(self):
        cfg = PRConfig()
        z = 2.5 * cmath.exp(1j * 1.2)
        s = PRState.from_complex(z, config=cfg)
        assert s.z == pytest.approx(z)

    def test_winding_round_rule_positive(self):
        cfg = PRConfig()
        s = PRState.from_complex(
            1.0 * cmath.exp(1j * (2 * math.pi + 0.1)),
            config=cfg,
            theta_lift=2 * math.pi + 0.1,
        )
        assert s.w == 1
        assert s.b == 1
        assert s.theta_R == pytest.approx(0.1)

    def test_winding_round_rule_negative(self):
        cfg = PRConfig()
        s = PRState.from_complex(
            1.0 * cmath.exp(1j * (-2 * math.pi - 0.2)),
            config=cfg,
            theta_lift=-2 * math.pi - 0.2,
        )
        assert s.w == -1
        assert s.b == 1  # -1 mod 2 = 1

    def test_near_zero_freezes_phase(self):
        cfg = PRConfig(freeze_near_zero=True)
        s = PRState.from_complex(1e-20 + 0j, config=cfg)
        assert s.theta == 0.0
        assert s.w == 0
        assert s.b == 0


class TestPhaseUnwrapping:
    """Tests for phase unwrapping."""

    def test_unwrap_simple(self):
        cfg = PRConfig()
        s = PRState.from_complex(1 + 0j, config=cfg)
        # Move to 45 degrees
        z_new = cmath.exp(1j * math.pi / 4)
        s2 = s.update_from_complex(z_new)
        assert s2.theta == pytest.approx(math.pi / 4)

    def test_unwrap_through_branch_cut(self):
        cfg = PRConfig()
        # Start near +π
        s = PRState.from_complex(cmath.exp(1j * 0.9 * math.pi), config=cfg)
        # Move past +π (should continue to >π, not jump to -π)
        s2 = s.update_from_complex(cmath.exp(1j * (-0.9 * math.pi)))
        assert s2.theta > math.pi  # Continued through, not wrapped

    def test_winding_accumulation_one_loop(self):
        cfg = PRConfig()
        # Go around counterclockwise
        n_points = 100
        z_path = [cmath.exp(1j * 2 * math.pi * k / n_points) for k in range(n_points + 1)]
        states = pr_unwrap_path(z_path, config=cfg)
        
        assert states[-1].w == 1
        assert states[-1].b == 1

    def test_cut_crossing_logged(self):
        cfg = PRConfig()
        s = PRState.from_complex(cmath.exp(1j * 0.9 * math.pi), config=cfg)
        s2 = s.update_from_complex(cmath.exp(1j * (-0.9 * math.pi)), idx=1)
        
        assert len(s2.crossings) == 1
        assert s2.crossings[0].delta_w == 1


class TestClosurePredicate:
    """Tests for closure checking."""

    def test_u1_identity_closure(self):
        cfg = PRConfig()
        assert check_closure(0.5, 0.5, cfg, ClosureClass.U1_IDENTITY)
        assert not check_closure(0.5, 0.5 + 2 * math.pi, cfg, ClosureClass.U1_IDENTITY)

    def test_central_phase_closure(self):
        cfg = PRConfig()
        assert check_closure(0.5, 0.5 + 2 * math.pi, cfg, ClosureClass.CENTRAL_PHASE)
        assert check_closure(0.5, 0.5 + 4 * math.pi, cfg, ClosureClass.CENTRAL_PHASE)

    def test_z2_closure(self):
        cfg = PRConfig()
        assert check_closure(0.5, 0.5 + math.pi, cfg, ClosureClass.Z2_CLASS)
        assert check_closure(0.5, 0.5 + 2 * math.pi, cfg, ClosureClass.Z2_CLASS)


class TestPRSqrt:
    """Tests for phase-resolved square root."""

    def test_sqrt_positive_real(self):
        cfg = PRConfig()
        s = PRState.from_complex(4.0 + 0j, config=cfg)
        root, parity = pr_sqrt(s)
        
        assert root.A == pytest.approx(2.0)
        assert abs(root.theta) < 1e-10
        assert parity == 0

    def test_sqrt_negative_real(self):
        cfg = PRConfig()
        s = PRState.from_complex(-1.0 + 0j, config=cfg)
        root, parity = pr_sqrt(s)
        
        # θ = π, so θ/2 = π/2 → result is i
        assert root.z == pytest.approx(1j, abs=1e-10)

    def test_sqrt_sheet_behavior(self):
        """After one winding, sqrt changes sign."""
        cfg = PRConfig()
        # z=1 with winding 0
        s0 = PRState.from_complex(1 + 0j, config=cfg, theta_lift=0)
        sqrt0, p0 = pr_sqrt(s0)
        
        # z=1 with winding 1 (θ = 2π)
        s1 = PRState.from_complex(1 + 0j, config=cfg, theta_lift=2 * math.pi)
        sqrt1, p1 = pr_sqrt(s1)
        
        assert p0 == 0 and p1 == 1
        assert sqrt0.z == pytest.approx(-sqrt1.z)  # Opposite signs


class TestPROperations:
    """Tests for other PR-Root operations."""

    def test_mul(self):
        cfg = PRConfig()
        a = PRState.from_complex(2.0 * cmath.exp(1j * math.pi / 4), config=cfg, theta_lift=math.pi / 4)
        b = PRState.from_complex(3.0 * cmath.exp(1j * math.pi / 3), config=cfg, theta_lift=math.pi / 3)
        
        result = pr_mul(a, b)
        
        assert result.A == pytest.approx(6.0)
        assert result.theta == pytest.approx(math.pi / 4 + math.pi / 3)

    def test_div(self):
        cfg = PRConfig()
        a = PRState.from_complex(6.0 * cmath.exp(1j * math.pi / 2), config=cfg, theta_lift=math.pi / 2)
        b = PRState.from_complex(2.0 * cmath.exp(1j * math.pi / 6), config=cfg, theta_lift=math.pi / 6)
        
        result = pr_div(a, b)
        
        assert result.A == pytest.approx(3.0)
        assert result.theta == pytest.approx(math.pi / 2 - math.pi / 6)

    def test_pow(self):
        cfg = PRConfig()
        s = PRState.from_complex(2.0 * cmath.exp(1j * math.pi / 4), config=cfg, theta_lift=math.pi / 4)
        
        result = pr_pow(s, 3)
        
        assert result.A == pytest.approx(8.0)
        assert result.theta == pytest.approx(3 * math.pi / 4)

    def test_pow_zero(self):
        cfg = PRConfig()
        s = PRState.from_complex(5.0 + 3j, config=cfg)
        result = pr_pow(s, 0)
        assert result.z == pytest.approx(1 + 0j)


class TestPathOperations:
    """Tests for batch path operations."""

    def test_unwrap_circular_path(self):
        cfg = PRConfig()
        n = 100
        z_path = [cmath.exp(1j * 2 * math.pi * k / n) for k in range(n + 1)]
        
        states = pr_unwrap_path(z_path, config=cfg)
        
        assert len(states) == n + 1
        assert states[0].w == 0
        assert states[-1].w == 1

    def test_two_loops(self):
        cfg = PRConfig()
        n = 200
        z_path = [cmath.exp(1j * 4 * math.pi * k / n) for k in range(n + 1)]
        
        states = pr_unwrap_path(z_path, config=cfg)
        
        assert states[-1].w == 2
        assert states[-1].b == 0  # Even winding, parity back to 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_large_theta(self):
        cfg = PRConfig()
        s = PRState.from_complex(1 + 0j, config=cfg, theta_lift=100 * math.pi)
        
        assert s.w == 50
        assert s.b == 0

    def test_negative_winding(self):
        cfg = PRConfig()
        s = PRState.from_complex(1 + 0j, config=cfg, theta_lift=-3 * math.pi)
        
        assert s.w == -2  # -3π → w=-2, θ_R=+π
        assert s.b == 0

    def test_config_immutable(self):
        cfg = PRConfig()
        with pytest.raises(Exception):
            cfg.pi_a = 1.0  # Should fail, frozen dataclass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
