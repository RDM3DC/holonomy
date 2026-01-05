"""
PR-Root Configuration v1.0.1
Holonomy-correct conventions with deterministic tie-breaks.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class ClosureClass(str, Enum):
    """Equivalence relation defining 'closure' for a loop/path end-state."""
    U1_IDENTITY = "U(1)-identity"
    CENTRAL_PHASE = "central phase"
    Z2_CLASS = "Z2-class"


@dataclass(frozen=True)
class PRConfig:
    """
    Deterministic conventions for PR-Root.

    pi_a:
        Adaptive πₐ (default: π). All periods are built from pi_a for extensibility.
    theta_cut:
        Branch cut anchor; the principal interval is (theta_cut - 2*pi_a, theta_cut].
        Default theta_cut=pi_a reproduces (-pi_a, pi_a] with tie -pi_a -> +pi_a.
    eps_tie:
        Tie-break epsilon for boundary decisions.
    delta_theta_max:
        Maximum allowed unwrap step; warn (or reject in strict mode) if exceeded.
    kappa, sigma, eps_mach:
        Near-zero threshold r_min = kappa * sigma * sqrt(eps_mach).
    strict_jumps:
        If True, reject phase updates that exceed delta_theta_max.
    freeze_near_zero:
        If True, freeze theta/w/b when A < r_min (but still update amplitude).
    """
    pi_a: float = math.pi
    theta_cut: float = math.pi
    eps_tie: float = 1e-12
    delta_theta_max: float = math.pi
    kappa: float = 10.0
    sigma: float = 1.0
    eps_mach: float = 2.0 ** -52
    strict_jumps: bool = False
    freeze_near_zero: bool = True

    @property
    def two_pi_a(self) -> float:
        return 2.0 * self.pi_a

    @property
    def r_min(self) -> float:
        return self.kappa * self.sigma * math.sqrt(self.eps_mach)


def prin(theta: float, cfg: PRConfig) -> float:
    """
    Deterministic principal reduction to (theta_cut - 2*pi_a, theta_cut].

    Let y = (theta - theta_cut) mod (2*pi_a) in [0, 2*pi_a).
    - If y is (near) 0, return theta_cut (tie-break: lower boundary maps to upper endpoint).
    - Else return theta_cut - 2*pi_a + y.

    Default gauge (theta_cut=pi_a):
        Range becomes (-pi_a, pi_a] and -pi_a -> +pi_a.
    """
    two_pi = cfg.two_pi_a
    y = (theta - cfg.theta_cut) % two_pi  # in [0, 2pi)
    if y <= cfg.eps_tie:
        return cfg.theta_cut
    return cfg.theta_cut - two_pi + y


def wrap_to_pi_a(delta: float, cfg: PRConfig) -> float:
    """
    Reduce an angular delta to (-pi_a, pi_a] with tie -pi_a -> +pi_a.
    """
    two_pi = cfg.two_pi_a
    x = (delta + cfg.pi_a) % two_pi - cfg.pi_a  # in [-pi_a, pi_a)
    if abs(x + cfg.pi_a) <= cfg.eps_tie:
        return cfg.pi_a
    if abs(x - cfg.pi_a) <= cfg.eps_tie:
        return cfg.pi_a
    return x


def _is_close_mod(x: float, y: float, modulus: float, atol: float) -> bool:
    """Check if x ≈ y (mod modulus)."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    d = (x - y) % modulus
    d = min(d, modulus - d)
    return d <= atol


def check_closure(
    theta0: float,
    theta1: float,
    cfg: PRConfig,
    closure: ClosureClass,
    atol: float = 1e-9,
) -> bool:
    """
    Closure predicate for lifts.

    - U(1)-identity: theta1 == theta0
    - central phase: theta1 == theta0 (mod 2*pi_a)
    - Z2-class: theta1 == theta0 (mod pi_a)
    """
    if closure == ClosureClass.U1_IDENTITY:
        return abs(theta1 - theta0) <= atol
    if closure == ClosureClass.CENTRAL_PHASE:
        return _is_close_mod(theta1, theta0, cfg.two_pi_a, atol)
    if closure == ClosureClass.Z2_CLASS:
        return _is_close_mod(theta1, theta0, cfg.pi_a, atol)
    raise ValueError(f"Unknown closure class: {closure}")
