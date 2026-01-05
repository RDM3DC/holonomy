"""
PR-Root Operations v1.0.1
Phase-resolved arithmetic operations.
"""
from __future__ import annotations

import math
import cmath
from typing import Tuple, List, Optional

from .state import PRState
from .config import PRConfig


def pr_mul(a: PRState, b: PRState) -> PRState:
    """
    Multiply states by composing amplitudes and lifted phases:
        (A1, θ1) * (A2, θ2) = (A1*A2, θ1+θ2)
    """
    if a.config != b.config:
        raise ValueError("Configs must match for holonomy-consistent composition.")
    cfg = a.config
    A = a.A * b.A
    theta = a.theta + b.theta
    return PRState.from_complex(A * cmath.exp(1j * theta), config=cfg, theta_lift=theta)


def pr_div(a: PRState, b: PRState) -> PRState:
    """
    Divide states by composing amplitudes and lifted phases:
        (A1, θ1) / (A2, θ2) = (A1/A2, θ1-θ2)
    """
    if a.config != b.config:
        raise ValueError("Configs must match for holonomy-consistent composition.")
    if b.A == 0:
        raise ZeroDivisionError("Division by zero amplitude.")
    cfg = a.config
    A = a.A / b.A
    theta = a.theta - b.theta
    return PRState.from_complex(A * cmath.exp(1j * theta), config=cfg, theta_lift=theta)


def pr_pow(a: PRState, n: int) -> PRState:
    """
    Integer power via θ -> nθ (single-valued for integer n).
    Non-integer exponents are multi-valued and belong in a cover-aware API.
    """
    if not isinstance(n, int):
        raise TypeError("pr_pow only supports integer exponents.")
    cfg = a.config
    if n == 0:
        return PRState.from_complex(1 + 0j, config=cfg, theta_lift=0.0)
    A = a.A ** n
    theta = a.theta * n
    return PRState.from_complex(A * cmath.exp(1j * theta), config=cfg, theta_lift=theta)


def pr_sqrt(a: PRState) -> Tuple[PRState, int]:
    """
    Holonomy-correct square root computed from the lift:
        sqrt(z) = sqrt(A) * exp(i*θ/2)

    Returns (root_state, parity), where parity is the *input* parity b = w mod 2.

    Interpretation:
      Let principal_sqrt be computed from the reduced rep θ_R:
         principal_sqrt(z) = sqrt(A) * exp(i*θ_R/2)
      Then:
         pr_sqrt(z) = (-1)^w * principal_sqrt(z)
      So parity tells you whether PR-sqrt equals (parity=0) or negates (parity=1) the principal sqrt.

    Note:
      We do not assert that an output integer 'winding' is the same invariant as the input winding,
      because the root naturally lives on a different cover (Z₂ grading).
    """
    cfg = a.config
    A_out = math.sqrt(a.A)
    theta_out = a.theta / 2.0
    root_state = PRState.from_complex(A_out * cmath.exp(1j * theta_out), config=cfg, theta_lift=theta_out)
    return root_state, a.b


def principal_sqrt_value(a: PRState) -> complex:
    """Standard principal sqrt at the same point (uses θ_R)."""
    return math.sqrt(a.A) * cmath.exp(1j * (a.theta_R / 2.0))


def pr_nthroot(a: PRState, n: int) -> Tuple[PRState, int]:
    """
    n-th root via θ -> θ/n.
    Returns (root_state, sheet_index) where sheet_index = w mod n.
    """
    if n <= 0:
        raise ValueError(f"Root degree must be positive, got {n}")
    cfg = a.config
    A_out = a.A ** (1.0 / n)
    theta_out = a.theta / n
    root_state = PRState.from_complex(A_out * cmath.exp(1j * theta_out), config=cfg, theta_lift=theta_out)
    sheet_index = a.w % n
    return root_state, sheet_index


# =============================================================================
# Path operations (convenience functions)
# =============================================================================

def pr_unwrap_path(z_path: List[complex], config: Optional[PRConfig] = None) -> List[PRState]:
    """
    Unwrap a path of complex numbers maintaining continuous phase.

    Args:
        z_path: List of complex numbers representing a path
        config: PR-Root configuration

    Returns:
        List of PRState objects with continuously unwrapped phase
    """
    if len(z_path) == 0:
        return []

    cfg = config or PRConfig()
    states = [PRState.from_complex(z_path[0], config=cfg)]

    for idx, z in enumerate(z_path[1:], start=1):
        new_state = states[-1].update_from_complex(z, idx=idx)
        states.append(new_state)

    return states


def pr_path_winding(states: List[PRState]) -> int:
    """Get the total winding number of a path."""
    if len(states) < 2:
        return 0
    return states[-1].w - states[0].w


def pr_path_parity_flips(states: List[PRState]) -> int:
    """Count the number of cut crossings (parity flips) along a path."""
    if len(states) == 0:
        return 0
    return len(states[-1].crossings)


# Convenience wrapper
def pr(z: complex, config: Optional[PRConfig] = None) -> PRState:
    """Wrap a complex number in a PRState."""
    return PRState.from_complex(z, config)
