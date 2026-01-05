"""
PR-Root State v1.0.1
Immutable, holonomy-correct phase state with cut-crossing log.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import cmath
from typing import List, Optional, Tuple

from .config import PRConfig, prin, wrap_to_pi_a


@dataclass(frozen=True)
class CutCrossingEvent:
    """
    Discrete winding/parity transition caused by principal reduction cut crossing.

    idx:
        User-provided sample index (or None).
    delta_w:
        w_new - w_old (often ±1; can be larger if sampling is coarse).
    w_new, b_new:
        New winding/parity after the update.
    theta_R_old/new:
        Reduced reps before/after update.
    theta_old/new:
        Lifted phases before/after update.
    """
    idx: Optional[int]
    delta_w: int
    w_new: int
    b_new: int
    theta_R_old: float
    theta_R_new: float
    theta_old: float
    theta_new: float


@dataclass(frozen=True)
class PRState:
    """
    Holonomy-correct phase state (immutable).

    Stored:
      A      : amplitude
      theta  : lifted phase θ ∈ ℝ
      w      : winding w ∈ ℤ
      b      : parity b = w mod 2

    Derived:
      theta_R = prin(theta)

    crossings:
      Cut crossing log, recording parity-flip history.
    warnings:
      Discontinuity warnings from jump guard.
    """
    A: float
    theta: float
    config: PRConfig = field(default_factory=PRConfig)
    w: int = 0
    b: int = 0
    crossings: Tuple[CutCrossingEvent, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def theta_R(self) -> float:
        """Reduced principal representative."""
        return prin(self.theta, self.config)

    @property
    def z(self) -> complex:
        """Complex value z = A * exp(i*theta)."""
        return self.A * cmath.exp(1j * self.theta)

    def _recompute_wb(self, theta: float) -> Tuple[int, int, float]:
        """
        Deterministically compute (w, b, theta_R) from theta using:
          theta_R = prin(theta)
          w = round((theta - theta_R)/(2*pi_a))
          b = w mod 2
        """
        theta_R = prin(theta, self.config)
        w = int(round((theta - theta_R) / self.config.two_pi_a))
        b = w % 2
        return w, b, theta_R

    @classmethod
    def from_complex(
        cls,
        z: complex,
        config: Optional[PRConfig] = None,
        theta_lift: Optional[float] = None,
    ) -> PRState:
        """
        Initialize from complex z.

        If theta_lift is provided, it is used directly as the lift θ.
        Otherwise we use the measured principal arg and reduce it into the
        configured principal interval via prin().
        """
        cfg = config or PRConfig()
        A = abs(z)

        if theta_lift is None:
            raw = math.atan2(z.imag, z.real)  # (-pi, pi] in the default gauge
            theta = prin(raw, cfg)            # reduced into (theta_cut-2pi_a, theta_cut]
        else:
            theta = float(theta_lift)

        # Near-zero: if phase is undefined and freeze policy is on, pick a neutral lift at init.
        if cfg.freeze_near_zero and A < cfg.r_min:
            theta = 0.0

        theta_R = prin(theta, cfg)
        w = int(round((theta - theta_R) / cfg.two_pi_a))
        b = w % 2
        return cls(A=A, theta=theta, config=cfg, w=w, b=b)

    def update_from_complex(self, z: complex, idx: Optional[int] = None) -> PRState:
        """
        Update from a new complex sample z:

        1) If A < r_min and freeze_near_zero: update A only (freeze theta/w/b)
        2) Measure theta_raw via atan2 and prin()
        3) Unwrap by nearest continuation:
              delta = wrap_to_pi_a(theta_raw - prin(theta_prev))
              theta_new = theta_prev + delta
        4) Apply jump guard, warn/reject depending on strict_jumps
        5) Recompute w/b via rounding identity
        6) Log a cut crossing event if w changes (parity flip history)
        """
        cfg = self.config
        A_new = abs(z)

        # Near-zero freeze policy.
        if cfg.freeze_near_zero and A_new < cfg.r_min:
            return replace(self, A=A_new)

        theta_raw = prin(math.atan2(z.imag, z.real), cfg)
        theta_R_prev = prin(self.theta, cfg)

        delta = wrap_to_pi_a(theta_raw - theta_R_prev, cfg)

        warnings: List[str] = list(self.warnings)
        if abs(delta) > cfg.delta_theta_max + cfg.eps_tie:
            msg = (
                f"DISCONTINUITY_WARNING: |delta|={abs(delta):.6g} "
                f"exceeds delta_theta_max={cfg.delta_theta_max:.6g}"
            )
            warnings.append(msg)
            if cfg.strict_jumps:
                # Reject phase update but still update amplitude.
                return replace(self, A=A_new, warnings=tuple(warnings))

        theta_new = self.theta + delta
        w_new, b_new, theta_R_new = self._recompute_wb(theta_new)

        crossings: List[CutCrossingEvent] = list(self.crossings)
        if w_new != self.w:
            crossings.append(
                CutCrossingEvent(
                    idx=idx,
                    delta_w=int(w_new - self.w),
                    w_new=int(w_new),
                    b_new=int(b_new),
                    theta_R_old=float(theta_R_prev),
                    theta_R_new=float(theta_R_new),
                    theta_old=float(self.theta),
                    theta_new=float(theta_new),
                )
            )

        return PRState(
            A=A_new,
            theta=theta_new,
            config=cfg,
            w=w_new,
            b=b_new,
            crossings=tuple(crossings),
            warnings=tuple(warnings),
        )

    def with_lift(self, theta: float) -> PRState:
        """Return a new state with updated lift theta and recomputed w/b."""
        w_new, b_new, _ = self._recompute_wb(float(theta))
        return replace(self, theta=float(theta), w=w_new, b=b_new)

    def copy_clear_logs(self) -> PRState:
        """Return the same state but with crossings/warnings cleared."""
        return replace(self, crossings=tuple(), warnings=tuple())
    
    # Convenience properties for backward compatibility
    @property
    def amplitude(self) -> float:
        return self.A
    
    @property
    def winding(self) -> int:
        return self.w
    
    @property
    def parity(self) -> int:
        return self.b
    
    @property
    def complex(self) -> complex:
        return self.z
