"""
PR-Root v1.0.1 - Holonomy-Correct Phase-Resolved Root Arithmetic
"""
from .config import PRConfig, ClosureClass, prin, wrap_to_pi_a, check_closure
from .state import PRState, CutCrossingEvent

__version__ = "1.0.1"
__all__ = [
    # Config
    "PRConfig",
    "ClosureClass",
    "prin",
    "wrap_to_pi_a",
    "check_closure",
    # State
    "PRState",
    "CutCrossingEvent",
]


# Lazy imports for optional modules
def __getattr__(name):
    if name in ("PRTensor", "pr_sqrt_diff", "pr_mul_diff", "pr_div_diff",
                "pr_unwrap_diff", "soft_parity", "HolonomyEmbedding",
                "PRSqrtLayer", "PRUnwrapLayer", "TORCH_AVAILABLE"):
        from . import differentiable
        return getattr(differentiable, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
