# PR-Root

**Phase-Resolved Root Arithmetic with Topological State Tracking**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pure Python](https://img.shields.io/badge/dependencies-none-green.svg)](https://github.com/)

PR-Root is a computational framework that elevates phase tracking from an after-the-fact diagnostic to a first-class component of machine state. By maintaining a continuous lift of the argument function and tracking topological invariants (winding number, branch parity), PR-Root enables deterministic, branch-aware complex arithmetic—particularly for multi-valued functions like the square root.

**Pure Python** — No external dependencies for core functionality.

## Installation

```bash
pip install -e .
```

For visualization (requires numpy + matplotlib):

```bash
pip install -e ".[viz]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import cmath
from prroot import PRConfig, PRState, prin, wrap_to_pi_a
from prroot.operations import pr_sqrt, pr_unwrap_path

# Create a configuration (immutable)
cfg = PRConfig()

# Create a phase-resolved state from a complex number
z = -1 + 0j
state = PRState.from_complex(z, config=cfg)
print(f"State: A={state.A}, θ={state.theta}, w={state.w}, b={state.b}")

# Phase-resolved square root
sqrt_state, parity = pr_sqrt(state)
print(f"√z = {sqrt_state.z}")  # Should be 1j

# Track phase through a circular path (one loop around origin)
n = 100
z_path = [cmath.exp(1j * 2 * cmath.pi * k / n) for k in range(n + 1)]
states = pr_unwrap_path(z_path, config=cfg)

print(f"After one loop: winding={states[-1].w}, parity={states[-1].b}")
# Output: winding=1, parity=1
```

## Core Concepts

### State Vector

PR-Root maintains the state vector **S = (A, θ, w, b)**:

| Component | Domain | Description |
|-----------|--------|-------------|
| A | ℝ≥0 | Amplitude (modulus) |
| θ | ℝ | **Lifted phase** (continuous, unwrapped) |
| w | ℤ | Winding number: `round((θ - θ_R) / 2π_a)` |
| b | {0, 1} | Branch parity: `w mod 2` |

### Key Identity

The lifted phase decomposes as:

```
θ = θ_R + 2π_a · w
```

where `θ_R = prin(θ)` is the principal (reduced) phase in `(-π_a, π_a]`.

### Key Innovation

Standard complex arithmetic loses phase information:
```python
z = -1
sqrt_standard = z ** 0.5  # Could be +i or -i depending on implementation
```

PR-Root tracks the continuous lift:
```python
cfg = PRConfig()

# -1 with θ = π (winding 0)
state = PRState.from_complex(-1, config=cfg)
sqrt_state, parity = pr_sqrt(state)
print(sqrt_state.z)  # 1j (deterministic)

# -1 with θ = 3π (winding 1) — same z, different history
state2 = PRState.from_complex(-1, config=cfg, theta_lift=3*3.14159)
sqrt_state2, parity2 = pr_sqrt(state2)
print(sqrt_state2.z)  # -1j (opposite sheet!)
```

### Holonomy Tracking

The branch parity `b` is the **ℤ₂ holonomy** of the square-root Riemann surface:

- Start at z=1, take √z = 1
- Go around the origin once (w → w+1)
- Return to z=1, but now √z = -1

PR-Root tracks this automatically:

```python
import cmath
from prroot import PRConfig, PRState
from prroot.operations import pr_sqrt, pr_unwrap_path

cfg = PRConfig()

# Circular path around origin
n = 100
z_path = [cmath.exp(1j * 2 * cmath.pi * k / n) for k in range(n + 1)]
states = pr_unwrap_path(z_path, config=cfg)

# The parity flip is logged
print(f"Cut crossings: {len(states[-1].crossings)}")  # 1
print(f"Final parity: {states[-1].b}")  # 1 (flipped)

# Square root at start vs end (both at z=1)
sqrt_start, _ = pr_sqrt(states[0])
sqrt_end, _ = pr_sqrt(states[-1])
print(f"√z at start: {sqrt_start.z}")  # ≈ 1
print(f"√z at end: {sqrt_end.z}")      # ≈ -1 (sign flip!)
```

## API Reference

### PRConfig (immutable)

```python
from prroot import PRConfig

cfg = PRConfig(
    pi_a=math.pi,          # Adaptive half-period
    theta_cut=math.pi,     # Branch cut location (upper boundary)
    eps_tie=1e-12,         # Tie-break threshold
    delta_theta_max=math.pi,  # Max allowed jump
    kappa=0.999,           # Near-boundary threshold
    sigma=1e-8,            # Jump detection threshold
    strict_jumps=False,    # Raise on large jumps
    freeze_near_zero=True, # Freeze phase near zero amplitude
)
```

### PRState (immutable)

```python
from prroot import PRState

# Create from complex number
state = PRState.from_complex(z, config=cfg)
state = PRState.from_complex(z, config=cfg, theta_lift=explicit_lift)

# Update with unwrapping (returns new state)
state2 = state.update_from_complex(z_new)
state2 = state.update_from_complex(z_new, idx=step_index)

# Properties
state.A        # Amplitude
state.theta    # Lifted phase
state.theta_R  # Principal phase: prin(theta)
state.w        # Winding number
state.b        # Branch parity
state.z        # Complex value: A * exp(i*theta)
state.crossings  # Tuple of CutCrossingEvent
state.warnings   # Tuple of warning messages
```

### Operations

```python
from prroot.operations import pr_sqrt, pr_mul, pr_div, pr_pow, pr_nthroot

pr_sqrt(state)          # Phase-resolved √z, returns (state, parity)
pr_mul(a, b)            # Phase-resolved multiplication
pr_div(a, b)            # Phase-resolved division  
pr_pow(state, n)        # Phase-resolved integer power
pr_nthroot(state, n)    # n-th root
```

### Path Operations

```python
from prroot.operations import pr_unwrap_path

states = pr_unwrap_path(z_path, config=cfg)
states = pr_unwrap_path(z_path, config=cfg, start_theta=initial_lift)
```

### Closure Checking

```python
from prroot import ClosureClass
from prroot.config import check_closure

check_closure(theta_start, theta_end, cfg, ClosureClass.U1_IDENTITY)    # θ unchanged
check_closure(theta_start, theta_end, cfg, ClosureClass.CENTRAL_PHASE)  # Δθ = 2πk
check_closure(theta_start, theta_end, cfg, ClosureClass.Z2_CLASS)       # Δθ = πk
```

### Visualization (optional)

```python
# Requires: pip install prroot[viz]
from prroot.viz import (
    plot_riemann_surface,
    plot_theta_evolution,
    plot_parity_timeline,
    plot_sqrt_branches,
    plot_holonomy_demo,
)

fig, ax = plot_riemann_surface(states)
fig, ax = plot_theta_evolution(states)
fig, ax = plot_parity_timeline(states)
fig, axes = plot_sqrt_branches(r=1.0, n_points=100)
fig, axes = plot_holonomy_demo()
```

## Testing

```bash
pytest tests/ -v
```

## Specification

See [PR-Root-Spec-v1.0.1.md](PR-Root-Spec-v1.0.1.md) for the complete formal specification.

## License

MIT
