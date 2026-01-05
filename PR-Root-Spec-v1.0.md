# PR-Root Phase-State Machine Specification v1.0

**Phase-Resolved Root Arithmetic with Topological State Tracking**

---

## Abstract

PR-Root is a computational framework that elevates phase tracking from an after-the-fact diagnostic to a first-class component of machine state. By maintaining a continuous lift of the argument function and tracking topological invariants (winding number, branch parity), PR-Root enables deterministic, branch-aware complex arithmetic—particularly for multi-valued functions like the square root.

---

## 1. State Vector

A PR-Root machine maintains the **phase-resolved state vector**:

$$
\mathbf{S} = (A, \theta_R, w, b)
$$

| Component | Domain | Description |
|-----------|--------|-------------|
| $A$ | $\mathbb{R}_{\geq 0}$ | Amplitude (modulus) |
| $\theta_R$ | $\mathbb{R}$ | **Lifted phase** (unwrapped argument, not reduced mod $2\pi$) |
| $w$ | $\mathbb{Z}$ | Winding number (cumulative full rotations) |
| $b$ | $\{0, 1\}$ | Branch parity ($w \bmod 2$) |

The complex value is recovered as:

$$
z = A \cdot e^{i\theta_R}
$$

---

## 2. Topological Invariants (Category A)

These are **path-invariant quantities**—constants for a given continuous history, derived from the topology of $\mathbb{C}^* = \mathbb{C} \setminus \{0\}$.

### 2.1 Phase Period

$$
T_\phi = 2\pi
$$

The fundamental period of the $\mathrm{U}(1)$ group. All angular arithmetic respects this periodicity; PR-Root's key innovation is tracking the **lift** $\theta_R \in \mathbb{R}$ rather than the reduced representative $\theta \in [0, 2\pi)$.

### 2.2 Winding Number

For any continuous path $z(t): [0,1] \to \mathbb{C}^*$ forming a closed loop ($z(0) = z(1)$):

$$
w = \frac{1}{2\pi} \oint \frac{dz}{iz} = \frac{\Delta\theta_R}{2\pi} \in \mathbb{Z}
$$

**PR-Root semantics:** The winding number is stored as part of machine state and updated incrementally, not computed post-hoc.

### 2.3 Branch Parity

$$
b := w \bmod 2 \in \mathbb{Z}_2
$$

This is the **holonomy** in the $\mathbb{Z}_2$ covering space of the square-root Riemann surface. Under the phase-resolved square root:

$$
\circledcirc\!\sqrt{z} = \sqrt{A} \cdot e^{i\theta_R / 2}
$$

One complete winding ($w \mapsto w + 1$) induces:

$$
e^{i\theta_R/2} \mapsto e^{i(\theta_R + 2\pi)/2} = -e^{i\theta_R/2}
$$

Hence $b$ tracks **which sheet** of the Riemann surface the computation resides on.

---

## 3. Implementation Conventions (Category B)

These are **gauge choices and numerical thresholds** that must be fixed for deterministic cross-platform behavior. They are not fundamental constants but standardized parameters.

### 3.1 Branch-Cut Anchor

$$
\theta_{\mathrm{cut}} \in [0, 2\pi)
$$

**Default:** $\theta_{\mathrm{cut}} = \pi$ (negative real axis)

The ray $\arg(z) = \theta_{\mathrm{cut}}$ defines where the principal branch discontinuity occurs. This is a **gauge choice**; making it explicit enables consistent behavior and future holonomy/matrix generalizations.

### 3.2 Near-Zero Threshold

$$
r_{\min} = \kappa \cdot \sigma \cdot \sqrt{\epsilon_{\mathrm{mach}}}
$$

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\kappa$ | $10$ | Safety factor |
| $\sigma$ | context-dependent | Typical amplitude scale of the computation |
| $\epsilon_{\mathrm{mach}}$ | $\approx 2.2 \times 10^{-16}$ (float64) | Machine epsilon |

**Semantics:** If $A < r_{\min}$, the phase $\theta_R$ is **undefined**. Operations must either:
- Preserve the previous $\theta_R$ (phase freeze), or
- Signal an indeterminate state.

### 3.3 Tie-Break Epsilon

$$
\epsilon_{\mathrm{tie}} > 0
$$

**Default:** $\epsilon_{\mathrm{tie}} = 10^{-12}$

When unwrapping via nearest-lift selection, exact half-way cases ($\Delta\theta = \pm\pi$ exactly) require a deterministic tie-break. Convention:

- If $|\Delta\theta - \pi| < \epsilon_{\mathrm{tie}}$, select the **positive** (counterclockwise) continuation.

### 3.4 Maximum Jump Guard

$$
\Delta\theta_{\max} \in (\pi, 2\pi)
$$

**Default:** $\Delta\theta_{\max} = \pi$

For discrete/sampled paths, any single-step phase change exceeding $\Delta\theta_{\max}$ triggers:
- A **discontinuity warning**, or
- Rejection of the update (depending on strictness mode).

**Note:** This guard is essential for noisy measured data but may be disabled for exact analytic paths.

---

## 4. Closure Class (Category C)

The closure class $\mathcal{C}$ specifies what equivalence relation defines "returning to the initial state."

$$
\mathcal{C} \in \{\mathrm{U}(1)\text{-identity},\ \mathrm{central\ phase},\ \mathbb{Z}_2\text{-class},\ \ldots\}
$$

| Class | Closure Condition | Use Case |
|-------|-------------------|----------|
| **U(1)-identity** | $\theta_R$ returns exactly | Strict phase tracking |
| **Central phase** | $\theta_R$ returns mod $2\pi$ | Standard complex arithmetic |
| **$\mathbb{Z}_2$-class** | $\theta_R$ returns mod $\pi$ | Square-root branch tracking |

**Future extension:** For $\mathrm{U}(N)$ holonomy, closure becomes $U(\gamma) \in G$ for target conjugacy class $G$.

---

## 5. State Update Rules

### 5.1 Phase Unwrapping (Continuous Lift)

Given previous state $\theta_R^{(n-1)}$ and new measured angle $\theta_{\mathrm{raw}} \in (-\pi, \pi]$:

```
θ_candidate := θ_R^(n-1) + wrap_to_pi(θ_raw - (θ_R^(n-1) mod 2π))

if |θ_candidate - θ_R^(n-1)| > Δθ_max:
    signal DISCONTINUITY_WARNING
    
θ_R^(n) := θ_candidate
```

Where `wrap_to_pi(x)` reduces $x$ to $(-\pi, \pi]$.

### 5.2 Winding Number Update

$$
w^{(n)} := \left\lfloor \frac{\theta_R^{(n)}}{2\pi} \right\rfloor
$$

Or incrementally:

$$
\Delta w := \left\lfloor \frac{\theta_R^{(n)} - \theta_R^{(n-1)} + \pi}{2\pi} \right\rfloor
$$

### 5.3 Branch Parity Update

$$
b^{(n)} := w^{(n)} \bmod 2
$$

### 5.4 Phase-Resolved Square Root

$$
\circledcirc\!\sqrt{(A, \theta_R, w, b)} = \left( \sqrt{A},\ \frac{\theta_R}{2},\ \left\lfloor \frac{w}{2} \right\rfloor,\ w \bmod 2 \right)
$$

The branch parity $b$ of the **input** determines the sign of the output on the standard sheet.

---

## 6. Invariant Preservation Theorem

**Theorem (Topological Consistency):** For any continuous path $z(t): [0,1] \to \mathbb{C}^*$ with $z(0) = z(1)$, the PR-Root state machine guarantees:

1. $w \in \mathbb{Z}$ is the true winding number of the path.
2. $b = w \bmod 2$ correctly identifies the Riemann sheet for $\sqrt{z}$.
3. The lifted phase satisfies $\theta_R(1) = \theta_R(0) + 2\pi w$.

**Proof sketch:** By construction, $\theta_R$ is a continuous lift of $\arg(z)$ to the universal cover $\mathbb{R}$ of $S^1$. The winding number is the degree of the map $z/|z|: S^1 \to S^1$, which equals $\Delta\theta_R / 2\pi$ by standard covering space theory. $\square$

---

## 7. Reference Values

| Constant | Symbol | Default Value | Category |
|----------|--------|---------------|----------|
| Phase period | $T_\phi$ | $2\pi$ | A (invariant) |
| Branch-cut anchor | $\theta_{\mathrm{cut}}$ | $\pi$ | B (convention) |
| Near-zero threshold factor | $\kappa$ | $10$ | B (convention) |
| Tie-break epsilon | $\epsilon_{\mathrm{tie}}$ | $10^{-12}$ | B (convention) |
| Maximum jump guard | $\Delta\theta_{\max}$ | $\pi$ | B (convention) |
| Default closure class | $\mathcal{C}$ | $\mathbb{Z}_2$-class | C (closure) |

---

## 8. Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\circledcirc\!\sqrt{\cdot}$ | Phase-resolved square root |
| $\theta_R$ | Lifted (unwrapped) phase in $\mathbb{R}$ |
| $w$ | Winding number |
| $b$ | Branch parity |
| $\mathbf{S}$ | Full state vector $(A, \theta_R, w, b)$ |

---

## Appendix A: Relationship to Standard Complex Arithmetic

Standard implementations use the **reduced** state:

$$
\mathbf{S}_{\mathrm{std}} = (A, \theta) \quad \text{where } \theta = \theta_R \bmod 2\pi
$$

PR-Root extends this by tracking the **full lift**, enabling:

1. Deterministic multi-valued function evaluation
2. Path-dependent computation (holonomy tracking)
3. Unambiguous branch selection without hidden state

---

## Appendix B: Extension Roadmap

| Version | Feature |
|---------|---------|
| v1.0 | U(1) phase tracking, $\sqrt{z}$ |
| v1.1 | $n$-th roots via $\mathbb{Z}_n$ holonomy |
| v2.0 | U(N) matrix holonomy, parallel transport |
| v2.1 | Non-abelian gauge fields, Wilson loops |

---

*Specification version: 1.0*  
*Date: 2026-01-05*
