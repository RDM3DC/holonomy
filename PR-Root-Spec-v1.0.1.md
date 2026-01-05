# PR-Root Phase-State Machine Specification v1.0.1

**Phase-Resolved Root Arithmetic with Topological State Tracking**  
*Holonomy-Correct Edition*

---

## Abstract

PR-Root is a computational framework that elevates phase tracking from an after-the-fact diagnostic to a first-class component of machine state. By maintaining a continuous lift of the argument function and tracking topological invariants (winding number, branch parity), PR-Root enables deterministic, branch-aware complex arithmetic—particularly for multi-valued functions like the square root.

**v1.0.1 changes:** Notation clarified to distinguish lift ($\theta$) from reduced representative ($\theta_R$); adaptive period $\pi_a$ introduced; winding computed via `round()` not `floor()`; square-root output semantics corrected for holonomy.

---

## 1. State Vector

A PR-Root machine maintains the **phase-resolved state vector**:

$$
\mathbf{S} = (A, \theta, w, b)
$$

| Component | Domain | Description |
|-----------|--------|-------------|
| $A$ | $\mathbb{R}_{\geq 0}$ | Amplitude (modulus) |
| $\theta$ | $\mathbb{R}$ | **Lifted phase** (continuous, unwrapped) |
| $w$ | $\mathbb{Z}$ | Winding number (cumulative full rotations) |
| $b$ | $\{0, 1\}$ | Branch parity ($w \bmod 2$) |

### 1.1 Derived Quantities

The **reduced principal representative** is computed on demand:

$$
\theta_R := \operatorname{prin}_{\pi_a}(\theta) \in (-\pi_a, \pi_a]
$$

The complex value is recovered as:

$$
z = A \cdot e^{i\theta}
$$

### 1.2 Fundamental Identity

The lift and reduced representative are related by:

$$
\theta = \theta_R + 2\pi_a \cdot w
$$

This identity defines $w$ given $\theta$ and $\theta_R$:

$$
w = \operatorname{round}\!\left(\frac{\theta - \theta_R}{2\pi_a}\right)
$$

---

## 2. Topological Invariants (Category A)

These are **path-invariant quantities**—constants for a given continuous history, derived from the topology of $\mathbb{C}^* = \mathbb{C} \setminus \{0\}$.

### 2.1 Adaptive Phase Period

$$
T_\phi = 2\pi_a
$$

where $\pi_a$ is the **adaptive half-period** constant. For standard U(1) arithmetic:

$$
\pi_a = \pi \approx 3.14159265358979...
$$

**Rationale:** Using $\pi_a$ (rather than hardcoding $\pi$) prevents "phase slippage" across implementations and enables future extension to other covering spaces.

### 2.2 Winding Number

For any continuous path $z(t): [0,1] \to \mathbb{C}^*$ forming a closed loop ($z(0) = z(1)$):

$$
w = \frac{1}{2\pi_a} \oint \frac{dz}{iz} = \frac{\Delta\theta}{2\pi_a} \in \mathbb{Z}
$$

**PR-Root semantics:** The winding number is stored as part of machine state and updated incrementally via cut-crossing events, not computed post-hoc.

### 2.3 Branch Parity

$$
b := w \bmod 2 \in \mathbb{Z}_2
$$

This is the **holonomy** in the $\mathbb{Z}_2$ covering space of the square-root Riemann surface. Under the phase-resolved square root:

$$
\circledcirc\!\sqrt{z} = \sqrt{A} \cdot e^{i\theta / 2}
$$

One complete winding ($w \mapsto w + 1$) induces:

$$
e^{i\theta/2} \mapsto e^{i(\theta + 2\pi_a)/2} = -e^{i\theta/2}
$$

Hence $b$ tracks **which sheet** of the Riemann surface the computation resides on.

---

## 3. Implementation Conventions (Category B)

These are **gauge choices and numerical thresholds** that must be fixed for deterministic cross-platform behavior. They are not fundamental constants but standardized parameters.

### 3.1 Principal Reduction Operator

$$
\operatorname{prin}_{\pi_a}(x) \in (-\pi_a, \pi_a]
$$

**Definition:** Reduce $x$ modulo $2\pi_a$ to the half-open interval $(-\pi_a, \pi_a]$.

**Tie-break rule (critical):** If $x \equiv -\pi_a \pmod{2\pi_a}$ within $\epsilon_{\mathrm{tie}}$, return $+\pi_a$.

```python
def prin(x, pi_a, eps_tie):
    reduced = ((x + pi_a) % (2 * pi_a)) - pi_a
    if abs(reduced + pi_a) < eps_tie:
        return pi_a
    return reduced
```

### 3.2 Branch-Cut Anchor

$$
\theta_{\mathrm{cut}} \in [0, 2\pi_a)
$$

**Default:** $\theta_{\mathrm{cut}} = \pi_a$ (negative real axis)

The ray $\arg(z) = \theta_{\mathrm{cut}}$ defines where the principal branch discontinuity occurs. This is a **gauge choice**; making it explicit enables consistent behavior and future holonomy/matrix generalizations.

### 3.3 Near-Zero Threshold

$$
r_{\min} = \kappa \cdot \sigma \cdot \sqrt{\epsilon_{\mathrm{mach}}}
$$

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\kappa$ | $10$ | Safety factor |
| $\sigma$ | context-dependent | Typical amplitude scale of the computation |
| $\epsilon_{\mathrm{mach}}$ | $\approx 2.2 \times 10^{-16}$ (float64) | Machine epsilon |

**Semantics:** If $A < r_{\min}$, the phase $\theta$ is **indeterminate**. Operations must:

1. **Freeze** $\theta$, $w$, and $b$ at their previous values (recommended for continuity), or
2. Set an explicit `PHASE_INDETERMINATE` flag.

**Critical:** Do **not** unwrap through near-zero without an explicit policy—this is where implementations silently diverge.

### 3.4 Tie-Break Epsilon

$$
\epsilon_{\mathrm{tie}} > 0
$$

**Default:** $\epsilon_{\mathrm{tie}} = 10^{-12}$

Used in `prin()` to resolve the $-\pi_a$ vs $+\pi_a$ ambiguity deterministically.

### 3.5 Maximum Jump Guard

$$
\Delta\theta_{\max} \in (\pi_a, 2\pi_a)
$$

**Default:** $\Delta\theta_{\max} = \pi_a$

For discrete/sampled paths, any single-step phase change exceeding $\Delta\theta_{\max}$ triggers:
- A **discontinuity warning**, or
- Rejection of the update (in strict mode).

**Note:** This guard is essential for noisy measured data but may be disabled for exact analytic paths.

---

## 4. Closure Class (Category C)

The closure class $\mathcal{C}$ specifies what equivalence relation defines "returning to the initial state."

$$
\mathcal{C} \in \{\mathrm{U}(1)\text{-identity},\ \mathrm{central\ phase},\ \mathbb{Z}_2\text{-class},\ \ldots\}
$$

| Class | Closure Predicate | Use Case |
|-------|-------------------|----------|
| **U(1)-identity** | $\theta(1) = \theta(0)$ | Strict phase tracking |
| **Central phase** | $\theta(1) = \theta(0) + 2\pi_a k$ for some $k \in \mathbb{Z}$ | Standard complex arithmetic |
| **$\mathbb{Z}_2$-class** | $\theta(1) = \theta(0) + \pi_a k$ for some $k \in \mathbb{Z}$ | Square-root branch tracking |

**Interpretation:** A single input winding breaks "U(1)-identity closure" for the square root but preserves the weaker $\mathbb{Z}_2$ closure classification.

**Future extension:** For $\mathrm{U}(N)$ holonomy, closure becomes $U(\gamma) \in G$ for target conjugacy class $G$.

---

## 5. State Update Rules

### 5.1 Phase Unwrapping (Continuous Lift)

Given previous lift $\theta^{(n-1)}$ and new raw measurement $\theta_{\mathrm{raw}}^{(n)} \in (-\pi_a, \pi_a]$:

**Step 1: Compute delta via principal reduction**

$$
\delta^{(n)} = \operatorname{prin}_{\pi_a}\!\left(\theta_{\mathrm{raw}}^{(n)} - \operatorname{prin}_{\pi_a}(\theta^{(n-1)})\right)
$$

**Step 2: Check for discontinuity**

If $|\delta^{(n)}| > \Delta\theta_{\max}$:
- Signal `DISCONTINUITY_WARNING`
- In strict mode: reject update and return error

**Step 3: Update the lift**

$$
\theta^{(n)} = \theta^{(n-1)} + \delta^{(n)}
$$

**Step 4: Recompute derived quantities**

$$
\theta_R^{(n)} = \operatorname{prin}_{\pi_a}(\theta^{(n)})
$$

$$
w^{(n)} = \operatorname{round}\!\left(\frac{\theta^{(n)} - \theta_R^{(n)}}{2\pi_a}\right)
$$

$$
b^{(n)} = w^{(n)} \bmod 2
$$

### 5.2 Cut-Crossing Events

A **cut crossing** occurs when $\theta_R$ jumps by approximately $\pm 2\pi_a$ under principal reduction. Equivalently:

$$
\text{CUT\_CROSSING} \iff \left| \theta_R^{(n)} - \theta_R^{(n-1)} \right| > \pi_a
$$

Each cut crossing:
- Increments or decrements $w$ by $\pm 1$
- Flips parity $b$

**Parity Flip Log:** For physical applications, maintain a log of cut-crossing events. Parity can flip multiple times within a globally trivial cycle ($\Delta w = 0$), and this flip history may be physically meaningful.

### 5.3 Near-Zero Handling

If $A^{(n)} < r_{\min}$:

```python
# Freeze policy (recommended)
θ^(n) := θ^(n-1)
w^(n) := w^(n-1)
b^(n) := b^(n-1)
phase_valid := False
```

---

## 6. Phase-Resolved Square Root

### 6.1 Point Evaluation

The phase-resolved square root at a single point uses the lift directly:

$$
\circledcirc\!\sqrt{z} = \sqrt{A} \cdot e^{i\theta/2}
$$

The sheet/sign is controlled by parity:
- When $w \mapsto w + 1$, the output $\sqrt{z}$ picks up a factor of $(-1)$
- This is exactly $b$ as the $\mathbb{Z}_2$ holonomy shadow

### 6.2 Path Semantics (Critical)

**Warning:** A closed loop in $z$-space with odd winding produces a **non-closed** path in $\sqrt{z}$-space (sheet swap). Therefore:

- An integer "output winding" for $\sqrt{z}$ is **not well-defined** under the original U(1) closure class
- The output lives on a **double cover** where "one turn" means $\pi_a$ (not $2\pi_a$)

### 6.3 Correct State Mapping

For the **input** state $(A, \theta, w, b)$:

| Output Component | Value | Notes |
|------------------|-------|-------|
| Amplitude | $\sqrt{A}$ | Standard |
| Phase (lift) | $\theta/2$ | Halved |
| Winding | *undefined* | See §6.2 |
| Parity | $b = w \bmod 2$ | Records sheet/sign |

**Interpretation:** The input parity $b$ determines which sheet of the Riemann surface the output resides on. Do not compute $\lfloor w/2 \rfloor$ as "output winding"—this is meaningful only if you change the period to $\pi_a$ (a v1.1+ feature).

---

## 7. Invariant Preservation Theorem

**Theorem (Topological Consistency):** For any continuous path $z(t): [0,1] \to \mathbb{C}^*$ with $z(0) = z(1)$, the PR-Root state machine guarantees:

1. $w \in \mathbb{Z}$ is the true winding number of the path.
2. $b = w \bmod 2$ correctly identifies the Riemann sheet for $\sqrt{z}$.
3. The lifted phase satisfies $\theta(1) = \theta(0) + 2\pi_a w$.

**Proof sketch:** By construction, $\theta$ is a continuous lift of $\arg(z)$ to the universal cover $\mathbb{R}$ of $S^1$. The winding number is the degree of the map $z/|z|: S^1 \to S^1$, which equals $\Delta\theta / 2\pi_a$ by standard covering space theory. $\square$

---

## 8. Reference Values

| Constant | Symbol | Default Value | Category |
|----------|--------|---------------|----------|
| Adaptive half-period | $\pi_a$ | $\pi$ | A (invariant) |
| Phase period | $T_\phi$ | $2\pi_a$ | A (invariant) |
| Branch-cut anchor | $\theta_{\mathrm{cut}}$ | $\pi_a$ | B (convention) |
| Near-zero safety factor | $\kappa$ | $10$ | B (convention) |
| Tie-break epsilon | $\epsilon_{\mathrm{tie}}$ | $10^{-12}$ | B (convention) |
| Maximum jump guard | $\Delta\theta_{\max}$ | $\pi_a$ | B (convention) |
| Default closure class | $\mathcal{C}$ | $\mathbb{Z}_2$-class | C (closure) |

---

## 9. Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Lifted (continuous, unwrapped) phase in $\mathbb{R}$ |
| $\theta_R$ | Reduced principal representative in $(-\pi_a, \pi_a]$ |
| $\pi_a$ | Adaptive half-period (= $\pi$ for standard U(1)) |
| $\operatorname{prin}_{\pi_a}$ | Principal reduction operator |
| $w$ | Winding number |
| $b$ | Branch parity ($w \bmod 2$) |
| $\circledcirc\!\sqrt{\cdot}$ | Phase-resolved square root |
| $\mathbf{S}$ | Full state vector $(A, \theta, w, b)$ |

---

## Appendix A: Relationship to Standard Complex Arithmetic

Standard implementations use the **reduced** state:

$$
\mathbf{S}_{\mathrm{std}} = (A, \theta_R) \quad \text{where } \theta_R = \operatorname{prin}_{\pi_a}(\theta)
$$

PR-Root extends this by tracking the **full lift** $\theta$, enabling:

1. Deterministic multi-valued function evaluation
2. Path-dependent computation (holonomy tracking)
3. Unambiguous branch selection without hidden state
4. Cut-crossing event logging for physical applications

---

## Appendix B: Extension Roadmap

| Version | Feature |
|---------|---------|
| v1.0.1 | Holonomy-correct U(1) phase tracking, $\sqrt{z}$ |
| v1.1 | $n$-th roots via $\mathbb{Z}_n$ holonomy (period $2\pi_a/n$) |
| v2.0 | U(N) matrix holonomy, parallel transport |
| v2.1 | Non-abelian gauge fields, Wilson loops |

---

## Appendix C: Migration from v1.0

| v1.0 | v1.0.1 | Change |
|------|--------|--------|
| $\theta_R$ (lift) | $\theta$ | Renamed for clarity |
| — | $\theta_R$ | Now means reduced representative |
| $2\pi$ | $2\pi_a$ | Adaptive period notation |
| $\lfloor \theta_R / 2\pi \rfloor$ | $\operatorname{round}((\theta - \theta_R)/(2\pi_a))$ | Winding formula |
| $(\sqrt{A}, \theta_R/2, \lfloor w/2 \rfloor, w \bmod 2)$ | $(\sqrt{A}, \theta/2, \text{undefined}, b)$ | Sqrt output |

---

*Specification version: 1.0.1*  
*Date: 2026-01-05*
