#!/usr/bin/env python3
"""
QUANTUM HOLONOMY & BLACK HOLE PHASE TWISTS
==========================================

PR-Root meets quantum mechanics and general relativity.
The same math that flips sqrt(1) also:
  - Gives electrons memory (Berry phase)
  - Lets particles sense fields they never touch (Aharonov-Bohm)
  - Twists spacetime around black holes (frame dragging)
  
This is not a metaphor. This is THE SAME HOLONOMY.
"""

import sys
import math
import cmath
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from prroot import PRState, PRConfig
from prroot.operations import pr_sqrt, pr_mul, pr_nthroot, pr_unwrap_path

# Check for visualization
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Note: Install numpy/matplotlib for visualizations")


def berry_phase_simulator():
    """
    BERRY PHASE: The quantum holonomy
    
    When a quantum state |psi> is slowly transported around a closed loop
    in parameter space, it acquires a geometric phase:
    
        |psi> -> e^{i*gamma} |psi>
    
    This gamma is the HOLONOMY of the Berry connection.
    It's exactly what PR-Root computes!
    """
    print("\n" + "="*70)
    print("QUANTUM MIND-BENDER #1: BERRY PHASE")
    print("="*70)
    print("\nA spin-1/2 particle in a magnetic field that slowly rotates.")
    print("The spin follows the field... but picks up a GEOMETRIC PHASE.\n")
    
    # Simulate a magnetic field rotating in a cone
    # The Berry phase = -Omega/2 where Omega is solid angle traced
    
    # For a cone of half-angle theta, solid angle = 2*pi*(1 - cos(theta))
    # Berry phase for spin-1/2 = -solid_angle / 2
    
    cone_angles = [30, 45, 60, 90, 120, 180]  # degrees
    
    print("Magnetic field traces a cone. Spin follows. Phase accumulates.\n")
    print(f"{'Cone Angle':>12} {'Solid Angle':>14} {'Berry Phase':>14} {'PR-Root w':>12}")
    print("-" * 56)
    
    for theta_deg in cone_angles:
        theta = math.radians(theta_deg)
        solid_angle = 2 * math.pi * (1 - math.cos(theta))
        berry_phase = -solid_angle / 2  # spin-1/2
        
        # This maps to winding! Berry phase = pi * w for half-integer spin
        effective_winding = berry_phase / math.pi
        
        print(f"{theta_deg:>10} deg {solid_angle:>12.4f} sr {berry_phase:>12.4f} rad {effective_winding:>12.2f}")
    
    print("\nAt theta=180 deg (full sphere), Berry phase = -pi")
    print("The state picks up a SIGN FLIP: |psi> -> -|psi>")
    print("This is EXACTLY sqrt(1) = -1 after one loop!")
    print("\nBerry phase IS PR-Root holonomy in Hilbert space.")
    
    # Demonstrate with PR-Root
    print("\n--- PR-Root Verification ---")
    z = 1 + 0j
    
    # Direct path (no rotation)
    state_direct, parity_direct = pr_sqrt(PRState.from_complex(z))
    
    # Full rotation (solid angle = 4*pi for full sphere traversal)
    state_rotated, parity_rotated = pr_sqrt(PRState(A=1.0, theta=2*math.pi, w=1, b=1))
    
    print(f"No rotation:   sqrt(1) = {state_direct.complex:+.4f}")
    print(f"Full rotation: sqrt(1) = {state_rotated.complex:+.4f}")
    print("\nThe quantum state remembers its journey through parameter space!")


def aharonov_bohm_effect():
    """
    AHARONOV-BOHM: The most beautiful experiment in physics
    
    An electron passes around a solenoid (magnetic field confined inside).
    The electron NEVER TOUCHES the field, yet its phase shifts!
    
    Why? Because the vector potential A has non-zero circulation,
    even where B = 0. The electron picks up holonomy.
    
    Phase shift = (e/hbar) * integral(A . dl) = (e/hbar) * Phi
    where Phi = magnetic flux through solenoid
    """
    print("\n" + "="*70)
    print("QUANTUM MIND-BENDER #2: AHARONOV-BOHM EFFECT")
    print("="*70)
    print("\nAn electron passes around a solenoid containing magnetic flux Phi.")
    print("The electron never enters the field region. B = 0 everywhere it goes.")
    print("Yet it acquires a phase shift: delta_phi = (e/hbar) * Phi\n")
    
    # Normalize: one flux quantum Phi_0 = h/e gives phase shift of 2*pi
    # So phase shift = 2*pi * (Phi / Phi_0)
    
    flux_values = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # in units of Phi_0
    
    print("Two paths: left of solenoid vs right of solenoid")
    print("They enclose different flux -> different phase -> INTERFERENCE!\n")
    
    print(f"{'Flux (Phi_0)':>12} {'Phase Shift':>14} {'Interference':>16}")
    print("-" * 46)
    
    for flux in flux_values:
        phase_shift = 2 * math.pi * flux
        # Interference: I = |e^{i*0} + e^{i*phase}|^2 = 2 + 2*cos(phase)
        interference = 2 + 2 * math.cos(phase_shift)
        pattern = "CONSTRUCTIVE" if interference > 3 else ("DESTRUCTIVE" if interference < 1 else "partial")
        
        print(f"{flux:>10.2f} {phase_shift:>12.4f} rad {pattern:>16}")
    
    print("\nAt Phi = 0.5 * Phi_0: COMPLETE DESTRUCTIVE INTERFERENCE")
    print("The electron 'knows' about flux it never touched!")
    print("\nThis is topology. The vector potential has winding around the solenoid.")
    
    # PR-Root model
    print("\n--- PR-Root Model ---")
    print("The electron's wavefunction is like sqrt(z) going around origin.")
    print("The solenoid is the branch point. Flux sets the 'winding charge'.")
    
    # Path that doesn't enclose solenoid (pass complex numbers to pr_unwrap_path)
    path_outside = [cmath.rect(2, t) + 3 for t in [0, 0.5, 1.0, 0.5, 0]]  # stays right
    states_outside = pr_unwrap_path(path_outside)
    
    # Path that encloses solenoid (at origin)  
    path_loop = [cmath.rect(1, t * 2 * math.pi) for t in [0, 0.25, 0.5, 0.75, 1.0]]
    states_loop = pr_unwrap_path(path_loop)
    
    print(f"Path not enclosing solenoid: winding = {states_outside[-1].w}")
    print(f"Path enclosing solenoid:     winding = {states_loop[-1].w}")
    print("\nThe topology of the path determines the phase. Always.")


def black_hole_holonomy():
    """
    BLACK HOLE HOLONOMY: Spacetime itself has winding
    
    Near a rotating (Kerr) black hole, spacetime is twisted.
    A vector parallel-transported around the horizon comes back ROTATED.
    
    This is called FRAME DRAGGING or gravitomagnetic effect.
    The rotation angle is the holonomy of the Levi-Civita connection.
    
    For a Schwarzschild (non-rotating) black hole at r = 2M:
    - Circumference = 2*pi*r but proper distance is infinite
    - Time dilation -> infinity
    - Vectors parallel-transported around come back unchanged (no rotation)
    
    For a Kerr (rotating) black hole:
    - Frame dragging angular velocity: omega = 2*M*a / r^3 (far field)
    - A gyroscope precesses as it orbits
    - The precession angle is HOLONOMY
    """
    print("\n" + "="*70)
    print("SPACETIME MIND-BENDER #1: BLACK HOLE FRAME DRAGGING")
    print("="*70)
    print("\nA Kerr black hole (mass M, spin a) drags spacetime with it.")
    print("A gyroscope orbiting the hole precesses - not from torque,")
    print("but from the CURVATURE OF SPACETIME ITSELF.\n")
    
    # Simplified model: precession angle per orbit at radius r
    # For far-field: Omega_prec ~ 2*G*M*a / (c^2 * r^3) per orbit
    # In geometric units (G=c=1): Omega_prec ~ 2*M*a / r^3
    
    # Let's use M = 1 (geometric units), vary spin parameter a
    M = 1.0
    r = 10.0  # radius in units of M (well outside horizon at ~2M for a<<M)
    
    spin_params = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.998]  # a/M ratio
    
    print(f"Black hole mass M = 1, orbit radius r = {r}M")
    print(f"Horizon at r ~ 2M for slow spin, r ~ M for maximal spin\n")
    
    print(f"{'Spin a/M':>10} {'Precession/orbit':>18} {'Equiv. PR-Root w':>18}")
    print("-" * 50)
    
    for a_over_M in spin_params:
        a = a_over_M * M
        # Lense-Thirring precession per orbit (simplified)
        # Full formula involves elliptic integrals, this is far-field approximation
        omega_prec = 4 * math.pi * M * a / (r ** 1.5 * (r - 2*M) ** 0.5) if r > 2*M else float('inf')
        
        # This maps to winding: angle / (2*pi)
        effective_w = omega_prec / (2 * math.pi)
        
        if omega_prec < 100:
            print(f"{a_over_M:>10.3f} {omega_prec:>16.4f} rad {effective_w:>16.4f}")
        else:
            print(f"{a_over_M:>10.3f} {'diverges':>16} {'->inf':>16}")
    
    print("\nAt a = 0.998M (near-maximal spin), spacetime is a blender!")
    print("Light itself is dragged around the hole.")
    print("\nThis frame-dragging precession IS holonomy of spacetime connection.")
    
    # Connect to PR-Root
    print("\n--- Connection to PR-Root ---")
    print("PR-Root tracks phase accumulation around branch points.")
    print("GR tracks vector rotation around curvature singularities.")
    print("SAME MATH. Different manifolds.")
    print("\nThe black hole IS a branch point in spacetime geometry!")


def event_horizon_phase():
    """
    EVENT HORIZON: Where phase goes to die (and be reborn)
    
    As you approach the event horizon:
    - Time dilation -> infinity
    - Your phase oscillates infinitely fast (from outside view)
    - Information is "frozen" at the horizon (for external observers)
    
    But in PR-Root terms:
    - The horizon is like the branch cut
    - Crossing it is like going to another Riemann sheet
    - The "inside" is a different branch of the manifold
    """
    print("\n" + "="*70)
    print("SPACETIME MIND-BENDER #2: THE HORIZON AS BRANCH CUT")
    print("="*70)
    print("\nThe event horizon is where sqrt(-1) becomes real.")
    print("(The metric signature flips: time becomes space-like)\n")
    
    # Model: infalling observer's phase
    # Proper time tau vs coordinate time t
    # As r -> 2M: t -> infinity, but tau stays finite
    
    # Phase accumulation in coordinate time vs proper time
    # omega * t -> infinity, but omega * tau stays finite
    
    print("An observer falls into a Schwarzschild black hole (M=1).")
    print("They carry a clock oscillating at frequency omega.\n")
    
    r_values = [10, 5, 3, 2.5, 2.1, 2.01, 2.001]  # approaching horizon at r=2M
    M = 1
    
    print(f"{'r/M':>8} {'Time dilation':>16} {'Phase rate (ext)':>18} {'PR-Root analogy':>20}")
    print("-" * 66)
    
    for r in r_values:
        if r > 2*M:
            # Time dilation factor: sqrt(1 - 2M/r)
            dilation = math.sqrt(1 - 2*M/r)
            phase_rate_ext = 1 / dilation  # external observer sees this
            
            # As r -> 2M, this diverges like 1/sqrt(r - 2M)
            # Similar to how phase diverges near z=0 in log(z)
            
            pr_analogy = f"|z| ~ {r - 2*M:.3f}"
            print(f"{r:>8.3f} {dilation:>16.6f} {phase_rate_ext:>16.2f}x {pr_analogy:>20}")
    
    print("\nAt the horizon: time dilation = 0, external phase rate = INFINITY")
    print("The infalling observer sees finite oscillations.")
    print("External observer sees infinite oscillations frozen at horizon.")
    print("\nThis is the holographic principle: horizon stores infinite information!")
    
    # PR-Root connection
    print("\n--- PR-Root Interpretation ---")
    print("Near z = 0: |log(z)| -> infinity (infinite winding capacity)")
    print("Near r = 2M: phase -> infinity (infinite time dilation)")
    print("\nThe event horizon is a BRANCH POINT in spacetime!")
    print("Crossing it is like crossing the branch cut: new sheet, new physics.")
    
    # Demonstrate with PR-Root
    print("\n--- Simulation: Approaching the 'Origin' ---")
    radii = [1.0, 0.5, 0.1, 0.01, 0.001]
    
    for radius in radii:
        z = complex(radius, 0.001)  # slightly off real axis to avoid cut
        state = PRState.from_complex(z)
        sqrt_state, _ = pr_sqrt(state)
        print(f"|z| = {radius:.3f}: sqrt(z) = {sqrt_state.complex:+.4f}, theta = {sqrt_state.theta:+.4f}")
    
    print("\nAs |z| -> 0, we approach the branch point.")
    print("The phase becomes increasingly sensitive to small perturbations.")
    print("This is the 'horizon' of complex analysis!")


def wormhole_parity_flip():
    """
    WORMHOLE TRAVERSAL: The ultimate parity flip
    
    A traversable wormhole connects two regions of spacetime.
    In some models, traversing the wormhole FLIPS YOUR PARITY.
    
    You go in right-handed, come out left-handed.
    Your heart moves from left to right.
    
    This is because the wormhole throat can have non-trivial topology
    (like a Klein bottle or Mobius strip in higher dimensions).
    """
    print("\n" + "="*70)
    print("SPACETIME MIND-BENDER #3: WORMHOLE PARITY FLIP")
    print("="*70)
    print("\nA traversable wormhole with non-orientable topology.")
    print("Enter right-handed, exit left-handed.")
    print("Your mirror image becomes YOU.\n")
    
    # Model this with PR-Root: traversing adds winding
    # After odd winding: parity flipped
    
    print("Modeling wormhole as a 'winding portal':")
    print("Each traversal adds w = 1 to your topological state.\n")
    
    # Start with a "normal" object at z = 1
    z = 1 + 0j
    state = PRState.from_complex(z)
    
    print(f"{'Traversals':>12} {'Winding w':>12} {'Parity b':>12} {'Handedness':>15}")
    print("-" * 55)
    
    for traversals in range(8):
        # Each wormhole traversal adds winding
        modified_state = PRState(A=state.A, theta=state.theta + traversals * math.pi, w=traversals, b=traversals % 2)
        
        handedness = "RIGHT-HANDED" if modified_state.b == 0 else "LEFT-HANDED"
        print(f"{traversals:>12} {modified_state.w:>12} {modified_state.b:>12} {handedness:>15}")
    
    print("\nOdd traversals: LEFT-HANDED (parity flipped)")
    print("Even traversals: RIGHT-HANDED (original parity)")
    print("\nThe wormhole acts as a PARITY OPERATOR: P|state> = (-1)^w |state>")
    print("This is the topological 'charge conjugation' of spacetime!")
    
    # Physical implications
    print("\n--- Physical Implications ---")
    print("If our universe has a wormhole with this topology:")
    print("  - An astronaut returns as their mirror image")
    print("  - Amino acids flip from L to D form")
    print("  - Matter could become 'anti-matter-like' (CPT considerations)")
    print("\nTopology doesn't just remember your path—it TRANSFORMS you.")


def quantum_gravity_holonomy():
    """
    LOOP QUANTUM GRAVITY: Holonomy as fundamental
    
    In Loop Quantum Gravity (LQG), the fundamental variables are
    HOLONOMIES of the Ashtekar connection around loops.
    
    Spacetime is not smooth—it's a spin network of quantized holonomies.
    Area and volume come in discrete quanta.
    
    PR-Root's discrete winding numbers mirror LQG's discrete geometry!
    """
    print("\n" + "="*70)
    print("QUANTUM GRAVITY: HOLONOMY AS THE FABRIC OF SPACE")
    print("="*70)
    print("\nIn Loop Quantum Gravity, spacetime is made of HOLONOMIES.")
    print("Not points, not strings—LOOPS with quantized rotation.\n")
    
    print("Key insight: Area comes in discrete units!")
    print("A_n = 8 * pi * l_P^2 * gamma * sum_j sqrt(j(j+1))")
    print("where j = half-integers (0, 1/2, 1, 3/2, ...)")
    print("and l_P = Planck length, gamma = Immirzi parameter\n")
    
    # Show the area spectrum
    print("Area Eigenvalues (in Planck units, gamma=1):")
    print("-" * 40)
    
    gamma = 0.2375  # Immirzi parameter (from black hole entropy)
    l_P_sq = 1  # Planck length squared = 1 in Planck units
    
    j_values = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    
    print(f"{'j':>6} {'j(j+1)':>10} {'sqrt(j(j+1))':>14} {'Area (l_P^2)':>14}")
    print("-" * 48)
    
    for j in j_values:
        jj1 = j * (j + 1)
        sqrt_jj1 = math.sqrt(jj1)
        area = 8 * math.pi * gamma * sqrt_jj1
        print(f"{j:>6.1f} {jj1:>10.2f} {sqrt_jj1:>14.4f} {area:>14.4f}")
    
    print("\nThese j values are like PR-Root's winding numbers!")
    print("j = 0, 1/2, 1, ... -> w = 0, 1, 2, ...")
    print("\nQuantized holonomy = quantized geometry = discrete spacetime!")
    
    # Connection to PR-Root
    print("\n--- PR-Root as Baby LQG ---")
    print("PR-Root: winding w in Z (integers)")
    print("LQG:     spin j in Z/2 (half-integers)")
    print("\nBoth track holonomy around loops.")
    print("Both have discrete spectra.")
    print("Both make the continuous discrete through topology!")
    
    # Demonstrate discrete phase structure
    print("\n--- Discrete Phase Structure ---")
    print("sqrt() has 2 sheets (j=1/2 holonomy)")
    print("Fourth root has 4 sheets (j=2 holonomy)")
    print("nth root has n sheets (j=n/2 holonomy)")
    
    for n in [2, 3, 4, 5, 6]:
        state = PRState.from_complex(1+0j)
        roots = []
        for k in range(n):
            s = PRState(A=1.0, theta=2*math.pi*k/n, w=k, b=k%2)
            root, _ = pr_nthroot(s, n)
            roots.append(root.complex)
        print(f"  {n}th root sheets: {[f'{r:+.2f}' for r in roots]}")


def visualize_quantum_holonomy():
    """Create mind-bending visualizations"""
    if not HAS_VIZ:
        print("\n[Skipping visualizations - numpy/matplotlib not available]")
        return
    
    print("\n" + "="*70)
    print("GENERATING QUANTUM HOLONOMY VISUALIZATIONS...")
    print("="*70)
    
    # 1. Berry Phase on Bloch Sphere
    fig = plt.figure(figsize=(16, 12))
    
    # Berry phase visualization
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Draw Bloch sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, alpha=0.1, color='blue')
    
    # Draw a path on the sphere (cone at 45 degrees)
    theta_cone = np.pi/4
    phi = np.linspace(0, 2*np.pi, 100)
    x_path = np.sin(theta_cone) * np.cos(phi)
    y_path = np.sin(theta_cone) * np.sin(phi)
    z_path = np.cos(theta_cone) * np.ones_like(phi)
    ax1.plot(x_path, y_path, z_path, 'r-', linewidth=3, label=f'Path (solid angle = {2*np.pi*(1-np.cos(theta_cone)):.2f})')
    
    # Show the solid angle
    ax1.plot([0, 0], [0, 0], [0, 1], 'k--', alpha=0.5)
    ax1.set_title('Berry Phase on Bloch Sphere\nSpin follows B-field around cone', fontsize=12)
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 2. Aharonov-Bohm interference pattern
    ax2 = fig.add_subplot(222)
    
    flux = np.linspace(0, 3, 300)
    phase_diff = 2 * np.pi * flux
    interference = 2 + 2 * np.cos(phase_diff)
    
    ax2.fill_between(flux, 0, interference, alpha=0.3, color='purple')
    ax2.plot(flux, interference, 'purple', linewidth=2)
    ax2.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='No interference')
    ax2.set_xlabel('Magnetic Flux (units of h/e)', fontsize=11)
    ax2.set_ylabel('Intensity', fontsize=11)
    ax2.set_title('Aharonov-Bohm Interference\nElectron senses flux without touching it', fontsize=12)
    ax2.set_xlim(0, 3)
    ax2.legend()
    
    # 3. Black hole frame dragging
    ax3 = fig.add_subplot(223)
    
    r = np.linspace(2.5, 20, 100)  # Outside horizon at r=2M
    M = 1
    
    spins = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.hot(np.linspace(0.2, 0.8, len(spins)))
    
    for a, c in zip(spins, colors):
        # Lense-Thirring precession rate
        omega = 2 * M * a / r**3
        ax3.plot(r, omega, color=c, linewidth=2, label=f'a/M = {a}')
    
    ax3.axvline(x=2, color='black', linestyle='--', label='Horizon (a=0)')
    ax3.set_xlabel('Radius r/M', fontsize=11)
    ax3.set_ylabel('Frame dragging rate (rad/M)', fontsize=11)
    ax3.set_title('Kerr Black Hole Frame Dragging\nSpacetime itself rotates', fontsize=12)
    ax3.set_xlim(2, 20)
    ax3.set_ylim(0, 0.1)
    ax3.legend(fontsize=9)
    
    # 4. LQG area spectrum
    ax4 = fig.add_subplot(224)
    
    gamma = 0.2375
    j_vals = np.arange(0, 5.5, 0.5)
    areas = 8 * np.pi * gamma * np.sqrt(j_vals * (j_vals + 1))
    
    ax4.stem(j_vals, areas, basefmt=' ', linefmt='b-', markerfmt='bo')
    ax4.set_xlabel('Spin quantum number j', fontsize=11)
    ax4.set_ylabel('Area (Planck units)', fontsize=11)
    ax4.set_title('Loop Quantum Gravity Area Spectrum\nSpacetime is discrete!', fontsize=12)
    ax4.set_xlim(-0.5, 5.5)
    ax4.grid(True, alpha=0.3)
    
    # Add annotation
    ax4.annotate('Smallest non-zero area\n(j = 1/2)', xy=(0.5, areas[1]), 
                xytext=(1.5, areas[1]+2), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('quantum_holonomy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: quantum_holonomy.png")
    
    # 5. Wormhole traversal visualization
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    # Draw two "universes" connected by wormhole
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Universe A (left)
    ax.fill(np.cos(theta)*2 - 4, np.sin(theta)*2, alpha=0.2, color='blue', label='Universe A')
    ax.plot(np.cos(theta)*2 - 4, np.sin(theta)*2, 'b-', linewidth=2)
    
    # Universe B (right)
    ax.fill(np.cos(theta)*2 + 4, np.sin(theta)*2, alpha=0.2, color='red', label='Universe B')
    ax.plot(np.cos(theta)*2 + 4, np.sin(theta)*2, 'r-', linewidth=2)
    
    # Wormhole throat (hyperbola connecting them)
    x_throat = np.linspace(-2, 2, 100)
    y_upper = np.sqrt(1 + x_throat**2) * 0.5
    y_lower = -y_upper
    
    ax.fill_between(x_throat, y_lower, y_upper, alpha=0.3, color='purple')
    ax.plot(x_throat, y_upper, 'purple', linewidth=2)
    ax.plot(x_throat, y_lower, 'purple', linewidth=2)
    
    # Traveler path
    x_path = np.linspace(-4, 4, 50)
    y_path = 0.3 * np.sin(x_path * np.pi / 4)
    ax.plot(x_path, y_path, 'g-', linewidth=3, label='Traveler path')
    ax.annotate('', xy=(4, 0), xytext=(-4, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Labels
    ax.text(-4, -2.5, 'Enter:\nRight-handed', ha='center', fontsize=11, color='blue')
    ax.text(4, -2.5, 'Exit:\nLeft-handed!', ha='center', fontsize=11, color='red')
    ax.text(0, -1.5, 'Wormhole\n(parity flip)', ha='center', fontsize=11, color='purple')
    
    ax.set_xlim(-8, 8)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title('Wormhole Parity Flip: Non-Orientable Topology\nTraverse once = mirror reflection', fontsize=14)
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('wormhole_parity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: wormhole_parity.png")


def run_all():
    """Run all quantum holonomy demonstrations"""
    print("\n" + "="*70)
    print("   QUANTUM HOLONOMY & BLACK HOLE PHASE TWISTS")
    print("   PR-Root meets Physics at the Edge of Reality")
    print("="*70)
    
    berry_phase_simulator()
    aharonov_bohm_effect()
    black_hole_holonomy()
    event_horizon_phase()
    wormhole_parity_flip()
    quantum_gravity_holonomy()
    visualize_quantum_holonomy()
    
    print("\n" + "="*70)
    print("THE PUNCHLINE")
    print("="*70)
    print("""
    Complex analysis:  sqrt(z) has holonomy around z=0
    Quantum mechanics: Berry phase is holonomy in parameter space  
    Electromagnetism:  Aharonov-Bohm is holonomy of vector potential
    General relativity: Frame dragging is holonomy of spacetime connection
    Quantum gravity:   LQG quantizes holonomy directly
    
    IT'S ALL THE SAME MATH.
    
    When you track sqrt(1) = +1 vs sqrt(1) = -1 based on path,
    you're doing the same thing the universe does:
    
        REMEMBERING TOPOLOGY.
    
    The universe is a holonomy machine.
    PR-Root is its arithmetic.
    """)


if __name__ == "__main__":
    run_all()
