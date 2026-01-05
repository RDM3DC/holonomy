"""
🌀 HOLONOMY MIND-BENDERS 🌀
Wild demonstrations of topological phase magic

"The same number is not the same number."
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from IPython.display import HTML
import cmath
import math

# Add parent to path for prroot
import sys
sys.path.insert(0, '..')

from prroot import PRConfig, PRState
from prroot.operations import pr_sqrt, pr_unwrap_path, pr_nthroot

cfg = PRConfig()

print("🌀 HOLONOMY MIND-BENDERS LOADED 🌀")
print("Prepare to question reality...")


# =============================================================================
# 🎭 MIND-BENDER #1: THE IMPOSSIBLE EQUATION
# =============================================================================

def impossible_equation():
    """
    Prove that 1 = -1 (sort of)
    
    We show that √1 can equal both +1 AND -1 depending on history.
    """
    print("=" * 60)
    print("🎭 MIND-BENDER #1: THE IMPOSSIBLE EQUATION")
    print("=" * 60)
    print("\nWe will now prove that √1 = +1 AND √1 = -1")
    print("...depending on how you got to 1.\n")
    
    # Path A: Direct to z=1
    z_direct = PRState.from_complex(1 + 0j, config=cfg)
    sqrt_direct, _ = pr_sqrt(z_direct)
    
    # Path B: Go around the origin once, end at z=1
    n = 100
    path_loop = [cmath.exp(2j * cmath.pi * k / n) for k in range(n + 1)]
    states_loop = pr_unwrap_path(path_loop, config=cfg)
    sqrt_loop, _ = pr_sqrt(states_loop[-1])
    
    print(f"Path A (direct to 1):     √1 = {sqrt_direct.z: .4f}")
    print(f"Path B (loop around 0):   √1 = {sqrt_loop.z: .4f}")
    print(f"\nSame point. Different answers.")
    print(f"The universe remembers where you've been. 🌀")
    
    return sqrt_direct, sqrt_loop


# =============================================================================
# 🔐 MIND-BENDER #2: THE TOPOLOGICAL LOCK
# =============================================================================

def topological_lock(secret_winding=3):
    """
    A lock that only opens if you've wound around the origin 
    exactly the right number of times.
    
    This is basically how topological quantum computing works.
    """
    print("\n" + "=" * 60)
    print("🔐 MIND-BENDER #2: THE TOPOLOGICAL LOCK")
    print("=" * 60)
    print(f"\nThe lock requires exactly {secret_winding} winds around the origin.")
    print("Let's try different paths to z=1...\n")
    
    def try_unlock(n_loops):
        n = 100 * max(1, abs(n_loops))
        if n_loops == 0:
            path = [1 + 0j]
        else:
            path = [cmath.exp(2j * cmath.pi * n_loops * k / n) for k in range(n + 1)]
        states = pr_unwrap_path(path, config=cfg)
        final_w = states[-1].w
        unlocked = (final_w == secret_winding)
        status = "🔓 UNLOCKED!" if unlocked else "🔒 locked"
        print(f"  {n_loops} loop(s) → winding = {final_w} → {status}")
        return unlocked
    
    for loops in [0, 1, 2, 3, 4, 5]:
        try_unlock(loops)
    
    print(f"\nOnly the path with w={secret_winding} opens the lock.")
    print("You can't fake topology. 🔐")


# =============================================================================
# 🎪 MIND-BENDER #3: THE FOURTH ROOT CIRCUS
# =============================================================================

def fourth_root_circus():
    """
    ⁴√1 has FOUR values: 1, i, -1, -i
    
    We visit ALL of them by winding 0, 1, 2, 3 times.
    After 4 winds, we're back to the first value.
    """
    print("\n" + "=" * 60)
    print("🎪 MIND-BENDER #3: THE FOURTH ROOT CIRCUS")
    print("=" * 60)
    print("\n⁴√1 has four values. We'll visit them ALL.\n")
    
    results = []
    for w in range(5):
        # Create z=1 with winding w
        theta_lift = 2 * math.pi * w
        state = PRState.from_complex(1 + 0j, config=cfg, theta_lift=theta_lift)
        root, sheet = pr_nthroot(state, 4)
        results.append(root.z)
        print(f"  Winding {w}: ⁴√1 = {root.z: .4f}  (sheet {sheet})")
    
    print(f"\nAfter 4 winds, we're back: {results[0]:.4f} ≈ {results[4]:.4f}")
    print("The fourth roots form a cycle: 1 → i → -1 → -i → 1")
    print("Holonomy creates a ℤ₄ group action! 🎪")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Plot the four roots
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['w=0: 1', 'w=1: i', 'w=2: -1', 'w=3: -i']
    for i, (z, c, l) in enumerate(zip(results[:4], colors, labels)):
        ax.plot(z.real, z.imag, 'o', color=c, markersize=20, label=l)
        ax.annotate(l, (z.real, z.imag), xytext=(10, 10), 
                   textcoords='offset points', fontsize=12)
    
    # Arrows showing the cycle
    for i in range(4):
        z1, z2 = results[i], results[(i+1) % 4]
        ax.annotate('', xy=(z2.real*0.8, z2.imag*0.8), 
                   xytext=(z1.real*0.8, z1.imag*0.8),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('⁴√1: Four Values, One Number\nHolonomy cycles through all roots', fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('fourth_root_circus.png', dpi=150)
    plt.show()
    
    return results


# =============================================================================
# 🌊 MIND-BENDER #4: THE HOLONOMY WAVE
# =============================================================================

def holonomy_wave():
    """
    Create a wave where the PHASE is the message,
    and the winding number is the "envelope."
    
    Two waves with the same frequency but different winding
    interfere destructively — they're on different sheets!
    """
    print("\n" + "=" * 60)
    print("🌊 MIND-BENDER #4: THE HOLONOMY WAVE")
    print("=" * 60)
    print("\nTwo waves, same frequency, different holonomy.")
    print("They should add... but do they?\n")
    
    t = np.linspace(0, 4*np.pi, 500)
    
    # Wave 1: winding 0
    wave1 = np.exp(1j * t)
    states1 = pr_unwrap_path(wave1.tolist(), config=cfg)
    
    # Wave 2: winding 1 (started after one loop)
    wave2_raw = np.exp(1j * t)
    # Give it initial winding of 1
    states2 = []
    init_state = PRState.from_complex(wave2_raw[0], config=cfg, theta_lift=2*np.pi)
    states2.append(init_state)
    for z in wave2_raw[1:]:
        states2.append(states2[-1].update_from_complex(z))
    
    # Get sqrt of each
    sqrt1 = [pr_sqrt(s)[0].z for s in states1]
    sqrt2 = [pr_sqrt(s)[0].z for s in states2]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original waves (same!)
    ax = axes[0, 0]
    ax.plot(t, [s.z.real for s in states1], 'b-', label='Wave 1', linewidth=2)
    ax.plot(t, [s.z.real for s in states2], 'r--', label='Wave 2', linewidth=2)
    ax.set_title('Original waves (identical!)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Their sqrt (DIFFERENT!)
    ax = axes[0, 1]
    ax.plot(t, [z.real for z in sqrt1], 'b-', label='√Wave1', linewidth=2)
    ax.plot(t, [z.real for z in sqrt2], 'r--', label='√Wave2', linewidth=2)
    ax.set_title('Square roots (OPPOSITE!)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sum of sqrt
    ax = axes[1, 0]
    sum_sqrt = [z1 + z2 for z1, z2 in zip(sqrt1, sqrt2)]
    ax.plot(t, [z.real for z in sum_sqrt], 'purple', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title('√Wave1 + √Wave2 ≈ 0 (destructive interference!)')
    ax.grid(True, alpha=0.3)
    
    # Winding numbers
    ax = axes[1, 1]
    ax.plot(t, [s.w for s in states1], 'b-', label='w₁', linewidth=2)
    ax.plot(t, [s.w for s in states2], 'r--', label='w₂', linewidth=2)
    ax.set_title('Winding numbers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('🌊 HOLONOMY WAVE: Same waves, opposite square roots!', fontsize=14)
    plt.tight_layout()
    plt.savefig('holonomy_wave.png', dpi=150)
    plt.show()
    
    print("The waves are IDENTICAL in value.")
    print("But their square roots are OPPOSITE.")
    print("Add them: destructive interference from pure topology! 🌊")


# =============================================================================
# 🎰 MIND-BENDER #5: THE HOLONOMY SLOT MACHINE
# =============================================================================

def holonomy_slot_machine():
    """
    A chaotic system where tiny changes in winding 
    lead to completely different outcomes.
    
    Butterfly effect, but topological.
    """
    print("\n" + "=" * 60)
    print("🎰 MIND-BENDER #5: THE HOLONOMY SLOT MACHINE")
    print("=" * 60)
    print("\nSame endpoint, 7 different paths, 7 different fates.\n")
    
    def compute_fate(n_loops):
        """Apply sqrt 10 times, see where we end up."""
        if n_loops == 0:
            state = PRState.from_complex(1 + 0j, config=cfg)
        else:
            n = 100 * abs(n_loops)
            path = [cmath.exp(2j * cmath.pi * n_loops * k / n) for k in range(n + 1)]
            states = pr_unwrap_path(path, config=cfg)
            state = states[-1]
        
        # Apply sqrt 10 times
        trajectory = [state.z]
        for _ in range(10):
            state, _ = pr_sqrt(state)
            trajectory.append(state.z)
        
        return trajectory
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, n_loops in enumerate(range(7)):
        traj = compute_fate(n_loops)
        ax = axes[i]
        
        # Plot trajectory in complex plane
        x = [z.real for z in traj]
        y = [z.imag for z in traj]
        
        # Color by iteration
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='viridis', linewidth=2)
        lc.set_array(np.arange(len(traj)))
        ax.add_collection(lc)
        
        ax.plot(x[0], y[0], 'go', markersize=10, label='start')
        ax.plot(x[-1], y[-1], 'r*', markersize=15, label='end')
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'w={n_loops}: end={traj[-1]:.3f}')
        ax.grid(True, alpha=0.3)
    
    # Hide last subplot
    axes[7].axis('off')
    axes[7].text(0.5, 0.5, 'Same z=1\nDifferent histories\nDifferent fates\n\n🎰', 
                 ha='center', va='center', fontsize=14, transform=axes[7].transAxes)
    
    plt.suptitle('🎰 HOLONOMY SLOT MACHINE: 10× √ from z=1 with different winding', fontsize=14)
    plt.tight_layout()
    plt.savefig('holonomy_slot_machine.png', dpi=150)
    plt.show()


# =============================================================================
# 🌀 MIND-BENDER #6: THE PHASE VORTEX
# =============================================================================

def phase_vortex():
    """
    Visualize the holonomy as a vortex in phase space.
    
    Points spiral around the origin, accumulating phase forever,
    but their complex value keeps cycling.
    """
    print("\n" + "=" * 60)
    print("🌀 MIND-BENDER #6: THE PHASE VORTEX")
    print("=" * 60)
    print("\nThe complex value cycles. The phase spirals forever.\n")
    
    # Create a path that winds 5 times
    n = 500
    t = np.linspace(0, 10*np.pi, n)
    z_path = np.exp(1j * t)
    
    states = pr_unwrap_path(z_path.tolist(), config=cfg)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Left: Complex plane (cycles)
    ax1 = fig.add_subplot(131)
    x = [s.z.real for s in states]
    y = [s.z.imag for s in states]
    ax1.plot(x, y, 'b-', linewidth=0.5)
    ax1.plot(x[0], y[0], 'go', markersize=10)
    ax1.plot(x[-1], y[-1], 'r*', markersize=15)
    ax1.set_aspect('equal')
    ax1.set_title('Complex plane: cycles forever')
    ax1.grid(True, alpha=0.3)
    
    # Middle: Phase vs time (spirals)
    ax2 = fig.add_subplot(132)
    theta = [s.theta for s in states]
    theta_R = [s.theta_R for s in states]
    ax2.plot(t, theta, 'b-', linewidth=2, label='θ (lifted)')
    ax2.plot(t, theta_R, 'r-', linewidth=1, alpha=0.5, label='θ_R (principal)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Phase')
    ax2.set_title('Phase: lifted grows forever')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Right: 3D vortex (z-plane + phase axis)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(x, y, theta, 'b-', linewidth=1)
    ax3.scatter([x[0]], [y[0]], [theta[0]], color='green', s=100, label='start')
    ax3.scatter([x[-1]], [y[-1]], [theta[-1]], color='red', s=100, marker='*', label='end')
    ax3.set_xlabel('Re(z)')
    ax3.set_ylabel('Im(z)')
    ax3.set_zlabel('θ (lifted)')
    ax3.set_title('3D: The Phase Vortex')
    ax3.legend()
    
    plt.suptitle('🌀 PHASE VORTEX: z cycles, θ spirals to infinity', fontsize=14)
    plt.tight_layout()
    plt.savefig('phase_vortex.png', dpi=150)
    plt.show()
    
    print(f"After 5 loops: z = {states[-1].z:.4f} (back to start)")
    print(f"But θ = {states[-1].theta:.4f} (NOT back to start)")
    print(f"Winding w = {states[-1].w}")
    print("\nThe complex number lies. The phase remembers. 🌀")


# =============================================================================
# RUN ALL MIND-BENDERS
# =============================================================================

def run_all():
    """Blow all the minds."""
    print("\n" + "🌀" * 30)
    print("   HOLONOMY MIND-BENDERS: PREPARE FOR REALITY DISTORTION")
    print("🌀" * 30 + "\n")
    
    impossible_equation()
    topological_lock()
    fourth_root_circus()
    holonomy_wave()
    holonomy_slot_machine()
    phase_vortex()
    
    print("\n" + "=" * 60)
    print("🎤 MIC DROP 🎤")
    print("=" * 60)
    print("""
    What you just witnessed:
    
    1. √1 = +1 AND √1 = -1 (same point, different history)
    2. A lock that only opens with the right topology
    3. Four different fourth roots, accessed by winding
    4. Identical waves with opposite square roots
    5. Chaos from topology alone
    6. The infinite phase vortex
    
    This is HOLONOMY.
    
    The path matters.
    The universe has memory.
    Mathematics is stranger than you thought.
    
    🌀 github.com/RDM3DC/holonomy 🌀
    """)


if __name__ == "__main__":
    run_all()
