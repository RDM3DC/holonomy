"""
PR-Root Visualization Module

Optional visualization tools for PR-Root states.
Requires: numpy, matplotlib (not part of core dependencies)
"""
from __future__ import annotations

import math
import cmath
from typing import List, Optional

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .state import PRState
from .config import PRConfig


def _require_matplotlib():
    """Raise ImportError if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Visualization requires numpy and matplotlib. "
            "Install with: pip install numpy matplotlib"
        )


def plot_riemann_surface(
    states: List[PRState],
    *,
    ax=None,
    show_z2_sheets: bool = True,
    color_by_winding: bool = True,
    title: Optional[str] = None,
):
    """
    Plot path on ℤ₂ Riemann surface with color-coded winding.
    
    Parameters
    ----------
    states : List[PRState]
        Sequence of phase states from pr_unwrap_path
    ax : matplotlib Axes, optional
        Existing axes to plot on
    show_z2_sheets : bool
        If True, shade the two sheets
    color_by_winding : bool
        If True, color path by winding number
    title : str, optional
        Plot title
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _require_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()
    
    # Extract data
    x = [s.z.real for s in states]
    y = [s.z.imag for s in states]
    windings = [s.w for s in states]
    parities = [s.b for s in states]
    
    if show_z2_sheets:
        # Light shading for sheet indication
        theta = np.linspace(0, 2 * np.pi, 100)
        max_r = max(abs(s.z) for s in states) * 1.2
        ax.fill_between(
            [0, max_r], [0, 0], [max_r, max_r],
            alpha=0.05, color='blue', label='Sheet 0'
        )
        ax.fill_between(
            [0, -max_r], [0, 0], [-max_r, -max_r],
            alpha=0.05, color='red', label='Sheet 1'
        )
    
    if color_by_winding:
        # Create line segments colored by winding
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Normalize windings for colormap
        w_array = np.array(windings)
        norm = plt.Normalize(w_array.min(), w_array.max())
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(w_array[:-1])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax, label='Winding number w')
    else:
        ax.plot(x, y, 'b-', linewidth=2)
    
    # Mark start and end
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax.plot(x[-1], y[-1], 'r*', markersize=12, label='End')
    
    # Mark cut crossings
    for i, s in enumerate(states):
        for crossing in s.crossings:
            if crossing.idx == i:
                ax.plot(x[i], y[i], 'mx', markersize=8, markeredgewidth=2)
    
    # Draw branch cut
    max_r = max(abs(s.z) for s in states) * 1.2
    cfg = states[0].config
    cut_angle = cfg.theta_cut
    ax.plot(
        [0, max_r * math.cos(cut_angle)],
        [0, max_r * math.sin(cut_angle)],
        'k--', linewidth=2, alpha=0.5, label='Branch cut'
    )
    
    # Draw unit circle
    circle = Circle((0, 0), 1, fill=False, color='gray', linestyle=':', alpha=0.5)
    ax.add_patch(circle)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(title or 'PR-Root Path on ℤ₂ Surface')
    
    # Adjust limits
    margin = max_r * 0.1
    ax.set_xlim(-max_r - margin, max_r + margin)
    ax.set_ylim(-max_r - margin, max_r + margin)
    
    return fig, ax


def plot_theta_evolution(
    states: List[PRState],
    *,
    ax=None,
    show_principal: bool = True,
    show_windings: bool = True,
    title: Optional[str] = None,
):
    """
    Plot phase evolution θ(t) with lifted and principal phases.
    
    Parameters
    ----------
    states : List[PRState]
        Sequence of phase states
    ax : matplotlib Axes, optional
        Existing axes to plot on
    show_principal : bool
        If True, also plot θ_R (principal phase)
    show_windings : bool
        If True, mark winding changes
    title : str, optional
        Plot title
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _require_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    t = np.arange(len(states))
    theta = [s.theta for s in states]
    theta_R = [s.theta_R for s in states]
    windings = [s.w for s in states]
    
    # Plot lifted phase
    ax.plot(t, theta, 'b-', linewidth=2, label='θ (lifted)')
    
    if show_principal:
        ax.plot(t, theta_R, 'r--', linewidth=1.5, alpha=0.7, label='θ_R (principal)')
    
    # Add winding markers
    if show_windings:
        for i, (w, w_prev) in enumerate(zip(windings[1:], windings[:-1]), 1):
            if w != w_prev:
                ax.axvline(i, color='green', alpha=0.3, linestyle=':')
                ax.annotate(
                    f'w={w}',
                    xy=(i, theta[i]),
                    xytext=(5, 10),
                    textcoords='offset points',
                    fontsize=8,
                    color='green'
                )
    
    # Add π reference lines
    cfg = states[0].config
    pi_a = cfg.pi_a
    for k in range(-5, 6):
        ax.axhline(k * pi_a, color='gray', alpha=0.2, linestyle=':')
    
    ax.set_xlabel('Step index')
    ax.set_ylabel('Phase (radians)')
    ax.set_title(title or 'Phase Evolution θ(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_parity_timeline(
    states: List[PRState],
    *,
    ax=None,
    title: Optional[str] = None,
):
    """
    Plot parity b ∈ {0,1} over time showing ℤ₂ sheet transitions.
    
    Parameters
    ----------
    states : List[PRState]
        Sequence of phase states
    ax : matplotlib Axes, optional
        Existing axes to plot on
    title : str, optional
        Plot title
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _require_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.get_figure()
    
    t = np.arange(len(states))
    parities = [s.b for s in states]
    
    # Step plot for parity
    ax.step(t, parities, 'b-', where='post', linewidth=2)
    ax.fill_between(t, parities, step='post', alpha=0.3)
    
    # Mark sheet changes
    for i, (b, b_prev) in enumerate(zip(parities[1:], parities[:-1]), 1):
        if b != b_prev:
            ax.axvline(i, color='red', alpha=0.5, linestyle='--')
    
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Sheet 0 (even)', 'Sheet 1 (odd)'])
    ax.set_xlabel('Step index')
    ax.set_title(title or 'ℤ₂ Parity (Sheet) Evolution')
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig, ax


def plot_sqrt_branches(
    *,
    r: float = 1.0,
    n_points: int = 100,
    config: Optional[PRConfig] = None,
    title: Optional[str] = None,
):
    """
    Visualize √z on both sheets of the ℤ₂ covering.
    
    Parameters
    ----------
    r : float
        Radius of the circle to trace
    n_points : int
        Number of points around the circle
    config : PRConfig, optional
        Configuration (uses default if None)
    title : str, optional
        Plot title
        
    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    _require_matplotlib()
    from .operations import pr_sqrt, pr_unwrap_path
    
    cfg = config or PRConfig()
    
    # Create path around origin (2 loops)
    z_path = [r * cmath.exp(1j * 4 * math.pi * k / n_points) for k in range(n_points + 1)]
    states = pr_unwrap_path(z_path, config=cfg)
    
    # Compute square roots
    sqrt_states = []
    parities = []
    for s in states:
        sqrt_s, p = pr_sqrt(s)
        sqrt_states.append(sqrt_s)
        parities.append(p)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original path in z-plane
    ax0 = axes[0]
    x = [s.z.real for s in states]
    y = [s.z.imag for s in states]
    ax0.plot(x, y, 'b-', linewidth=2)
    ax0.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax0.set_aspect('equal')
    ax0.set_title('z-plane (2 loops)')
    ax0.set_xlabel('Re(z)')
    ax0.set_ylabel('Im(z)')
    ax0.grid(True, alpha=0.3)
    
    # Plot 2: √z path colored by parity
    ax1 = axes[1]
    sqrt_x = [s.z.real for s in sqrt_states]
    sqrt_y = [s.z.imag for s in sqrt_states]
    
    # Separate by parity for coloring
    points = np.array([sqrt_x, sqrt_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    colors = ['blue' if p == 0 else 'red' for p in parities[:-1]]
    for seg, c in zip(segments, colors):
        ax1.plot(seg[:, 0], seg[:, 1], color=c, linewidth=2)
    
    ax1.plot(sqrt_x[0], sqrt_y[0], 'go', markersize=10)
    ax1.set_aspect('equal')
    ax1.set_title('√z plane (blue=sheet 0, red=sheet 1)')
    ax1.set_xlabel('Re(√z)')
    ax1.set_ylabel('Im(√z)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 3: Phase evolution
    ax2 = axes[2]
    t = np.arange(len(states))
    theta = [s.theta for s in states]
    sqrt_theta = [s.theta for s in sqrt_states]
    
    ax2.plot(t, theta, 'b-', linewidth=2, label='θ(z)')
    ax2.plot(t, sqrt_theta, 'r--', linewidth=2, label='θ(√z)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Phase (radians)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Phase evolution')
    
    fig.suptitle(title or 'Square Root on ℤ₂ Covering', fontsize=14)
    fig.tight_layout()
    
    return fig, axes


def plot_holonomy_demo(
    *,
    n_points: int = 100,
    config: Optional[PRConfig] = None,
):
    """
    Complete holonomy demonstration showing one loop ≠ identity.
    
    Returns a figure showing:
    - Path around origin
    - Phase accumulation
    - Parity flip
    - √z sign change
    """
    _require_matplotlib()
    from .operations import pr_sqrt, pr_unwrap_path
    
    cfg = config or PRConfig()
    
    # One loop
    z_path = [cmath.exp(1j * 2 * math.pi * k / n_points) for k in range(n_points + 1)]
    states = pr_unwrap_path(z_path, config=cfg)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Path in z-plane
    plot_riemann_surface(states, ax=axes[0, 0], title='Path in z-plane')
    
    # Phase evolution
    plot_theta_evolution(states, ax=axes[0, 1], title='Lifted phase θ(t)')
    
    # Parity evolution
    plot_parity_timeline(states, ax=axes[1, 0], title='Parity (ℤ₂ sheet)')
    
    # √z values
    ax3 = axes[1, 1]
    sqrt_states = [pr_sqrt(s)[0] for s in states]
    t = np.arange(len(states))
    sqrt_x = [s.z.real for s in sqrt_states]
    sqrt_y = [s.z.imag for s in sqrt_states]
    
    ax3.plot(t, sqrt_x, 'b-', label='Re(√z)')
    ax3.plot(t, sqrt_y, 'r-', label='Im(√z)')
    ax3.axhline(0, color='gray', linestyle=':')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Value')
    ax3.set_title('√z components (note sign flip at end)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle('ℤ₂ Holonomy: One Loop Changes Sign of √z', fontsize=14)
    fig.tight_layout()
    
    return fig, axes
