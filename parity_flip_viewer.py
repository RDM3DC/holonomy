#!/usr/bin/env python3
"""
PARITY-FLIP VIEWER: Toggle Between Topological States
======================================================

This script generates the multi-sheet visualization that lets you
"slide" between different Riemann sheets of the phase configuration.

The key insight: shifting the winding number w -> w + k changes which
"world" you're viewing, while keeping the same underlying complex values.

Physical interpretation:
  - w = 0: Principal branch (standard sqrt)
  - w = 1: First sheet (sqrt flipped)
  - w = 2: Back to original (for sqrt), or second sheet (for 4th root)
  
The parity indicator b = w mod 2 determines the SIGN of the root.
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from prroot import PRState, PRConfig
from prroot.operations import pr_sqrt, pr_nthroot

# Optional visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
    from matplotlib.colors import hsv_to_rgb, TwoSlopeNorm
    from mpl_toolkits.mplot3d import Axes3D
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Note: Install matplotlib for interactive visualization")


class ParityFlipViewer:
    """
    Interactive viewer for parity-dependent topological configurations.
    
    Allows sliding between Riemann sheets to see how the same complex
    values produce different roots depending on winding history.
    """
    
    def __init__(self, phase_field: np.ndarray, conductance_field: np.ndarray,
                 curvature_field: np.ndarray):
        """
        Initialize viewer with simulation data.
        
        Args:
            phase_field: The lifted phase θ (unwrapped)
            conductance_field: ARP conductance G
            curvature_field: Holonomy curvature |F_xy|
        """
        self.phase = phase_field
        self.conductance = conductance_field
        self.curvature = curvature_field
        self.shape = phase_field.shape
        
        # Current winding offset (which sheet we're viewing)
        self.winding_offset = 0
        
        # Parity threshold for visualization
        self.parity_threshold = 0.5004
        
        # Compute derived fields
        self.update_derived_fields()
    
    def update_derived_fields(self):
        """Recompute parity and sqrt fields based on current winding offset."""
        # Shifted phase (viewing different sheet)
        self.shifted_phase = self.phase + 2 * np.pi * self.winding_offset
        
        # Compute winding number at each point
        self.winding = np.round(self.shifted_phase / (2 * np.pi)).astype(int)
        
        # Parity indicator: b = w mod 2
        self.parity = self.winding % 2
        
        # Compute sqrt with current winding
        self.sqrt_real = np.zeros(self.shape)
        self.sqrt_imag = np.zeros(self.shape)
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Create PR state with current winding
                A = self.conductance[i, j]
                theta = self.shifted_phase[i, j]
                w = self.winding[i, j]
                b = w % 2
                
                state = PRState(A=max(A, 0.01), theta=theta, w=w, b=b)
                sqrt_state, _ = pr_sqrt(state)
                
                z = sqrt_state.complex
                self.sqrt_real[i, j] = z.real
                self.sqrt_imag[i, j] = z.imag
        
        # Branch cut locations: where parity changes
        parity_grad_y = np.abs(np.diff(self.parity, axis=0, prepend=self.parity[:1, :]))
        parity_grad_x = np.abs(np.diff(self.parity, axis=1, prepend=self.parity[:, :1]))
        self.branch_cuts = (parity_grad_y > 0) | (parity_grad_x > 0)
    
    def set_winding_offset(self, offset: int):
        """Change which Riemann sheet we're viewing."""
        self.winding_offset = offset
        self.update_derived_fields()
    
    def get_field_at_point(self, x: int, y: int) -> dict:
        """Get all field values at a specific point."""
        return {
            'phase': self.shifted_phase[y, x],
            'conductance': self.conductance[y, x],
            'curvature': self.curvature[y, x],
            'winding': self.winding[y, x],
            'parity': self.parity[y, x],
            'sqrt': complex(self.sqrt_real[y, x], self.sqrt_imag[y, x]),
            'on_branch_cut': self.branch_cuts[y, x]
        }
    
    def export_fields(self, output_dir: str):
        """Export all fields as .npy files for the viewer."""
        output_path = Path(output_dir)
        fields_dir = output_path / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)
        
        # Export fields matching scene.json expectations
        np.save(fields_dir / "f000_phase.npy", self.shifted_phase)
        np.save(fields_dir / "f001_metrics.npy", self.conductance)
        np.save(fields_dir / "f002_curvature.npy", self.curvature)
        np.save(fields_dir / "f003_parity.npy", self.parity.astype(np.float32))
        np.save(fields_dir / "f004_winding.npy", self.winding.astype(np.float32))
        np.save(fields_dir / "f005_sqrt_real.npy", self.sqrt_real)
        np.save(fields_dir / "f005_sqrt_imag.npy", self.sqrt_imag)
        np.save(fields_dir / "f006_branch_cuts.npy", self.branch_cuts.astype(np.float32))
        
        print(f"[Export] Fields saved to {fields_dir}")
        print(f"[Export] Winding offset: {self.winding_offset}")
        print(f"[Export] Parity distribution: {np.mean(self.parity):.1%} odd")


def create_interactive_viewer(viewer: ParityFlipViewer):
    """Create matplotlib interactive visualization."""
    if not HAS_VIZ:
        print("Matplotlib not available for interactive viewer")
        return
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('PR-Root Parity-Flip Viewer: Slide Between Riemann Sheets', 
                 fontsize=14, fontweight='bold')
    
    # Create axes
    ax_phase = fig.add_subplot(231)
    ax_parity = fig.add_subplot(232)
    ax_sqrt = fig.add_subplot(233)
    ax_cuts = fig.add_subplot(234)
    ax_conduct = fig.add_subplot(235)
    ax_3d = fig.add_subplot(236, projection='3d')
    
    # Initial plots
    im_phase = ax_phase.imshow(viewer.shifted_phase, cmap='hsv', origin='lower')
    ax_phase.set_title('Phase θ (lifted)')
    plt.colorbar(im_phase, ax=ax_phase, label='rad')
    
    im_parity = ax_parity.imshow(viewer.parity, cmap='RdYlGn', origin='lower', 
                                  vmin=0, vmax=1)
    ax_parity.set_title('Parity b = w mod 2')
    plt.colorbar(im_parity, ax=ax_parity, label='0=even, 1=odd')
    
    sqrt_mag = np.sqrt(viewer.sqrt_real**2 + viewer.sqrt_imag**2)
    im_sqrt = ax_sqrt.imshow(sqrt_mag, cmap='viridis', origin='lower')
    ax_sqrt.set_title('|√z| magnitude')
    plt.colorbar(im_sqrt, ax=ax_sqrt, label='amplitude')
    
    im_cuts = ax_cuts.imshow(viewer.branch_cuts.astype(float), cmap='Reds', 
                              origin='lower', vmin=0, vmax=1)
    ax_cuts.set_title('Branch Cuts (parity jumps)')
    
    im_conduct = ax_conduct.imshow(viewer.conductance, cmap='viridis', origin='lower')
    ax_conduct.set_title('Conductance G (structure)')
    plt.colorbar(im_conduct, ax=ax_conduct, label='G')
    
    # 3D surface of sqrt real part
    X, Y = np.meshgrid(range(viewer.shape[1]), range(viewer.shape[0]))
    surf = ax_3d.plot_surface(X, Y, viewer.sqrt_real, cmap='coolwarm', alpha=0.8)
    ax_3d.set_title('√z real part (3D)')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('Re(√z)')
    
    # Slider for winding offset
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Winding Offset', -3, 3, valinit=0, valstep=1)
    
    def update(val):
        offset = int(slider.val)
        viewer.set_winding_offset(offset)
        
        # Update all plots
        im_phase.set_data(viewer.shifted_phase)
        im_phase.set_clim(viewer.shifted_phase.min(), viewer.shifted_phase.max())
        
        im_parity.set_data(viewer.parity)
        
        sqrt_mag = np.sqrt(viewer.sqrt_real**2 + viewer.sqrt_imag**2)
        im_sqrt.set_data(sqrt_mag)
        im_sqrt.set_clim(sqrt_mag.min(), sqrt_mag.max())
        
        im_cuts.set_data(viewer.branch_cuts.astype(float))
        
        ax_3d.clear()
        ax_3d.plot_surface(X, Y, viewer.sqrt_real, cmap='coolwarm', alpha=0.8)
        ax_3d.set_title(f'√z real part (w offset = {offset})')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('Re(√z)')
        
        fig.suptitle(f'PR-Root Parity-Flip Viewer | Sheet w = {offset} | '
                     f'Parity odd: {np.mean(viewer.parity):.1%}', 
                     fontsize=14, fontweight='bold')
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Export button
    ax_export = plt.axes([0.85, 0.02, 0.1, 0.03])
    btn_export = Button(ax_export, 'Export')
    
    def export_clicked(event):
        viewer.export_fields("ama_export")
        print("Exported fields to ama_export/fields/")
    
    btn_export.on_clicked(export_clicked)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('parity_flip_viewer.png', dpi=150, bbox_inches='tight')
    plt.show()


def generate_parity_sweep_frames(viewer: ParityFlipViewer, output_dir: str = "parity_sweep"):
    """Generate frames for all winding offsets (for animation)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating parity sweep frames...")
    
    for offset in range(-3, 4):
        viewer.set_winding_offset(offset)
        
        # Create frame
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Riemann Sheet w = {offset} | Parity odd: {np.mean(viewer.parity):.1%}',
                     fontsize=14, fontweight='bold')
        
        # Phase
        axes[0, 0].imshow(viewer.shifted_phase, cmap='hsv', origin='lower')
        axes[0, 0].set_title('Phase θ')
        
        # Parity
        im_par = axes[0, 1].imshow(viewer.parity, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Parity b ({"ODD" if np.mean(viewer.parity) > 0.5 else "EVEN"} dominant)')
        
        # Branch cuts
        axes[0, 2].imshow(viewer.branch_cuts, cmap='Reds', origin='lower')
        axes[0, 2].set_title('Branch Cuts')
        
        # Sqrt real
        vmax = max(abs(viewer.sqrt_real.min()), abs(viewer.sqrt_real.max()))
        axes[1, 0].imshow(viewer.sqrt_real, cmap='coolwarm', origin='lower',
                          vmin=-vmax, vmax=vmax)
        axes[1, 0].set_title('Re(√z)')
        
        # Sqrt imag
        vmax = max(abs(viewer.sqrt_imag.min()), abs(viewer.sqrt_imag.max()))
        axes[1, 1].imshow(viewer.sqrt_imag, cmap='coolwarm', origin='lower',
                          vmin=-vmax, vmax=vmax)
        axes[1, 1].set_title('Im(√z)')
        
        # Conductance
        axes[1, 2].imshow(viewer.conductance, cmap='viridis', origin='lower')
        axes[1, 2].set_title('Conductance G')
        
        for ax in axes.flat:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        plt.tight_layout()
        frame_path = output_path / f"sheet_w{offset:+d}.png"
        plt.savefig(frame_path, dpi=100)
        plt.close()
        
        print(f"  Saved: {frame_path}")
    
    print(f"\nParity sweep complete. {7} frames in {output_dir}/")


def demonstrate_parity_flip():
    """
    Demonstrate the parity flip at a single point.
    
    Shows how √1 changes sign as winding increases.
    """
    print("\n" + "=" * 70)
    print("PARITY FLIP DEMONSTRATION: √1 on Different Sheets")
    print("=" * 70)
    print("\nThe same point z = 1, viewed from different Riemann sheets:\n")
    
    print(f"{'Winding w':>12} {'Parity b':>12} {'√1':>20} {'Sign':>10}")
    print("-" * 58)
    
    for w in range(-3, 4):
        b = w % 2
        # Create state at z=1 with given winding
        state = PRState(A=1.0, theta=2*np.pi*w, w=w, b=b)
        sqrt_state, parity = pr_sqrt(state)
        
        z = sqrt_state.complex
        sign = "+" if z.real > 0 else "-"
        
        print(f"{w:>12} {b:>12} {z:>20.4f} {sign:>10}")
    
    print("\nNotice: parity b = w mod 2 determines the SIGN of √1")
    print("Even w -> √1 = +1")
    print("Odd w  -> √1 = -1")
    print("\nThis is the 'Many Worlds' of square roots!")


def main():
    """Main entry point for parity flip viewer."""
    print("\n" + "=" * 70)
    print("   PR-ROOT PARITY-FLIP VIEWER")
    print("   Sliding Between Topological Worlds")
    print("=" * 70 + "\n")
    
    # First demonstrate the concept
    demonstrate_parity_flip()
    
    # Try to load existing simulation data
    from pathlib import Path
    
    # Check if we have data from holonomy_arp_bridge
    try:
        from src.io.ama_v0_1 import AMAReader
        reader = AMAReader("vibration_support_50Hz.ama")
        
        phase = reader.get_field("phase_spine")
        conductance = reader.get_field("conductance_geometry")
        curvature = reader.get_field("falsifier_residual")
        
        print(f"\nLoaded data from vibration_support_50Hz.ama")
        print(f"Shape: {phase.shape}")
        
    except (FileNotFoundError, Exception) as e:
        print(f"\nNo existing .ama file found, generating synthetic data...")
        
        # Generate synthetic phase field
        shape = (32, 32)
        x = np.linspace(-np.pi, np.pi, shape[0])
        y = np.linspace(-np.pi, np.pi, shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Phase with spiral pattern
        R = np.sqrt(X**2 + Y**2)
        phase = np.arctan2(Y, X) + 0.5 * R
        
        # Conductance peaked in center
        conductance = 2.0 * np.exp(-R**2 / 4) + 0.5
        
        # Curvature (fake for demo)
        curvature = np.abs(np.gradient(np.gradient(phase)[0])[1])
    
    # Create viewer
    viewer = ParityFlipViewer(phase, conductance, curvature)
    
    # Export initial fields
    viewer.export_fields("ama_export")
    
    # Generate sweep frames
    if HAS_VIZ:
        generate_parity_sweep_frames(viewer)
        
        # Launch interactive viewer
        print("\nLaunching interactive viewer...")
        print("Use slider to change winding offset (Riemann sheet)")
        print("Click 'Export' to save current configuration\n")
        
        create_interactive_viewer(viewer)


if __name__ == "__main__":
    main()
