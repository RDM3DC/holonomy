#!/usr/bin/env python3
"""
Force Density Heatmap & Energy Stability Analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

print('='*70)
print('FORCE DENSITY HEATMAP & ENERGY STABILITY ANALYSIS')
print('='*70)

# Run simulation capturing multiple snapshots
shape = (32, 32)
phi = np.zeros(shape)
G = np.ones(shape) * 1.9184

params = {'alpha': 0.1, 'mu': 0.05, 'dt': 0.5, 'coupling': 0.05}

# Storage
energy_history = []
force_history = []
snapshots = {}

for step in range(200):
    grad_phi_y, grad_phi_x = np.gradient(phi)
    flux_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2)
    
    dG = params['alpha'] * flux_mag - params['mu'] * G
    G = G + params['dt'] * dG
    G = np.clip(G, 0.01, 10.0)
    
    laplacian = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
                 np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4*phi)
    phi = phi + params['dt'] * params['coupling'] * G * laplacian
    
    center = np.array(shape) // 2
    r = np.sqrt((np.arange(shape[0])[:,None] - center[0])**2 +
                (np.arange(shape[1])[None,:] - center[1])**2)
    phi[r < 3] += params['dt'] * 0.5
    
    E_total = np.sum(G)
    energy_history.append(E_total)
    
    grad_G_y, grad_G_x = np.gradient(G)
    force_density = grad_G_x**2 + grad_G_y**2
    F_total = np.sum(force_density)
    force_history.append(F_total)
    
    if step in [0, 25, 50, 100, 150, 199]:
        snapshots[step] = {
            'G': G.copy(), 
            'force_density': force_density.copy(),
            'phi': phi.copy(), 
            'E': E_total, 
            'F': F_total
        }

print()
print('ENERGY STABILITY CHECK')
print('-' * 40)

E_init, E_final = energy_history[0], energy_history[-1]
E_max, E_min = max(energy_history), min(energy_history)

print(f'E_initial = {E_init:.2f}')
print(f'E_final   = {E_final:.2f}')
print(f'E_range   = {E_max - E_min:.2f} ({100*(E_max-E_min)/E_init:.1f}% of initial)')

E_late = energy_history[-50:]
E_late_std, E_late_mean = np.std(E_late), np.mean(E_late)
stability = 'STABLE' if E_late_std < 0.1*E_late_mean else 'OSCILLATING'

print(f'Late-stage: E_mean={E_late_mean:.2f}, std={E_late_std:.4f}')
print(f'Stability: {stability}')

print()
print('FORCE DENSITY STATISTICS')
print('-' * 40)

final_fd = snapshots[199]['force_density']
print(f'f_max={np.max(final_fd):.4f}, f_mean={np.mean(final_fd):.4f}, F_total={np.sum(final_fd):.2f}')

threshold = np.mean(final_fd) + 2*np.std(final_fd)
hotspots = final_fd > threshold
n_hotspots = np.sum(hotspots)
hotspot_force_pct = 100*np.sum(final_fd[hotspots])/np.sum(final_fd)
print(f'Hotspots: {n_hotspots} cells ({100*n_hotspots/1024:.1f}%) produce {hotspot_force_pct:.1f}% of force')

# VISUALIZATION
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Force Density & Energy Stability Analysis', fontsize=16, fontweight='bold')

# Row 1: Force density evolution
for i, step in enumerate([0, 50, 100, 199]):
    ax = fig.add_subplot(3, 4, i+1)
    fd = snapshots[step]['force_density']
    vmax = max(0.01, np.max(fd))
    im = ax.imshow(fd, cmap='hot', origin='lower', norm=LogNorm(vmin=0.001, vmax=vmax))
    ax.set_title('Step %d: F=%.1f' % (step, snapshots[step]['F']))
    plt.colorbar(im, ax=ax, label='f = |grad G|^2')

# Row 2: Conductance evolution
for i, step in enumerate([0, 50, 100, 199]):
    ax = fig.add_subplot(3, 4, i+5)
    im = ax.imshow(snapshots[step]['G'], cmap='viridis', origin='lower')
    ax.set_title('Step %d: E=%.1f' % (step, snapshots[step]['E']))
    plt.colorbar(im, ax=ax, label='G')

# Row 3: Analysis
ax_e = fig.add_subplot(3, 4, 9)
ax_e.plot(energy_history, 'b-', linewidth=2, label='E = integral(G) dA')
ax_e.axhline(E_late_mean, color='r', linestyle='--', alpha=0.7, label='Equilib (%.1f)' % E_late_mean)
ax_e.fill_between(range(150, 200), E_late_mean - E_late_std, E_late_mean + E_late_std, alpha=0.3, color='red')
ax_e.set_xlabel('Step')
ax_e.set_ylabel('Total Energy')
ax_e.set_title('Energy Evolution')
ax_e.legend()
ax_e.grid(True, alpha=0.3)

ax_f = fig.add_subplot(3, 4, 10)
ax_f.plot(force_history, 'r-', linewidth=2, label='F = integral(|grad G|^2) dA')
ax_f.axhline(262, color='g', linestyle='--', alpha=0.7, label='Target (262)')
ax_f.set_xlabel('Step')
ax_f.set_ylabel('Total Force')
ax_f.set_title('Force Evolution')
ax_f.legend()
ax_f.grid(True, alpha=0.3)

ax_3d = fig.add_subplot(3, 4, 11, projection='3d')
X, Y = np.meshgrid(range(32), range(32))
ax_3d.plot_surface(X, Y, final_fd, cmap='hot', alpha=0.9)
ax_3d.set_xlabel('x')
ax_3d.set_ylabel('y')
ax_3d.set_zlabel('f')
ax_3d.set_title('Force Density 3D')

ax_hot = fig.add_subplot(3, 4, 12)
ax_hot.imshow(snapshots[199]['G'], cmap='Greys', origin='lower', alpha=0.5)
ax_hot.imshow(np.ma.masked_where(~hotspots, final_fd), cmap='hot', origin='lower')
ax_hot.contour(hotspots, colors='cyan', linewidths=1.5)
ax_hot.set_title('Hotspots: %d cells = %.0f%% force' % (n_hotspots, hotspot_force_pct))

plt.tight_layout()
plt.savefig('force_density_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print()
print('='*70)
print('PHYSICAL INTERPRETATION')
print('='*70)
print('''
Force Density Heatmap shows WHERE the "pull" comes from:
  - Hot spots (red/yellow): High |grad G|^2 = strong local tension
  - These are the "ferrofluid spike tips"
  - %d cells produce %.0f%% of total force

Energy Stability:
  - System reaches equilibrium at E ~ %.1f
  - Fluctuation: +/- %.2f (%.2f%%)
  - %s configuration = reliable pull force

Ferrofluid Analog:
  - Total E = "magnetization" x volume
  - Hotspots = spike tips under field gradient
  - Equilibrium E = self-consistent magnetic structure
''' % (n_hotspots, hotspot_force_pct, E_late_mean, E_late_std, 
       100*E_late_std/E_late_mean, stability))

print('Saved: force_density_heatmap.png')
