#!/usr/bin/env python3
"""
plot_validation.py — Compare solver results against Mie-series analytical solutions.

Produces publication-quality matplotlib figures for every boundary condition
type the solver handles. Each plot shows the solver result overlaid on the
exact Mie solution, with the error plotted below.

Usage:
    python plot_validation.py                   # run all tests, display plots
    python plot_validation.py --save validation  # save to validation_*.png
    python plot_validation.py --quick            # fast run (fewer points)
    python plot_validation.py --fmm              # include FMM vs dense comparison

Requires: rcs_solver.py, test_validation.py, fmm_helmholtz_2d.py (for --fmm)
"""

import argparse
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

from rcs_solver import (
    solve_monostatic_rcs_2d,
    solve_bistatic_rcs_2d,
    C0,
)
from test_validation import (
    _build_pec_cylinder_snapshot,
    _build_dielectric_cylinder_snapshot,
    _build_ibc_cylinder_snapshot,
    _mie_backscatter_2d_pec_cylinder,
    _mie_backscatter_2d_dielectric_cylinder,
    _mie_backscatter_2d_ibc_cylinder,
    _mie_bistatic_2d_pec_cylinder,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mie helpers
# ═══════════════════════════════════════════════════════════════════════════════

def mie_bistatic_pattern_pec(ka, pol, theta_inc_deg, n_obs=361):
    """Full 360-degree bistatic Mie pattern for PEC cylinder."""
    obs = np.linspace(0, 360, n_obs)
    rcs = np.array([_mie_bistatic_2d_pec_cylinder(ka, pol, theta_inc_deg, float(a))
                    for a in obs])
    return obs, rcs


def mie_monostatic_sweep_pec(ka_values, pol):
    """Monostatic RCS vs ka for PEC cylinder."""
    rcs = []
    for ka in ka_values:
        _, sigma = _mie_backscatter_2d_pec_cylinder(ka, pol)
        rcs.append(sigma)
    return np.array(rcs)


def mie_bistatic_dielectric(ka, eps_r, mu_r, pol, theta_inc_deg, n_obs=361):
    """Full bistatic Mie for dielectric cylinder (computed from series)."""
    from scipy.special import jv, jvp, hankel2, h2vp
    n_terms = max(int(ka) + 20, 40)
    obs = np.linspace(0, 360, n_obs)
    rcs = np.zeros(n_obs)
    k1a = ka * np.sqrt(complex(eps_r) * complex(mu_r))
    for idx, theta_obs in enumerate(obs):
        rel = math.radians(180.0 - (theta_obs - theta_inc_deg))
        amp = 0.0 + 0.0j
        for n in range(-n_terms, n_terms + 1):
            if pol.upper() == 'TE':
                # TE: match Ez and dEz/dr at r=a
                num = mu_r * jv(n, ka) * jvp(n, k1a) - jvp(n, ka) * jv(n, k1a)
                den = mu_r * hankel2(n, ka) * jvp(n, k1a) - h2vp(n, ka) * jv(n, k1a)
            else:
                # TM: match Hz and dHz/dr at r=a
                num = eps_r * jv(n, ka) * jvp(n, k1a) - jvp(n, ka) * jv(n, k1a)
                den = eps_r * hankel2(n, ka) * jvp(n, k1a) - h2vp(n, ka) * jv(n, k1a)
            a_n = -num / den
            amp += a_n * np.exp(-1j * n * rel)
        amp *= 4.0
        rcs[idx] = float(np.abs(amp) ** 2) / (4.0 * ka)
    return obs, rcs


# ═══════════════════════════════════════════════════════════════════════════════
# Solver wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def solve_monostatic(snap, freq_ghz, angles, pol, method="auto"):
    r = solve_monostatic_rcs_2d(
        snap, [freq_ghz], list(angles), pol, "meters",
        solver_method=method,
    )
    rcs_db = [s["rcs_db"] for s in r["samples"]]
    rcs_lin = [s["rcs_linear"] for s in r["samples"]]
    return np.array(rcs_db), np.array(rcs_lin)


def solve_bistatic(snap, freq_ghz, theta_inc, obs_angles, pol, method="auto"):
    r = solve_bistatic_rcs_2d(
        snap, [freq_ghz], [theta_inc], list(obs_angles), pol, "meters",
        solver_method=method,
    )
    rcs_lin = [s["rcs_linear"] for s in r["samples"]]
    return np.array(rcs_lin)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot functions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_monostatic_vs_ka(fig_num, ka_range, n_sides, quick=False):
    """Plot 1: Monostatic RCS vs ka for PEC cylinder, TE and TM."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), num=fig_num)
    fig.suptitle("PEC Cylinder — Monostatic RCS vs Electrical Size (ka)", fontsize=13)

    for col, pol in enumerate(["TE", "TM"]):
        mie_rcs = mie_monostatic_sweep_pec(ka_range, pol)
        mie_db = 10 * np.log10(np.maximum(mie_rcs, 1e-30))

        solver_db = []
        for ka in ka_range:
            freq = (ka / 1.0) * C0 / (2 * math.pi * 1e9)
            snap = _build_pec_cylinder_snapshot(1.0, n_sides)
            db, _ = solve_monostatic(snap, freq, [0.0], pol)
            solver_db.append(db[0])
        solver_db = np.array(solver_db)

        # RCS comparison
        ax = axes[0, col]
        ax.plot(ka_range, mie_db, "k-", linewidth=2, label="Mie (exact)")
        ax.plot(ka_range, solver_db, "ro", markersize=5, label=f"Solver ({n_sides} panels)")
        ax.set_xlabel("ka")
        ax.set_ylabel("RCS (dBsm/m)")
        ax.set_title(f"{pol} Polarization")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Error
        ax2 = axes[1, col]
        err = np.abs(solver_db - mie_db)
        ax2.semilogy(ka_range, err, "b.-")
        ax2.set_xlabel("ka")
        ax2.set_ylabel("|Error| (dB)")
        ax2.set_title(f"{pol} Error vs ka")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0.1, color="r", linestyle="--", alpha=0.5, label="0.1 dB")
        ax2.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_bistatic_pec(fig_num, ka, n_sides, n_obs=181):
    """Plot 2: Bistatic RCS pattern for PEC cylinder."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), num=fig_num)
    fig.suptitle(f"PEC Cylinder — Bistatic RCS Pattern (ka={ka})", fontsize=13)

    freq = (ka / 1.0) * C0 / (2 * math.pi * 1e9)
    snap = _build_pec_cylinder_snapshot(1.0, n_sides)
    obs_angles = np.linspace(0, 360, n_obs)

    for col, pol in enumerate(["TE", "TM"]):
        # Mie
        mie_obs, mie_rcs = mie_bistatic_pattern_pec(ka, pol, 0.0, n_obs)
        mie_db = 10 * np.log10(np.maximum(mie_rcs, 1e-30))

        # Solver
        solver_rcs = solve_bistatic(snap, freq, 0.0, obs_angles, pol)
        solver_db = 10 * np.log10(np.maximum(solver_rcs, 1e-30))

        ax = axes[col]
        ax.plot(mie_obs, mie_db, "k-", linewidth=1.5, label="Mie (exact)")
        ax.plot(obs_angles, solver_db, "r--", linewidth=1.0, label=f"Solver ({n_sides}-gon)")
        ax.set_xlabel("Observation angle (deg)")
        ax.set_ylabel("RCS (dBsm/m)")
        ax.set_title(f"{pol} Polarization")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_bistatic_polar(fig_num, ka, n_sides, n_obs=361):
    """Plot 3: Bistatic pattern in polar coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), num=fig_num,
                              subplot_kw={"projection": "polar"})
    fig.suptitle(f"PEC Cylinder — Polar Bistatic Pattern (ka={ka})", fontsize=13)

    freq = (ka / 1.0) * C0 / (2 * math.pi * 1e9)
    snap = _build_pec_cylinder_snapshot(1.0, n_sides)
    obs_angles = np.linspace(0, 360, n_obs)

    for col, pol in enumerate(["TE", "TM"]):
        mie_obs, mie_rcs = mie_bistatic_pattern_pec(ka, pol, 0.0, n_obs)
        mie_db = 10 * np.log10(np.maximum(mie_rcs, 1e-30))

        solver_rcs = solve_bistatic(snap, freq, 0.0, obs_angles, pol)
        solver_db = 10 * np.log10(np.maximum(solver_rcs, 1e-30))

        ax = axes[col]
        ax.plot(np.deg2rad(mie_obs), mie_db, "k-", linewidth=1.5, label="Mie")
        ax.plot(np.deg2rad(obs_angles), solver_db, "r--", linewidth=1.0, label="Solver")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_title(f"{pol}", fontsize=11, pad=15)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def plot_material_comparison(fig_num, ka, n_sides):
    """Plot 4: All material types compared to Mie at one ka."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), num=fig_num)
    fig.suptitle(f"All Material Types — Monostatic RCS Comparison (ka={ka})", fontsize=13)

    freq = (ka / 1.0) * C0 / (2 * math.pi * 1e9)
    angles = np.linspace(0, 180, 37)

    cases = [
        ("PEC TE", "TE", _build_pec_cylinder_snapshot(1.0, n_sides),
         lambda a: _mie_bistatic_2d_pec_cylinder(ka, "TE", float(a), float(a))),
        ("PEC TM", "TM", _build_pec_cylinder_snapshot(1.0, n_sides),
         lambda a: _mie_bistatic_2d_pec_cylinder(ka, "TM", float(a), float(a))),
        ("Dielectric TE\nεr=4", "TE", _build_dielectric_cylinder_snapshot(1.0, n_sides, 4.0, 1.0),
         lambda a: None),  # monostatic only
        ("Dielectric TM\nεr=4", "TM", _build_dielectric_cylinder_snapshot(1.0, n_sides, 4.0, 1.0),
         lambda a: None),
        ("IBC TE\nZs=1+j", "TE", _build_ibc_cylinder_snapshot(1.0, n_sides, 1.0, 1.0),
         lambda a: None),
        ("IBC TM\nZs=1+j", "TM", _build_ibc_cylinder_snapshot(1.0, n_sides, 1.0, 1.0),
         lambda a: None),
    ]

    for idx, (title, pol, snap, mie_fn) in enumerate(cases):
        ax = axes[idx // 3, idx % 3]

        # Solver monostatic sweep
        solver_db, solver_lin = solve_monostatic(snap, freq, angles, pol)

        # Mie reference
        if "PEC" in title:
            mie_lin = np.array([_mie_bistatic_2d_pec_cylinder(ka, pol, float(a), float(a))
                                for a in angles])
        elif "Dielectric" in title:
            _, mie_sigma = _mie_backscatter_2d_dielectric_cylinder(ka, 4.0, 1.0, pol)
            # Monostatic for dielectric is angle-independent (circular symmetry)
            mie_lin = np.full(len(angles), mie_sigma)
        elif "IBC" in title:
            mie_sigma = _mie_backscatter_2d_ibc_cylinder(ka, 1.0 + 1.0j, pol)
            mie_lin = np.full(len(angles), mie_sigma)

        mie_db = 10 * np.log10(np.maximum(mie_lin, 1e-30))

        ax.plot(angles, mie_db, "k-", linewidth=2, label="Mie")
        ax.plot(angles, solver_db, "ro", markersize=4, label="Solver")
        ax.set_xlabel("Elevation (deg)")
        ax.set_ylabel("RCS (dB)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate max error
        err = np.abs(solver_db - mie_db)
        ax.text(0.98, 0.02, f"max err: {np.max(err):.3f} dB",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_mesh_convergence(fig_num, ka):
    """Plot 5: Error vs mesh density for PEC cylinder."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), num=fig_num)
    fig.suptitle(f"Mesh Convergence — PEC Cylinder (ka={ka})", fontsize=13)

    freq = (ka / 1.0) * C0 / (2 * math.pi * 1e9)
    n_sides_list = [20, 30, 40, 60, 80, 120, 160, 240]

    for col, pol in enumerate(["TE", "TM"]):
        _, mie_sigma = _mie_backscatter_2d_pec_cylinder(ka, pol)
        mie_db = 10 * math.log10(mie_sigma)

        errors = []
        ppw = []  # panels per wavelength
        for n in n_sides_list:
            snap = _build_pec_cylinder_snapshot(1.0, n)
            db, _ = solve_monostatic(snap, freq, [0.0], pol)
            err = abs(db[0] - mie_db)
            errors.append(err)
            circumference = 2 * math.pi * 1.0  # radius=1
            panel_length = circumference / n
            wavelength = 2 * math.pi / (ka / 1.0)  # k = ka/a, lambda = 2pi/k
            ppw.append(wavelength / panel_length)

        ax = axes[col]
        ax.loglog(ppw, errors, "bo-", markersize=6, linewidth=1.5)
        ax.set_xlabel("Panels per wavelength")
        ax.set_ylabel("|RCS error| (dB)")
        ax.set_title(f"{pol} Polarization")
        ax.grid(True, alpha=0.3, which="both")
        ax.axhline(0.1, color="r", linestyle="--", alpha=0.5, label="0.1 dB target")
        ax.axhline(0.01, color="g", linestyle="--", alpha=0.5, label="0.01 dB target")
        ax.legend(fontsize=9)

        # Add reference slope
        if len(ppw) > 2:
            x1, x2 = ppw[1], ppw[-2]
            y1 = errors[1]
            y2 = y1 * (x1 / x2) ** 2  # O(h²) reference line
            ax.plot([x1, x2], [y1, y2], "k--", alpha=0.3, linewidth=0.8)
            ax.text(x2 * 1.1, y2, "O(h²)", fontsize=8, alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_solver_method_comparison(fig_num, ka, n_sides):
    """Plot 6: Dense vs FMM comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), num=fig_num)
    fig.suptitle(f"Solver Method Comparison — PEC Cylinder (ka={ka}, {n_sides} panels)", fontsize=13)

    freq = (ka / 1.0) * C0 / (2 * math.pi * 1e9)
    snap = _build_pec_cylinder_snapshot(1.0, n_sides)
    angles = np.linspace(0, 180, 37)

    # Mie reference
    mie_rcs = np.array([_mie_bistatic_2d_pec_cylinder(ka, "TM", float(a), float(a))
                        for a in angles])
    mie_db = 10 * np.log10(np.maximum(mie_rcs, 1e-30))

    methods = [("auto", "Direct (LU)", "b"), ("fmm", "FMM+GMRES", "r")]
    timings = {}

    for method, label, color in methods:
        try:
            t0 = time.time()
            db, lin = solve_monostatic(snap, freq, angles, "TM", method=method)
            elapsed = time.time() - t0
            timings[label] = elapsed

            axes[0].plot(angles, db, f"{color}o-", markersize=3, linewidth=1, label=f"{label} ({elapsed:.1f}s)")
            err = np.abs(db - mie_db)
            axes[1].semilogy(angles, err, f"{color}.-", label=f"{label} (max={np.max(err):.4f} dB)")
        except Exception as e:
            print(f"  {label} failed: {e}")

    axes[0].plot(angles, mie_db, "k-", linewidth=2, label="Mie (exact)")
    axes[0].set_xlabel("Elevation (deg)")
    axes[0].set_ylabel("RCS (dB)")
    axes[0].set_title("TM Monostatic RCS")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Elevation (deg)")
    axes[1].set_ylabel("|Error| (dB)")
    axes[1].set_title("Error vs Mie")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0.1, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_3d_expansion(fig_num, ka, n_sides, n_obs=361):
    """Plot 7: Angular harmonic expansion — shows how many Fourier modes
    the solver captures vs the Mie series.

    The 2D far-field pattern can be expanded in angular harmonics:
        A(θ) = Σ aₙ eⁱⁿᶿ

    The Mie series gives the exact coefficients. The solver's pattern
    is Fourier-transformed to get its coefficients. Comparing them
    validates the angular resolution of the solver.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), num=fig_num)
    fig.suptitle(f"Angular Harmonic Expansion — PEC Cylinder (ka={ka})", fontsize=13)

    freq = (ka / 1.0) * C0 / (2 * math.pi * 1e9)
    snap = _build_pec_cylinder_snapshot(1.0, n_sides)
    obs_angles = np.linspace(0, 360, n_obs, endpoint=False)

    for col, pol in enumerate(["TE", "TM"]):
        # Mie bistatic pattern
        mie_obs, mie_rcs = mie_bistatic_pattern_pec(ka, pol, 0.0, n_obs)
        mie_amplitude = np.sqrt(mie_rcs * 4 * ka)  # recover |A| from sigma = |A|²/4k

        # Solver bistatic pattern
        solver_rcs = solve_bistatic(snap, freq, 0.0, obs_angles, pol)
        solver_amplitude = np.sqrt(np.maximum(solver_rcs, 0) * 4 * ka)

        # Fourier transform of the amplitude patterns
        mie_fft = np.fft.fft(mie_amplitude) / n_obs
        solver_fft = np.fft.fft(solver_amplitude) / n_obs
        n_modes = min(30, n_obs // 2)
        modes = np.arange(n_modes)

        # Spectrum plot
        ax = axes[0, col]
        ax.semilogy(modes, np.abs(mie_fft[:n_modes]), "k-o", markersize=4,
                     linewidth=1.5, label="Mie (exact)")
        ax.semilogy(modes, np.abs(solver_fft[:n_modes]), "r--s", markersize=3,
                     linewidth=1.0, label="Solver")
        ax.set_xlabel("Harmonic order n")
        ax.set_ylabel("|aₙ|")
        ax.set_title(f"{pol} — Angular Harmonic Spectrum")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Reconstructed pattern comparison
        ax2 = axes[1, col]
        mie_db = 10 * np.log10(np.maximum(mie_rcs, 1e-30))
        solver_db = 10 * np.log10(np.maximum(solver_rcs, 1e-30))

        ax2.plot(obs_angles, mie_db, "k-", linewidth=1.5, label="Mie")
        ax2.plot(obs_angles, solver_db, "r--", linewidth=1.0, label="Solver")
        ax2.set_xlabel("Observation angle (deg)")
        ax2.set_ylabel("RCS (dB)")
        ax2.set_title(f"{pol} — Bistatic Pattern")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Solver validation plots vs Mie series")
    parser.add_argument("--save", help="Save prefix (e.g. 'validation' → validation_1.png ...)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--quick", action="store_true", help="Fewer points for faster run")
    parser.add_argument("--fmm", action="store_true", help="Include FMM vs dense comparison")
    args = parser.parse_args()

    n_sides = 60 if args.quick else 100
    ka = 1.0

    figs = []

    # Plot 1: Monostatic vs ka
    print("Plot 1: Monostatic RCS vs ka ...")
    ka_range = np.arange(0.5, 4.1, 0.5) if args.quick else np.arange(0.5, 5.1, 0.25)
    figs.append(("monostatic_vs_ka", plot_monostatic_vs_ka(1, ka_range, n_sides, args.quick)))

    # Plot 2: Bistatic pattern (Cartesian)
    print("Plot 2: Bistatic RCS pattern ...")
    n_obs = 91 if args.quick else 181
    figs.append(("bistatic_cartesian", plot_bistatic_pec(2, ka, n_sides, n_obs)))

    # Plot 3: Bistatic pattern (Polar)
    print("Plot 3: Polar bistatic pattern ...")
    figs.append(("bistatic_polar", plot_bistatic_polar(3, ka, n_sides, n_obs * 2)))

    # Plot 4: All material types
    print("Plot 4: Material type comparison ...")
    figs.append(("material_comparison", plot_material_comparison(4, ka, n_sides)))

    # Plot 5: Mesh convergence
    print("Plot 5: Mesh convergence ...")
    figs.append(("mesh_convergence", plot_mesh_convergence(5, ka)))

    # Plot 6: Solver method comparison (optional)
    if args.fmm:
        print("Plot 6: Dense vs FMM comparison ...")
        figs.append(("method_comparison", plot_solver_method_comparison(6, ka, n_sides)))

    # Plot 7: Angular harmonic expansion (3D-like validation)
    print("Plot 7: Angular harmonic expansion ...")
    n_obs_3d = 91 if args.quick else 181
    figs.append(("harmonic_expansion", plot_3d_expansion(7, ka, n_sides, n_obs_3d)))

    # Save or show
    if args.save:
        for name, fig in figs:
            path = f"{args.save}_{name}.png"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            print(f"  Saved {path}")
    else:
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
