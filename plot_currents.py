#!/usr/bin/env python3
"""
plot_currents.py — Visualize surface current output from the 2D BIE/MoM RCS solver.

Reads a surface_currents.json (or current.json) file produced by
compute_surface_currents() and generates publication-quality plots:

  1. Geometry outline colored by current magnitude (dB scale)
  2. Current magnitude vs arc length
  3. Current phase vs arc length
  4. Polar RCS pattern (if rcs.json is also provided)

Usage:
    python plot_currents.py surface_currents.json
    python plot_currents.py surface_currents.json --rcs rcs_output.json
    python plot_currents.py surface_currents.json --save plots.png --dpi 200
"""

import argparse
import json
import math
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_arc_length(cx, cy):
    """Compute cumulative arc length along the panel centers."""
    cx = np.asarray(cx)
    cy = np.asarray(cy)
    ds = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
    s = np.zeros(len(cx))
    s[1:] = np.cumsum(ds)
    return s


def plot_geometry_current(ax, data, db_range=40):
    """Plot geometry outline colored by current magnitude."""
    cx = np.asarray(data["centers_x"])
    cy = np.asarray(data["centers_y"])
    nx = np.asarray(data["normals_x"])
    ny = np.asarray(data["normals_y"])
    lengths = np.asarray(data["lengths"])
    mag = np.asarray(data["density_abs"])

    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-30))
    vmax = np.max(mag_db)
    vmin = vmax - db_range

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Draw each panel as a colored line segment.
    for i in range(len(cx)):
        half = 0.5 * lengths[i]
        tx = -ny[i]  # tangent = 90 deg CW from normal
        ty = nx[i]
        x0 = cx[i] - half * tx
        y0 = cy[i] - half * ty
        x1 = cx[i] + half * tx
        y1 = cy[i] + half * ty
        color = cmap(norm(mag_db[i]))
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.5, solid_capstyle="round")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("|J| (dB)", fontsize=10)

    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    freq = data.get("frequency_ghz", "?")
    elev = data.get("elevation_deg", "?")
    pol = data.get("polarization", "?")
    form = data.get("formulation", "")
    ax.set_title(f"Surface Current — {freq} GHz, {pol}, θ={elev}°\n{form}", fontsize=10)


def plot_magnitude_vs_arc(ax, data):
    """Plot current magnitude vs arc length."""
    cx = np.asarray(data["centers_x"])
    cy = np.asarray(data["centers_y"])
    mag = np.asarray(data["density_abs"])
    s = compute_arc_length(cx, cy)

    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-30))

    ax.plot(s, mag_db, "b-", linewidth=1.2)
    ax.set_xlabel("Arc length (m)", fontsize=10)
    ax.set_ylabel("|J| (dB)", fontsize=10)
    ax.set_title("Current Magnitude vs Position", fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_phase_vs_arc(ax, data):
    """Plot current phase vs arc length."""
    cx = np.asarray(data["centers_x"])
    cy = np.asarray(data["centers_y"])
    phase = np.asarray(data["density_phase_deg"])
    s = compute_arc_length(cx, cy)

    ax.plot(s, phase, "r-", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Arc length (m)", fontsize=10)
    ax.set_ylabel("Phase (deg)", fontsize=10)
    ax.set_title("Current Phase vs Position", fontsize=10)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.grid(True, alpha=0.3)


def plot_rcs_polar(ax, rcs_data):
    """Plot monostatic RCS pattern in polar coordinates."""
    angles = []
    rcs_db = []
    for s in rcs_data.get("samples", []):
        angles.append(float(s["theta_inc_deg"]))
        rcs_db.append(float(s["rcs_db"]))

    if not angles:
        ax.text(0.5, 0.5, "No RCS data", transform=ax.transAxes,
                ha="center", va="center")
        return

    angles_rad = np.deg2rad(angles)
    rcs_db = np.array(rcs_db)

    ax.plot(angles_rad, rcs_db, "b-o", markersize=3, linewidth=1.2)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xlabel("RCS (dB)")

    freq = rcs_data.get("metadata", {}).get("frequencies_ghz", ["?"])
    pol = rcs_data.get("metadata", {}).get("polarization", "?")
    if isinstance(freq, list) and freq:
        freq = freq[0]
    ax.set_title(f"Monostatic RCS — {freq} GHz {pol}", fontsize=10, pad=15)


def plot_current_arrows(ax, data, arrow_scale=0.3):
    """Plot geometry with current direction arrows."""
    cx = np.asarray(data["centers_x"])
    cy = np.asarray(data["centers_y"])
    nx = np.asarray(data["normals_x"])
    ny = np.asarray(data["normals_y"])
    mag = np.asarray(data["density_abs"])
    phase_rad = np.deg2rad(data["density_phase_deg"])

    # Tangent direction (current flows along surface)
    tx = -ny
    ty = nx

    # Current phasor projected onto real part for visualization
    jr = mag * np.cos(phase_rad)
    mag_max = np.max(np.abs(jr)) if np.max(np.abs(jr)) > 0 else 1.0

    # Plot geometry
    ax.plot(cx, cy, "k-", linewidth=0.8, alpha=0.5)

    # Plot arrows showing current direction and magnitude
    scale = arrow_scale * np.max(np.sqrt((cx.max()-cx.min())**2 + (cy.max()-cy.min())**2)) / mag_max
    step = max(1, len(cx) // 40)  # show ~40 arrows

    for i in range(0, len(cx), step):
        dx = jr[i] * tx[i] * scale
        dy = jr[i] * ty[i] * scale
        ax.annotate("", xy=(cx[i]+dx, cy[i]+dy), xytext=(cx[i], cy[i]),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=1.0))

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.set_title("Current Direction (Re{J})", fontsize=10)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Plot surface currents from the 2D BIE/MoM RCS solver.")
    parser.add_argument("current_json", help="Path to surface_currents.json")
    parser.add_argument("--rcs", help="Optional path to RCS output JSON for polar plot")
    parser.add_argument("--save", help="Save figure to file instead of displaying")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default 150)")
    parser.add_argument("--db-range", type=float, default=40,
                        help="Dynamic range in dB for color map (default 40)")
    args = parser.parse_args()

    data = load_json(args.current_json)
    rcs_data = load_json(args.rcs) if args.rcs else None

    n_plots = 4 if rcs_data else 3
    fig = plt.figure(figsize=(14, 10))

    if rcs_data:
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4, polar=True)
    else:
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

    plot_geometry_current(ax1, data, db_range=args.db_range)
    plot_magnitude_vs_arc(ax2, data)
    plot_phase_vs_arc(ax3, data)

    if rcs_data:
        plot_rcs_polar(ax4, rcs_data)
    else:
        plot_current_arrows(ax4, data)

    fig.tight_layout(pad=2.0)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
