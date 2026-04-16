#!/usr/bin/env python3
"""
Headless 2D BIE/MoM RCS solver — edit the configuration block below and run.

Usage:
    python run_solver.py

All settings are in the CONFIG section below.  No command-line arguments.
Designed for HPC batch jobs with high memory and many cores.
"""

import json
import math
import os
import sys
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this section
# ═══════════════════════════════════════════════════════════════════════════════

# --- Geometry ---
GEOMETRY_FILE   = "my_geometry.grim"          # path to .grim geometry file
GEOMETRY_UNITS  = "inches"                    # "inches" or "meters"
MATERIAL_DIR    = "."                         # directory containing fort.* material files

# --- Frequencies (GHz) ---
# Option A: explicit list
FREQUENCIES     = [1.0, 2.0, 5.0, 10.0]
# Option B: linspace — uncomment to use instead of the list above
# FREQUENCIES   = list(__import__('numpy').linspace(1.0, 18.0, 35))

# --- Incidence angles (degrees, "coming-from" convention) ---
ELEVATIONS      = [0.0]
# ELEVATIONS    = list(range(0, 181, 2))      # 0 to 180 in 2-degree steps

# --- Polarization ---
POLARIZATION    = "TE"                        # "TE" (E_z / VV) or "TM" (H_z / HH)

# --- Scattering mode ---
#   "monostatic"  — backscatter, obs = inc (default)
#   "bistatic"    — specify OBSERVATION_ANGLES separately
#   "adaptive"    — auto-refine frequency sweep (ignores FREQUENCIES, uses FREQ_START/STOP)
#   "currents"    — compute surface currents at first freq/elev, export to JSON
MODE            = "monostatic"

# --- Bistatic observation angles (only used when MODE = "bistatic") ---
OBSERVATION_ANGLES = [0, 30, 60, 90, 120, 150, 180]

# --- Adaptive sweep settings (only used when MODE = "adaptive") ---
FREQ_START      = 1.0                         # GHz
FREQ_STOP       = 18.0                        # GHz
ADAPTIVE_INITIAL_POINTS = 21
ADAPTIVE_MAX_REFINEMENTS = 4
ADAPTIVE_THRESHOLD_DB = 1.0
ADAPTIVE_MAX_POINTS = 301

# --- Solver settings ---
SOLVER_METHOD   = "auto"                      # "auto", "direct" (LU), "gmres", or "fmm"
CFIE_ALPHA      = 0.2                         # 0 = disabled, 0.2 = default
MESH_REF_GHZ    = None                        # None = re-mesh per freq, float = cache at this freq
MAX_PANELS      = 50_000                      # hard cap on panel count

# --- Output ---
OUTPUT_JSON     = "rcs_result.json"           # solver result (samples + metadata)
OUTPUT_GRIM     = "rcs_output.grim"           # GRIM export (set to None to skip)
OUTPUT_CURRENTS = "surface_currents.json"     # surface current export (MODE="currents")

# ═══════════════════════════════════════════════════════════════════════════════
# END CONFIG — no edits needed below this line
# ═══════════════════════════════════════════════════════════════════════════════


def load_geometry(path: str) -> dict:
    """Load a .grim geometry file and return a solver snapshot dict."""
    from geometry_io import parse_geometry, build_geometry_snapshot

    with open(path, "r") as f:
        text = f.read()

    title, segments, ibcs, dielectrics = parse_geometry(text)
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    snapshot["source_path"] = os.path.abspath(path)
    return snapshot


def print_header(snapshot: dict) -> None:
    n_seg = snapshot.get("segment_count", 0)
    n_ibc = len(snapshot.get("ibcs", []))
    n_diel = len(snapshot.get("dielectrics", []))
    title = snapshot.get("title", "")
    print(f"  Geometry : {title} ({n_seg} segments, {n_ibc} IBC, {n_diel} dielectric)")
    print(f"  Units    : {GEOMETRY_UNITS}")
    print(f"  Mode     : {MODE}")
    print(f"  Pol      : {POLARIZATION}")
    print(f"  Solver   : {SOLVER_METHOD}  CFIE α={CFIE_ALPHA}")
    if MODE == "adaptive":
        print(f"  Freq     : {FREQ_START}–{FREQ_STOP} GHz (adaptive, up to {ADAPTIVE_MAX_POINTS} pts)")
    else:
        print(f"  Freq     : {len(FREQUENCIES)} points, {min(FREQUENCIES):.4g}–{max(FREQUENCIES):.4g} GHz")
    print(f"  Elev     : {len(ELEVATIONS)} angles")
    if MODE == "bistatic":
        print(f"  Obs      : {len(OBSERVATION_ANGLES)} angles")
    print(f"  Max panels: {MAX_PANELS}")
    print()


def progress_callback(done: int, total: int, message: str) -> None:
    if total > 0:
        pct = 100.0 * done / total
    else:
        pct = 0.0
    print(f"  [{pct:5.1f}%] {message}", flush=True)


def run_monostatic(snapshot: dict) -> dict:
    from rcs_solver import solve_monostatic_rcs_2d

    return solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=FREQUENCIES,
        elevations_deg=ELEVATIONS,
        polarization=POLARIZATION,
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=MATERIAL_DIR,
        progress_callback=progress_callback,
        max_panels=MAX_PANELS,
        cfie_alpha=CFIE_ALPHA,
        mesh_reference_ghz=MESH_REF_GHZ,
        solver_method=SOLVER_METHOD,
    )


def run_bistatic(snapshot: dict) -> dict:
    from rcs_solver import solve_bistatic_rcs_2d

    return solve_bistatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=FREQUENCIES,
        incidence_angles_deg=ELEVATIONS,
        observation_angles_deg=OBSERVATION_ANGLES,
        polarization=POLARIZATION,
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=MATERIAL_DIR,
        progress_callback=progress_callback,
        max_panels=MAX_PANELS,
        cfie_alpha=CFIE_ALPHA,
        mesh_reference_ghz=MESH_REF_GHZ,
        solver_method=SOLVER_METHOD,
    )


def run_adaptive(snapshot: dict) -> dict:
    from rcs_solver import solve_adaptive_frequency_sweep

    return solve_adaptive_frequency_sweep(
        geometry_snapshot=snapshot,
        freq_start_ghz=FREQ_START,
        freq_stop_ghz=FREQ_STOP,
        elevations_deg=ELEVATIONS,
        polarization=POLARIZATION,
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=MATERIAL_DIR,
        progress_callback=progress_callback,
        max_panels=MAX_PANELS,
        cfie_alpha=CFIE_ALPHA,
        solver_method=SOLVER_METHOD,
        initial_points=ADAPTIVE_INITIAL_POINTS,
        max_refinements=ADAPTIVE_MAX_REFINEMENTS,
        db_threshold=ADAPTIVE_THRESHOLD_DB,
        max_total_points=ADAPTIVE_MAX_POINTS,
    )


def run_currents(snapshot: dict) -> dict:
    from rcs_solver import compute_surface_currents

    freq = FREQUENCIES[0] if FREQUENCIES else 1.0
    elev = ELEVATIONS[0] if ELEVATIONS else 0.0
    print(f"  Computing surface currents at {freq} GHz, {elev} deg, {POLARIZATION}")

    return compute_surface_currents(
        geometry_snapshot=snapshot,
        frequency_ghz=freq,
        elevation_deg=elev,
        polarization=POLARIZATION,
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=MATERIAL_DIR,
        cfie_alpha=CFIE_ALPHA,
        max_panels=MAX_PANELS,
    )


def print_summary(result: dict) -> None:
    samples = result.get("samples", [])
    if not samples:
        print("  No samples produced.")
        return

    meta = result.get("metadata", {})
    formulation = meta.get("formulation", "unknown")
    print(f"  Formulation : {formulation}")
    print(f"  Samples     : {len(samples)}")

    residual_max = meta.get("residual_norm_max", 0)
    if residual_max:
        print(f"  Residual max: {residual_max:.2e}")

    warnings = meta.get("warnings", [])
    if warnings:
        print(f"  Warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    - {w}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more")

    # Print RCS table (compact).
    freqs = sorted(set(s["frequency_ghz"] for s in samples))
    is_bistatic = any(
        abs(s.get("theta_inc_deg", 0) - s.get("theta_scat_deg", 0)) > 0.01
        for s in samples
    )

    if is_bistatic:
        # Bistatic: group by (freq, inc), show obs angles as columns.
        inc_angles = sorted(set(s.get("theta_inc_deg", 0) for s in samples))
        obs_angles = sorted(set(s.get("theta_scat_deg", 0) for s in samples))

        if len(freqs) * len(inc_angles) <= 20 and len(obs_angles) <= 12:
            for f in freqs:
                for inc in inc_angles:
                    print(f"\n  Freq={f:.4f} GHz  Inc={inc:.1f}°")
                    header = f"  {'Obs':>8s}"
                    for o in obs_angles:
                        header += f"  {o:>8.1f}°"
                    print(header)
                    row = f"  {'RCS dB':>8s}"
                    for o in obs_angles:
                        match = [s for s in samples
                                 if abs(s["frequency_ghz"] - f) < 1e-10
                                 and abs(s.get("theta_inc_deg", 0) - inc) < 0.01
                                 and abs(s.get("theta_scat_deg", 0) - o) < 0.01]
                        if match:
                            row += f"  {match[0]['rcs_db']:>8.2f}"
                        else:
                            row += f"  {'---':>8s}"
                    print(row)
        else:
            rcs_vals = [s["rcs_db"] for s in samples]
            print(f"  RCS range: {min(rcs_vals):.2f} to {max(rcs_vals):.2f} dB")
    else:
        # Monostatic: freq × elevation table.
        elevs = sorted(set(s.get("theta_inc_deg", 0) for s in samples))

        if len(freqs) <= 20 and len(elevs) <= 10:
            print()
            header = f"  {'Freq GHz':>10s}"
            for e in elevs:
                header += f"  {e:>8.1f}°"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for f in freqs:
                row = f"  {f:10.4f}"
                for e in elevs:
                    match = [s for s in samples
                             if abs(s["frequency_ghz"] - f) < 1e-10
                             and abs(s.get("theta_inc_deg", 0) - e) < 0.01]
                    if match:
                        row += f"  {match[0]['rcs_db']:>8.2f}"
                    else:
                        row += f"  {'---':>8s}"
                print(row)
        else:
            rcs_vals = [s["rcs_db"] for s in samples]
            print(f"  RCS range: {min(rcs_vals):.2f} to {max(rcs_vals):.2f} dB")


def print_currents_summary(result: dict) -> None:
    import numpy as np

    n = result.get("element_count", 0)
    formulation = result.get("formulation", "unknown")
    abs_vals = result.get("density_abs", [])
    print(f"  Formulation : {formulation}")
    print(f"  Elements    : {n}")
    if abs_vals:
        print(f"  |J| range   : {min(abs_vals):.6f} to {max(abs_vals):.6f}")
        print(f"  |J| mean    : {sum(abs_vals)/len(abs_vals):.6f}")


def export_grim(result: dict, snapshot: dict, path: str) -> None:
    from grim_io import export_result_to_grim
    export_result_to_grim(result, path, geometry_snapshot=snapshot)
    print(f"  GRIM export : {path}")


def main() -> None:
    print("=" * 70)
    print("2D BIE/MoM RCS Solver")
    print("=" * 70)

    # Load geometry.
    if not os.path.isfile(GEOMETRY_FILE):
        print(f"ERROR: Geometry file not found: {GEOMETRY_FILE}")
        sys.exit(1)

    snapshot = load_geometry(GEOMETRY_FILE)
    print_header(snapshot)

    # Solve.
    t0 = time.time()

    if MODE == "monostatic":
        result = run_monostatic(snapshot)
    elif MODE == "bistatic":
        result = run_bistatic(snapshot)
    elif MODE == "adaptive":
        result = run_adaptive(snapshot)
    elif MODE == "currents":
        result = run_currents(snapshot)
    else:
        print(f"ERROR: Unknown mode '{MODE}'. Use monostatic/bistatic/adaptive/currents.")
        sys.exit(1)

    elapsed = time.time() - t0
    print()
    print(f"  Solve time: {elapsed:.2f} s")

    # Summary.
    print()
    if MODE == "currents":
        print_currents_summary(result)
    else:
        print_summary(result)

    # Export.
    print()
    if MODE == "currents":
        with open(OUTPUT_CURRENTS, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Currents JSON: {OUTPUT_CURRENTS}")
    else:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Result JSON  : {OUTPUT_JSON}")

        if OUTPUT_GRIM and MODE in ("monostatic", "bistatic"):
            try:
                export_grim(result, snapshot, OUTPUT_GRIM)
            except Exception as exc:
                print(f"  GRIM export failed: {exc}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
