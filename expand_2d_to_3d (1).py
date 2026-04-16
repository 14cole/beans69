#!/usr/bin/env python3
"""
expand_2d_to_3d.py — Map 2D cross-section RCS onto a 3D body for 3D .grim output.

Edit the CONFIG section below and run:
    python expand_2d_to_3d.py

Takes:
  1. A 2D .grim file (monostatic RCS vs azimuth from the 2D solver)
  2. A .stp/.step CAD file defining the 3D body surface
  3. A CSV/text file with x,y,z coordinates defining the cross-section curve

Produces:
  A 3D .grim file with RCS vs (azimuth, elevation) expanded from the 2D pattern
  using surface normals from the STEP geometry. Shadowed regions default to -200 dBsm.

The CSV can be either:
    x, y, z                     (normals computed from STEP or finite differences)
    x, y, z, nx, ny, nz         (normals provided explicitly)

STEP loading requires one of: cadquery, build123d, pythonocc-core, or FreeCAD.
If none are available, normals are computed from the curve geometry itself.
"""

import json
import math
import os
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this section
# ═══════════════════════════════════════════════════════════════════════════════

# --- Input files ---
GRIM_2D_FILE    = "rcs_output.grim"             # 2D .grim from the solver
STEP_FILE       = "body.stp"                     # 3D STEP/STL file (set to "none" to skip)
CURVE_CSV       = "curve.csv"                    # CSV with x,y,z [,nx,ny,nz]

# --- Output ---
OUTPUT_GRIM     = "output_3d.grim"               # 3D .grim output path

# --- 3D angle grid ---
AZIMUTHS        = list(range(0, 360, 5))         # azimuth angles (degrees)
ELEVATIONS      = list(range(-90, 91, 5))        # elevation angles (degrees)
# AZIMUTHS      = list(range(0, 360, 2))         # finer grid
# ELEVATIONS    = list(range(-90, 91, 2))

# --- Body axis (unit vector along the long axis of the body) ---
#   None = auto-detect from curve endpoints
#   [1, 0, 0] = body runs along the x-axis
BODY_AXIS       = None
# BODY_AXIS     = [1, 0, 0]

# --- Shadowing ---
SHADOW_DBSM     = -200.0                         # RCS for shadowed facets (dBsm)

# ═══════════════════════════════════════════════════════════════════════════════
# END CONFIG — no edits needed below this line
# ═══════════════════════════════════════════════════════════════════════════════

C0 = 299_792_458.0
SHADOW_LINEAR = 10.0 ** (SHADOW_DBSM / 10.0)


def load_grim_2d(path):
    data = np.load(path, allow_pickle=True)
    azimuths = np.array(data["azimuths"], dtype=float)
    frequencies = np.array(data["frequencies"], dtype=float)
    rcs_power = np.array(data["rcs_power"], dtype=float)
    rcs_phase = np.array(data["rcs_phase"], dtype=float)
    pols = data["polarizations"] if "polarizations" in data else np.array(["TE"])
    return {
        "azimuths_deg": azimuths,
        "frequencies_ghz": frequencies,
        "rcs_linear": rcs_power[:, 0, :, 0],
        "rcs_phase": rcs_phase[:, 0, :, 0],
        "polarization": str(pols[0]) if len(pols) > 0 else "TE",
    }


def save_grim_3d(path, azimuths_deg, elevations_deg, frequencies_ghz,
                  rcs_linear_4d, rcs_phase_4d, polarization, source_2d_path, history=""):
    if not path.lower().endswith(".grim"):
        path += ".grim"
    np.savez(
        path,
        azimuths=np.asarray(azimuths_deg, dtype=float),
        elevations=np.asarray(elevations_deg, dtype=float),
        frequencies=np.asarray(frequencies_ghz, dtype=float),
        polarizations=np.array([polarization]),
        rcs_power=rcs_linear_4d.astype(np.float32),
        rcs_phase=rcs_phase_4d.astype(np.float32),
        rcs_domain="power_phase",
        power_domain="linear_rcs",
        source_path=source_2d_path,
        history=history,
        units=json.dumps({
            "azimuth": "deg", "elevation": "deg",
            "frequency": "GHz", "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        }),
        phase_reference="2D-to-3D expansion",
    )
    return os.path.abspath(path)


def load_curve_csv(path):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("x"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                lines.append([float(v) for v in parts])
    arr = np.array(lines, dtype=float)
    points = arr[:, :3]
    normals = arr[:, 3:6] if arr.shape[1] >= 6 else None
    if normals is not None:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-15)
    return points, normals


def compute_normals_from_curve(points):
    N = len(points)
    tangents = np.zeros_like(points)
    for i in range(N):
        if i == 0:
            tangents[i] = points[1] - points[0]
        elif i == N - 1:
            tangents[i] = points[-1] - points[-2]
        else:
            tangents[i] = points[i + 1] - points[i - 1]
    t_len = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(t_len, 1e-15)
    centroid = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    plane_normal = Vt[2]
    normals = np.cross(tangents, plane_normal)
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(n_len, 1e-15)
    for i in range(N):
        if np.dot(normals[i], centroid - points[i]) > 0:
            normals[i] = -normals[i]
    return normals


def load_step_and_compute_normals(step_path, query_points):
    for loader in [_try_ocp_normals, _try_freecad_normals, _try_trimesh_normals]:
        normals = loader(step_path, query_points)
        if normals is not None:
            return normals
    return None


def _try_ocp_normals(step_path, query_points):
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.gp import gp_Pnt
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    except ImportError:
        return None
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        return None
    reader.TransferRoots()
    shape = reader.OneShape()
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        faces.append(explorer.Current())
        explorer.Next()
    normals = np.zeros((len(query_points), 3), dtype=float)
    for i, pt in enumerate(query_points):
        vtx = BRepBuilderAPI_MakeVertex(gp_Pnt(*pt.tolist())).Vertex()
        best_dist = float("inf")
        for face in faces:
            dist_tool = BRepExtrema_DistShapeShape(vtx, face)
            if dist_tool.NbSolution() > 0 and dist_tool.Value() < best_dist:
                best_dist = dist_tool.Value()
                u, v = dist_tool.ParOnFaceS2(1)
                adaptor = BRepAdaptor_Surface(face)
                gp_n = adaptor.DN(u, v, 1, 0).Crossed(adaptor.DN(u, v, 0, 1))
                mag = gp_n.Magnitude()
                if mag > 1e-10:
                    normals[i] = [gp_n.X()/mag, gp_n.Y()/mag, gp_n.Z()/mag]
    return normals


def _try_freecad_normals(step_path, query_points):
    try:
        import FreeCAD, Part
    except ImportError:
        return None
    try:
        shape = Part.Shape()
        shape.read(step_path)
    except Exception:
        return None
    normals = np.zeros((len(query_points), 3), dtype=float)
    for i, pt in enumerate(query_points):
        try:
            dist, pts, info = shape.distToShape(Part.Vertex(FreeCAD.Vector(*pt.tolist())))
            if info and hasattr(info[0], "normalAt"):
                n = info[0].normalAt(0, 0)
                normals[i] = [n.x, n.y, n.z]
        except Exception:
            normals[i] = [0, 0, 1]
    return normals


def _try_trimesh_normals(step_path, query_points):
    try:
        import trimesh
    except ImportError:
        return None
    ext = os.path.splitext(step_path)[1].lower()
    if ext in (".stp", ".step"):
        warnings.warn(
            "trimesh cannot load STEP directly. Convert to STL first, "
            "or install cadquery/build123d. Using curve-based normals.")
        return None
    try:
        mesh = trimesh.load(step_path)
    except Exception:
        return None
    _, _, face_ids = mesh.nearest.on_surface(query_points)
    normals = mesh.face_normals[face_ids]
    centroid = np.mean(query_points, axis=0)
    for i in range(len(normals)):
        if np.dot(normals[i], query_points[i] - centroid) < 0:
            normals[i] = -normals[i]
    return normals


def interpolate_2d_rcs(grim_2d, angle_deg, freq_idx):
    az = grim_2d["azimuths_deg"]
    rcs_lin = grim_2d["rcs_linear"][:, freq_idx]
    rcs_db = 10.0 * np.log10(np.maximum(rcs_lin, 1e-30))
    if len(az) < 2:
        return rcs_lin[0], 0.0
    periodic = (az[-1] - az[0]) > 350
    db_val = float(np.interp(angle_deg, az, rcs_db,
                              period=360.0 if periodic else None))
    phase_data = grim_2d["rcs_phase"][:, freq_idx]
    ph_val = float(np.interp(angle_deg, az, phase_data,
                              period=360.0 if periodic else None))
    return 10.0 ** (db_val / 10.0), ph_val


def expand_2d_to_3d(grim_2d, curve_points, curve_normals,
                     azimuths_3d, elevations_3d, body_axis=None):
    n_az = len(azimuths_3d)
    n_el = len(elevations_3d)
    n_freq = len(grim_2d["frequencies_ghz"])
    n_pts = len(curve_points)

    if body_axis is None:
        body_axis = curve_points[-1] - curve_points[0]
        body_axis = body_axis / max(np.linalg.norm(body_axis), 1e-15)
    body_axis = np.asarray(body_axis, dtype=float)

    ds = np.zeros(n_pts)
    for i in range(n_pts):
        if i == 0:
            ds[i] = np.linalg.norm(curve_points[1] - curve_points[0])
        elif i == n_pts - 1:
            ds[i] = np.linalg.norm(curve_points[-1] - curve_points[-2])
        else:
            ds[i] = 0.5 * (np.linalg.norm(curve_points[i+1] - curve_points[i]) +
                            np.linalg.norm(curve_points[i] - curve_points[i-1]))

    normals_cs = curve_normals - np.outer(np.dot(curve_normals, body_axis), body_axis)
    n_len = np.linalg.norm(normals_cs, axis=1, keepdims=True)
    normals_cs = normals_cs / np.maximum(n_len, 1e-15)

    rcs_3d = np.full((n_az, n_el, n_freq, 1), SHADOW_LINEAR, dtype=float)
    phase_3d = np.zeros((n_az, n_el, n_freq, 1), dtype=float)

    for fi in range(n_freq):
        freq_ghz = grim_2d["frequencies_ghz"][fi]
        k0 = 2.0 * math.pi * freq_ghz * 1e9 / C0
        for ai, az_deg in enumerate(azimuths_3d):
            for ei, el_deg in enumerate(elevations_3d):
                az_rad = math.radians(az_deg)
                el_rad = math.radians(el_deg)
                d_3d = np.array([
                    math.cos(az_rad) * math.cos(el_rad),
                    math.sin(az_rad) * math.cos(el_rad),
                    math.sin(el_rad),
                ])
                d_cs = d_3d - np.dot(d_3d, body_axis) * body_axis
                d_cs_mag = np.linalg.norm(d_cs)
                if d_cs_mag < 1e-10:
                    continue
                d_cs = d_cs / d_cs_mag

                total_rcs = 0.0
                total_weight = 0.0
                any_visible = False
                for pi in range(n_pts):
                    cos_inc = np.dot(normals_cs[pi], d_cs)
                    if cos_inc <= 0:
                        continue
                    any_visible = True
                    look_angle = math.degrees(math.atan2(d_cs[1], d_cs[0]))
                    rcs_2d_local, _ = interpolate_2d_rcs(grim_2d, look_angle, fi)
                    weight = cos_inc * ds[pi]
                    total_rcs += rcs_2d_local * weight
                    total_weight += weight

                if any_visible and total_weight > 0:
                    cos_el = math.cos(el_rad)
                    rcs_3d_val = k0 * total_rcs * cos_el * cos_el
                    rcs_3d[ai, ei, fi, 0] = max(rcs_3d_val, SHADOW_LINEAR)

    return rcs_3d, phase_3d


def main():
    print("=" * 60)
    print("  2D-to-3D RCS Expansion")
    print("=" * 60)

    print(f"\nLoading 2D .grim: {GRIM_2D_FILE}")
    grim_2d = load_grim_2d(GRIM_2D_FILE)
    print(f"  {len(grim_2d['azimuths_deg'])} azimuths, "
          f"{len(grim_2d['frequencies_ghz'])} frequencies, "
          f"pol={grim_2d['polarization']}")

    print(f"Loading curve: {CURVE_CSV}")
    points, csv_normals = load_curve_csv(CURVE_CSV)
    print(f"  {len(points)} points, "
          f"normals={'from CSV' if csv_normals is not None else 'to compute'}")

    normals = csv_normals
    if normals is None and STEP_FILE.lower() != "none":
        print(f"Loading STEP: {STEP_FILE}")
        normals = load_step_and_compute_normals(STEP_FILE, points)
        if normals is not None:
            print(f"  Normals from CAD surface ({len(normals)} points)")
        else:
            print("  No CAD backend found (install cadquery, build123d, or FreeCAD)")
    if normals is None:
        print("  Computing normals from curve finite differences")
        normals = compute_normals_from_curve(points)

    azimuths_3d = np.asarray(AZIMUTHS, dtype=float)
    elevations_3d = np.asarray(ELEVATIONS, dtype=float)
    print(f"3D grid: {len(azimuths_3d)} az x {len(elevations_3d)} el "
          f"x {len(grim_2d['frequencies_ghz'])} freq")

    body_axis = None
    if BODY_AXIS is not None:
        body_axis = np.array(BODY_AXIS, dtype=float)
        body_axis = body_axis / max(np.linalg.norm(body_axis), 1e-15)
        print(f"Body axis: [{body_axis[0]:.3f}, {body_axis[1]:.3f}, {body_axis[2]:.3f}]")
    else:
        ba = points[-1] - points[0]
        ba = ba / max(np.linalg.norm(ba), 1e-15)
        print(f"Body axis (auto): [{ba[0]:.3f}, {ba[1]:.3f}, {ba[2]:.3f}]")

    print("Expanding 2D -> 3D ...")
    rcs_3d, phase_3d = expand_2d_to_3d(
        grim_2d, points, normals, azimuths_3d, elevations_3d, body_axis=body_axis)

    rcs_db = 10.0 * np.log10(np.maximum(rcs_3d[:, :, :, 0], 1e-30))
    visible = rcs_db > SHADOW_DBSM + 1
    if np.any(visible):
        print(f"  Visible: {np.sum(visible)} of {rcs_db.size} entries")
        print(f"  RCS range: {np.min(rcs_db[visible]):+.1f} to "
              f"{np.max(rcs_db[visible]):+.1f} dBsm")
        print(f"  Shadowed: {np.sum(~visible)} entries -> {SHADOW_DBSM:.0f} dBsm")
    else:
        print("  WARNING: all entries shadowed — check body axis and normals")

    out = save_grim_3d(
        OUTPUT_GRIM, azimuths_3d, elevations_3d, grim_2d["frequencies_ghz"],
        rcs_3d, phase_3d, grim_2d["polarization"],
        source_2d_path=os.path.abspath(GRIM_2D_FILE),
        history=f"2D-to-3D expansion from {os.path.basename(GRIM_2D_FILE)} "
                f"via {os.path.basename(STEP_FILE)}")
    print(f"\nSaved: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
