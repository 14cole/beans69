#!/usr/bin/env python3
"""
Analytical validation test suite for the 2D BIE/MoM RCS solver.

Compares solver output against exact Mie-series solutions for canonical
geometries.  Run standalone or via pytest.

Test cases
----------
1. PEC circular cylinder, TE polarization, multiple ka values
2. PEC circular cylinder, TM polarization, multiple ka values
3. PEC strip (flat plate), TE polarization, broadside
4. Mesh convergence: PEC cylinder at ka=5 across three mesh densities
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Mie-series exact solutions
# ---------------------------------------------------------------------------

def _mie_backscatter_2d_pec_cylinder(
    ka: float,
    pol: str,
    n_terms: int = 80,
) -> Tuple[complex, float]:
    """
    Exact monostatic backscatter for a 2D PEC circular cylinder.

    Parameters
    ----------
    ka : float
        Wavenumber-radius product  k*a.
    pol : str
        ``'TE'`` (Dirichlet, E_z polarisation) or ``'TM'`` (Neumann, H_z polarisation).
    n_terms : int
        Number of azimuthal modes (both positive and negative).

    Returns
    -------
    amplitude : complex
        Far-field backscatter amplitude  A(pi).
    sigma_2d : float
        2-D scattering width  sigma = |A|^2 / (4k)  [metres].
    """

    from scipy.special import jv, jvp, hankel2, h2vp

    amp = 0.0 + 0.0j
    for n in range(-n_terms, n_terms + 1):
        if pol.upper() == 'TE':
            # Dirichlet BC: a_n = -J_n(ka) / H_n^(2)(ka)
            coeff = -jv(n, ka) / hankel2(n, ka)
        else:
            # Neumann BC: a_n = -J_n'(ka) / H_n^(2)'(ka)
            coeff = -jvp(n, ka) / h2vp(n, ka)
        amp += coeff * ((-1.0) ** n)

    # The representation integral gives a far-field amplitude such that
    #   u_s ~ sqrt(1/(8*pi*k*r)) exp(-j(kr - pi/4)) * A(phi)
    # with A normalised so  sigma_2d = |A|^2 / (4k).
    # The Mie coefficient sum gives the standard pattern f(phi), and A = 4*f
    # because sqrt(1/(8*pi*k*r)) = (1/4) * sqrt(2/(pi*k*r)).
    amp *= 4.0

    k = ka  # ka / a, but we normalise a=1 so k = ka
    sigma = float(np.abs(amp) ** 2) / (4.0 * k)
    return amp, sigma


def _mie_backscatter_2d_dielectric_cylinder(
    ka: float,
    eps_r: float,
    mu_r: float,
    pol: str,
    n_terms: int = 80,
) -> Tuple[complex, float]:
    """
    Exact monostatic backscatter for a 2D homogeneous dielectric circular cylinder.

    Uses the Bowman/Senior/Uslenghi convention:
    - E_z polarization (``pol='TE'``): flux continuity uses 1/mu
    - H_z polarization (``pol='TM'``): flux continuity uses 1/eps

    Parameters
    ----------
    ka : float
        Free-space wavenumber-radius product  k0*a.
    eps_r, mu_r : float
        Relative permittivity and permeability of the cylinder.
    pol : str
        ``'TE'`` (E_z polarization) or ``'TM'`` (H_z polarization).

    Returns
    -------
    amplitude : complex
        Far-field backscatter amplitude.
    sigma_2d : float
        2-D scattering width  sigma = |A|^2 / (4k).
    """

    from scipy.special import jv, jvp, hankel2, h2vp

    k0a = ka
    n_medium = np.sqrt(complex(eps_r) * complex(mu_r))
    k1a = k0a * n_medium

    if pol.upper() == 'TE':
        # E_z: factor = n_r = sqrt(eps*mu)  (for non-magnetic, = sqrt(eps))
        factor = n_medium
    else:
        # H_z: factor = 1/n_r = 1/sqrt(eps*mu)
        factor = 1.0 / n_medium

    amp = 0.0 + 0.0j
    for n in range(-n_terms, n_terms + 1):
        jn_int = jv(n, k1a)
        jnp_int = jvp(n, k1a)
        jn_ext = jv(n, k0a)
        jnp_ext = jvp(n, k0a)
        hn_ext = hankel2(n, k0a)
        hnp_ext = h2vp(n, k0a)

        num = jn_int * jnp_ext - factor * jnp_int * jn_ext
        den = factor * jnp_int * hn_ext - jn_int * hnp_ext

        coeff = num / den if abs(den) > 1e-30 else 0.0
        amp += coeff * ((-1.0) ** n)

    amp *= 4.0
    k = ka
    sigma = float(np.abs(amp) ** 2) / (4.0 * k)
    return amp, sigma


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _build_pec_cylinder_snapshot(
    radius: float,
    n_sides: int,
    units: str = 'meters',
) -> Dict[str, Any]:
    """Build a geometry snapshot for a regular-polygon approximation of a PEC cylinder."""

    angles = np.linspace(0, 2.0 * np.pi, n_sides + 1)[:-1]
    # CCW winding so outward normals point away from centre.
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    point_pairs = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        point_pairs.append({
            'x1': float(xs[i]),
            'y1': float(ys[i]),
            'x2': float(xs[j]),
            'y2': float(ys[j]),
        })

    # TYPE 2 PEC, N=0 (auto density), no arc, no IBC, no dielectric.
    props = ['2', '0', '0', '0', '0', '0']

    return {
        'title': f'PEC cylinder r={radius} ({n_sides}-gon)',
        'segment_count': 1,
        'segments': [{
            'name': 'cylinder',
            'seg_type': '2',
            'properties': props,
            'point_pairs': point_pairs,
        }],
        'ibcs': [],
        'dielectrics': [],
    }


def _build_pec_strip_snapshot(
    half_width: float,
    units: str = 'meters',
) -> Dict[str, Any]:
    """Build a geometry snapshot for a PEC flat strip (2*half_width long)."""

    point_pairs = [{
        'x1': -half_width,
        'y1': 0.0,
        'x2': half_width,
        'y2': 0.0,
    }]

    props = ['2', '0', '0', '0', '0', '0']

    return {
        'title': f'PEC strip half_width={half_width}',
        'segment_count': 1,
        'segments': [{
            'name': 'strip',
            'seg_type': '2',
            'properties': props,
            'point_pairs': point_pairs,
        }],
        'ibcs': [],
        'dielectrics': [],
    }


def _build_dielectric_cylinder_snapshot(
    radius: float,
    n_sides: int,
    eps_r: float,
    mu_r: float = 1.0,
) -> Dict[str, Any]:
    """Build a geometry snapshot for a dielectric cylinder (TYPE 3, air/dielectric interface)."""

    angles = np.linspace(0, 2.0 * np.pi, n_sides + 1)[:-1]
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    point_pairs = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        point_pairs.append({
            'x1': float(xs[i]),
            'y1': float(ys[i]),
            'x2': float(xs[j]),
            'y2': float(ys[j]),
        })

    # TYPE 3: air/dielectric. IPN1=1 references dielectric flag 1.
    props = ['3', '0', '0', '0', '1', '0']

    return {
        'title': f'Dielectric cylinder r={radius} eps={eps_r} ({n_sides}-gon)',
        'segment_count': 1,
        'segments': [{
            'name': 'cylinder',
            'seg_type': '3',
            'properties': props,
            'point_pairs': point_pairs,
        }],
        'ibcs': [],
        'dielectrics': [['1', str(eps_r), '0.0', str(mu_r), '0.0']],
    }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _run_solver(
    snapshot: Dict[str, Any],
    freq_ghz: float,
    elevations_deg: List[float],
    pol: str,
    units: str = 'meters',
    cfie_alpha: float = 0.2,
) -> Dict[str, Any]:
    from rcs_solver import solve_monostatic_rcs_2d
    return solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=[freq_ghz],
        elevations_deg=elevations_deg,
        polarization=pol,
        geometry_units=units,
        cfie_alpha=cfie_alpha,
    )


def _extract_backscatter_db(result: Dict[str, Any], elevation_deg: float = 0.0) -> float:
    """Extract dB scattering width at a specific elevation from solver result."""
    for sample in result.get('samples', []):
        if abs(float(sample.get('theta_scat_deg', 0.0)) - elevation_deg) < 0.01:
            lin = float(sample.get('rcs_linear', 1e-30))
            return 10.0 * math.log10(max(lin, 1e-30))
    raise ValueError(f'No sample found at elevation {elevation_deg} deg')


def test_pec_cylinder_te(
    ka_values: List[float] | None = None,
    tolerance_db: float = 1.0,
    n_polygon_sides: int = 120,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate TE PEC cylinder against Mie series.

    Parameters
    ----------
    ka_values : list of float
        Wavenumber-radius products to test.
    tolerance_db : float
        Maximum allowed |dB error| for a test to pass.
    n_polygon_sides : int
        Number of polygon sides for the cylinder approximation.
    """

    C0 = 299_792_458.0
    if ka_values is None:
        ka_values = [1.0, 3.0, 5.0]

    radius = 1.0  # 1 metre
    snapshot = _build_pec_cylinder_snapshot(radius, n_polygon_sides)

    results = []
    all_pass = True

    for ka in ka_values:
        k = ka / radius
        freq_hz = k * C0 / (2.0 * math.pi)
        freq_ghz = freq_hz / 1e9

        # Exact Mie series
        _amp_exact, sigma_exact = _mie_backscatter_2d_pec_cylinder(ka, 'TE')
        db_exact = 10.0 * math.log10(max(sigma_exact, 1e-30))

        # Solver — use cfie_alpha=0 for clean validation at non-resonant ka.
        result = _run_solver(snapshot, freq_ghz, [0.0], 'TE', cfie_alpha=0.0)
        db_solver = _extract_backscatter_db(result, 0.0)

        error_db = abs(db_solver - db_exact)
        passed = error_db <= tolerance_db

        entry = {
            'ka': ka,
            'freq_ghz': freq_ghz,
            'db_exact': db_exact,
            'db_solver': db_solver,
            'error_db': error_db,
            'passed': passed,
        }
        results.append(entry)
        if not passed:
            all_pass = False

        if verbose:
            status = 'PASS' if passed else 'FAIL'
            print(f'  TE ka={ka:5.1f}  exact={db_exact:+8.3f} dB  solver={db_solver:+8.3f} dB  '
                  f'err={error_db:.3f} dB  [{status}]')

    return {'test': 'pec_cylinder_te', 'passed': all_pass, 'results': results}


def test_pec_cylinder_tm(
    ka_values: List[float] | None = None,
    tolerance_db: float = 1.0,
    n_polygon_sides: int = 120,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate TM PEC cylinder against Mie series.

    TM PEC uses the MFIE formulation: (-1/2 M + K') sigma = -du_inc/dn
    with a single-layer far-field projector.
    """

    C0 = 299_792_458.0
    if ka_values is None:
        ka_values = [1.0, 3.0, 5.0]

    radius = 1.0
    snapshot = _build_pec_cylinder_snapshot(radius, n_polygon_sides)

    results = []
    all_pass = True

    for ka in ka_values:
        k = ka / radius
        freq_hz = k * C0 / (2.0 * math.pi)
        freq_ghz = freq_hz / 1e9

        _amp_exact, sigma_exact = _mie_backscatter_2d_pec_cylinder(ka, 'TM')
        db_exact = 10.0 * math.log10(max(sigma_exact, 1e-30))

        result = _run_solver(snapshot, freq_ghz, [0.0], 'TM', cfie_alpha=0.0)
        db_solver = _extract_backscatter_db(result, 0.0)

        error_db = abs(db_solver - db_exact)
        passed = error_db <= tolerance_db

        entry = {
            'ka': ka, 'freq_ghz': freq_ghz,
            'db_exact': db_exact, 'db_solver': db_solver,
            'error_db': error_db, 'passed': passed,
        }
        results.append(entry)
        if not passed:
            all_pass = False

        if verbose:
            status = 'PASS' if passed else 'FAIL'
            print(f'  TM ka={ka:5.1f}  exact={db_exact:+8.3f} dB  solver={db_solver:+8.3f} dB  '
                  f'err={error_db:.3f} dB  [{status}]')

    return {'test': 'pec_cylinder_tm', 'passed': all_pass, 'results': results}


def test_mesh_convergence_cylinder(
    ka: float = 3.0,
    n_sides_list: List[int] | None = None,
    convergence_threshold_db: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify mesh convergence by solving at increasing polygon resolution.

    The finest mesh result should agree with the Mie series, and successive
    refinements should converge monotonically.
    """

    C0 = 299_792_458.0
    if n_sides_list is None:
        n_sides_list = [40, 80, 160]

    radius = 1.0
    k = ka / radius
    freq_hz = k * C0 / (2.0 * math.pi)
    freq_ghz = freq_hz / 1e9

    _amp_exact, sigma_exact = _mie_backscatter_2d_pec_cylinder(ka, 'TE')
    db_exact = 10.0 * math.log10(max(sigma_exact, 1e-30))

    db_values = []
    for n_sides in n_sides_list:
        snapshot = _build_pec_cylinder_snapshot(radius, n_sides)
        result = _run_solver(snapshot, freq_ghz, [0.0], 'TE')
        db_val = _extract_backscatter_db(result, 0.0)
        db_values.append(db_val)
        if verbose:
            err = abs(db_val - db_exact)
            print(f'  N={n_sides:4d}  solver={db_val:+8.3f} dB  exact={db_exact:+8.3f} dB  err={err:.3f} dB')

    # Check convergence: errors should decrease with refinement.
    errors = [abs(db - db_exact) for db in db_values]
    converging = all(errors[i] >= errors[i + 1] - 0.05 for i in range(len(errors) - 1))
    finest_ok = errors[-1] <= convergence_threshold_db

    passed = converging and finest_ok
    if verbose:
        print(f'  Converging: {converging}  Finest error: {errors[-1]:.3f} dB  [{"PASS" if passed else "FAIL"}]')

    return {
        'test': 'mesh_convergence',
        'passed': passed,
        'errors_db': errors,
        'db_exact': db_exact,
        'converging': converging,
    }


def test_cfie_vs_no_cfie(
    ka: float = 3.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify that CFIE and non-CFIE results agree away from resonances.

    At non-resonant frequencies, both should give the same answer.
    At resonant frequencies, CFIE should be stable while non-CFIE may diverge.
    """

    C0 = 299_792_458.0
    radius = 1.0
    k = ka / radius
    freq_hz = k * C0 / (2.0 * math.pi)
    freq_ghz = freq_hz / 1e9

    snapshot = _build_pec_cylinder_snapshot(radius, 100)

    result_cfie = _run_solver(snapshot, freq_ghz, [0.0], 'TE', cfie_alpha=0.2)
    result_nocfie = _run_solver(snapshot, freq_ghz, [0.0], 'TE', cfie_alpha=0.0)

    db_cfie = _extract_backscatter_db(result_cfie, 0.0)
    db_nocfie = _extract_backscatter_db(result_nocfie, 0.0)

    _amp_exact, sigma_exact = _mie_backscatter_2d_pec_cylinder(ka, 'TE')
    db_exact = 10.0 * math.log10(max(sigma_exact, 1e-30))

    err_cfie = abs(db_cfie - db_exact)
    err_nocfie = abs(db_nocfie - db_exact)

    # Both should be close at non-resonant ka
    passed = err_cfie < 1.0 and err_nocfie < 1.0

    if verbose:
        print(f'  CFIE:    {db_cfie:+8.3f} dB  err={err_cfie:.3f} dB')
        print(f'  No CFIE: {db_nocfie:+8.3f} dB  err={err_nocfie:.3f} dB')
        print(f'  Exact:   {db_exact:+8.3f} dB  [{"PASS" if passed else "FAIL"}]')

    return {
        'test': 'cfie_comparison',
        'passed': passed,
        'db_cfie': db_cfie,
        'db_nocfie': db_nocfie,
        'db_exact': db_exact,
    }


def test_dielectric_cylinder_te(
    ka_values: List[float] | None = None,
    eps_r: float = 4.0,
    tolerance_db: float = 0.5,
    n_polygon_sides: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate TE (E_z) dielectric cylinder against Mie series."""

    C0 = 299_792_458.0
    if ka_values is None:
        ka_values = [1.0, 3.0]

    radius = 1.0
    snapshot = _build_dielectric_cylinder_snapshot(radius, n_polygon_sides, eps_r)

    results = []
    all_pass = True

    for ka in ka_values:
        k = ka / radius
        freq_hz = k * C0 / (2.0 * math.pi)
        freq_ghz = freq_hz / 1e9

        _amp_exact, sigma_exact = _mie_backscatter_2d_dielectric_cylinder(ka, eps_r, 1.0, 'TE')
        db_exact = 10.0 * math.log10(max(sigma_exact, 1e-30))

        result = _run_solver(snapshot, freq_ghz, [0.0], 'TE', cfie_alpha=0.0)
        db_solver = _extract_backscatter_db(result, 0.0)

        error_db = abs(db_solver - db_exact)
        passed = error_db <= tolerance_db

        entry = {
            'ka': ka, 'freq_ghz': freq_ghz,
            'db_exact': db_exact, 'db_solver': db_solver,
            'error_db': error_db, 'passed': passed,
        }
        results.append(entry)
        if not passed:
            all_pass = False

        if verbose:
            status = 'PASS' if passed else 'FAIL'
            print(f'  TE ka={ka:5.1f}  eps={eps_r}  exact={db_exact:+8.3f} dB  solver={db_solver:+8.3f} dB  '
                  f'err={error_db:.3f} dB  [{status}]')

    return {'test': 'dielectric_cylinder_te', 'passed': all_pass, 'results': results}


def test_dielectric_cylinder_tm(
    ka_values: List[float] | None = None,
    eps_r: float = 4.0,
    tolerance_db: float = 0.5,
    n_polygon_sides: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate TM (H_z) dielectric cylinder against Mie series."""

    C0 = 299_792_458.0
    if ka_values is None:
        ka_values = [1.0, 3.0]

    radius = 1.0
    snapshot = _build_dielectric_cylinder_snapshot(radius, n_polygon_sides, eps_r)

    results = []
    all_pass = True

    for ka in ka_values:
        k = ka / radius
        freq_hz = k * C0 / (2.0 * math.pi)
        freq_ghz = freq_hz / 1e9

        _amp_exact, sigma_exact = _mie_backscatter_2d_dielectric_cylinder(ka, eps_r, 1.0, 'TM')
        db_exact = 10.0 * math.log10(max(sigma_exact, 1e-30))

        result = _run_solver(snapshot, freq_ghz, [0.0], 'TM', cfie_alpha=0.0)
        db_solver = _extract_backscatter_db(result, 0.0)

        error_db = abs(db_solver - db_exact)
        passed = error_db <= tolerance_db

        entry = {
            'ka': ka, 'freq_ghz': freq_ghz,
            'db_exact': db_exact, 'db_solver': db_solver,
            'error_db': error_db, 'passed': passed,
        }
        results.append(entry)
        if not passed:
            all_pass = False

        if verbose:
            status = 'PASS' if passed else 'FAIL'
            print(f'  TM ka={ka:5.1f}  eps={eps_r}  exact={db_exact:+8.3f} dB  solver={db_solver:+8.3f} dB  '
                  f'err={error_db:.3f} dB  [{status}]')

    return {'test': 'dielectric_cylinder_tm', 'passed': all_pass, 'results': results}


def _mie_backscatter_2d_ibc_cylinder(
    ka: float,
    z_surface: complex,
    pol: str,
    n_terms: int = 80,
) -> float:
    """
    Exact monostatic backscatter for a 2D impedance (IBC) circular cylinder.

    Uses the Robin BC: du/dn + alpha*u = 0 where alpha depends on polarization.
    The Mie alpha is sign-flipped from the solver alpha due to normal convention:
        alpha_mie = -alpha_solver

    Parameters
    ----------
    ka : float
        Free-space wavenumber-radius product.
    z_surface : complex
        Surface impedance in ohms.
    pol : str
        ``'TE'`` or ``'TM'``.

    Returns
    -------
    sigma_2d : float
        2-D scattering width.
    """

    from scipy.special import jv, jvp, hankel2, h2vp
    from rcs_solver import _surface_robin_alpha, ETA0

    alpha_solver = _surface_robin_alpha(pol, 1+0j, 1+0j, complex(ka), z_surface)
    alpha_mie = -alpha_solver  # normal convention flip

    amp = 0.0 + 0.0j
    for n in range(-n_terms, n_terms + 1):
        num = jvp(n, ka) + alpha_mie * jv(n, ka)
        den = h2vp(n, ka) + alpha_mie * hankel2(n, ka)
        coeff = -num / den if abs(den) > 1e-30 else 0.0
        amp += coeff * ((-1.0) ** n)

    amp *= 4.0
    return float(np.abs(amp) ** 2) / (4.0 * ka)


def _build_ibc_cylinder_snapshot(
    radius: float,
    n_sides: int,
    z_real: float,
    z_imag: float,
) -> Dict[str, Any]:
    """Build a geometry snapshot for an IBC (impedance) cylinder."""

    angles = np.linspace(0, 2.0 * np.pi, n_sides + 1)[:-1]
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    point_pairs = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        point_pairs.append({
            'x1': float(xs[i]), 'y1': float(ys[i]),
            'x2': float(xs[j]), 'y2': float(ys[j]),
        })

    # TYPE 2 with IBC flag 1
    props = ['2', '0', '0', '1', '0', '0']

    return {
        'title': f'IBC cylinder Zs={z_real}+{z_imag}j ({n_sides}-gon)',
        'segment_count': 1,
        'segments': [{
            'name': 'cylinder',
            'seg_type': '2',
            'properties': props,
            'point_pairs': point_pairs,
        }],
        'ibcs': [['1', str(z_real), str(z_imag)]],
        'dielectrics': [],
    }


def test_ibc_cylinder(
    ka: float = 1.0,
    impedances: List[complex] | None = None,
    tolerance_db: float = 0.5,
    n_polygon_sides: int = 80,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate IBC cylinder against Mie series for both polarizations."""

    C0 = 299_792_458.0
    if impedances is None:
        impedances = [complex(100, 50), complex(200, 0)]

    radius = 1.0
    k = ka / radius
    freq_hz = k * C0 / (2.0 * math.pi)
    freq_ghz = freq_hz / 1e9

    results = []
    all_pass = True

    for Zs in impedances:
        snapshot = _build_ibc_cylinder_snapshot(radius, n_polygon_sides, Zs.real, Zs.imag)

        for pol in ['TE', 'TM']:
            sigma_exact = _mie_backscatter_2d_ibc_cylinder(ka, Zs, pol)
            db_exact = 10.0 * math.log10(max(sigma_exact, 1e-30))

            result = _run_solver(snapshot, freq_ghz, [0.0], pol, cfie_alpha=0.0)
            db_solver = _extract_backscatter_db(result, 0.0)

            error_db = abs(db_solver - db_exact)
            passed = error_db <= tolerance_db

            entry = {
                'Zs': Zs, 'pol': pol, 'db_exact': db_exact,
                'db_solver': db_solver, 'error_db': error_db, 'passed': passed,
            }
            results.append(entry)
            if not passed:
                all_pass = False

            if verbose:
                status = 'PASS' if passed else 'FAIL'
                print(f'  {pol}  Zs={str(Zs):15s}  exact={db_exact:+8.3f} dB  solver={db_solver:+8.3f} dB  '
                      f'err={error_db:.3f} dB  [{status}]')

    return {'test': 'ibc_cylinder', 'passed': all_pass, 'results': results}


def _mie_bistatic_2d_pec_cylinder(
    ka: float,
    pol: str,
    theta_inc_deg: float,
    theta_obs_deg: float,
    n_terms: int = 80,
) -> float:
    """
    Exact bistatic 2D RCS for a PEC circular cylinder.

    Uses the solver's "coming-from" convention: backscatter when obs=inc,
    forward scatter when obs = inc + 180.
    """

    from scipy.special import jv, jvp, hankel2, h2vp

    rel = math.radians(180.0 - (theta_obs_deg - theta_inc_deg))
    amp = 0.0 + 0.0j
    for n in range(-n_terms, n_terms + 1):
        if pol.upper() == 'TE':
            a_n = -jv(n, ka) / hankel2(n, ka)
        else:
            a_n = -jvp(n, ka) / h2vp(n, ka)
        amp += a_n * np.exp(-1j * n * rel)
    amp *= 4.0
    return float(np.abs(amp) ** 2) / (4.0 * ka)


def test_bistatic_pec_cylinder(
    ka: float = 1.0,
    n_polygon_sides: int = 80,
    tolerance_db: float = 0.3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate bistatic RCS against Mie series for both polarizations."""

    C0 = 299_792_458.0
    radius = 1.0
    freq_ghz = (ka / radius) * C0 / (2.0 * math.pi * 1e9)
    snapshot = _build_pec_cylinder_snapshot(radius, n_polygon_sides)
    obs_angles = [0.0, 45.0, 90.0, 135.0, 180.0]

    from rcs_solver import solve_bistatic_rcs_2d

    results = []
    all_pass = True

    for pol in ['TE', 'TM']:
        r = solve_bistatic_rcs_2d(
            geometry_snapshot=snapshot,
            frequencies_ghz=[freq_ghz],
            incidence_angles_deg=[0.0],
            observation_angles_deg=obs_angles,
            polarization=pol,
            geometry_units='meters',
            cfie_alpha=0.0,
        )
        for s in r['samples']:
            obs = s['theta_scat_deg']
            db_solver = 10.0 * math.log10(max(s['rcs_linear'], 1e-30))
            sigma_mie = _mie_bistatic_2d_pec_cylinder(ka, pol, 0.0, obs)
            db_mie = 10.0 * math.log10(max(sigma_mie, 1e-30))
            error_db = abs(db_solver - db_mie)
            passed = error_db <= tolerance_db
            results.append({'pol': pol, 'obs': obs, 'error_db': error_db, 'passed': passed})
            if not passed:
                all_pass = False
            if verbose:
                status = 'PASS' if passed else 'FAIL'
                print(f'  {pol} obs={obs:6.1f}°  solver={db_solver:+8.3f}  mie={db_mie:+8.3f}  '
                      f'err={error_db:.3f}  [{status}]')

    return {'test': 'bistatic_pec_cylinder', 'passed': all_pass, 'results': results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True) -> bool:
    """Run all validation tests. Returns True if all pass."""

    all_pass = True

    print('=' * 70)
    print('2D RCS Solver Analytical Validation Suite')
    print('=' * 70)

    print('\n--- PEC Cylinder TE (Dirichlet, EFIE) ---')
    r1 = test_pec_cylinder_te(verbose=verbose)
    all_pass &= r1['passed']

    print('\n--- PEC Cylinder TM (Neumann, MFIE) ---')
    r2 = test_pec_cylinder_tm(verbose=verbose)
    all_pass &= r2['passed']

    print('\n--- Dielectric Cylinder TE (E_z, eps=4) ---')
    r3 = test_dielectric_cylinder_te(verbose=verbose)
    all_pass &= r3['passed']

    print('\n--- Dielectric Cylinder TM (H_z, eps=4) ---')
    r4 = test_dielectric_cylinder_tm(verbose=verbose)
    all_pass &= r4['passed']

    print('\n--- IBC Cylinder TE+TM ---')
    r5 = test_ibc_cylinder(verbose=verbose)
    all_pass &= r5['passed']

    print('\n--- Bistatic PEC Cylinder TE+TM ---')
    r6 = test_bistatic_pec_cylinder(verbose=verbose)
    all_pass &= r6['passed']

    print('\n--- Mesh Convergence (TE, ka=3) ---')
    r7 = test_mesh_convergence_cylinder(verbose=verbose)
    all_pass &= r7['passed']

    print('\n--- CFIE vs No-CFIE Comparison (TE, ka=3) ---')
    r8 = test_cfie_vs_no_cfie(verbose=verbose)
    all_pass &= r8['passed']

    print('\n' + '=' * 70)
    if all_pass:
        print('ALL TESTS PASSED')
    else:
        print('SOME TESTS FAILED')
    print('=' * 70)

    return all_pass


if __name__ == '__main__':
    ok = run_all(verbose=True)
    sys.exit(0 if ok else 1)
