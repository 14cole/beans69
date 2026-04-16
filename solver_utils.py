"""
Shared utility functions for the 2D RCS solver project.

Consolidates polarization handling, unit conversion, and constants
that were previously duplicated across rcs_solver.py and grim_io.py.
"""

C0 = 299_792_458.0
ETA0 = 376.730313668
EPS = 1e-12


def canonical_polarization(label: str | None) -> str:
    """
    Normalize polarization label to 'TE' or 'TM'.

    Accepted aliases:
    - TE, VV, V, VERTICAL → 'TE'
    - TM, HH, H, HORIZONTAL → 'TM'
    - None or empty → 'TE' (default)

    Raises ValueError for unrecognized labels.
    """

    text = str(label or '').strip().upper()
    if text in {'TE', 'VV', 'V', 'VERTICAL'}:
        return 'TE'
    if text in {'TM', 'HH', 'H', 'HORIZONTAL'}:
        return 'TM'
    if not text:
        return 'TE'
    raise ValueError(
        f"Unsupported polarization '{label}'. "
        "Use TE/VV/V/VERTICAL or TM/HH/H/HORIZONTAL."
    )


def primary_polarization_alias(label: str) -> str:
    """Return 'VV' for TE, 'HH' for TM."""
    return 'VV' if canonical_polarization(label) == 'TE' else 'HH'


def polarization_alias_list(label: str) -> list[str]:
    """Return all accepted aliases for the given polarization."""
    canonical = canonical_polarization(label)
    if canonical == 'TE':
        return ['TE', 'VV', 'V', 'VERTICAL']
    return ['TM', 'HH', 'H', 'HORIZONTAL']


def unit_scale_to_meters(units: str) -> float:
    """Convert geometry unit string to meters scale factor."""
    value = (units or '').strip().lower()
    if value in {'inch', 'inches', 'in'}:
        return 0.0254
    if value in {'meter', 'meters', 'm'}:
        return 1.0
    raise ValueError(f"Unsupported geometry units '{units}'. Use inches or meters.")
