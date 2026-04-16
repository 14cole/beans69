import json
import cmath
import math
import os
from typing import Any, Dict, List

import numpy as np

C0 = 299_792_458.0
EPS = 1e-12

def _canonical_user_polarization_label(label: str | None) -> str:
    text = str(label or '').strip().upper()
    if text in {'TE', 'VV', 'V', 'VERTICAL'}:
        return 'TE'
    if text in {'TM', 'HH', 'H', 'HORIZONTAL'}:
        return 'TM'
    if not text:
        return 'TE'
    raise ValueError(
        f"Unsupported polarization '{label}'. Use TE/VV/V/VERTICAL or TM/HH/H/HORIZONTAL."
    )

def _primary_alias_for_user_polarization(label: str) -> str:
    return 'VV' if _canonical_user_polarization_label(label) == 'TE' else 'HH'

def _alias_list_for_user_polarization(label: str) -> list[str]:
    canonical = _canonical_user_polarization_label(label)
    return ['TE', 'VV', 'V', 'VERTICAL'] if canonical == 'TE' else ['TM', 'HH', 'H', 'HORIZONTAL']

def _ensure_grim_ext(path: str) -> str:
    return path if path.lower().endswith('.grim') else f'{path}.grim'

def _suffix_for_incidence(theta_inc_deg: float) -> str:
    value = f'{theta_inc_deg:.6f}'.rstrip('0').rstrip('.')
    value = value.replace('-', 'm').replace('.', 'p')
    return f'inc_{value or "0"}'

def _freq_value_to_hz(freq_value: float, unit: str = 'GHz') -> float:
    unit_key = str(unit or 'GHz').strip().lower()
    scale = {
        'hz': 1.0,
        'khz': 1.0e3,
        'mhz': 1.0e6,
        'ghz': 1.0e9,
    }.get(unit_key, 1.0e9)
    return float(freq_value) * scale

def compute_dbke_from_linear(
    rcs_linear: float,
    frequency_value: float,
    frequency_unit: str = 'GHz',
) -> float:
    """
    Convert linear 2D scattering width to absolute dBke.

    Absolute dBke uses the knife-edge normalization:
        dBke = 10*log10((2*pi/lambda) * sigma_2d)
             = 10*log10((2*pi*f/c0) * sigma_2d)
    """

    lin = float(rcs_linear)
    if (not math.isfinite(lin)) or lin <= 0.0:
        lin = EPS
    freq_hz = _freq_value_to_hz(frequency_value, unit=frequency_unit)
    if (not math.isfinite(freq_hz)) or freq_hz <= 0.0:
        raise ValueError('frequency_value must be a positive finite frequency.')
    return 10.0 * math.log10(((2.0 * math.pi * freq_hz) / C0) * lin)

def compute_linear_from_dbke(dbke_value: float, frequency_value: float, frequency_unit: str = 'GHz') -> float:
    """Convert absolute dBke to linear 2D scattering width sigma_2d."""

    dbke = float(dbke_value)
    freq_hz = _freq_value_to_hz(frequency_value, unit=frequency_unit)
    if (not math.isfinite(freq_hz)) or freq_hz <= 0.0:
        raise ValueError('frequency_value must be a positive finite frequency.')
    return (C0 / (2.0 * math.pi * freq_hz)) * (10.0 ** (dbke / 10.0))

def _build_grid_for_samples(
    samples: List[Dict[str, Any]],
    polarization: str,
    source_path: str = '',
    history: str = '',
    preserve_raw_complex_amplitude: bool = True,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError('No samples available to export.')

    azimuths = np.asarray(sorted({float(row['theta_scat_deg']) for row in samples}), dtype=float)
    elevations = np.asarray([0.0], dtype=float)
    frequencies = np.asarray(sorted({float(row['frequency_ghz']) for row in samples}), dtype=float)
    polarization_label = _canonical_user_polarization_label(polarization)
    polarizations = np.asarray([polarization_label], dtype=str)

    shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
    rcs_phase = np.full(shape, np.nan, dtype=np.float32)
    rcs_power = np.full(shape, np.nan, dtype=np.float32)
    rcs_amp_real = np.full(shape, np.nan, dtype=np.float32) if preserve_raw_complex_amplitude else None
    rcs_amp_imag = np.full(shape, np.nan, dtype=np.float32) if preserve_raw_complex_amplitude else None

    az_index = {value: i for i, value in enumerate(azimuths)}
    f_index = {value: i for i, value in enumerate(frequencies)}

    for row in samples:
        az = float(row['theta_scat_deg'])
        freq = float(row['frequency_ghz'])
        lin = float(row.get('rcs_linear', 0.0))
        if not math.isfinite(lin) or lin < 0.0:
            lin = 0.0

        amp_real = float(row.get('rcs_amp_real', 0.0))
        amp_imag = float(row.get('rcs_amp_imag', 0.0))
        if not math.isfinite(amp_real):
            amp_real = 0.0
        if not math.isfinite(amp_imag):
            amp_imag = 0.0

        idx = (az_index[az], 0, f_index[freq], 0)
        amp_value_raw = complex(amp_real, amp_imag)
        phase_value = float(cmath.phase(amp_value_raw)) if abs(amp_value_raw) > EPS else 0.0

        existing_power = rcs_power[idx]
        if not np.isnan(existing_power):
            if abs(existing_power - lin) > EPS:
                raise ValueError(
                    f'Duplicate sample conflict at az={az}, el=0, f={freq}, pol={polarization}.'
                )
            existing_phase = rcs_phase[idx]
            if np.isfinite(existing_phase) and abs(existing_phase - phase_value) > EPS:
                raise ValueError(
                    f'Duplicate amplitude conflict at az={az}, el=0, f={freq}, pol={polarization}.'
                )
            continue

        rcs_power[idx] = float(max(lin, 0.0))
        rcs_phase[idx] = phase_value
        if preserve_raw_complex_amplitude:
            rcs_amp_real[idx] = float(amp_value_raw.real)
            rcs_amp_imag[idx] = float(amp_value_raw.imag)

    units_payload = {
        'azimuth': 'deg',
        'elevation': 'deg',
        'frequency': 'GHz',
        'rcs_log_unit': 'dBke',
        'rcs_linear_quantity': 'sigma_2d',
    }

    payload = {
        'azimuths': azimuths,
        'elevations': elevations,
        'frequencies': frequencies,
        'polarizations': polarizations,
        'polarization_alias_primary': _primary_alias_for_user_polarization(polarization_label),
        'polarization_aliases_json': json.dumps(_alias_list_for_user_polarization(polarization_label)),
        'rcs_power': rcs_power,
        'rcs_phase': rcs_phase,
        'rcs_domain': 'power_phase',
        'power_domain': 'linear_rcs',
        'source_path': source_path,
        'history': history,
        'units': json.dumps(units_payload),
        'phase_reference': 'origin=(0,0), convention=exp(-jwt), monostatic far-field amplitude',
        'raw_complex_amplitude_preserved': bool(preserve_raw_complex_amplitude),
    }
    if preserve_raw_complex_amplitude:
        payload['rcs_amp_real'] = rcs_amp_real
        payload['rcs_amp_imag'] = rcs_amp_imag
        payload['complex_field_domain'] = 'solver_raw_far_field_amplitude'
    return payload

def _save_grim_npz(payload: Dict[str, Any], path: str) -> str:
    out = _ensure_grim_ext(path)
    with open(out, 'wb') as f:
        save_payload = dict(
            azimuths=payload['azimuths'],
            elevations=payload['elevations'],
            frequencies=payload['frequencies'],
            polarizations=payload['polarizations'],
            polarization_alias_primary=payload.get('polarization_alias_primary', ''),
            polarization_aliases_json=payload.get('polarization_aliases_json', ''),
            rcs_power=payload['rcs_power'],
            rcs_phase=payload['rcs_phase'],
            rcs_domain=payload['rcs_domain'],
            power_domain=payload['power_domain'],
            source_path=payload['source_path'],
            history=payload['history'],
            units=payload['units'],
            phase_reference=payload['phase_reference'],
            raw_complex_amplitude_preserved=payload.get('raw_complex_amplitude_preserved', False),
        )
        if 'rcs_amp_real' in payload and 'rcs_amp_imag' in payload:
            save_payload['rcs_amp_real'] = payload['rcs_amp_real']
            save_payload['rcs_amp_imag'] = payload['rcs_amp_imag']
            save_payload['complex_field_domain'] = payload.get('complex_field_domain', 'solver_raw_far_field_amplitude')
        np.savez(f, **save_payload)
    return out

def export_result_to_grim(
    result: Dict[str, Any],
    output_path: str,
    polarization: str | None = None,
    source_path: str = '',
    history: str = '',
    preserve_raw_complex_amplitude: bool = True,
) -> List[str]:
    samples = result.get('samples', []) or []
    if not samples:
        raise ValueError('No solver samples were returned, nothing to export.')

    pol = _canonical_user_polarization_label(
        polarization or result.get('polarization_export') or result.get('polarization') or 'TE'
    )
    mode = str(result.get('scattering_mode', 'monostatic')).strip().lower()

    if mode != 'bistatic':
        payload = _build_grid_for_samples(
            samples,
            pol,
            source_path=source_path,
            history=history,
            preserve_raw_complex_amplitude=preserve_raw_complex_amplitude,
        )
        return [os.path.abspath(_save_grim_npz(payload, output_path))]

    by_inc: Dict[float, List[Dict[str, Any]]] = {}
    for row in samples:
        inc = float(row.get('theta_inc_deg', 0.0))
        by_inc.setdefault(inc, []).append(row)

    rootspec = _ensure_grim_ext(output_path)
    root_no_ext = rootspec[:-5]
    written: List[str] = []
    for inc in sorted(by_inc.keys()):
        payload = _build_grid_for_samples(
            by_inc[inc],
            pol,
            source_path=source_path,
            history=(history + f' | theta_inc_deg={inc:g}').strip(' |'),
            preserve_raw_complex_amplitude=preserve_raw_complex_amplitude,
        )
        out = f'{root_no_ext}_{_suffix_for_incidence(inc)}.grim'
        written.append(os.path.abspath(_save_grim_npz(payload, out)))
    return written

def _ensure_csv_ext(path: str) -> str:
    return path if path.lower().endswith('.csv') else f'{path}.csv'

def export_result_to_dbke_csv(
    result: Dict[str, Any],
    output_path: str,
    source_path: str = '',
    history: str = '',
) -> str:
    """
    Export solver samples to CSV with an absolute dBke column.

    """

    samples = result.get('samples', []) or []
    if not samples:
        raise ValueError('No solver samples were returned, nothing to export.')

    out = _ensure_csv_ext(output_path)
    rows = sorted(
        samples,
        key=lambda row: (
            float(row.get('frequency_ghz', 0.0)),
            float(row.get('theta_inc_deg', 0.0)),
            float(row.get('theta_scat_deg', 0.0)),
        ),
    )
    header = [
        'frequency_ghz',
        'theta_inc_deg',
        'theta_scat_deg',
        'rcs_linear',
        'rcs_db',
        'dbke',
        'rcs_amp_real',
        'rcs_amp_imag',
        'rcs_amp_phase_deg',
        'source_path',
        'history',
    ]
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            lin = float(row.get('rcs_linear', 0.0))
            if not math.isfinite(lin) or lin <= 0.0:
                lin = EPS
            freq_ghz = float(row.get('frequency_ghz', 0.0))
            rcs_db = float(row.get('rcs_db', 10.0 * math.log10(lin)))
            dbke = compute_dbke_from_linear(lin, freq_ghz, frequency_unit='GHz')
            vals = [
                f"{freq_ghz:.12g}",
                f"{float(row.get('theta_inc_deg', 0.0)):.12g}",
                f"{float(row.get('theta_scat_deg', 0.0)):.12g}",
                f"{lin:.12g}",
                f"{rcs_db:.12g}",
                f"{dbke:.12g}",
                f"{float(row.get('rcs_amp_real', 0.0)):.12g}",
                f"{float(row.get('rcs_amp_imag', 0.0)):.12g}",
                f"{float(row.get('rcs_amp_phase_deg', 0.0)):.12g}",
                source_path.replace(',', ';'),
                history.replace(',', ';'),
            ]
            f.write(','.join(vals) + '\n')
    return os.path.abspath(out)
