"""Validate and merge experiment records without overwriting conflicting samples."""
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from .data import mixture, scalar, text
from .schema import FEATURES, validate_features

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'experiment_record.csv'


def atomic_csv(frame, path):
    """Replace a generated CSV only after its full contents have been written."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def to_model_input(row):
    if row['architecture'] != 'nip' or row['deposition_method'] != 'Spin-coating':
        raise ValueError('Outside nip / single-step Spin-coating scope')
    if row['composition_b_ions'] != 'Pb' or not .95 <= scalar(row['composition_b_coefficients']) <= 1.05:
        raise ValueError('Outside single-layer Pb scope')
    if row['light_spectrum'] != 'AM 1.5' or abs(scalar(row['light_intensity_mw_cm2']) - 100) > .01 or row['scan_direction'] != 'Reversed':
        raise ValueError('Measurement conditions do not match model scope')
    if row['solvent_ratio_basis'] != 'volume':
        raise ValueError('Lab data require confirmed volume ratios for this interface')
    a = mixture(row['composition_a_ions'], row['composition_a_coefficients'], ['Cs', 'FA', 'MA'], (.8, 1.2))
    x = mixture(row['composition_x_ions'], row['composition_x_coefficients'], ['Br', 'I'], (2.5, 3.5))
    solvents = mixture(row['solvents'], row['solvent_ratios'], ['DMF', 'DMSO', 'GBL', 'NMP'])
    recipe = {'a_cs': a['Cs'], 'a_fa': a['FA'], 'a_ma': a['MA'], 'x_br': x['Br'], 'x_i': x['I'],
              **{'solvent_' + key.lower(): value for key, value in solvents.items()},
              'anneal_temperature_c': scalar(row['anneal_temperature_c']),
              'anneal_time_min': scalar(row['anneal_time_min']), 'area_cm2': scalar(row['area_measured_cm2']),
              **{key: text(row[key]) for key in ['etl', 'htl', 'quenching_media', 'additives']}}
    validate_features(pd.DataFrame([recipe]))
    return recipe


def validate_records(frame):
    columns = pd.read_csv(TEMPLATE).columns.tolist()
    if set(frame.columns) != set(columns):
        raise ValueError(f'Use the experiment template. Missing={sorted(set(columns) - set(frame.columns))}, extra={sorted(set(frame.columns) - set(columns))}')
    frame = frame[columns].fillna('').astype(str).apply(lambda column: column.str.strip())
    if frame.empty:
        raise ValueError('Template contains no experiment records.')
    if frame.sample_id.eq('').any() or frame.sample_id.duplicated().any():
        raise ValueError('sample_id must be nonempty and unique within the import.')
    for _, row in frame.iterrows():
        for key in ['batch_id', 'operator', 'source']:
            if not row[key]:
                raise ValueError(f"{row.sample_id}: {key} is required.")
        try:
            date.fromisoformat(row['date'])
        except ValueError as error:
            raise ValueError(f'{row.sample_id}: date must be YYYY-MM-DD.') from error
        if row.status not in ['success', 'failed', 'unmeasured']:
            raise ValueError(f'{row.sample_id}: status must be success, failed or unmeasured.')
        if row.status == 'failed' and not row.failure_reason:
            raise ValueError(f'{row.sample_id}: record the failure reason.')
        if row.status == 'success' and not row.pce_percent:
            raise ValueError(f'{row.sample_id}: successful measurement requires PCE.')
        if row.status == 'unmeasured' and row.pce_percent:
            raise ValueError(f'{row.sample_id}: unmeasured record cannot contain PCE.')
        for key, bounds in {'pce_percent': (0, 100), 'ff_fraction': (0, 1),
                            'ambient_rh_percent': (0, 100), 'coverage_fraction': (0, 1)}.items():
            if row[key] and not bounds[0] <= scalar(row[key]) <= bounds[1]:
                raise ValueError(f'{row.sample_id}: {key} must be in {bounds}.')
        if row.solvent_ratio_basis and row.solvent_ratio_basis not in ['volume', 'mass', 'molar']:
            raise ValueError(f'{row.sample_id}: solvent_ratio_basis must be volume, mass or molar.')
    return frame


def ingest(input_path, output_directory):
    input_path, output_directory = Path(input_path), Path(output_directory)
    incoming = validate_records(pd.read_csv(input_path, dtype=str, keep_default_na=False))
    master_path = output_directory / 'experiments.csv'
    existing = validate_records(pd.read_csv(master_path, dtype=str, keep_default_na=False)) if master_path.exists() else incoming.iloc[:0].copy()
    previous = existing.set_index('sample_id')
    new_rows, duplicate_count = [], 0
    for _, row in incoming.iterrows():
        if row.sample_id in previous.index:
            if not row.drop('sample_id').equals(previous.loc[row.sample_id]):
                raise ValueError(f'{row.sample_id}: conflicting existing sample; no records were written. Use a new ID for a new measurement.')
            duplicate_count += 1
        else:
            new_rows.append(row.to_dict())
    master = pd.concat([existing, pd.DataFrame(new_rows, columns=incoming.columns)], ignore_index=True)
    normalized, eligibility = [], []
    for _, row in master.iterrows():
        try:
            if not row.pce_percent:
                raise ValueError('No measured PCE yet; record retained for tracking')
            recipe = to_model_input(row)
            normalized.append({'sample_id': row.sample_id, 'batch_id': row.batch_id, 'pce': scalar(row.pce_percent), **recipe})
            eligibility.append({'sample_id': row.sample_id, 'eligible': True, 'reason': ''})
        except ValueError as error:
            eligibility.append({'sample_id': row.sample_id, 'eligible': False, 'reason': str(error)})
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_csv(master, master_path)
    atomic_csv(pd.DataFrame(normalized, columns=['sample_id', 'batch_id', 'pce'] + FEATURES), output_directory / 'model_inputs.csv')
    atomic_csv(pd.DataFrame(eligibility), output_directory / 'eligibility.csv')
    report = {'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(), 'incoming_rows': len(incoming),
              'added_rows': len(new_rows), 'identical_existing_rows_skipped': duplicate_count,
              'master_rows': len(master), 'model_eligible_rows': len(normalized),
              'automatic_retraining': False}
    (output_directory / 'latest_import.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
