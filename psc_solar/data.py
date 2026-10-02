"""Conservative parsing based on the official database field description v5.4."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FEATURES, MISSING, SCOPE, validate_features

PINNED_SHA256 = 'e722ded60e6162a47b00746c6215498dcf0ce9dd004703fd01ab560dfc213ba2'
NUMBER = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


def scalar(value):
    text = str(value).strip()
    if not NUMBER.fullmatch(text):
        raise ValueError('not a scalar number')
    number = float(text)
    if not np.isfinite(number):
        raise ValueError('nonfinite number')
    return number


def mixture(names, coefficients, allowed, sum_bounds=None):
    if any(token in str(names) + str(coefficients) for token in ('|', '>>')):
        raise ValueError('multiple layers or steps')
    labels = [part.strip().strip('()') for part in str(names).split(';')]
    values = [scalar(part) for part in str(coefficients).split(';')]
    if len(labels) != len(values) or len(set(labels)) != len(labels) or not set(labels) <= set(allowed):
        raise ValueError('unsupported or mismatched mixture')
    total = sum(values)
    if min(values) < 0 or total <= 0:
        raise ValueError('invalid coefficients')
    if sum_bounds and not sum_bounds[0] <= total <= sum_bounds[1]:
        raise ValueError('nonstoichiometric coefficients outside scope')
    parsed = dict(zip(labels, values))
    return {name: parsed.get(name, 0.0) / total for name in allowed}


def text(value):
    value = str(value).strip()
    return 'Unreported' if value.lower() in MISSING else value


def normalize_doi(value):
    value = str(value).strip().lower()
    value = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', value)
    return value


def prepare(raw_path, output_directory):
    raw_path, output_directory = Path(raw_path), Path(output_directory)
    with raw_path.open('rb') as source:
        sha = hashlib.file_digest(source, 'sha256').hexdigest()
    if sha != PINNED_SHA256:
        raise ValueError('Raw CSV does not match pinned version. Check the source before training.')
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    counts = Counter()
    excluded, retained = [], []
    for position, row in raw.iterrows():
        reason = None
        try:
            if row['Cell_architecture'].strip() != 'nip':
                reason = 'architecture_not_nip'
            elif row['Perovskite_deposition_procedure'].strip() != 'Spin-coating' or scalar(row['Perovskite_deposition_number_of_deposition_steps']) != 1:
                reason = 'not_single_step_spin_coating'
            elif row['Perovskite_composition_b_ions'].strip() != 'Pb' or not 0.95 <= scalar(row['Perovskite_composition_b_ions_coefficients']) <= 1.05:
                reason = 'not_single_layer_stoichiometric_Pb'
            elif row['Perovskite_composition_perovskite_ABC3_structure'].lower() != 'true':
                reason = 'not_perovskite_structure'
            elif row['JV_light_spectra'].strip() != 'AM 1.5' or not np.isclose(scalar(row['JV_light_intensity']), 100, atol=0.01, rtol=0):
                reason = 'nonstandard_illumination'
            elif row['JV_default_PCE_scan_direction'].strip() != 'Reversed':
                reason = 'not_reverse_scan_label'
            if reason:
                raise ValueError(reason)
            doi = normalize_doi(row['Ref_DOI_number'])
            if not re.fullmatch(r'10\.\d{4,9}/\S+', doi):
                reason = 'missing_or_invalid_doi'
                raise ValueError(reason)
            reason = 'invalid_PCE'
            pce = scalar(row['JV_default_PCE'])
            if not 0 <= pce <= 100:
                raise ValueError(reason)
            reason = 'unsupported_A_or_X_composition'
            a = mixture(row['Perovskite_composition_a_ions'], row['Perovskite_composition_a_ions_coefficients'], ['Cs', 'FA', 'MA'], (0.8, 1.2))
            x = mixture(row['Perovskite_composition_c_ions'], row['Perovskite_composition_c_ions_coefficients'], ['Br', 'I'], (2.5, 3.5))
            reason = 'unsupported_or_unknown_solvent_ratio'
            solvent = mixture(row['Perovskite_deposition_solvents'], row['Perovskite_deposition_solvents_mixing_ratios'], ['DMF', 'DMSO', 'GBL', 'NMP'])
            reason = 'non_scalar_or_invalid_annealing'
            temperature = scalar(row['Perovskite_deposition_thermal_annealing_temperature'])
            time = scalar(row['Perovskite_deposition_thermal_annealing_time'])
            if not -10 <= temperature <= 400 or not 0 <= time <= 1440:
                raise ValueError(reason)
            reason = 'area_outside_small_cell_scope'
            area = scalar(row['Cell_area_measured'])
            if not 0 < area <= 1:
                raise ValueError(reason)
            retained.append({
                'source_row': position + 2, 'doi': doi, 'pce': pce,
                'a_cs': a['Cs'], 'a_fa': a['FA'], 'a_ma': a['MA'], 'x_br': x['Br'], 'x_i': x['I'],
                **{'solvent_' + key.lower(): value for key, value in solvent.items()},
                'anneal_temperature_c': temperature, 'anneal_time_min': time, 'area_cm2': area,
                'etl': text(row['ETL_stack_sequence']), 'htl': text(row['HTL_stack_sequence']),
                'quenching_media': text(row['Perovskite_deposition_quenching_media']),
                'additives': text(row['Perovskite_additives_compounds']),
            })
        except (ValueError, TypeError):
            reason = reason or 'malformed_scope_field'
            counts[reason] += 1
            excluded.append({'source_row': position + 2, 'reason': reason})
    clean = pd.DataFrame(retained)
    if clean.empty:
        raise ValueError('No rows meet the configured scope.')
    validate_features(clean)
    # Same-paper identical feature+label records may be duplicate extraction; keep the first.
    duplicate = clean.duplicated(subset=['doi', 'pce'] + FEATURES)
    for source_row in clean.loc[duplicate, 'source_row']:
        excluded.append({'source_row': int(source_row), 'reason': 'same_paper_exact_feature_label_duplicate'})
    counts['same_paper_exact_feature_label_duplicate'] = int(duplicate.sum())
    clean = clean.loc[~duplicate].reset_index(drop=True)
    # Keep identical recipes with different outcomes: they capture actual variability.
    report = {'raw_sha256': sha, 'raw_rows': len(raw), 'retained_rows': len(clean),
              'retained_dois': int(clean.doi.nunique()), 'scope': SCOPE,
              'exclusion_first_reason_counts': dict(counts),
              'pce_range': [float(clean.pce.min()), float(clean.pce.max())],
              'caveats': ['Measured area may include historical database default 0.1 cm².',
                          'Solvent ratios follow archive convention; volume basis is not guaranteed.',
                          'Unreported additives are not proven additive-free.',
                          'Records may represent averages, not individual devices.']}
    output_directory.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_directory / 'clean.csv', index=False)
    pd.DataFrame(excluded).to_csv(output_directory / 'exclusions.csv', index=False)
    (output_directory / 'cleaning_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return clean, report
