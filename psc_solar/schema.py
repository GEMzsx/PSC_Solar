"""Feature contract shared by cleaning, training, inference and the UI."""
NUMERIC = [
    'a_cs', 'a_fa', 'a_ma', 'x_br', 'x_i',
    'solvent_dmf', 'solvent_dmso', 'solvent_gbl', 'solvent_nmp',
    'anneal_temperature_c', 'anneal_time_min', 'area_cm2',
]
CATEGORICAL = ['etl', 'htl', 'quenching_media', 'additives']
FEATURES = NUMERIC + CATEGORICAL
SCOPE = {
    'architecture': 'nip', 'deposition': 'single-step Spin-coating',
    'absorber': 'single-layer Pb-based; A=Cs/FA/MA, X=I/Br',
    'illumination': 'AM 1.5, 100 mW/cm²',
    'label': 'JV_default_PCE, Reversed scan only',
    'area': '0 < measured area <= 1 cm²; historical default area may be present',
    'annealing': 'one numeric temperature and time, no multistage program',
    'solvents': 'DMF/DMSO/GBL/NMP with known relative mixing ratios',
}
MISSING = {'', 'nan', 'none', 'null', 'n/a', 'na', 'unknown', 'unkown', 'not reported'}


def validate_features(frame):
    """Return an independent normalized frame; never silently repair invalid inputs."""
    import numpy as np
    import pandas as pd
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    result = frame[FEATURES].copy()
    if result.empty:
        raise ValueError('Provide at least one input row.')
    for key in NUMERIC:
        result[key] = pd.to_numeric(result[key], errors='raise')
        if not np.isfinite(result[key]).all():
            raise ValueError(f'{key} must contain finite numbers.')
    for cols in [['a_cs', 'a_fa', 'a_ma'], ['x_br', 'x_i'],
                 ['solvent_dmf', 'solvent_dmso', 'solvent_gbl', 'solvent_nmp']]:
        if ((result[cols] < 0) | (result[cols] > 1)).any().any():
            raise ValueError(f'Fractions must be in [0, 1]: {cols}')
        if not np.allclose(result[cols].sum(axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError(f'Fractions must sum to 1: {cols}')
    bounds = {'anneal_temperature_c': (-10, 400), 'anneal_time_min': (0, 1440), 'area_cm2': (0, 1)}
    for key, (lo, hi) in bounds.items():
        if ((result[key] < lo) | (result[key] > hi)).any() or (key == 'area_cm2' and (result[key] <= 0).any()):
            raise ValueError(f'{key} is outside supported scope {lo}..{hi}.')
    for key in CATEGORICAL:
        result[key] = result[key].fillna('Unreported').astype(str).str.strip().replace('', 'Unreported')
    return result
