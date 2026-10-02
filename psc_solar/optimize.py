"""Model-based annealing exploration; recommendations require laboratory review."""
import itertools

import numpy as np
import pandas as pd

from .model import predict
from .schema import CATEGORICAL, FEATURES, validate_features


def explore(bundle, recipe, temperature_bounds, time_bounds, target_pce=15., temperature_tolerance=5., time_tolerance_fraction=.1, grid_size=15, batch_size=6):
    recipe = validate_features(pd.DataFrame([recipe])).iloc[0].to_dict()
    if not 3 <= grid_size <= 30 or not 1 <= batch_size <= 20:
        raise ValueError('Grid size must be 3..30 and batch size 1..20.')
    if not 0 <= temperature_tolerance <= 30 or not 0 <= time_tolerance_fraction <= .5:
        raise ValueError('Invalid perturbation tolerances.')
    for key, bounds in [('anneal_temperature_c', temperature_bounds), ('anneal_time_min', time_bounds)]:
        lo, hi = bounds
        support = bundle['metadata']['feature_ranges'][key]
        if not np.isfinite([lo, hi]).all() or not support['min'] <= lo < hi <= support['max']:
            raise ValueError(f'{key}: search bounds must lie inside observed training range.')
    if not np.isfinite(target_pce) or not 0 <= target_pce <= 100:
        raise ValueError('Target PCE must be between 0 and 100 percent.')
    history = bundle['training_features']
    composition = ['a_cs', 'a_fa', 'a_ma', 'x_br', 'x_i']
    solvents = ['solvent_dmf', 'solvent_dmso', 'solvent_gbl', 'solvent_nmp']
    matching = np.ones(len(history), dtype=bool)
    for key in CATEGORICAL:
        matching &= history[key].eq(recipe[key]).to_numpy()
    matching &= (history[composition].subtract(pd.Series(recipe)[composition]).abs().max(axis=1) <= .05).to_numpy()
    matching &= (history[solvents].subtract(pd.Series(recipe)[solvents]).abs().max(axis=1) <= .10).to_numpy()
    matching &= np.isclose(history.area_cm2, recipe['area_cm2'], rtol=.2, atol=.005)
    local = history.loc[matching]
    if len(local) < 5:
        raise ValueError(f'Only {len(local)} comparable training records. At least 5 are required to explore this recipe.')
    centers = list(itertools.product(np.linspace(*temperature_bounds, grid_size), np.linspace(*time_bounds, grid_size)))
    trials, center_ids = [], []
    for index, (temperature, time) in enumerate(centers):
        for delta_t, factor in itertools.product([-temperature_tolerance, 0., temperature_tolerance], [1 - time_tolerance_fraction, 1., 1 + time_tolerance_fraction]):
            trials.append({**recipe, 'anneal_temperature_c': temperature + delta_t, 'anneal_time_min': time * factor})
            center_ids.append(index)
    trial_frame = pd.DataFrame(trials)
    # Validate the perturbations as well as the centers; reject physically invalid requests.
    validate_features(trial_frame)
    prediction = predict(bundle, trial_frame)
    comparison_keys = ['anneal_temperature_c', 'anneal_time_min']
    scale = np.maximum(local[comparison_keys].max().to_numpy() - local[comparison_keys].min().to_numpy(), [10., 5.])
    distance = np.linalg.norm((trial_frame[comparison_keys].to_numpy()[:, None, :] - local[comparison_keys].to_numpy()[None, :, :]) / scale, axis=2).min(axis=1)
    trial_result = prediction.assign(center_id=center_ids, local_distance=distance)
    summary = trial_result.groupby('center_id').agg(
        worst_case_prediction=('pce_prediction', 'min'),
        conservative_lower=('pce_lower', 'min'),
        max_local_distance=('local_distance', 'max'),
        all_within_marginal_support=('within_marginal_training_support', 'all'),
    )
    summary['anneal_temperature_c'] = [point[0] for point in centers]
    summary['anneal_time_min'] = [point[1] for point in centers]
    summary['supported'] = summary.all_within_marginal_support & summary.max_local_distance.le(.25)
    summary['model_passes_target'] = summary.supported & summary.conservative_lower.ge(target_pce)
    center_frame = pd.DataFrame([{**recipe, 'anneal_temperature_c': t, 'anneal_time_min': time} for t, time in centers])
    center_predictions = predict(bundle, center_frame)
    summary['pce_prediction'] = center_predictions.pce_prediction.to_numpy()
    summary['pce_lower'] = center_predictions.pce_lower.to_numpy()
    summary['pce_upper'] = center_predictions.pce_upper.to_numpy()
    regressor = bundle['model'].named_steps['regressor']
    if hasattr(regressor, 'estimators_'):
        transformed = bundle['model'].named_steps['features'].transform(center_frame[FEATURES])
        summary['ensemble_disagreement'] = np.std([tree.predict(transformed) for tree in regressor.estimators_], axis=0)
    else:
        summary['ensemble_disagreement'] = np.nan
    center_distance = np.linalg.norm((center_frame[comparison_keys].to_numpy()[:, None, :] - local[comparison_keys].to_numpy()[None, :, :]) / scale, axis=2).min(axis=1)
    summary['center_local_distance'] = center_distance
    candidates = summary.loc[summary.supported & summary.center_local_distance.gt(.02)].copy()
    selected = []
    # Alternate conservative performance and exploratory disagreement/novelty.
    remaining = candidates.copy()
    for position in range(min(batch_size, len(candidates))):
        objective = 'conservative_lower' if position % 2 == 0 else ('ensemble_disagreement' if remaining.ensemble_disagreement.notna().any() else 'center_local_distance')
        ranked = remaining.sort_values(objective, ascending=False)
        if ranked.empty:
            break
        choice = ranked.iloc[0].copy()
        choice['recommendation_reason'] = 'performance' if position % 2 == 0 else 'exploration_proxy'
        selected.append(choice)
        chosen_point = choice[comparison_keys].to_numpy(dtype=float)
        separation = np.linalg.norm((remaining[comparison_keys].to_numpy() - chosen_point) / scale, axis=1)
        remaining = remaining.loc[separation > .08]
    recommendations = pd.DataFrame(selected)
    if not recommendations.empty:
        for key, value in recipe.items():
            if key not in comparison_keys:
                recommendations[key] = value
    return {'grid': summary.reset_index(drop=True), 'recommendations': recommendations.reset_index(drop=True),
            'comparable_training_records': len(local),
            'qualification': 'Model-screened points, not experimentally validated windows. Tree disagreement and novelty are exploration proxies, not calibrated uncertainty or proven information gain.'}
