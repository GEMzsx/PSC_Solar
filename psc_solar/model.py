"""Paper-separated model selection, group calibration and auditable inference."""
import json
import hashlib
import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL, FEATURES, NUMERIC, SCOPE, validate_features


def independent_groups(frame):
    """Connect papers sharing exact feature+label duplicates before splitting."""
    parents = {doi: doi for doi in frame.doi.unique()}

    def root(doi):
        while parents[doi] != doi:
            parents[doi] = parents[parents[doi]]
            doi = parents[doi]
        return doi

    first = {}
    for row in frame.itertuples(index=False):
        values = row._asdict()
        signature = tuple(values[key] for key in FEATURES + ['pce'])
        if signature in first:
            a, b = root(values['doi']), root(first[signature])
            if a != b:
                parents[max(a, b)] = min(a, b)
        else:
            first[signature] = values['doi']
    return frame.doi.map(root)


def split_frame(frame, seed):
    frame = frame.copy()
    frame['split_group'] = independent_groups(frame)
    if frame.split_group.nunique() < 40:
        raise ValueError('At least 40 independent paper groups are needed for this four-way split.')
    remaining, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed).split(frame, groups=frame.split_group))
    pool = frame.iloc[remaining]
    development, calibration = next(GroupShuffleSplit(n_splits=1, test_size=10 / 85, random_state=seed + 1).split(pool, groups=pool.split_group))
    development = pool.iloc[development]
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed + 2).split(development, groups=development.split_group))
    parts = {'train': development.iloc[train].copy(), 'validation': development.iloc[validation].copy(),
             'calibration': pool.iloc[calibration].copy(), 'test': frame.iloc[test].copy()}
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not set(parts[left].doi) & set(parts[right].doi)
            assert not set(parts[left].split_group) & set(parts[right].split_group)
    return parts


def build_pipeline(estimator):
    transform = ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=True), CATEGORICAL),
    ])
    return Pipeline([('features', transform), ('regressor', estimator)])


def metrics(actual, predicted, groups):
    errors = pd.DataFrame({'absolute_error': np.abs(np.asarray(actual) - predicted), 'group': np.asarray(groups)})
    k = max(1, math.ceil(len(actual) * .25))
    # Rank membership avoids expanding top quartile when target values tie.
    top_actual = set(np.argsort(-np.asarray(actual), kind='stable')[:k])
    top_predicted = set(np.argsort(-predicted, kind='stable')[:k])
    return {'mae_percentage_points': float(mean_absolute_error(actual, predicted)),
            'paper_balanced_mae_percentage_points': float(errors.groupby('group').absolute_error.mean().mean()),
            'rmse_percentage_points': float(np.sqrt(mean_squared_error(actual, predicted))),
            'r2': float(r2_score(actual, predicted)),
            'top_quartile_precision': None if np.ptp(predicted) < 1e-12 else len(top_actual & top_predicted) / k,
            'top_quartile_random_expected': k / len(actual), 'top_quartile_selected_count': k}


def group_radius(errors, groups, alpha=.1):
    # One score per independent group, calibrated to cover all rows of a new group.
    scores = pd.DataFrame({'error': errors, 'group': np.asarray(groups)}).groupby('group').error.max().to_numpy()
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError('Insufficient calibration groups for a finite interval at this coverage level.')
    return float(np.sort(scores)[rank - 1]), len(scores)


def feature_importance(model, validation, seed=42):
    """Permute complete constrained mixtures together, keeping fractions aligned."""
    sample = validation.sample(n=min(800, len(validation)), random_state=seed).reset_index(drop=True)
    groups = {'A_site_composition': ['a_cs', 'a_fa', 'a_ma'], 'X_site_composition': ['x_br', 'x_i'],
              'solvent_mixture': ['solvent_dmf', 'solvent_dmso', 'solvent_gbl', 'solvent_nmp'],
              **{key: [key] for key in ['anneal_temperature_c', 'anneal_time_min', 'area_cm2'] + CATEGORICAL}}
    baseline = mean_absolute_error(sample.pce, model.predict(sample[FEATURES]))
    rng, records = np.random.default_rng(seed), []
    for name, keys in groups.items():
        scores = []
        for _ in range(5):
            altered = sample[FEATURES].copy()
            altered[keys] = altered[keys].iloc[rng.permutation(len(altered))].to_numpy()
            scores.append(mean_absolute_error(sample.pce, model.predict(altered)) - baseline)
        records.append({'feature': name, 'mae_increase_mean': float(np.mean(scores)), 'mae_increase_std': float(np.std(scores))})
    return pd.DataFrame(records).sort_values('mae_increase_mean', ascending=False)


def train(frame, output_directory, cleaning_report, seed=42):
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    parts = split_frame(frame, seed)
    train_set, validation = parts['train'], parts['validation']
    candidates = {
        'mean_baseline': DummyRegressor(strategy='mean'),
        'ridge_alpha_100': Ridge(alpha=100, solver='lsqr'),
        'extra_trees_leaf_3': ExtraTreesRegressor(n_estimators=160, min_samples_leaf=3, max_features=1., random_state=seed, n_jobs=2),
        'extra_trees_leaf_10': ExtraTreesRegressor(n_estimators=160, min_samples_leaf=10, max_features=1., random_state=seed, n_jobs=2),
    }
    models, validation_results = {}, {}
    for name, estimator in candidates.items():
        print(f'Training {name} on {len(train_set)} records...', flush=True)
        pipeline = build_pipeline(estimator)
        pipeline.fit(train_set[FEATURES], train_set.pce)
        models[name] = pipeline
        prediction = pipeline.predict(validation[FEATURES])
        validation_results[name] = metrics(validation.pce, prediction, validation.doi)
    selected = min(validation_results, key=lambda name: validation_results[name]['paper_balanced_mae_percentage_points'])
    # Keep the chosen training-only fit: validation data are not folded into calibration.
    model = models[selected]
    calibration, test = parts['calibration'], parts['test']
    cal_prediction = model.predict(calibration[FEATURES])
    radius, calibration_groups = group_radius(np.abs(calibration.pce.to_numpy() - cal_prediction), calibration.split_group)
    test_prediction = model.predict(test[FEATURES])
    covered = np.abs(test.pce.to_numpy() - test_prediction) <= radius
    paper_covered = pd.DataFrame({'group': test.split_group, 'covered': covered}).groupby('group').covered.all()
    baseline_prediction = models['mean_baseline'].predict(test[FEATURES])
    test_metrics = metrics(test.pce, test_prediction, test.doi)
    test_metrics.update({'row_interval_coverage': float(covered.mean()),
                         'independent_group_all_rows_coverage': float(paper_covered.mean()),
                         'interval_width_percentage_points': 2 * radius})
    importance_frame = feature_importance(model, validation, seed)
    importance_frame.to_csv(directory / 'feature_importance.csv', index=False)
    ranges = {key: {'min': float(train_set[key].min()), 'max': float(train_set[key].max())} for key in NUMERIC}
    categories = {key: sorted(train_set[key].unique().tolist()) for key in CATEGORICAL}
    # Pick a real, common recipe for the demo; never substitute a fabricated result.
    recipe_keys = [key for key in FEATURES if key not in ['anneal_temperature_c', 'anneal_time_min', 'area_cm2']]
    recipe_sizes = train_set.groupby(recipe_keys, dropna=False)['pce'].transform('size')
    example = train_set.loc[[recipe_sizes.idxmax()], FEATURES]
    example.to_csv(directory / 'example_input.csv', index=False)
    manifest = pd.concat([part[['source_row', 'doi', 'split_group']].assign(split=name) for name, part in parts.items()])
    manifest.to_csv(directory / 'split_manifest.csv', index=False)
    test_output = test[['source_row', 'doi', 'pce']].copy()
    test_output['prediction'] = test_prediction
    test_output['lower'] = test_prediction - radius
    test_output['upper'] = test_prediction + radius
    test_output.to_csv(directory / 'test_predictions.csv', index=False)
    report = {
        'seed': seed, 'raw_sha256': cleaning_report['raw_sha256'], 'scope': SCOPE,
        'retained_rows': len(frame), 'retained_dois': int(frame.doi.nunique()),
        'splits': {name: {'rows': len(part), 'papers': int(part.doi.nunique()), 'independent_groups': int(part.split_group.nunique())} for name, part in parts.items()},
        'selection_metric': 'validation paper-balanced MAE', 'selected_model': selected,
        'candidate_parameters': {name: estimator.get_params() for name, estimator in candidates.items()},
        'source_code_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob('*.py'))},
        'validation_results': validation_results, 'test': test_metrics,
        'test_mean_baseline': metrics(test.pce, baseline_prediction, test.doi),
        'interval': {'nominal_group_coverage': .9, 'radius_percentage_points': radius,
                     'calibration_groups': calibration_groups, 'method': 'finite-sample quantile of calibration group maximum absolute residuals',
                     'qualification': 'Coverage requires exchangeability of independent paper groups, including group sizes. It is not a per-recipe guarantee or a guarantee for a different laboratory.'},
        'feature_ranges': ranges, 'feature_categories': categories,
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
        'limitations': ['Public historical literature data, not group-specific experimental validation.',
                       'Small-area spin-coated cells; no large-area or coating-method extrapolation.',
                       'Permutation importance is predictive association, not causality.',
                       'No film-quality labels or verified physical solvent descriptors yet.',
                       'Duplicates within papers removed; cross-paper identical feature+label records kept in the same split group.',
                       'Measured area can include historical defaults; reported ratios/coefficients can be estimates.'],
    }
    (directory / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    bundle = {'model': model, 'radius': radius, 'metadata': report, 'training_features': train_set[FEATURES].reset_index(drop=True)}
    joblib.dump(bundle, directory / 'model.joblib', compress=3)
    print(json.dumps({'selected_model': selected, 'test': test_metrics, 'baseline': report['test_mean_baseline'], 'splits': report['splits']}, ensure_ascii=False, indent=2), flush=True)
    return report


def load_bundle(path):
    # Load only locally trained artifacts; joblib is not a safe interchange format.
    bundle = joblib.load(path)
    # Small interactive requests do not benefit from a Windows multiprocessing pipe.
    # Keep inference single-threaded for portability in restricted environments.
    regressor = bundle['model'].named_steps['regressor']
    if hasattr(regressor, 'n_jobs'):
        regressor.set_params(n_jobs=1)
    return bundle


def predict(bundle, frame):
    inputs = validate_features(frame)
    values = bundle['model'].predict(inputs)
    radius = bundle['radius']
    result = pd.DataFrame({'pce_prediction': values, 'pce_lower': values - radius,
                           'pce_upper': values + radius}, index=inputs.index)
    warnings = []
    metadata = bundle['metadata']
    for _, row in inputs.iterrows():
        messages = []
        for key in NUMERIC:
            limits = metadata['feature_ranges'][key]
            if row[key] < limits['min'] - 1e-8 or row[key] > limits['max'] + 1e-8:
                messages.append(f'{key}: outside training range')
        for key in CATEGORICAL:
            if row[key] not in metadata['feature_categories'][key]:
                messages.append(f'{key}: unseen category')
        warnings.append('; '.join(messages))
    result['warnings'] = warnings
    result['within_marginal_training_support'] = result.warnings.eq('')
    return result
