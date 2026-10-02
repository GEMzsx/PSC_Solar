"""Public-data small-sample rehearsal, evaluated only on development validation papers."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from .model import build_pipeline, metrics, split_frame
from .schema import FEATURES


def run(clean_path, output_path, sizes=(50, 100, 200), seeds=(42, 43, 44)):
    frame = pd.read_csv(clean_path, keep_default_na=False)
    parts = split_frame(frame, 42)
    pool, validation = parts['train'], parts['validation']
    records = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(pool.split_group.unique())
        ordered = pd.concat([pool.loc[pool.split_group.eq(group)] for group in order])
        for size in sizes:
            small = ordered.iloc[:size]
            if len(small) < size:
                raise ValueError('Not enough training records for requested sample budget.')
            assert not set(small.split_group) & set(validation.split_group)
            candidates = {'mean_baseline': DummyRegressor(strategy='mean'),
                          'ridge_alpha_100': Ridge(alpha=100, solver='lsqr'),
                          'extra_trees_leaf_3': ExtraTreesRegressor(n_estimators=160, min_samples_leaf=3, random_state=seed, n_jobs=1)}
            for name, estimator in candidates.items():
                model = build_pipeline(estimator)
                model.fit(small[FEATURES], small.pce)
                score = metrics(validation.pce, model.predict(validation[FEATURES]), validation.doi)
                records.append({'training_rows': size, 'training_papers': small.doi.nunique(), 'seed': seed, 'model': name, **score})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(output, index=False)
    print(result.groupby(['training_rows', 'model']).mae_percentage_points.agg(['mean', 'std']).to_string(), flush=True)
    return result
