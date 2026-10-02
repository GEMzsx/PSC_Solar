import unittest

import numpy as np
import pandas as pd

from psc_solar.data import mixture, normalize_doi, scalar
from psc_solar.model import group_radius, independent_groups, load_bundle, predict, split_frame
from psc_solar.optimize import explore
from psc_solar.schema import CATEGORICAL, FEATURES, NUMERIC, validate_features


class ParsingTests(unittest.TestCase):
    def test_scalar_rejects_multistage_and_unknown(self):
        for value in ['70; 100', '100 >> 150', 'nan', '1 mg/ml', '25 | 100', 'inf']:
            with self.assertRaises(ValueError):
                scalar(value)
        self.assertEqual(scalar('1e2'), 100)

    def test_mixture_alignment_and_normalization(self):
        result = mixture('Cs; FA; MA', '0.05; 0.8; 0.15', ['Cs', 'FA', 'MA'], (.8, 1.2))
        self.assertAlmostEqual(result['FA'], .8)
        for names, values in [('DMF; DMSO', '1'), ('DMF >> DMSO', '1 >> 1'), ('DMF; DMSO', '-1; 2'), ('DMF; DMF', '1; 1')]:
            with self.assertRaises(ValueError):
                mixture(names, values, ['DMF', 'DMSO'])

    def test_doi_urls_normalized(self):
        self.assertEqual(normalize_doi(' HTTPS://DX.DOI.ORG/10.1234/ABC '), '10.1234/abc')

    def test_interval_uses_paper_max_and_finite_sample_rank(self):
        groups = np.repeat(np.arange(10), 2)
        errors = np.column_stack([np.zeros(10), np.arange(1, 11)]).ravel()
        radius, count = group_radius(errors, groups)
        self.assertEqual((radius, count), (10., 10))
        with self.assertRaises(ValueError):
            group_radius([1, 2], ['a', 'b'])


class ActualArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle('artifacts/baseline/model.joblib')
        cls.example = pd.read_csv('artifacts/baseline/example_input.csv', keep_default_na=False)

    def test_split_has_no_paper_or_exact_duplicate_leakage(self):
        clean = pd.read_csv('data/processed/clean.csv', keep_default_na=False)
        parts = split_frame(clean, 42)
        manifest = pd.read_csv('artifacts/baseline/split_manifest.csv')
        self.assertEqual(manifest.groupby('doi').split.nunique().max(), 1)
        self.assertEqual(manifest.groupby('split_group').split.nunique().max(), 1)
        self.assertEqual(len(manifest), len(clean))
        seen = set()
        for part in parts.values():
            signatures = set(map(tuple, part[FEATURES + ['pce']].to_numpy()))
            self.assertFalse(seen & signatures)
            seen |= signatures

    def test_duplicate_papers_connected_transitively(self):
        row = {**self.example.iloc[0].to_dict(), 'pce': 10.}
        other = {**row, 'pce': 11.}
        frame = pd.DataFrame([{**row, 'doi': 'a'}, {**row, 'doi': 'b'},
                              {**other, 'doi': 'b'}, {**other, 'doi': 'c'}])
        self.assertEqual(independent_groups(frame).nunique(), 1)

    def test_invalid_input_does_not_silently_normalize(self):
        for key, value in [('a_ma', .5), ('area_cm2', 0), ('anneal_time_min', np.inf)]:
            bad = self.example.copy()
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_features(bad)

    def test_prediction_interval_and_unseen_category_warning(self):
        result = predict(self.bundle, self.example)
        self.assertTrue(np.isfinite(result[['pce_prediction', 'pce_lower', 'pce_upper']]).all().all())
        np.testing.assert_allclose(result.pce_upper - result.pce_lower, self.bundle['radius'] * 2)
        bad = self.example.copy()
        bad['etl'] = 'not_in_training_set'
        result = predict(self.bundle, bad)
        self.assertIn('unseen category', result.warnings.iloc[0])
        self.assertFalse(result.within_marginal_training_support.iloc[0])

    def test_exploration_preserves_recipe_and_filters_support(self):
        recipe = self.example.iloc[0].to_dict()
        t, time = recipe['anneal_temperature_c'], recipe['anneal_time_min']
        result = explore(self.bundle, recipe, (t - 5, t + 5), (max(.1, time * .9), max(time * 1.1, .2)), temperature_tolerance=1., time_tolerance_fraction=.01, grid_size=5)
        self.assertEqual(len(result['grid']), 25)
        suggestions = result['recommendations']
        if not suggestions.empty:
            self.assertTrue(suggestions.supported.all())
            self.assertTrue(suggestions.center_local_distance.gt(.02).all())
            for key in FEATURES:
                if key not in ['anneal_temperature_c', 'anneal_time_min']:
                    self.assertTrue(suggestions[key].eq(recipe[key]).all())


if __name__ == '__main__':
    unittest.main()
