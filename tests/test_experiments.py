import tempfile
import unittest
from pathlib import Path

import pandas as pd

from psc_solar.experiments import TEMPLATE, ingest, validate_records


def simulated_record():
    # Synthetic QA fixture only; not a laboratory observation or training dataset.
    row = {key: '' for key in pd.read_csv(TEMPLATE).columns}
    row.update(sample_id='SYNTHETIC_QA_001', batch_id='SYNTHETIC_QA_BATCH', operator='test', date='2026-10-02',
               architecture='nip', deposition_method='Spin-coating', composition_a_ions='MA', composition_a_coefficients='1',
               composition_b_ions='Pb', composition_b_coefficients='1', composition_x_ions='I', composition_x_coefficients='3',
               solvents='DMF', solvent_ratios='1', solvent_ratio_basis='volume', anneal_temperature_c='100', anneal_time_min='10',
               area_measured_cm2='0.1', etl='TiO2-c', htl='Spiro-MeOTAD', pce_percent='0', ff_fraction='0',
               scan_direction='Reversed', light_intensity_mw_cm2='100', light_spectrum='AM 1.5',
               status='failed', failure_reason='Synthetic QA fixture', source='SYNTHETIC_QA')
    return row


class ExperimentTests(unittest.TestCase):
    def test_failed_zero_pce_retained_and_idempotent_import(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            output = Path(folder) / 'lab'
            pd.DataFrame([simulated_record()]).to_csv(source, index=False)
            self.assertEqual(ingest(source, output)['model_eligible_rows'], 1)
            self.assertEqual(ingest(source, output)['added_rows'], 0)
            stored = pd.read_csv(output / 'model_inputs.csv')
            self.assertEqual(stored.pce.iloc[0], 0)
            self.assertEqual(len(stored), 1)

    def test_conflicting_sample_does_not_overwrite_master(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            output = Path(folder) / 'lab'
            row = simulated_record()
            pd.DataFrame([row]).to_csv(source, index=False)
            ingest(source, output)
            before = (output / 'experiments.csv').read_bytes()
            row['pce_percent'] = '1'
            pd.DataFrame([row]).to_csv(source, index=False)
            with self.assertRaises(ValueError):
                ingest(source, output)
            self.assertEqual((output / 'experiments.csv').read_bytes(), before)

    def test_ff_percent_cannot_be_mistaken_for_fraction(self):
        row = simulated_record()
        row['ff_fraction'] = '80'
        with self.assertRaises(ValueError):
            validate_records(pd.DataFrame([row]))
