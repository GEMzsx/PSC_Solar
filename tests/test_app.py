import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_prediction_and_optimization_flow(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label == '预测 PCE').click().run()
        self.assertFalse(app.exception)
        self.assertIn('pce_prediction', app.session_state['prediction'])
        next(button for button in app.button if button.label == '搜索工艺候选点').click().run()
        self.assertFalse(app.exception)
        self.assertIn('grid', app.session_state['exploration'])
        self.assertEqual(len(app.session_state['exploration']['grid']), 225)

    def test_invalid_fractions_display_error(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
        next(button for button in app.button if button.label == '预测 PCE').click().run()
        self.assertIn('pce_prediction', app.session_state['prediction'])
        app.number_input(key='a_cs').set_value(80.)
        app.number_input(key='a_fa').set_value(50.)
        next(button for button in app.button if button.label == '预测 PCE').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('不能超过' in error.value for error in app.error))
        with self.assertRaises(KeyError):
            _ = app.session_state['prediction']


if __name__ == '__main__':
    unittest.main()
