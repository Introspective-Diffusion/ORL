import json
from pathlib import Path
import unittest
from inference.adapters.qwen_dlm import decoder
from repro.paper.render_config import render


class RenderTests(unittest.TestCase):
    def test_bundled_source_references_resolve(self):
        root = Path(__file__).resolve().parents[2]
        for path in (root / 'configs/paper/decoding').glob('*.yaml'):
            config = json.loads(path.read_text())
            for reference in config['source_references']:
                with self.subTest(config=path.name, reference=reference):
                    target = (root / reference).resolve()
                    self.assertTrue(target.is_relative_to(root))
                    self.assertTrue(target.is_file())
            historical = config.get('historical_source_references_not_bundled', [])
            self.assertFalse(set(historical) & set(config['source_references']))

    def test_all_qwen_dlm_cells_match_native_adapter(self):
        for role in ('qwen-dlm-base', 'qwen-dlm-orl'):
            for benchmark in ('gsm8k', 'he', 'math500', 'mbpp'):
                self.assertEqual(render('main', role, benchmark, 1234), decoder(role, benchmark))

    def test_llada_seeds_and_temperature(self):
        config = render('transfer', 'llada-orl', 'aime', 12352)
        self.assertEqual(config['randtok_seed'], 12352)
        self.assertEqual(config['prob_threshold'], .85)
        self.assertEqual(config['temperature'], 0)
        self.assertEqual(config['temperature_max'], 0)

    def test_invalid_or_dg_selection(self):
        for model, seed in [('llada-base', 999), ('dg-base', 1234)]:
            with self.assertRaises(ValueError):
                render('main', model, 'he', seed)


if __name__ == '__main__':
    unittest.main()
