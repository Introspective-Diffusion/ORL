import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("paper_plan", Path(__file__).with_name("plan.py"))
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


class PlanTests(unittest.TestCase):
    def test_full_counts(self):
        for suite, expected in (("main", (40, 40, 24, 192, 96, 24100)),
                                ("transfer", (68, 100, 24, 188, 188, 106280)),
                                ("tps", (12, 12, 12, 12, 3, 108))):
            with self.subTest(suite=suite):
                result = plan.expand(suite)
                keys = ("generation_runs", "scored_run_cells", "displayed_cells", "benchmark_shards", "worker_groups", "generated_rows")
                self.assertEqual(tuple(result["counts"][k] for k in keys), expected)
                self.assertFalse(result["launches_resources"])

    def test_partitions(self):
        self.assertEqual([len(x) for x in plan.partition_indices(500, 8, "contiguous_balanced")], [63]*4+[62]*4)
        self.assertEqual(plan.partition_indices(5, 2, "modulo"), [[0, 2, 4], [1, 3]])
        for kind in ("modulo", "contiguous_balanced"):
            parts = plan.partition_indices(1319, 8, kind)
            self.assertEqual(sorted(i for p in parts for i in p), list(range(1319)))

    def test_filters_and_years(self):
        result = plan.expand("transfer", ["llada-orl"], ["aime24"])
        self.assertEqual(result["counts"]["generation_runs"], 8)
        self.assertEqual(result["counts"]["scored_run_cells"], 8)
        self.assertTrue(all(r["generated_rows"] == 60 for r in result["runs"]))
        self.assertTrue(all(c["n"] == 30 for c in result["score_cells"]))
        result = plan.expand("main", ["dg-orl"], ["mbpp427"])
        self.assertEqual(result["counts"]["generation_runs"], 3)
        self.assertTrue(all(r["parameters"]["max_new_tokens"] == 4096 for r in result["runs"]))

    def test_qwen_dlm_caps_and_seed(self):
        result = plan.expand("main", ["qwen-dlm-base", "qwen-dlm-orl"], ["gsm8k", "he"])
        by_key = {r["id"]: r for r in result["runs"]}
        self.assertEqual(by_key["qwen-dlm-base.gsm8k.s1234"]["parameters"]["max_new_tokens"], 49152)
        self.assertEqual(by_key["qwen-dlm-orl.gsm8k.s1234"]["parameters"]["max_new_tokens"], 32768)
        self.assertTrue(all(r["parameters"]["decoder"]["randtok_seed"] == 12345 for r in result["runs"]))
        self.assertTrue(all(r["parameters"]["decoder"]["temperature"] == 0 for r in result["runs"]))

    def test_transfer_seed_and_context(self):
        runs = plan.expand("transfer", ["llada-base", "dg-base"], ["gpqa"])["runs"]
        self.assertEqual(len(runs), 6)
        for run in runs:
            if run["family"] == "llada":
                self.assertEqual(run["parameters"]["decoder"]["randtok_seed"], run["seed"])
            else:
                self.assertEqual(run["parameters"]["engine"]["context_length"], 32768)

    def test_diffusion_gemma_transfer_protocol(self):
        expected_decoders = {
            "base": {
                "canvas_length": 256, "max_denoising_steps": 48,
                "confidence_threshold": 0.005, "stability_threshold": 1,
                "sampler_patch": "alignment", "t_min": 0.4, "t_max": 0.8,
                "sampler_config": {"entropy_bound": 0.1},
            },
            "orl": {
                "canvas_length": 256, "max_denoising_steps": 48,
                "confidence_threshold": 0.005, "stability_threshold": 1,
                "sampler_patch": "adaptive_scalar", "t_min": 0.5, "t_max": 0.8,
                "sampler_config": {"entropy_bound": 0.12},
                "temperature_mode": "adaptive_scalar", "adaptive_power": 0.5,
            },
        }
        for role in ("base", "orl"):
            for benchmark, cap, context in (("gpqa", 16384, 32768), ("lcb", 4096, 8192)):
                with self.subTest(role=role, benchmark=benchmark):
                    runs = plan.expand("transfer", [f"dg-{role}"], [benchmark])["runs"]
                    self.assertEqual([run["seed"] for run in runs], [1234, 1235, 1236])
                    self.assertTrue(all(run["dataset_n"] == (198 if benchmark == "gpqa" else 175)
                                        for run in runs))
                    self.assertTrue(all(run["parameters"]["max_new_tokens"] == cap for run in runs))
                    self.assertTrue(all(run["parameters"]["engine"]["context_length"] == context
                                        for run in runs))
                    self.assertTrue(all(run["parameters"]["decoder"] == expected_decoders[role]
                                        for run in runs))

    def test_quick_selection(self):
        result = plan.expand("tps")
        for run in result["runs"]:
            self.assertEqual(run["warmup_indices"], [713] if run["benchmark"] == "gsm8k" else [110])
            self.assertEqual(len(run["measured_indices"]), 8)
        self.assertEqual(result["counts"]["measured_rows"], 96)

    def test_invalid_filters(self):
        for args in (("unknown",), ("main", ["unknown"]), ("transfer", ["qwen-dlm-base"]), ("main", None, ["mbpp500"])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                plan.expand(*args)

    def test_configs_no_private_locations(self):
        files = list((plan.ROOT / "configs/paper").rglob("*.yaml")) + list(Path(__file__).parent.glob("*.yaml"))
        for file in files:
            raw = file.read_text()
            json.loads(raw)
            self.assertNotIn("s3://", raw)
            self.assertNotIn("/Users/", raw)
            self.assertNotIn("AWS_", raw)


if __name__ == "__main__":
    unittest.main()
