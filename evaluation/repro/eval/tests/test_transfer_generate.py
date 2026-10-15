import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.adapters import transfer


class TransferGenerationTests(unittest.TestCase):
    def problem(self, idx):
        return {"task_id": f"gpqa/{idx}", "messages": [{"role": "user", "content": "trusted fixture"}],
                "gold": "A", "meta": {"question_id": idx}}

    def native(self, problem):
        return {**problem, "raw": "trusted fixture response", "finish": "stop", "completion_tokens": 5,
                "fwd_denoise": 2, "fwd_total": 3, "wall_s": 0.2}

    def test_normalization(self):
        problems = [self.problem(0), self.problem(1)]
        result = transfer.normalize([self.native(p) for p in reversed(problems)], problems, "gpqa", 12345)
        self.assertEqual([r["task_id"] for r in result], [p["task_id"] for p in problems])
        self.assertTrue(all(r["scoring_status"] == "not_run" for r in result))
        self.assertEqual(result[0]["meta"], problems[0]["meta"])

    def test_records_without_meta_bind_to_scorer(self):
        from repro.eval.portable.score_benchmarks import registry, validate_inputs
        problems = [{k: v for k, v in self.problem(i).items() if k != "meta"} for i in range(198)]
        records = transfer.normalize([self.native(p) for p in problems], problems, "gpqa", 1234)
        profile = registry()["profiles"]["transfer.gpqa"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {}
            for name, rows in (("records.jsonl", records), ("original.jsonl", problems)):
                data = "".join(json.dumps(r) + "\n" for r in rows).encode()
                (root / name).write_bytes(data)
                files[name] = {"path": name, "sha256": transfer.sha(data)}
            manifest = {"schema_version": 1, "scorer_protocol_id": profile["protocol_id"], "scope": "full_cell",
                        "benchmark": "gpqa", "expected_task_ids": [p["task_id"] for p in problems],
                        "records": files["records.jsonl"], "original_problems": files["original.jsonl"]}
            self.assertEqual(len(validate_inputs(manifest, root, profile)), 198)

    def test_fail_closed_outputs(self):
        p = self.problem(0)
        for key, value in (("gold", "B"), ("meta", {}), ("finish", "error"),
                           ("wall_s", float("inf")), ("completion_tokens", True), ("fwd_denoise", 0)):
            row = self.native(p)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                transfer.normalize([row], [p], "gpqa", 1234)
        with self.assertRaises(ValueError):
            transfer.normalize([], [p], "gpqa", 1234)

    def test_pinned_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inputs.jsonl"
            path.write_text(json.dumps(self.problem(0)) + "\n")
            with self.assertRaises(ValueError):
                transfer.read_inputs(path, "gpqa")
            with patch.dict(transfer.INPUTS, {"gpqa": (1, transfer.sha(path.read_bytes()))}):
                rows, _ = transfer.read_inputs(path, "gpqa")
                self.assertEqual(len(rows), 1)

    def test_native_sources_are_exact(self):
        for family, (name, digest) in transfer.CLIENTS.items():
            source = transfer.ROOT / "inference/clients" / name
            self.assertEqual(transfer.sha(source.read_bytes()), digest)

    def test_engine_directory_and_source_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            engine = runtime / "engine"
            engine.mkdir(parents=True)
            client = runtime / "multi_gen.py"
            client.write_text("# inert client\n")
            source = engine / "source.py"
            source.write_text("# inert engine\n")
            manifest = root / "runtime/sglang/source_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"base_revision": "fixture", "files": {"source.py": {
                "kind": "file", "sha256": transfer.sha(source.read_bytes())}}}))
            (engine / "engine_manifest.json").write_text(json.dumps({"profile": "llada", "engine_revision": "fixture",
                "source_manifest_sha256": transfer.sha(manifest.read_bytes())}))
            with patch.object(transfer, "ROOT", root), patch.dict(transfer.CLIENTS,
                    {"llada": ("multi_gen.py", transfer.sha(client.read_bytes()))}):
                self.assertEqual(transfer.validate_runtime(runtime, "llada"), client.resolve())
                source.write_text("# drift\n")
                with self.assertRaises(ValueError):
                    transfer.validate_runtime(runtime, "llada")

    def test_server_receipt(self):
        run = transfer.expand("transfer", ["dg-orl"], ["aime"])["runs"][0]
        p = run["parameters"]
        receipt = {"schema_version": 1, "family": "dg", "role": "orl", "image_digest": p["engine"]["image"],
                   "model_manifest_sha256": "abc", "diffusion_config": {"canvas_length": 256, "max_denoising_steps": 48},
                   "max_model_len": 32768, "seed_protocol": "reset_each_request", "sampler_role": "adaptive_scalar",
                   "sampling_overrides": {k: v for k, v in p["decoder"].items() if k not in ("canvas_length", "max_denoising_steps", "sampler_patch")},
                   "gpu_memory_utilization": 0.95, "attention_backend": "TRITON_ATTN", "base_url": "http://127.0.0.1:8000",
                   "server_started": True, "server_ready": True}
        transfer.validate_server(receipt, p, "orl", receipt["base_url"], "abc")
        for key, value in (("max_model_len", 8192), ("server_started", False), ("server_ready", False), ("sampler_role", "alignment")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                transfer.validate_server({**receipt, key: value}, p, "orl", receipt["base_url"], "abc")
        with self.assertRaises(ValueError):
            transfer.validate_server(receipt, p, "orl", "https://example.invalid", "abc")

    def test_execute_inert_native_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"fixture-not-a-model")
            runtime = root / "runtime"
            runtime.mkdir()
            client = runtime / "multi_gen.py"
            client.write_text("# inert fixture\n")
            args = argparse.Namespace(model=model, runtime=runtime, out=root / "output", model_id="llada-base",
                benchmark="gpqa", seed=12345, shard_index=0, smoke_limit=1, server_receipt=None)
            run = transfer.expand("transfer", ["llada-base"], ["gpqa"])["runs"][0]
            problem = self.problem(0)
            def inert(command, **kwargs):
                (args.out / "native/shard_w0.jsonl").write_text(json.dumps(self.native(problem)) + "\n")
            with patch.object(transfer.subprocess, "run", side_effect=inert) as fake:
                result = transfer.execute(args, (run, [problem], "fixture", client))
            self.assertEqual(fake.call_count, 1)
            self.assertEqual(result["scope"], "smoke_only")
            self.assertFalse(result["generated_code_executed"])
            self.assertTrue((args.out / "records.jsonl").is_file())

    def test_server_launcher_contract_inert(self):
        from runtime.vllm.serve import plan as server_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "generation_config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"fixture-not-a-model")
            for role in ("base", "orl"):
                for benchmark in ("aime", "ifeval"):
                    command, config, receipt = server_plan(model, role, benchmark, root / "server")
                    receipt.update(server_started=True, server_ready=True)
                    run = transfer.expand("transfer", ["dg-" + role], [benchmark])["runs"][0]
                    transfer.validate_server(receipt, run["parameters"], role, receipt["base_url"],
                                             transfer.model_identity(model)["sha256"])
                    self.assertIsNone(config["max_new_tokens"])
                    self.assertIn("--no-enable-prefix-caching", command)

    def test_reject_exited_server_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server_receipt.json"
            path.write_text("{}")
            self.assertEqual(transfer.read_server_receipt(path), {})
            path.with_name("server_exit.json").write_text('{"exit_code": 0}')
            with self.assertRaises(ValueError):
                transfer.read_server_receipt(path)


if __name__ == "__main__":
    unittest.main()
