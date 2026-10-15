# Pinned SGLang source profiles

QwenDLM runs on SGLang with the SDAR engine patch (`sdar-engine.patch`); LLaDA runs on the
base tree. Do not replace one with the other or apply the SDAR engine patch directly to
the repository's shared engine.

```sh
python3 -m runtime.stage --family qwen-dlm --out outputs/runtime-qwen-dlm
python3 -m runtime.stage --family llada --out outputs/runtime-llada
```

These commands verify every tracked source file against the bundled inventory
and create separate source trees. No private runtime archive or Git history is
needed at preparation time. The QwenDLM runtime applies the bundled patch only
inside its new tree and checks the resulting content hashes. Existing output
directories are refused. The non-runtime formatter symlink is left out of staged
trees, and source symlinks outside the inventory are not admitted.

Base tree: `f2230a5c2fcca72440d6c3fea4b50441b7d7516c`.
QwenDLM engine (SDAR patch applied): `b974f47818b5c8e6ab855da8f19e1cad0c92464f`.
The `Dockerfile` pins the image used for generation and the adapter YAML
dependency. Run it with a separately mounted staged source tree, local model
weights and the frozen inputs. It does not contain checkpoints or benchmark
data. CUDA/PyTorch/attention dependencies come from that image, not the host
evaluation Python environment.

Staging was checked against both actual source profiles. For the release, a few
of these files had debug output removed, one warning added and comments
rewritten; the computation is unchanged. Rebuilt container and new-entrypoint
GPU acceptance remain separate from the original reproduction.
