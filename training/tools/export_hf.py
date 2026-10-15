"""Merge an FSDP actor checkpoint into a Hugging Face directory without instantiating the model class.

verl's merger builds the model through transformers, which fails for the custom QwenDLM/LLaDA/DiffusionGemma code
under transformers 5.x. Here the merged state dict is written directly as sharded safetensors, the huggingface/ files of
the checkpoint are copied, and key names are aligned with a reference export of the same family (for example the
released ORL model): top-level modules nested under "model.", DiffusionGemma's shared encoder/decoder modules under
model.decoder, tied embeddings. A reference key that the checkpoint cannot supply is an error.

usage, with the Python of the training environment (inside $SIF with $PY, or the --no-container Python; see
training/README.md):
    PYTHONPATH=training/verl $PY training/tools/export_hf.py --actor-dir RUN/checkpoints/global_step_N/actor \
        --target OUT_DIR --reference RELEASED_MODEL_DIR
DiffusionGemma checkpoints need the runtime Transformers overlay first on the path: PYTHONPATH=$OVERLAY:training/verl
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open

from verl.model_merger.base_model_merger import generate_config_from_args, parse_args
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.utils.transformers_compat import ensure_transformers_runtime_compat


def reference_tensors(directory):
    index = directory / 'model.safetensors.index.json'
    files = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    layout = {}
    for name in files:
        with safe_open(directory / name, framework='pt') as handle:
            for key in handle.keys():
                layout[key] = (name, handle.get_slice(key).get_shape(), handle.get_slice(key).get_dtype())
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-dir', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='export of the same family whose key set is matched')
    parser.add_argument('--max-shard-size', default='5GB')
    parser.add_argument('--allow-drop', action='store_true', help='drop checkpoint tensors the reference does not name')
    args = parser.parse_args()
    if args.target.exists() and any(args.target.iterdir()):  # a failed export leaves the merger's empty target behind
        raise SystemExit(f'target exists: {args.target}')

    # the auto classes only know diffusion_gemma (from the overlay) once verl registers it, as the trainer does at start-up
    ensure_transformers_runtime_compat()
    sys.argv = ['merger', 'merge', '--backend', 'fsdp', '--trust-remote-code',
                '--local_dir', str(args.actor_dir), '--target_dir', str(args.target)]
    merger = FSDPModelMerger(generate_config_from_args(parse_args()))
    world_size = merger._get_world_size()
    mesh, mesh_dim_names = merger._extract_device_mesh_info(merger._load_rank_zero_state_dict(world_size), world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    state = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)

    report = {'checkpoint_keys': len(state), 'renamed': [], 'dropped': []}
    if args.reference:
        layout = reference_tensors(args.reference)
        # The paper exports nest top-level modules under "model." (e.g. LLaDA self_conditioning.* -> model.self_conditioning.*).
        # DiffusionGemma runs one set of embed_tokens/layers/norm modules as both its text encoder and its diffusion
        # decoder; the checkpoint names them under model.encoder.language_model, the paper export under model.decoder.
        for key in sorted(set(state) - set(layout)):
            for candidate in ('model.' + key, key.removeprefix('model.'),
                              key.replace('model.encoder.language_model.', 'model.decoder.', 1)):
                if candidate in layout and candidate not in state:
                    state[candidate] = state.pop(key)
                    report['renamed'].append(f'{key} -> {candidate}')
                    break
        for key in sorted(set(state) - set(layout)):
            report['dropped'].append(key)
            del state[key]
        if report['dropped'] and not args.allow_drop:
            raise SystemExit(f"the reference names none of these checkpoint tensors (pass --allow-drop to drop them): "
                             f"{report['dropped'][:10]}{' ...' if len(report['dropped']) > 10 else ''}")
        report['tied'] = []
        if 'lm_head.weight' in layout and 'lm_head.weight' not in state:
            # Tied embeddings (DG: tie_word_embeddings=true; the paper export's lm_head equals decoder.embed_tokens).
            for suffix in ('decoder.embed_tokens.weight', 'embed_tokens.weight', 'word_embeddings.weight'):
                source = next((k for k in state if k.endswith(suffix)), None)
                if source and list(state[source].shape) == list(layout['lm_head.weight'][1]):
                    state['lm_head.weight'] = state[source]
                    report['tied'].append(f'lm_head.weight <- {source}')
                    break
        for key in sorted(set(layout) - set(state)):
            # keys the paper export stores under both names (DiffusionGemma's per-layer layer_scalar)
            twin = key.replace('model.decoder.', 'model.encoder.language_model.', 1)
            if key.startswith('model.decoder.') and twin in state:
                state[key] = state[twin].clone()
                report['tied'].append(f'{key} <- {twin}')
        missing = sorted(set(layout) - set(state))
        if missing:
            raise SystemExit(f'{len(missing)} reference keys missing from the checkpoint: {missing[:5]}')
        # verl concatenates non-DTensor entries over ranks; replicated tensors (e.g. LLaDA expert_bias) come out
        # world_size times too long and are collapsed back when all copies agree.
        report['deduplicated'] = []
        for key in [k for k in state if list(state[k].shape) != list(layout[k][1])]:
            tensor, shape = state[key], list(layout[key][1])
            if shape and tensor.shape[0] == world_size * shape[0]:
                copies = tensor.chunk(world_size, dim=0)
                if all(torch.equal(copies[0], c) for c in copies[1:]):
                    state[key] = copies[0].clone()
                    report['deduplicated'].append(key)
        mismatched = [k for k in state if list(state[k].shape) != list(layout[k][1])]
        if mismatched:
            raise SystemExit(f'shape mismatch against the reference: {mismatched[:5]}')

    staging = args.target.with_name(args.target.name + '.partial')
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(args.actor_dir / 'huggingface', staging)
    save_torch_state_dict({k: v.contiguous() for k, v in state.items()}, staging, max_shard_size=args.max_shard_size,
                          metadata={'format': 'pt'})
    report['metadata_from_reference'] = []
    if args.reference:
        for name in ('generation_config.json', 'processor_config.json', 'chat_template.jinja', 'special_tokens_map.json',
                     'tokenizer.json', 'tokenizer_config.json', 'added_tokens.json'):
            if (args.reference / name).exists() and not (staging / name).exists():
                shutil.copyfile(args.reference / name, staging / name)
                report['metadata_from_reference'].append(name)
    report.update(exported_keys=len(state), dtypes=sorted({str(v.dtype) for v in state.values()}))
    (staging / 'export_report.json').write_text(json.dumps(report, indent=2) + '\n')
    staging.rename(args.target)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}))


if __name__ == '__main__':
    main()
