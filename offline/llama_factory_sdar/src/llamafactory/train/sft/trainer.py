# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch.nn.functional as F
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ..opd import topk_distillation_loss


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

    def _distributed_mean_item(self, value: "torch.Tensor") -> float:
        """Return a scalar averaged across distributed ranks for consistent logging."""

        value = value.detach()
        if value.dim() != 0:
            value = value.mean()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value = value.clone()
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.AVG)
        return float(value.item())

    def _build_shifted_labels(self, labels: "torch.Tensor", inputs: dict[str, Any]) -> "torch.Tensor":
        seq_len = inputs["input_ids"].size(-1)
        shifted_labels = labels[:, 1 : min(seq_len + 1, labels.shape[1])].contiguous()
        if shifted_labels.shape[1] < seq_len:
            shifted_labels = F.pad(shifted_labels, (0, seq_len - shifted_labels.shape[1]), value=IGNORE_INDEX)

        position_ids = inputs.get("position_ids")
        if position_ids is not None:
            for batch_idx in range(position_ids.size(0)):
                boundary_mask = position_ids[batch_idx, :-1] >= position_ids[batch_idx, 1:]
                shifted_labels[batch_idx, :-1] = torch.where(
                    boundary_mask,
                    torch.full_like(shifted_labels[batch_idx, :-1], IGNORE_INDEX),
                    shifted_labels[batch_idx, :-1],
                )

        return shifted_labels

    def _select_shifted_block_hidden(
        self,
        hidden_row: "torch.Tensor",
        prefix_len: int,
        block_len: int,
        segment_start: int,
        token_indices: "torch.Tensor",
    ) -> "torch.Tensor":
        """Select predictor hidden states for Dream-style shifted block targets."""

        predictor_indices = torch.where(
            token_indices == 0,
            torch.full_like(token_indices, prefix_len - 1),
            segment_start + token_indices - 1,
        )
        return hidden_row.index_select(0, predictor_indices)

    def _select_shifted_block_hidden_all(
        self,
        hidden_row: "torch.Tensor",
        prefix_len: int,
        block_len: int,
        segment_start: int,
    ) -> "torch.Tensor":
        token_indices = torch.arange(block_len, device=hidden_row.device)
        return self._select_shifted_block_hidden(hidden_row, prefix_len, block_len, segment_start, token_indices)

    def _select_shifted_kv_block_hidden(
        self,
        block_hidden: "torch.Tensor",
        prefix_last_hidden: "torch.Tensor",
        token_indices: "torch.Tensor",
    ) -> "torch.Tensor":
        shifted = torch.empty(token_indices.numel(), block_hidden.size(-1), dtype=block_hidden.dtype, device=block_hidden.device)
        first_mask = token_indices == 0
        if first_mask.any():
            shifted[first_mask] = prefix_last_hidden.to(dtype=block_hidden.dtype, device=block_hidden.device)
        if (~first_mask).any():
            shifted[~first_mask] = block_hidden.index_select(0, token_indices[~first_mask] - 1)
        return shifted

    def _select_opd_loss_mask(
        self,
        current_step_mask: "torch.Tensor",
        step_map: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        scope = getattr(self.finetuning_args, "opd_loss_scope", "accepted_only")
        if scope == "accepted_only":
            return current_step_mask
        if scope == "all_block":
            return torch.ones_like(current_step_mask, dtype=torch.bool)
        if scope == "unresolved":
            if step_map is None:
                raise ValueError("opd_loss_scope='unresolved' requires step_map.")
            current_steps = step_map[current_step_mask]
            current_steps = current_steps[current_steps >= 0]
            if current_steps.numel() == 0:
                return torch.zeros_like(current_step_mask, dtype=torch.bool)
            current_step = int(current_steps.min().item())
            return step_map >= current_step
        raise ValueError(f"Unsupported opd_loss_scope: {scope}")

    def _compute_opd_trajectory_loss(self, model, inputs: dict[str, Any]) -> "torch.Tensor":
        unwrapped_model = getattr(model, "module", model)
        if getattr(self.finetuning_args, "opd_target_source", "teacher_topk") != "clean_student":
            raise ValueError("Trajectory-level OPD currently supports only opd_target_source=clean_student.")

        input_ids = inputs["input_ids"]
        position_ids = inputs.get("position_ids")
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        prompt_lens = inputs["trajectory_prompt_len"].to(device=device)
        response_lens = inputs["trajectory_response_len"].to(device=device)
        step_maps = inputs["trajectory_step_map"].to(device=device, dtype=torch.long)
        block_sizes = inputs["trajectory_block_size"].to(device=device)
        if not torch.all(block_sizes == block_sizes[0]):
            raise ValueError("Trajectory-level OPD batches with variable block sizes are not supported yet.")

        block_size = int(block_sizes[0].item())
        clean_top_k = int(getattr(self.finetuning_args, "opd_clean_top_k", 16))
        mask_token_id = int(getattr(getattr(unwrapped_model, "config", None), "mask_token_id", 151669))

        total_lens = prompt_lens + response_lens
        max_total_len = int(total_lens.max().item())
        input_ids = input_ids[:, :max_total_len]
        position_ids = position_ids[:, :max_total_len]

        clean_mask = torch.zeros(bsz, 1, max_total_len, max_total_len, dtype=torch.bool, device=device)
        for batch_idx in range(bsz):
            total_len_i = int(total_lens[batch_idx].item())
            if total_len_i > 0:
                clean_mask[batch_idx, :, :total_len_i, :total_len_i] = torch.tril(
                    torch.ones(total_len_i, total_len_i, dtype=torch.bool, device=device)
                )
            if total_len_i < max_total_len:
                pad_idx = torch.arange(total_len_i, max_total_len, device=device)
                clean_mask[batch_idx, :, pad_idx, pad_idx] = True

        model_was_training = unwrapped_model.training
        if hasattr(unwrapped_model, "gradient_checkpointing_disable"):
            unwrapped_model.gradient_checkpointing_disable()
        if hasattr(unwrapped_model.model, "gradient_checkpointing"):
            unwrapped_model.model.gradient_checkpointing = False
        unwrapped_model.eval()
        clean_outputs = unwrapped_model.model(
            input_ids=input_ids,
            attention_mask=clean_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
        )

        per_state_losses = []
        hidden_size = unwrapped_model.model.embed_tokens.weight.shape[-1]
        for batch_idx in range(bsz):
            prompt_len = int(prompt_lens[batch_idx].item())
            response_len = int(response_lens[batch_idx].item())
            if response_len <= 0:
                continue

            response_token_indices = torch.arange(response_len, device=device)
            clean_predictor_hidden = self._select_shifted_block_hidden(
                clean_outputs.last_hidden_state[batch_idx],
                prefix_len=prompt_len,
                block_len=response_len,
                segment_start=prompt_len,
                token_indices=response_token_indices,
            )
            with torch.no_grad():
                clean_logits = unwrapped_model.lm_head(clean_predictor_hidden).detach()
                clean_logprobs = torch.log_softmax(clean_logits.float(), dim=-1)
                top_k = min(clean_top_k, clean_logits.size(-1))
                clean_topk_logprobs, clean_topk_ids = torch.topk(clean_logprobs, k=top_k, dim=-1)

            num_blocks = (response_len + block_size - 1) // block_size
            for block_idx in range(num_blocks):
                block_start = block_idx * block_size
                block_end = min(block_start + block_size, response_len)
                cur_block_len = block_end - block_start
                if cur_block_len <= 0:
                    continue

                prefix_ids = input_ids[batch_idx, : prompt_len + block_start]
                clean_block = input_ids[batch_idx, prompt_len + block_start : prompt_len + block_end]
                block_step_map = step_maps[batch_idx, block_start:block_end]
                valid_steps = sorted(int(x) for x in torch.unique(block_step_map[block_step_map >= 0]).detach().cpu().tolist())
                previous_signal_block = None

                for step in valid_steps:
                    unresolved = block_step_map >= step
                    if not unresolved.any():
                        continue

                    xt_block = clean_block.clone()
                    xt_block[unresolved] = mask_token_id
                    state_ids = torch.cat([prefix_ids, xt_block], dim=0).unsqueeze(0)
                    state_len = state_ids.size(1)
                    state_prefix_len = prefix_ids.numel()
                    state_pos = torch.arange(state_len, device=device).unsqueeze(0)

                    state_mask = torch.zeros(1, 1, state_len, state_len, dtype=torch.bool, device=device)
                    if state_prefix_len > 0:
                        state_mask[:, :, :state_prefix_len, :state_prefix_len] = torch.tril(
                            torch.ones(state_prefix_len, state_prefix_len, dtype=torch.bool, device=device)
                        )
                    state_mask[:, :, state_prefix_len:state_len, :state_len] = True

                    if previous_signal_block is None:
                        state_outputs = unwrapped_model.model(
                            input_ids=state_ids,
                            attention_mask=state_mask,
                            position_ids=state_pos,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    else:
                        signal = torch.zeros(
                            1,
                            state_len,
                            hidden_size,
                            dtype=unwrapped_model.model.embed_tokens.weight.dtype,
                            device=device,
                        )
                        signal[:, state_prefix_len : state_prefix_len + cur_block_len] = previous_signal_block.unsqueeze(0)
                        embeds = unwrapped_model.model.embed_tokens(state_ids)
                        embeds = unwrapped_model.self_conditioning(embeds, signal.to(dtype=embeds.dtype))
                        state_outputs = unwrapped_model.model(
                            inputs_embeds=embeds,
                            attention_mask=state_mask,
                            position_ids=state_pos,
                            output_hidden_states=False,
                            return_dict=True,
                        )

                    token_indices = torch.arange(cur_block_len, device=device)
                    predictor_hidden = self._select_shifted_block_hidden(
                        state_outputs.last_hidden_state[0],
                        prefix_len=state_prefix_len,
                        block_len=cur_block_len,
                        segment_start=state_prefix_len,
                        token_indices=token_indices,
                    )
                    previous_signal_block = unwrapped_model._build_self_conditioning_signal(
                        predictor_hidden.unsqueeze(0),
                        torch.ones(1, cur_block_len, dtype=torch.bool, device=device),
                    )[0]

                    loss_indices = torch.nonzero(unresolved, as_tuple=False).flatten()
                    student_logits = unwrapped_model.lm_head(predictor_hidden.index_select(0, loss_indices))
                    target_indices = block_start + loss_indices
                    token_losses = topk_distillation_loss(
                        student_logits,
                        clean_topk_ids.index_select(0, target_indices),
                        clean_topk_logprobs.index_select(0, target_indices),
                        mode=self.finetuning_args.opd_distill_mode,
                    )
                    per_state_losses.append(token_losses.mean())

        opd_loss = clean_outputs.last_hidden_state.sum() * 0.0 if not per_state_losses else torch.stack(per_state_losses).mean()
        if model_was_training:
            unwrapped_model.train()

        combined_loss = self.finetuning_args.opd_weight * opd_loss
        opd_loss_local = opd_loss.detach().item()
        combined_loss_local = combined_loss.detach().item()
        self.log(
            {
                "train/opd_loss": self._distributed_mean_item(opd_loss),
                "train/combined_loss": self._distributed_mean_item(combined_loss),
                "train/opd_loss_local": opd_loss_local,
                "train/combined_loss_local": combined_loss_local,
                "train/opd_weight": self.finetuning_args.opd_weight,
                "train/opd_self_conditioning": 1.0,
                "train/opd_target_source": 1.0,
                "train/opd_trajectory_level": 1.0,
            }
        )
        return combined_loss

    def _build_concat_layout_masks(self, inputs: dict[str, Any]) -> tuple["torch.Tensor", "torch.Tensor"]:
        seq_len = inputs["input_ids"].size(-1)
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            bsz = inputs["input_ids"].size(0)
            noisy_mask = torch.zeros(bsz, 2 * seq_len, dtype=torch.bool, device=inputs["input_ids"].device)
            noisy_mask[:, :seq_len] = True
            clean_mask = ~noisy_mask
            return noisy_mask, clean_mask

        noisy_masks, clean_masks = [], []
        for pids in position_ids:
            zero_indices = torch.nonzero(pids == 0).flatten()
            split_points = torch.cat(
                [
                    zero_indices,
                    torch.tensor([seq_len], device=pids.device, dtype=zero_indices.dtype),
                ]
            )
            lengths = torch.diff(split_points)
            router_noisy = (torch.arange(lengths.numel() * 2, device=pids.device) % 2 == 0)
            router_noisy = router_noisy.repeat_interleave(lengths.repeat_interleave(2))
            noisy_masks.append(router_noisy)
            clean_masks.append(~router_noisy)

        return torch.stack(noisy_masks, dim=0), torch.stack(clean_masks, dim=0)

    def _select_clean_hidden_states(self, hidden_states: "torch.Tensor", inputs: dict[str, Any]) -> "torch.Tensor":
        """Select x0/clean hidden states from SDAR's training concat layout.

        Non-packed batches are laid out as [noisy | clean]. Packed batches are
        laid out per logical sample: [noisy_1 | clean_1 | noisy_2 | clean_2 | ...].
        """
        _, clean_mask = self._build_concat_layout_masks(inputs)
        return hidden_states[clean_mask]

    def _maybe_log_position_losses(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Any],
        hidden_states: "torch.Tensor",
        shifted_labels: "torch.Tensor",
        clean_logits: "torch.Tensor",
    ) -> None:
        log_every = self.finetuning_args.log_position_loss_steps
        if log_every <= 0 or self.state.global_step % log_every != 0:
            return

        model_config = getattr(getattr(model, "module", model), "config", None)
        block_size = max(1, int(getattr(model_config, "block_size", 1)))
        max_offsets = min(self.finetuning_args.position_loss_max_offsets, block_size)
        if max_offsets <= 0:
            return

        unwrapped_model = getattr(model, "module", model)
        with torch.no_grad():
            clean_losses = F.cross_entropy(
                clean_logits.detach().float().view(-1, clean_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(shifted_labels)

            noisy_mask, _ = self._build_concat_layout_masks(inputs)
            task_hidden = hidden_states[noisy_mask]
            task_logits = unwrapped_model.lm_head(task_hidden).float()
            task_labels = shifted_labels.masked_fill(inputs["labels"] == IGNORE_INDEX, IGNORE_INDEX)
            task_losses = F.cross_entropy(
                task_logits,
                task_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(shifted_labels)

            log_dict = {}
            positions = torch.arange(shifted_labels.size(-1), device=shifted_labels.device).unsqueeze(0)
            block_offsets = positions % block_size
            valid_masks = {
                "task": task_labels != IGNORE_INDEX,
                "clean": shifted_labels != IGNORE_INDEX,
            }
            for name, losses in (("task", task_losses), ("clean", clean_losses)):
                for offset in range(max_offsets):
                    offset_mask = valid_masks[name] & (block_offsets == offset)
                    if offset_mask.any():
                        log_dict[f"train/position_loss/{name}/block_offset_{offset:04d}"] = (
                            losses[offset_mask].mean().item()
                        )

        if log_dict:
            self.log(log_dict)

    def _compute_noisy_entropy_loss(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Any],
        hidden_states: "torch.Tensor",
        shifted_labels: "torch.Tensor",
    ) -> tuple[Optional["torch.Tensor"], dict[str, float]]:
        alpha = getattr(self.finetuning_args, "entropy_loss_alpha", 0.0)
        if alpha <= 0:
            return None, {}

        noisy_mask, _ = self._build_concat_layout_masks(inputs)
        noisy_hidden_states = hidden_states[noisy_mask]
        labels_flat = inputs["labels"].reshape(-1)
        targets_flat = shifted_labels.reshape(-1)
        trainable_mask = (labels_flat != IGNORE_INDEX) & (targets_flat != IGNORE_INDEX)
        if not trainable_mask.any():
            return hidden_states.float().sum() * 0.0, {"train/entropy_loss_gate_mean": 0.0}

        selected_hidden = noisy_hidden_states[trainable_mask]
        selected_targets = targets_flat[trainable_mask]
        unwrapped_model = getattr(model, "module", model)
        chunk_size = max(1, int(getattr(self.finetuning_args, "entropy_loss_chunk_size", 64)))
        loss_type = getattr(self.finetuning_args, "entropy_loss_type", "plain")
        p_easy = float(getattr(self.finetuning_args, "entropy_gate_p_easy", 0.6))
        tau = float(getattr(self.finetuning_args, "entropy_gate_tau", 0.05))

        entropy_sum = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        gate_sum = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        for start in range(0, selected_hidden.size(0), chunk_size):
            end = min(start + chunk_size, selected_hidden.size(0))
            logits = unwrapped_model.lm_head(selected_hidden[start:end]).float()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)

            if loss_type == "gated":
                targets = selected_targets[start:end].unsqueeze(1)
                with torch.no_grad():
                    p_y = probs.gather(1, targets).squeeze(1)
                    pred = logits.argmax(dim=-1)
                    is_correct_top1 = pred.eq(targets.squeeze(1)).to(dtype=torch.float32)
                    gate = torch.sigmoid((p_y - p_easy) / tau) * is_correct_top1

                entropy_sum = entropy_sum + (entropy * gate).sum()
                gate_sum = gate_sum + gate.sum()
            else:
                entropy_sum = entropy_sum + entropy.sum()

        denom = max(int(selected_targets.numel()), 1)
        log_dict = {}
        if loss_type == "gated":
            log_dict["train/entropy_loss_gate_mean"] = (gate_sum / denom).item()

        return entropy_sum / denom, log_dict

    def _compute_opd_loss(self, model, inputs: dict[str, Any]) -> "torch.Tensor":
        if "trajectory_step_map" in inputs:
            return self._compute_opd_trajectory_loss(model, inputs)

        required = {
            "input_ids",
            "position_ids",
            "current_step_mask",
            "prefix_len",
            "block_len",
        }
        opd_target_source = getattr(self.finetuning_args, "opd_target_source", "teacher_topk")
        if opd_target_source == "teacher_topk":
            required.update({"teacher_topk_ids", "teacher_topk_logprobs"})
        missing = required - set(inputs)
        if missing:
            raise ValueError(f"Missing OPD inputs: {sorted(missing)}")

        unwrapped_model = getattr(model, "module", model)
        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        prefix_lens = inputs["prefix_len"].to(device=device)
        block_lens = inputs["block_len"].to(device=device)
        if not torch.all(block_lens == block_lens[0]):
            raise ValueError("OPD batches with variable block lengths are not supported yet.")

        block_len = int(block_lens[0].item())
        real_seq_lens = prefix_lens + 2 * block_len
        real_seq_len = int(real_seq_lens.max().item())
        if seq_len < real_seq_len:
            raise ValueError(f"Expected at least seq_len={real_seq_len}, got {seq_len}.")
        if seq_len > real_seq_len:
            input_ids = input_ids[:, :real_seq_len]
            position_ids = position_ids[:, :real_seq_len]
            seq_len = real_seq_len

        pad_to_multiple = int(getattr(self.finetuning_args, "opd_pad_to_multiple", 1))
        if pad_to_multiple > 1 and seq_len % pad_to_multiple:
            padded_seq_len = ((seq_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
            pad_len = padded_seq_len - seq_len
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            position_ids = F.pad(position_ids, (0, pad_len), value=0)
            seq_len = padded_seq_len

        try:
            from torch.nn.attention.flex_attention import create_block_mask
        except Exception as err:
            raise ImportError("OPD training requires torch flex_attention create_block_mask.") from err

        def opd_mask_mod(prefix_lens_tensor):
            def mask_mod(b, h, q_idx, kv_idx):
                prefix_len_b = prefix_lens_tensor[b]
                clean_start = prefix_len_b
                noisy_start_b = prefix_len_b + block_len
                real_len_b = prefix_len_b + 2 * block_len

                q_prompt = q_idx < clean_start
                q_clean = (q_idx >= clean_start) & (q_idx < noisy_start_b)
                q_noisy = (q_idx >= noisy_start_b) & (q_idx < real_len_b)
                q_pad = q_idx >= real_len_b

                kv_prompt = kv_idx < clean_start
                kv_clean = (kv_idx >= clean_start) & (kv_idx < noisy_start_b)
                kv_noisy = (kv_idx >= noisy_start_b) & (kv_idx < real_len_b)

                prompt_vis = q_prompt & kv_prompt & (kv_idx <= q_idx)
                clean_vis = q_clean & (kv_prompt | (kv_clean & (kv_idx <= q_idx)))
                noisy_vis = q_noisy & (kv_prompt | kv_noisy)
                pad_vis = q_pad & (q_idx == kv_idx)
                return prompt_vis | clean_vis | noisy_vis | pad_vis

            return mask_mod

        block_mask = create_block_mask(
            opd_mask_mod(prefix_lens),
            B=bsz,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

        supports_self_conditioning = all(
            hasattr(unwrapped_model, attr)
            for attr in ("_build_self_conditioning_signal", "self_conditioning")
        ) and hasattr(unwrapped_model.model, "embed_tokens")

        opd_training_backend = getattr(self.finetuning_args, "opd_training_backend", "full_replay")
        if "step_map" in inputs and opd_training_backend == "prefix_kv" and opd_target_source in {"teacher_topk", "clean_student"}:
            if not torch.all(prefix_lens == prefix_lens[0]):
                raise ValueError("OPD prefix_kv backend requires uniform prefix lengths; use full_replay for padded batches.")
            prefix_len = int(prefix_lens[0].item())
            if hasattr(unwrapped_model, "gradient_checkpointing_disable"):
                unwrapped_model.gradient_checkpointing_disable()
            if hasattr(unwrapped_model.model, "gradient_checkpointing"):
                unwrapped_model.model.gradient_checkpointing = False
            prefix_kv_was_training = unwrapped_model.training
            # Dense bool masks in eval mode use the same stable SDPA path as
            # rollout, avoiding flex-attention NoValidChoices on arbitrary
            # prefix lengths while still allowing gradients through the current
            # noisy-block forward.
            unwrapped_model.eval()

            teacher_topk_ids = inputs.get("teacher_topk_ids")
            teacher_topk_logprobs = inputs.get("teacher_topk_logprobs")
            if teacher_topk_ids is not None:
                teacher_topk_ids = teacher_topk_ids.to(device=device, dtype=torch.long)
            if teacher_topk_logprobs is not None:
                teacher_topk_logprobs = teacher_topk_logprobs.to(device=device)
            clean_top_k = int(getattr(self.finetuning_args, "opd_clean_top_k", 16))
            current_step_mask = inputs["current_step_mask"].to(device=device, dtype=torch.bool)
            step_maps = inputs["step_map"].to(device=device, dtype=torch.long)
            model_config = getattr(unwrapped_model, "config", None)
            mask_token_id = int(getattr(model_config, "mask_token_id", 151669))

            prefix_mask = torch.zeros(1, 1, prefix_len, prefix_len, dtype=torch.bool, device=device)
            prefix_mask[:, :, :, :] = torch.tril(torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device))
            block_kv_mask = torch.ones(1, 1, block_len, prefix_len + block_len, dtype=torch.bool, device=device)
            clean_block_kv_mask = torch.zeros(1, 1, block_len, prefix_len + block_len, dtype=torch.bool, device=device)
            clean_block_kv_mask[:, :, :, :prefix_len] = True
            clean_block_kv_mask[:, :, :, prefix_len:] = torch.tril(
                torch.ones(block_len, block_len, dtype=torch.bool, device=device)
            )
            block_position_ids = torch.arange(prefix_len, prefix_len + block_len, device=device).unsqueeze(0)

            per_state_losses = []
            loss_token_counts = []
            zero_anchor = unwrapped_model.model.embed_tokens.weight.sum() * 0.0
            for batch_idx in range(bsz):
                token_mask = self._select_opd_loss_mask(
                    current_step_mask[batch_idx, :block_len],
                    step_maps[batch_idx, :block_len],
                )
                if not token_mask.any():
                    continue

                token_indices = torch.nonzero(token_mask, as_tuple=False).flatten()
                loss_token_counts.append(token_indices.numel())
                current_step_values = step_maps[batch_idx, :block_len][current_step_mask[batch_idx, :block_len]]
                current_step_values = current_step_values[current_step_values >= 0]
                if current_step_values.numel() == 0:
                    continue
                current_step = int(current_step_values[0].item())

                prefix_ids = input_ids[batch_idx, :prefix_len].unsqueeze(0)
                prefix_position_ids = position_ids[batch_idx, :prefix_len].unsqueeze(0)
                clean_block = input_ids[batch_idx, prefix_len : prefix_len + block_len]
                # DeepSpeed/ZeRO and trainable DynamicCache graphs can hang in
                # backward. Keep the prefix KV as rollout-aligned context and
                # train the noisy-block path that receives OPD supervision.
                with torch.no_grad():
                    prefix_outputs = unwrapped_model.model(
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        position_ids=prefix_position_ids,
                        use_cache=True,
                        store_kv=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                prefix_cache = prefix_outputs.past_key_values
                if prefix_cache is None:
                    raise RuntimeError("OPD prefix-KV training expected past_key_values from prefix prefill.")
                prefix_last_hidden = prefix_outputs.last_hidden_state[0, prefix_len - 1]

                valid_steps = sorted(
                    int(x)
                    for x in torch.unique(step_maps[batch_idx, :block_len][step_maps[batch_idx, :block_len] >= 0])
                    .detach()
                    .cpu()
                    .tolist()
                )
                previous_signal_block = None
                if supports_self_conditioning:
                    for replay_step in [step for step in valid_steps if step < current_step]:
                        with torch.no_grad():
                            replay_xt = clean_block.clone()
                            replay_xt[step_maps[batch_idx, :block_len] >= replay_step] = mask_token_id
                            if previous_signal_block is None:
                                replay_outputs = unwrapped_model.model(
                                    input_ids=replay_xt.unsqueeze(0),
                                    attention_mask=block_kv_mask,
                                    position_ids=block_position_ids,
                                    past_key_values=prefix_cache,
                                    use_cache=True,
                                    store_kv=False,
                                    output_hidden_states=False,
                                    return_dict=True,
                                )
                            else:
                                replay_embeds = unwrapped_model.model.embed_tokens(replay_xt.unsqueeze(0))
                                replay_embeds = unwrapped_model.self_conditioning(
                                    replay_embeds,
                                    previous_signal_block.unsqueeze(0).to(dtype=replay_embeds.dtype),
                                )
                                replay_outputs = unwrapped_model.model(
                                    inputs_embeds=replay_embeds,
                                    attention_mask=block_kv_mask,
                                    position_ids=block_position_ids,
                                    past_key_values=prefix_cache,
                                    use_cache=True,
                                    store_kv=False,
                                    output_hidden_states=False,
                                    return_dict=True,
                                )

                            replay_all_indices = torch.arange(block_len, device=device)
                            replay_predictor_hidden = self._select_shifted_kv_block_hidden(
                                replay_outputs.last_hidden_state[0],
                                prefix_last_hidden,
                                replay_all_indices,
                            ).unsqueeze(0)
                            previous_signal_block = unwrapped_model._build_self_conditioning_signal(
                                replay_predictor_hidden,
                                torch.ones(1, block_len, dtype=torch.bool, device=device),
                            )[0]

                current_xt = clean_block.clone()
                current_xt[step_maps[batch_idx, :block_len] >= current_step] = mask_token_id
                if previous_signal_block is None:
                    current_outputs = unwrapped_model.model(
                        input_ids=current_xt.unsqueeze(0),
                        attention_mask=block_kv_mask,
                        position_ids=block_position_ids,
                        past_key_values=prefix_cache,
                        use_cache=True,
                        store_kv=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                else:
                    current_embeds = unwrapped_model.model.embed_tokens(current_xt.unsqueeze(0))
                    current_embeds = unwrapped_model.self_conditioning(
                        current_embeds,
                        previous_signal_block.unsqueeze(0).to(dtype=current_embeds.dtype),
                    )
                    current_outputs = unwrapped_model.model(
                        inputs_embeds=current_embeds,
                        attention_mask=block_kv_mask,
                        position_ids=block_position_ids,
                        past_key_values=prefix_cache,
                        use_cache=True,
                        store_kv=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )

                selected_hidden = self._select_shifted_kv_block_hidden(
                    current_outputs.last_hidden_state[0],
                    prefix_last_hidden,
                    token_indices,
                )
                student_logits = unwrapped_model.lm_head(selected_hidden)
                if opd_target_source == "clean_student":
                    with torch.no_grad():
                        clean_outputs = unwrapped_model.model(
                            input_ids=clean_block.unsqueeze(0),
                            attention_mask=clean_block_kv_mask,
                            position_ids=block_position_ids,
                            past_key_values=prefix_cache,
                            use_cache=True,
                            store_kv=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                        clean_hidden = self._select_shifted_kv_block_hidden(
                            clean_outputs.last_hidden_state[0],
                            prefix_last_hidden,
                            token_indices,
                        )
                        clean_logits = unwrapped_model.lm_head(clean_hidden).detach()
                        top_k = min(clean_top_k, clean_logits.size(-1))
                        clean_logprobs = torch.log_softmax(clean_logits.float(), dim=-1)
                        target_topk_logprobs, target_topk_ids = torch.topk(clean_logprobs, k=top_k, dim=-1)
                    token_losses = topk_distillation_loss(
                        student_logits,
                        target_topk_ids,
                        target_topk_logprobs,
                        mode=self.finetuning_args.opd_distill_mode,
                    )
                else:
                    token_losses = topk_distillation_loss(
                        student_logits,
                        teacher_topk_ids[batch_idx].index_select(0, token_indices),
                        teacher_topk_logprobs[batch_idx].index_select(0, token_indices),
                        mode=self.finetuning_args.opd_distill_mode,
                    )
                per_state_losses.append(token_losses.mean())

            if not per_state_losses:
                opd_loss = zero_anchor
            else:
                opd_loss = torch.stack(per_state_losses).mean()

            combined_loss = self.finetuning_args.opd_weight * opd_loss
            opd_loss_local = opd_loss.detach().item()
            combined_loss_local = combined_loss.detach().item()
            self.log(
                {
                    "train/opd_loss": self._distributed_mean_item(opd_loss),
                    "train/combined_loss": self._distributed_mean_item(combined_loss),
                    "train/opd_loss_local": opd_loss_local,
                    "train/combined_loss_local": combined_loss_local,
                    "train/opd_weight": self.finetuning_args.opd_weight,
                    "train/opd_self_conditioning": float(supports_self_conditioning),
                    "train/opd_target_source": 1.0 if opd_target_source == "clean_student" else 0.0,
                    "train/opd_prefix_kv_training": 1.0,
                    "train/opd_prefix_kv_cache_grad": 0.0,
                    "train/opd_loss_scope_unresolved": float(getattr(self.finetuning_args, "opd_loss_scope", "accepted_only") == "unresolved"),
                    "train/opd_loss_scope_all_block": float(getattr(self.finetuning_args, "opd_loss_scope", "accepted_only") == "all_block"),
                    "train/opd_reformulated_full_len": real_seq_lens.to(torch.float).mean().detach().item(),
                    "train/opd_reformulated_kv_len": (prefix_lens + block_lens).to(torch.float).mean().detach().item(),
                    "train/opd_prefix_len": prefix_lens.to(torch.float).mean().detach().item(),
                    "train/opd_block_len": float(block_len),
                    "train/opd_loss_tokens_per_state": float(sum(loss_token_counts) / max(len(loss_token_counts), 1)),
                }
            )
            if prefix_kv_was_training:
                unwrapped_model.train()
            return combined_loss

        if supports_self_conditioning:
            hidden_size = unwrapped_model.model.embed_tokens.weight.shape[-1]
            self_conditioning_signal = torch.zeros(
                bsz,
                seq_len,
                hidden_size,
                dtype=unwrapped_model.model.embed_tokens.weight.dtype,
                device=device,
            )

            if "step_map" in inputs:
                step_maps = inputs["step_map"].to(device=device, dtype=torch.long)
                model_config = getattr(unwrapped_model, "config", None)
                mask_token_id = int(getattr(model_config, "mask_token_id", 151669))
                current_step_mask = inputs["current_step_mask"].to(device=device, dtype=torch.bool)

                # Strict previous-step self-conditioning: reconstruct and replay
                # the same block states before the selected state. This avoids
                # storing full rollout residuals while matching inference's
                # "step k consumes residual from step k-1" contract.
                with torch.no_grad():
                    for batch_idx in range(bsz):
                        token_mask = current_step_mask[batch_idx, :block_len]
                        if not token_mask.any():
                            continue

                        prefix_len_i = int(prefix_lens[batch_idx].item())
                        noisy_start_i = prefix_len_i + block_len
                        real_len_i = int(real_seq_lens[batch_idx].item())
                        current_step_values = step_maps[batch_idx, :block_len][token_mask]
                        current_step_values = current_step_values[current_step_values >= 0]
                        if current_step_values.numel() == 0:
                            continue
                        current_step = int(current_step_values[0].item())

                        valid_steps = sorted(
                            int(x)
                            for x in torch.unique(step_maps[batch_idx, :block_len][step_maps[batch_idx, :block_len] >= 0])
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        previous_signal_block = None
                        replay_block_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)
                        real_len_i = int(real_seq_lens[batch_idx].item())
                        replay_block_mask[:, :, :real_len_i, :real_len_i] = torch.zeros(
                            real_len_i, real_len_i, dtype=torch.bool, device=device
                        )
                        # Same OPD full-layout visibility as `opd_mask_mod`, but
                        # materialized only for this no-grad replay to avoid flex
                        # autotune failures on arbitrary single-sample lengths.
                        replay_block_mask[:, :, :prefix_len_i, :prefix_len_i] = torch.tril(
                            torch.ones(prefix_len_i, prefix_len_i, dtype=torch.bool, device=device)
                        )
                        clean_start_i = prefix_len_i
                        noisy_start_i = prefix_len_i + block_len
                        replay_block_mask[:, :, clean_start_i:noisy_start_i, :prefix_len_i] = True
                        clean_causal = torch.tril(torch.ones(block_len, block_len, dtype=torch.bool, device=device))
                        replay_block_mask[:, :, clean_start_i:noisy_start_i, clean_start_i:noisy_start_i] = clean_causal
                        replay_block_mask[:, :, noisy_start_i:real_len_i, :prefix_len_i] = True
                        replay_block_mask[:, :, noisy_start_i:real_len_i, noisy_start_i:real_len_i] = True
                        if real_len_i < seq_len:
                            pad_idx = torch.arange(real_len_i, seq_len, device=device)
                            replay_block_mask[:, :, pad_idx, pad_idx] = True
                        replay_was_training = unwrapped_model.training
                        unwrapped_model.eval()
                        for replay_step in [step for step in valid_steps if step < current_step]:
                            replay_ids = input_ids[batch_idx, :seq_len].clone()
                            clean_block = replay_ids[prefix_len_i : prefix_len_i + block_len].clone()
                            replay_xt = clean_block.clone()
                            unresolved = step_maps[batch_idx, :block_len] >= replay_step
                            replay_xt[unresolved] = mask_token_id
                            replay_ids[noisy_start_i : noisy_start_i + block_len] = replay_xt
                            replay_position_ids = position_ids[batch_idx, :seq_len].unsqueeze(0)

                            if previous_signal_block is None:
                                replay_outputs = unwrapped_model.model(
                                    input_ids=replay_ids.unsqueeze(0),
                                    attention_mask=replay_block_mask,
                                    position_ids=replay_position_ids,
                                    output_hidden_states=False,
                                    return_dict=True,
                                )
                            else:
                                replay_signal = torch.zeros(
                                    1,
                                    seq_len,
                                    hidden_size,
                                    dtype=self_conditioning_signal.dtype,
                                    device=device,
                                )
                                replay_signal[:, noisy_start_i : noisy_start_i + block_len] = previous_signal_block.unsqueeze(0)
                                replay_embeds = unwrapped_model.model.embed_tokens(replay_ids.unsqueeze(0))
                                replay_embeds = unwrapped_model.self_conditioning(
                                    replay_embeds,
                                    replay_signal.to(dtype=replay_embeds.dtype),
                                )
                                replay_outputs = unwrapped_model.model(
                                    inputs_embeds=replay_embeds,
                                    attention_mask=replay_block_mask,
                                    position_ids=replay_position_ids,
                                    output_hidden_states=False,
                                    return_dict=True,
                                )

                            replay_hidden_block = self._select_shifted_block_hidden_all(
                                replay_outputs.last_hidden_state[0],
                                prefix_len_i,
                                block_len,
                                noisy_start_i,
                            ).unsqueeze(0)
                            previous_signal_block = unwrapped_model._build_self_conditioning_signal(
                                replay_hidden_block,
                                torch.ones(1, block_len, dtype=torch.bool, device=device),
                            )[0]
                        if replay_was_training:
                            unwrapped_model.train()

                        if previous_signal_block is not None:
                            self_conditioning_signal[
                                batch_idx,
                                noisy_start_i : noisy_start_i + block_len,
                            ] = previous_signal_block
            else:
                # Backward-compatible fallback for old expanded OPD JSONL that
                # lacks step_map: use the current-state two-loop signal.
                with torch.no_grad():
                    first_outputs = unwrapped_model.model(
                        input_ids=input_ids,
                        attention_mask=block_mask,
                        position_ids=position_ids,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    self_conditioning_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                    for batch_idx in range(bsz):
                        prefix_len_i = int(prefix_lens[batch_idx].item())
                        noisy_start_i = prefix_len_i + block_len
                        self_conditioning_mask[batch_idx, noisy_start_i : noisy_start_i + block_len] = True
                    self_conditioning_signal = unwrapped_model._build_self_conditioning_signal(
                        first_outputs.last_hidden_state,
                        self_conditioning_mask,
                    )

            inputs_embeds = unwrapped_model.model.embed_tokens(input_ids)
            inputs_embeds = unwrapped_model.self_conditioning(
                inputs_embeds,
                self_conditioning_signal.to(dtype=inputs_embeds.dtype),
            )
            outputs = unwrapped_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=block_mask,
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
            )
        else:
            outputs = unwrapped_model.model(
                input_ids=input_ids,
                attention_mask=block_mask,
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
            )
        hidden_states = outputs.last_hidden_state

        current_step_mask = inputs["current_step_mask"].to(device=device, dtype=torch.bool)
        teacher_topk_ids = inputs.get("teacher_topk_ids")
        teacher_topk_logprobs = inputs.get("teacher_topk_logprobs")
        if teacher_topk_ids is not None:
            teacher_topk_ids = teacher_topk_ids.to(device=device, dtype=torch.long)
        if teacher_topk_logprobs is not None:
            teacher_topk_logprobs = teacher_topk_logprobs.to(device=device)
        clean_top_k = int(getattr(self.finetuning_args, "opd_clean_top_k", 16))

        per_state_losses = []
        for batch_idx in range(bsz):
            step_map_row = inputs["step_map"].to(device=device, dtype=torch.long)[batch_idx, :block_len] if "step_map" in inputs else None
            token_mask = self._select_opd_loss_mask(current_step_mask[batch_idx, :block_len], step_map_row)
            if not token_mask.any():
                continue
            token_indices = torch.nonzero(token_mask, as_tuple=False).flatten()
            prefix_len_i = int(prefix_lens[batch_idx].item())
            clean_start_i = prefix_len_i
            noisy_start_i = prefix_len_i + block_len
            selected_hidden = self._select_shifted_block_hidden(
                hidden_states[batch_idx],
                prefix_len_i,
                block_len,
                noisy_start_i,
                token_indices,
            )
            student_logits = unwrapped_model.lm_head(selected_hidden)
            if opd_target_source == "clean_student":
                with torch.no_grad():
                    clean_hidden = self._select_shifted_block_hidden(
                        hidden_states[batch_idx],
                        prefix_len_i,
                        block_len,
                        clean_start_i,
                        token_indices,
                    )
                    clean_logits = unwrapped_model.lm_head(clean_hidden).detach()
                    top_k = min(clean_top_k, clean_logits.size(-1))
                    clean_logprobs = torch.log_softmax(clean_logits.float(), dim=-1)
                    target_topk_logprobs, target_topk_ids = torch.topk(clean_logprobs, k=top_k, dim=-1)
                token_losses = topk_distillation_loss(
                    student_logits,
                    target_topk_ids,
                    target_topk_logprobs,
                    mode=self.finetuning_args.opd_distill_mode,
                )
                per_state_losses.append(token_losses.mean())
                continue

            token_losses = topk_distillation_loss(
                student_logits,
                teacher_topk_ids[batch_idx].index_select(0, token_indices),
                teacher_topk_logprobs[batch_idx].index_select(0, token_indices),
                mode=self.finetuning_args.opd_distill_mode,
            )
            per_state_losses.append(token_losses.mean())

        if not per_state_losses:
            opd_loss = hidden_states.sum() * 0.0
        else:
            opd_loss = torch.stack(per_state_losses).mean()

        combined_loss = self.finetuning_args.opd_weight * opd_loss
        opd_loss_local = opd_loss.detach().item()
        combined_loss_local = combined_loss.detach().item()
        log_dict = {
            "train/opd_loss": self._distributed_mean_item(opd_loss),
            "train/combined_loss": self._distributed_mean_item(combined_loss),
            "train/opd_loss_local": opd_loss_local,
            "train/combined_loss_local": combined_loss_local,
            "train/opd_weight": self.finetuning_args.opd_weight,
            "train/opd_self_conditioning": float(supports_self_conditioning),
            "train/opd_target_source": 1.0 if opd_target_source == "clean_student" else 0.0,
            "train/opd_loss_scope_unresolved": float(getattr(self.finetuning_args, "opd_loss_scope", "accepted_only") == "unresolved"),
            "train/opd_loss_scope_all_block": float(getattr(self.finetuning_args, "opd_loss_scope", "accepted_only") == "all_block"),
            "train/opd_reformulated_full_len": real_seq_lens.to(torch.float).mean().detach().item(),
            "train/opd_reformulated_kv_len": (prefix_lens + block_lens).to(torch.float).mean().detach().item(),
            "train/opd_prefix_len": prefix_lens.to(torch.float).mean().detach().item(),
            "train/opd_block_len": float(block_len),
        }
        self.log(log_dict)
        return combined_loss

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # NOTE: The "ar" loss assumes an SDAR model that internally concatenates [noisy|clean],
        # producing hidden_states of length 2*seq_len. This will not work with standard AR models.
        if self.finetuning_args.idlm_loss_type == "opd":
            return self._compute_opd_loss(model, inputs)

        if self.finetuning_args.idlm_loss_type != "ar":
            return super().compute_loss(model, inputs, *args, **kwargs)

        alpha = self.finetuning_args.ce_alpha
        labels = inputs.get("labels")
        assert labels is not None

        # Forward pass — the SDAR model internally concatenates [noisy_xt | clean_x0],
        # so input_ids has length L but hidden_states has length 2L.
        outputs = model(**inputs, output_hidden_states=True)
        task_loss = outputs["loss"]  # CE on all positions (noisy + clean)

        # Unwrap DeepSpeedEngine/FSDP to access lm_head directly
        unwrapped_model = getattr(model, "module", model)

        # seq_len = L (original input length); clean positions depend on whether
        # the sample is packed or not.
        seq_len = inputs["input_ids"].size(-1)

        # AR CE loss on clean (x0) region with Dream-shift-aligned labels.
        # hidden[i] predicts token[i+1], so shift labels by 1.
        shifted_labels = self._build_shifted_labels(labels, inputs)

        last_hidden_states = outputs["hidden_states"][-1]
        clean_hidden_states = self._select_clean_hidden_states(last_hidden_states, inputs)
        clean_logits = unwrapped_model.lm_head(clean_hidden_states)
        ce_logits = clean_logits.view(-1, clean_logits.size(-1))
        ce_labels = shifted_labels.view(-1)
        clean_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(ce_logits, ce_labels)
        self._maybe_log_position_losses(model, inputs, last_hidden_states, shifted_labels, clean_logits)
        entropy_loss, entropy_log_dict = self._compute_noisy_entropy_loss(
            model,
            inputs,
            last_hidden_states,
            shifted_labels,
        )

        xt_rkl_loss = None
        if self.finetuning_args.xt_rkl_alpha > 0 or self.finetuning_args.three_way_loss_auto_balance:
            noisy_mask, _ = self._build_concat_layout_masks(inputs)
            noisy_hidden_states = last_hidden_states[noisy_mask]
            valid_mask = (shifted_labels != IGNORE_INDEX).view(-1)
            if valid_mask.any():
                xt_valid_hidden = noisy_hidden_states[valid_mask]
                clean_valid_logits = clean_logits.detach()[valid_mask]
                chunk_size = int(getattr(self.finetuning_args, "xt_rkl_chunk_size", 128))
                chunk_losses = []
                for start in range(0, xt_valid_hidden.size(0), chunk_size):
                    end = min(start + chunk_size, xt_valid_hidden.size(0))
                    xt_logits = unwrapped_model.lm_head(xt_valid_hidden[start:end]).float()
                    clean_logits_chunk = clean_valid_logits[start:end].float()

                    xt_logprobs = F.log_softmax(xt_logits, dim=-1)
                    with torch.no_grad():
                        sampled_ids = torch.multinomial(xt_logprobs.exp(), num_samples=1)
                        clean_sample_logprobs = F.log_softmax(clean_logits_chunk, dim=-1).gather(1, sampled_ids)

                    xt_sample_logprobs = xt_logprobs.gather(1, sampled_ids)
                    # K2 estimator for reverse KL: 0.5 * (log q(y) - log p(y))^2, y ~ q.
                    chunk_losses.append(0.5 * (xt_sample_logprobs - clean_sample_logprobs).square().mean())

                xt_rkl_loss = torch.stack(chunk_losses).mean()
            else:
                xt_rkl_loss = task_loss * 0.0

        xt_x0_jsd_loss = None
        if self.finetuning_args.xt_x0_jsd_alpha > 0 or self.finetuning_args.xt_x0_jsd_auto_balance:
            noisy_mask, _ = self._build_concat_layout_masks(inputs)
            noisy_hidden_states = last_hidden_states[noisy_mask]
            valid_mask = (shifted_labels != IGNORE_INDEX).view(-1)
            if valid_mask.any():
                xt_valid_hidden = noisy_hidden_states[valid_mask]
                clean_valid_logits = clean_logits.detach()[valid_mask]
                chunk_size = int(getattr(self.finetuning_args, "xt_x0_jsd_chunk_size", 128))
                chunk_losses = []
                for start in range(0, xt_valid_hidden.size(0), chunk_size):
                    end = min(start + chunk_size, xt_valid_hidden.size(0))
                    xt_logits = unwrapped_model.lm_head(xt_valid_hidden[start:end]).float()
                    clean_logits_chunk = clean_valid_logits[start:end].float()

                    xt_logprobs = F.log_softmax(xt_logits, dim=-1)
                    with torch.no_grad():
                        clean_logprobs = F.log_softmax(clean_logits_chunk, dim=-1)
                        xt_sample_ids = torch.multinomial(xt_logprobs.exp(), num_samples=1)
                        clean_sample_ids = torch.multinomial(clean_logprobs.exp(), num_samples=1)

                        clean_logprobs_on_xt = clean_logprobs.gather(1, xt_sample_ids)
                        clean_logprobs_on_clean = clean_logprobs.gather(1, clean_sample_ids)

                    xt_logprobs_on_xt = xt_logprobs.gather(1, xt_sample_ids)
                    xt_logprobs_on_clean = xt_logprobs.gather(1, clean_sample_ids)
                    log_m_on_xt = torch.logaddexp(xt_logprobs_on_xt, clean_logprobs_on_xt) - torch.log(
                        xt_logprobs_on_xt.new_tensor(2.0)
                    )
                    xt_k2 = 0.5 * (xt_logprobs_on_xt - log_m_on_xt).square()

                    log_m_on_clean = torch.logaddexp(xt_logprobs_on_clean, clean_logprobs_on_clean) - torch.log(
                        xt_logprobs_on_clean.new_tensor(2.0)
                    )
                    clean_k2 = 0.5 * (clean_logprobs_on_clean - log_m_on_clean).square()

                    # JSD = 0.5 KL(xt||m) + 0.5 KL(x0||m). Clean/x0 logits are detached.
                    chunk_losses.append(0.5 * xt_k2.mean() + 0.5 * clean_k2.mean())

                xt_x0_jsd_loss = torch.stack(chunk_losses).mean()
            else:
                xt_x0_jsd_loss = task_loss * 0.0

        if self.finetuning_args.three_way_loss_auto_balance:
            clean_scale = task_loss.detach() / (clean_ce_loss.detach() + 1e-8)
            rkl_scale = task_loss.detach() / (xt_rkl_loss.detach() + 1e-8)
            combined_loss = task_loss + clean_scale * clean_ce_loss + rkl_scale * xt_rkl_loss
        elif self.finetuning_args.loss_auto_balance:
            scale = task_loss.detach() / (clean_ce_loss.detach() + 1e-8)
            combined_loss = task_loss + scale * clean_ce_loss
        else:
            combined_loss = task_loss + alpha * clean_ce_loss
            if xt_rkl_loss is not None and self.finetuning_args.xt_rkl_alpha > 0:
                combined_loss = combined_loss + self.finetuning_args.xt_rkl_alpha * xt_rkl_loss
        if entropy_loss is not None:
            combined_loss = combined_loss + self.finetuning_args.entropy_loss_alpha * entropy_loss
        if xt_x0_jsd_loss is not None:
            if self.finetuning_args.xt_x0_jsd_auto_balance:
                jsd_scale = task_loss.detach() / (xt_x0_jsd_loss.detach() + 1e-8)
                combined_loss = combined_loss + jsd_scale * xt_x0_jsd_loss
            elif self.finetuning_args.xt_x0_jsd_alpha > 0:
                combined_loss = combined_loss + self.finetuning_args.xt_x0_jsd_alpha * xt_x0_jsd_loss

        # Logging
        log_dict = {
            "train/task_loss": task_loss.item(),
            "train/clean_ce_loss": clean_ce_loss.item(),
            "train/combined_loss": combined_loss.item(),
            "train/alpha": alpha,
        }
        if xt_rkl_loss is not None:
            log_dict["train/xt_rkl_loss"] = xt_rkl_loss.item()
        if xt_x0_jsd_loss is not None:
            log_dict["train/xt_x0_jsd_loss"] = xt_x0_jsd_loss.item()
            if self.finetuning_args.xt_x0_jsd_auto_balance:
                log_dict["train/xt_x0_jsd_scale"] = jsd_scale.item()
            else:
                log_dict["train/xt_x0_jsd_alpha"] = self.finetuning_args.xt_x0_jsd_alpha
        if entropy_loss is not None:
            log_dict["train/entropy_loss"] = entropy_loss.item()
            log_dict["train/entropy_loss_alpha"] = self.finetuning_args.entropy_loss_alpha
            log_dict.update(entropy_log_dict)
        sc_module = getattr(unwrapped_model, "self_conditioning", None)
        if sc_module is not None:
            for attr, key in (
                ("_last_input_rms", "train/self_conditioning/input_rms"),
                ("_last_input_absmax", "train/self_conditioning/input_absmax"),
                ("_last_residual_rms", "train/self_conditioning/residual_rms"),
                ("_last_residual_absmax", "train/self_conditioning/residual_absmax"),
                ("_last_combined_rms", "train/self_conditioning/combined_rms"),
                ("_last_combined_absmax", "train/self_conditioning/combined_absmax"),
                ("_last_signal_rms", "train/self_conditioning/signal_rms"),
                ("_last_signal_absmax", "train/self_conditioning/signal_absmax"),
                ("_last_residual_to_input_rms", "train/self_conditioning/residual_to_input_rms"),
            ):
                value = getattr(sc_module, attr, None)
                if value is not None:
                    log_dict[key] = value.item()

            with torch.no_grad():
                log_dict["train/self_conditioning/gate_proj_weight_rms"] = (
                    sc_module.gate_proj.weight.detach().float().square().mean().sqrt().item()
                )
                log_dict["train/self_conditioning/up_proj_weight_rms"] = (
                    sc_module.up_proj.weight.detach().float().square().mean().sqrt().item()
                )
                log_dict["train/self_conditioning/down_proj_weight_rms"] = (
                    sc_module.down_proj.weight.detach().float().square().mean().sqrt().item()
                )
                log_dict["train/self_conditioning/down_proj_weight_absmax"] = (
                    sc_module.down_proj.weight.detach().float().abs().max().item()
                )
        if self.finetuning_args.three_way_loss_auto_balance:
            log_dict["train/loss_balance_clean_scale"] = clean_scale.item()
            log_dict["train/loss_balance_rkl_scale"] = rkl_scale.item()
        self.log(log_dict)

        if self.state.global_step % self.args.logging_steps == 0:
            logger.info_rank0(
                f"Step {self.state.global_step}: "
                f"task_loss={task_loss.item():.4f}, "
                f"clean_ce_loss={clean_ce_loss.item():.4f}, "
                f"combined={combined_loss.item():.4f}"
            )

        return combined_loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
