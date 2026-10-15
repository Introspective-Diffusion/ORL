import torch

from jetengine_ext.config import Config
from jetengine_ext.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine_ext.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.consistent_sampling_params = False
        self.last_schedule_no_work_reason: str | None = None

    @staticmethod
    def _tensor_int_list(tensor: torch.Tensor, limit: int | None = None) -> list[int]:
        if limit is not None:
            tensor = tensor[:limit]
        return [int(x) for x in tensor.detach().cpu().tolist()]

    @staticmethod
    def _tensor_float_list(tensor: torch.Tensor, limit: int | None = None) -> list[float]:
        if limit is not None:
            tensor = tensor[:limit]
        return [float(x) for x in tensor.detach().cpu().tolist()]

    @staticmethod
    def _tensor_bool_list(tensor: torch.Tensor, limit: int | None = None) -> list[bool]:
        if limit is not None:
            tensor = tensor[:limit]
        return [bool(x) for x in tensor.detach().cpu().tolist()]

    def _selected_trace_entries(
        self,
        mask: torch.Tensor,
        proposal: torch.Tensor,
        raw_logprobs: torch.Tensor,
        sample_logprobs: torch.Tensor,
        proposal_probs: torch.Tensor,
        limit: int = 64,
    ) -> list[dict]:
        positions = torch.nonzero(mask, as_tuple=False).flatten()[:limit]
        entries = []
        for pos in positions.detach().cpu().tolist():
            entries.append(
                {
                    "pos": int(pos),
                    "token_id": int(proposal[pos].detach().cpu().item()),
                    "raw_logprob": float(raw_logprobs[pos].detach().cpu().item()),
                    "sample_logprob": float(sample_logprobs[pos].detach().cpu().item()),
                    "sample_prob": float(proposal_probs[pos].detach().cpu().item()),
                }
            )
        return entries

    def add(self, seq: Sequence):
        self.running.append(seq)

    def is_finished(self):
        return not self.running

    def _safe_num_new_blocks_needed(self, seq: Sequence) -> int | str:
        try:
            return int(seq.num_new_blocks_needed(self.block_manager.block_size))
        except Exception as exc:
            return f"error:{type(exc).__name__}:{exc}"

    def _sequence_debug_state(self, seq: Sequence) -> dict:
        accepted_mask = getattr(seq, "block_accepted_mask", None) or []
        intermediate_tokens = getattr(seq, "intermediate_block_tokens", None) or []
        status = getattr(seq, "status", None)
        return {
            "seq_id": int(getattr(seq, "seq_id", -1)),
            "status": status.name if hasattr(status, "name") else str(status),
            "num_tokens": int(getattr(seq, "num_tokens", -1)),
            "num_prompt_tokens": int(getattr(seq, "num_prompt_tokens", -1)),
            "num_completion_tokens": int(getattr(seq, "num_completion_tokens", -1)),
            "max_tokens": int(getattr(seq, "max_tokens", -1)),
            "current_denoising_step": int(getattr(seq, "current_denoising_step", -1)),
            "max_steps_per_block": int(getattr(seq, "max_steps_per_block", -1)),
            "block_table_len": len(getattr(seq, "block_table", []) or []),
            "num_blocks": int(getattr(seq, "num_blocks", -1)),
            "num_new_blocks_needed": self._safe_num_new_blocks_needed(seq),
            "intermediate_block_tokens_len": len(intermediate_tokens),
            "block_accepted_count": sum(1 for x in accepted_mask if x),
            "block_unaccepted_count": sum(1 for x in accepted_mask if not x),
        }

    def debug_state(self, pending_idx: int | None = None, total: int | None = None) -> dict:
        block_manager = self.block_manager
        state = {
            "pending_idx": pending_idx,
            "total": total,
            "running": len(self.running),
            "max_num_seqs": int(self.max_num_seqs),
            "free_blocks": len(block_manager.free_block_ids),
            "used_blocks": len(block_manager.used_block_ids),
            "total_blocks": len(block_manager.blocks),
            "block_size": int(block_manager.block_size),
            "last_schedule_no_work_reason": self.last_schedule_no_work_reason,
            "sequences": [self._sequence_debug_state(seq) for seq in self.running],
        }
        return state

    def _describe_no_work(
        self,
        prefill_candidates: list[Sequence],
        denoise_candidates: list[Sequence],
    ) -> str:
        if not self.running:
            return "idle"
        if prefill_candidates and not any(self.block_manager.can_allocate(seq) for seq in prefill_candidates):
            return "prefill_block_allocation_blocked"
        if denoise_candidates:
            can_run = any(
                self.block_manager.can_append_blocks(seq.num_new_blocks_needed(self.block_manager.block_size))
                for seq in denoise_candidates
            )
            if not can_run:
                return "denoise_block_allocation_blocked"
        statuses = sorted({seq.status.name for seq in self.running})
        return f"no_schedulable_sequences statuses={statuses}"

    def _sample_block(self, logits: torch.Tensor, seq: Sequence):
        logits = logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        if seq.temperature == 0:
            token_ids = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
        else:
            filtered_logits = self._filter_logits(logits / seq.temperature, seq.top_k, seq.top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        token_probs = torch.gather(probs, -1, token_ids.unsqueeze(-1)).squeeze(-1)
        token_logprobs = torch.gather(logprobs, -1, token_ids.unsqueeze(-1)).squeeze(-1)
        return token_ids, token_probs, token_logprobs, probs

    def _filter_logits(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        if top_k > 0 and top_k < logits.size(-1):
            values, _ = torch.topk(logits, top_k, dim=-1)
            threshold = values[:, -1:].expand_as(logits)
            logits = torch.where(logits >= threshold, logits, torch.full_like(logits, -float("inf")))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, sorted_logits)
        return logits

    def _select_transfer_indices(
        self,
        seq: Sequence,
        mask_index: torch.Tensor,
        token_probs: torch.Tensor,
        probs: torch.Tensor,
        num_to_transfer: int,
    ) -> torch.Tensor:
        transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
        if num_to_transfer <= 0 or not mask_index.any():
            return transfer_index

        unresolved_indices = mask_index.nonzero(as_tuple=True)[0]
        num_to_transfer = min(int(num_to_transfer), unresolved_indices.numel())
        strategy = seq.remasking_strategy

        if strategy == 'sequential':
            selected = unresolved_indices[:num_to_transfer]
        elif strategy == 'confidence_threshold_leftmost':
            confidence = torch.where(mask_index, token_probs, torch.full_like(token_probs, -float("inf")))
            selected = torch.nonzero(confidence > seq.dynamic_threshold, as_tuple=False).flatten()
            if selected.numel() == 0:
                selected = unresolved_indices[:num_to_transfer]
        elif strategy in {'low_confidence_dynamic', 'entropy_dynamic'}:
            confidence = self._transfer_score(strategy, mask_index, token_probs, probs)
            selected = torch.nonzero(confidence > seq.dynamic_threshold, as_tuple=False).flatten()
            if selected.numel() < num_to_transfer:
                selected = torch.topk(confidence, k=num_to_transfer).indices
        elif strategy in {'low_confidence_static', 'entropy_static'}:
            confidence = self._transfer_score(strategy, mask_index, token_probs, probs)
            selected = torch.topk(confidence, k=num_to_transfer).indices
        elif strategy in {'entropy_bound', 'entropy_bounded'}:
            block_probs = probs.float()
            masked_probs = block_probs[mask_index]
            eps = 1e-12
            entropies = -(masked_probs.clamp_min(eps) * masked_probs.clamp_min(eps).log()).sum(dim=-1)
            ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
            within_budget = (torch.cumsum(ent_sorted, dim=0) - ent_sorted) <= seq.eb_threshold
            selected = unresolved_indices[order[within_budget]]
        else:
            raise ValueError(f"Unsupported remasking strategy: {strategy}")

        if selected.numel() == 0 and seq.force_accept_leftmost:
            selected = unresolved_indices[:1]
        transfer_index[selected] = True
        return transfer_index & mask_index

    def _transfer_score(self, strategy: str, mask_index: torch.Tensor, token_probs: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if strategy.startswith('entropy_'):
            eps = 1e-12
            block_probs = probs.float().clamp_min(eps)
            entropy = -(block_probs * block_probs.log()).sum(dim=-1)
            if probs.size(-1) > 1:
                entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device, dtype=entropy.dtype))
            score = -entropy
        else:
            score = token_probs
        return torch.where(mask_index, score, torch.full_like(score, -float("inf")))

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        # 1. Schedule new sequences for prefill
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            # Simple batching: take as many as fit
            for seq in prefill_candidates:
                # num_tokens for a waiting seq is its prefill length
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    prefill_batch.append(seq)
            if prefill_batch:
                self.last_schedule_no_work_reason = None
                return prefill_batch, RunType.PREFILL   
        # 2. If no prefilling, create a DENOISE batch.
        denoise_candidates = [s for s in self.running if s.status == SequenceStatus.DENOISING or s.status == SequenceStatus.SAVING]
        if denoise_candidates:
            denoise_batch = []
            for seq in denoise_candidates:
                num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                if len(denoise_batch) < self.max_num_seqs and self.block_manager.can_append_blocks(num_new_blocks):
                    self.block_manager.append_blocks(seq, num_new_blocks)
                    denoise_batch.append(seq)
            if denoise_batch:
                self.last_schedule_no_work_reason = None
                return denoise_batch, RunType.DENOISE

        self.last_schedule_no_work_reason = self._describe_no_work(prefill_candidates, denoise_candidates)
        return None, None     

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        if run_type == RunType.PREFILL:
            start_idx = 0
            for seq in seqs:
                end_idx = start_idx + len(seq)
                if end_idx > start_idx:
                    seq.prefix_last_hidden = logits[end_idx - 1].detach()
                start_idx = end_idx
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        
        elif run_type == RunType.DENOISE:
            start_idx = 0
            for seq in seqs:
                # Extract the part of the tensors relevant to this sequence
                if seq.status == SequenceStatus.DENOISING:
                    block_len = seq.block_length
                    block_logits = logits[start_idx : start_idx + block_len]
                    seq_x0, seq_x0_p, seq_x0_lp, block_probs = self._sample_block(block_logits, seq)
                    current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                    accepted_mask = torch.tensor(seq.block_accepted_mask, device=logits.device, dtype=torch.bool)
                    prompt_mask = torch.tensor(seq.block_prompt_mask, device=logits.device, dtype=torch.bool)
                    mask_index = ~accepted_mask
                    schedule_idx = min(
                        seq.current_denoising_step,
                        len(seq.num_transfer_tokens_per_step) - 1,
                    )
                    num_to_transfer = seq.num_transfer_tokens_per_step[schedule_idx]
                    transfer_index = self._select_transfer_indices(seq, mask_index, seq_x0_p, block_probs, num_to_transfer)
                    num_to_transfer = int(transfer_index.sum().item())
                    edit_index = torch.zeros_like(transfer_index)
                    if seq.allow_accepted_change:
                        edit_index = (
                            accepted_mask
                            & (~prompt_mask)
                            & seq_x0_p.gt(seq.edit_threshold)
                            & seq_x0.to(current_block_tensor.dtype).ne(current_block_tensor)
                        )

                    # update
                    new_block_list = current_block_tensor.tolist()
                    accepted_tokens = seq_x0[transfer_index].tolist()
                    accepted_logprobs = seq_x0_lp[transfer_index].tolist()
                    original_indices = transfer_index.nonzero(as_tuple=True)[0].tolist()
                    edited_tokens = seq_x0[edit_index].tolist()
                    edited_logprobs = seq_x0_lp[edit_index].tolist()
                    edit_indices = edit_index.nonzero(as_tuple=True)[0].tolist()
                    sample_logprobs = seq_x0_p.float().clamp_min(
                        torch.finfo(seq_x0_p.float().dtype).tiny
                    ).log()
                    entropy = -(
                        block_probs.float()
                        * block_probs.float()
                        .clamp_min(torch.finfo(block_probs.float().dtype).tiny)
                        .log()
                    ).sum(dim=-1)
                    replay_event = None
                    if seq.record_replay_trace:
                        new_accept_mask = transfer_index
                        accepted_after = accepted_mask | new_accept_mask
                        block_idx = seq.num_completion_tokens // seq.block_length
                        replay_event = {
                            "kind": "jetengine_sdar_step",
                            "format": "jetengine_sdar_replay_trace_v1",
                            "block_idx": int(block_idx),
                            "step": int(seq.current_denoising_step + 1),
                            "denoise_step": int(seq.current_denoising_step),
                            "schedule_step": int(schedule_idx),
                            "max_steps_per_block": int(seq.max_steps_per_block),
                            "global_step": int(seq.global_denoising_step + 1),
                            "position": [int(seq.num_tokens), int(seq.num_tokens + block_len)],
                            "prompt_count": int(prompt_mask.sum().item()),
                            "accepted_count": int(accepted_after.sum().item()),
                            "new_accept_count": int(new_accept_mask.sum().item()),
                            "edit_change_count": int(edit_index.sum().item()),
                            "accepted_positions": self._tensor_int_list(
                                torch.nonzero(accepted_after, as_tuple=False).flatten(), 64
                            ),
                            "new_accept_positions": [int(x) for x in original_indices[:64]],
                            "edit_change_positions": [int(x) for x in edit_indices[:64]],
                            "new_accept_token_logprobs": self._selected_trace_entries(
                                new_accept_mask,
                                seq_x0,
                                seq_x0_lp,
                                sample_logprobs,
                                seq_x0_p,
                            ),
                            "edit_token_logprobs": self._selected_trace_entries(
                                edit_index,
                                seq_x0,
                                seq_x0_lp,
                                sample_logprobs,
                                seq_x0_p,
                            ),
                            "max_prob_mean": float(seq_x0_p.float().mean().item()),
                            "entropy_mean": float(entropy.mean().item()),
                            "input_head": self._tensor_int_list(current_block_tensor, 32),
                            "proposal_head": self._tensor_int_list(seq_x0, 32),
                            "maxp_head": self._tensor_float_list(seq_x0_p, 16),
                            "random_token_mode": seq.noise_mode,
                            "random_token_high": int(seq.random_token_high),
                            "deterministic_randtok": bool(seq.deterministic_randtok),
                            "randtok_seed": int(seq.randtok_seed),
                        }
                        if seq.replay_trace_full:
                            replay_event.update(
                                {
                                    "input_ids": self._tensor_int_list(current_block_tensor),
                                    "proposal_ids": self._tensor_int_list(seq_x0),
                                    "prompt_mask": self._tensor_bool_list(prompt_mask),
                                    "accepted_mask_before": self._tensor_bool_list(accepted_mask),
                                    "accepted_mask": self._tensor_bool_list(accepted_after),
                                    "new_accept_mask": self._tensor_bool_list(new_accept_mask),
                                    "edit_mask": self._tensor_bool_list(edit_index),
                                    "proposal_probs": self._tensor_float_list(seq_x0_p),
                                    "proposal_raw_logprobs": self._tensor_float_list(seq_x0_lp),
                                    "proposal_sample_logprobs": self._tensor_float_list(sample_logprobs),
                                }
                            )





                    # newly added
                    if seq.block_first_unmask_steps is None or len(seq.block_first_unmask_steps) != block_len:
                        seq.block_first_unmask_steps = [0] * block_len
                    first_time_global = seq.global_denoising_step + 1
                    for idx in original_indices:
                        if seq.block_first_unmask_steps[idx] == 0:
                            seq.block_first_unmask_steps[idx] = first_time_global

                    # NEW: record token probability at the moment it is decoded/accepted (first time only)
                    if seq.block_first_token_logprobs is None or len(seq.block_first_token_logprobs) != block_len:
                        seq.block_first_token_logprobs = [0.0] * block_len

                    for idx, lp in zip(original_indices, accepted_logprobs):
                        if seq.block_first_token_logprobs[idx] == 0.0:
                            seq.block_first_token_logprobs[idx] = float(lp)

                    for idx, token in zip(original_indices, accepted_tokens):
                        new_block_list[idx] = token
                        seq.block_accepted_mask[idx] = True
                    for idx, token in zip(edit_indices, edited_tokens):
                        new_block_list[idx] = token
                    for idx, lp in zip(edit_indices, edited_logprobs):
                        # SGLang overwrites q-logprobs when an accepted token is
                        # edited; keep PPO/replay logprobs aligned to final ids.
                        seq.block_first_token_logprobs[idx] = float(lp)
                    seq.intermediate_block_tokens = new_block_list
                    
                    seq.current_denoising_step += 1
                    seq.global_denoising_step += 1
                    
                    # Check if block is fully denoised
                    seq._refresh_unaccepted_inputs(
                        block_idx=seq.num_completion_tokens // seq.block_length,
                        step_idx=seq.current_denoising_step,
                    )
                    if replay_event is not None:
                        refreshed_block = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                        replay_event["current_ids_head"] = self._tensor_int_list(refreshed_block, 32)
                        if seq.replay_trace_full:
                            replay_event["current_ids"] = self._tensor_int_list(refreshed_block)
                        seq.replay_trace.append(replay_event)

                    is_fully_denoised = all(seq.block_accepted_mask) or \
                                        (seq.current_denoising_step >= seq.max_steps_per_block)

                    if is_fully_denoised:
                        # Block is done, commit it and check if generation is finished
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = num_to_transfer
                    
                elif seq.status == SequenceStatus.SAVING:
                    # If saving, commit the block and start a new one
                    if seq.last_denoise_hidden is not None:
                        seq.prefix_last_hidden = seq.last_denoise_hidden[-1].detach()
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()

                start_idx += seq.block_length
                
        # Filter out finished sequences from the running list
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)