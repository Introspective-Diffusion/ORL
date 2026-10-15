from enum import Enum, auto
from itertools import count
from jetengine_ext.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()      # Has a prompt part to prefill
    PREFILLING = auto()   # Is currently in a prefill model run
    DENOISING = auto()    # Is ready for or in a denoise model run
    SAVING = auto()       # Is ready for or in a save model run
    FINISHED = auto()
    
class RunType(Enum):
    PREFILL = auto()
    DENOISE = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, prompt_token_ids: list[int], mask_token_id: int, sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.block_length = sampling_params.block_length
        self.prompt_token_ids = prompt_token_ids
        prompt_len = len(self.prompt_token_ids)

        if sampling_params.prefill_full_prompt:
            self.num_prefill_tokens = prompt_len
        else:
            self.num_prefill_tokens = (prompt_len // self.block_length) * self.block_length
        prefill_part = self.prompt_token_ids[:self.num_prefill_tokens]
        
        first_denoise_part = self.prompt_token_ids[self.num_prefill_tokens:]
        
        self.token_ids = prefill_part
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = prompt_len # Keep track of the original full prompt length
        self.mask_token_id = mask_token_id

        self.noise_mode = sampling_params.noise_mode
        if self.noise_mode not in {"mask", "random_vocab"}:
            raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")
        self.random_token_high = sampling_params.random_token_high or mask_token_id
        self.deterministic_randtok = sampling_params.deterministic_randtok
        self.randtok_seed = sampling_params.randtok_seed
        self.allow_accepted_change = sampling_params.allow_accepted_change
        self.edit_threshold = (
            sampling_params.edit_threshold
            if sampling_params.edit_threshold is not None
            else sampling_params.dynamic_threshold
        )
        self.exclude_prompt_tail_from_stop = sampling_params.exclude_prompt_tail_from_stop
        self.truncate_on_stop = sampling_params.truncate_on_stop
        self.record_replay_trace = bool(sampling_params.record_replay_trace)
        self.replay_trace_full = bool(sampling_params.replay_trace_full)
        self.replay_trace: list[dict] = []

        self.intermediate_block_tokens = first_denoise_part + [mask_token_id] * (self.block_length - len(first_denoise_part))
        self.block_accepted_mask = [True] * len(first_denoise_part) + [False] * (self.block_length - len(first_denoise_part))
        self.block_prompt_mask = [True] * len(first_denoise_part) + [False] * (self.block_length - len(first_denoise_part))
        self._refresh_unaccepted_inputs(block_idx=0, step_idx=0)
        self.num_to_transfer = 0
        self.current_denoising_step = 0


        self.first_unmask_steps: list[int] = []
        self.block_first_unmask_steps: list[int] | None = [0] * len(self.intermediate_block_tokens)
        self.global_denoising_step = 0

        # NEW: store token probabilities ONLY when a token is decoded/accepted.
        # token_logprobs aligns with completion_token_ids (same length).
        self.token_logprobs: list[float] = []
        # block_first_token_logprobs aligns with current block positions; 0.0 means "not decoded yet".
        self.block_first_token_logprobs: list[float] | None = [0.0] * len(self.intermediate_block_tokens)

        # initial status based on whether prefill is needed.
        if self.num_prefill_tokens > 0:
            self.status = SequenceStatus.WAITING
        else:
            self.status = SequenceStatus.DENOISING

        # Block Diffusion parameters
        self.temperature = sampling_params.temperature
        self.stop_words = sampling_params.stop_words if sampling_params.stop_words is not None else []
        self.top_k = sampling_params.topk
        self.top_p = sampling_params.topp
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.denoising_steps = sampling_params.denoising_steps
        self.max_steps_per_block = (
            sampling_params.max_steps_per_block
            if sampling_params.max_steps_per_block is not None
            else self.denoising_steps
        )
        self.remasking_strategy = sampling_params.remasking_strategy
        self.dynamic_threshold = sampling_params.dynamic_threshold
        self.eb_threshold = sampling_params.eb_threshold
        self.force_accept_leftmost = sampling_params.force_accept_leftmost
        self.num_transfer_per_step = sampling_params.num_transfer_per_step
        self.num_transfer_tokens_per_step = self._get_num_transfer_tokens()

        # State for KV Caching
        self.num_cached_tokens = 0
        self.block_table = []

        # {1: {"position": (24, 28), "input_hidden_states": torch.Tensor}}
        self.input_hidden_states = {}
        self.prefix_last_hidden = None
        self.last_denoise_hidden = None
        self.current_logits_source_hidden = None

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    def _get_num_transfer_tokens(self):
        if self.num_transfer_per_step > 0:
            return [self.num_transfer_per_step] * self.denoising_steps
        base = self.block_length // self.denoising_steps
        remainder = self.block_length % self.denoising_steps
        num_tokens = [base] * self.denoising_steps
        for i in range(remainder):
            num_tokens[i] += 1
        return num_tokens

    def _random_token(self, pos: int, block_idx: int, step_idx: int) -> int:
        if self.deterministic_randtok:
            return int(
                ((pos + block_idx * self.block_length + 1) * 1103515245
                 + (step_idx + 1) * 12345
                 + self.randtok_seed)
                % self.random_token_high
            )
        import random
        return random.randrange(self.random_token_high)

    def _refresh_unaccepted_inputs(self, block_idx: int = 0, step_idx: int = 0):
        for pos, accepted in enumerate(self.block_accepted_mask):
            if accepted:
                continue
            if self.noise_mode == "mask":
                self.intermediate_block_tokens[pos] = self.mask_token_id
            elif self.noise_mode == "random_vocab":
                self.intermediate_block_tokens[pos] = self._random_token(pos, block_idx, step_idx)
            else:
                raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")

    def start_new_block(self):
        self.current_denoising_step = 0
        self.intermediate_block_tokens = [self.mask_token_id] * self.block_length
        self.block_accepted_mask = [False] * self.block_length
        self.block_prompt_mask = [False] * self.block_length
        self._refresh_unaccepted_inputs(block_idx=self.num_completion_tokens // self.block_length, step_idx=0)
        self.status = SequenceStatus.DENOISING
        # NEW: reset per-block probability buffer
        self.block_first_token_logprobs = [0.0] * self.block_length

    '''
    def commit_block(self, block_tokens: list[int]):
        # Trim block if it exceeds max_tokens or contains EOS
        final_block = []
        for token_id in block_tokens:
            if not self.ignore_eos and (token_id == self.eos_token_id or token_id in self.stop_words):
                final_block.append(token_id)
                self.status = SequenceStatus.FINISHED
                break
            if self.num_completion_tokens + len(final_block) >= self.max_tokens:
                self.status = SequenceStatus.FINISHED
                break
            final_block.append(token_id)

        self.token_ids.extend(final_block)
        self.num_tokens = len(self.token_ids)
        self.intermediate_block_tokens = []

        if self.num_tokens >= self.num_prompt_tokens + self.max_tokens:
             self.status = SequenceStatus.FINISHED'''
    





    def commit_block(self, block_tokens: list[int]):
        # 1) take token one by one, stop when EOS / reach max_tokens
        final_block = []
        k = 0
        before_ntok = self.num_tokens                     # pre length
        prompt_gap_for_stop = (
            max(0, self.num_prompt_tokens - before_ntok)
            if self.exclude_prompt_tail_from_stop
            else 0
        )
        completion_before_for_stop = (
            max(0, self.num_completion_tokens)
            if self.exclude_prompt_tail_from_stop
            else self.num_completion_tokens
        )
        for token_id in block_tokens:
            if k >= prompt_gap_for_stop:
                completion_k = k - prompt_gap_for_stop if self.exclude_prompt_tail_from_stop else k
                if self.truncate_on_stop and completion_before_for_stop + completion_k >= self.max_tokens:
                    self.status = SequenceStatus.FINISHED
                    break
                if not self.ignore_eos and (token_id == self.eos_token_id or token_id in self.stop_words):
                    self.status = SequenceStatus.FINISHED
                    final_block.append(token_id)
                    k += 1
                    if self.truncate_on_stop:
                        break
                    continue
                if completion_before_for_stop + completion_k >= self.max_tokens:
                    self.status = SequenceStatus.FINISHED
            final_block.append(token_id)
            k += 1

        # 2) self.token_ids
        self.token_ids.extend(final_block)
        self.num_tokens = len(self.token_ids)
        self.intermediate_block_tokens = []

        # 3) merge first unmask step into list
        # only completion 
        if self.block_first_unmask_steps is not None:
            prompt_gap = max(0, self.num_prompt_tokens - before_ntok)
            # completion start index
            start = min(prompt_gap, k)
            if start < k:
                self.first_unmask_steps.extend(self.block_first_unmask_steps[start:k])
            self.block_first_unmask_steps = None
        self.block_accepted_mask = []
        self.block_prompt_mask = []

        # NEW: merge per-token probs into completion-aligned list (only completion)
        if self.block_first_token_logprobs is not None:
            prompt_gap = max(0, self.num_prompt_tokens - before_ntok)
            start = min(prompt_gap, k)
            if start < k:
                self.token_logprobs.extend(self.block_first_token_logprobs[start:k])
            self.block_first_token_logprobs = None

        if self.num_tokens >= self.num_prompt_tokens + self.max_tokens:
            self.status = SequenceStatus.FINISHED

        if self.record_replay_trace:
            before_completion = max(0, before_ntok - self.num_prompt_tokens)
            block_idx = before_completion // self.block_length
            prompt_gap = max(0, self.num_prompt_tokens - before_ntok)
            start = min(prompt_gap, k)
            self.replay_trace.append(
                {
                    "kind": "jetengine_sdar_commit",
                    "format": "jetengine_sdar_replay_trace_v1",
                    "block_idx": int(block_idx),
                    "prompt_tail_tokens": int(prompt_gap),
                    "committed_response_start": int(start),
                    "final_ids": [int(x) for x in block_tokens],
                    "committed_response_ids": [int(x) for x in final_block[start:k]],
                    "random_token_mode": self.noise_mode,
                    "random_token_high": int(self.random_token_high),
                    "deterministic_randtok": bool(self.deterministic_randtok),
                    "randtok_seed": int(self.randtok_seed),
                    "logprob_source": "proposal_raw_logprob_overwritten_on_edit",
                }
            )


    




    def get_len_for_next_step(self):
        return self.num_tokens + self.block_length

    def num_new_blocks_needed(self, block_size: int) -> int:
        target_blocks = (self.num_tokens + self.block_length + block_size - 1) // block_size
        return max(0, target_blocks - len(self.block_table))

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    '''
    def __getstate__(self):
        # Simplified for multiprocessing; customize as needed
        return (self.seq_id, self.status, self.token_ids, self.num_tokens, self.num_prompt_tokens, 
                self.num_cached_tokens, self.block_table, self.intermediate_block_tokens, self.current_denoising_step)

    def __setstate__(self, state):
        (self.seq_id, self.status, self.token_ids, self.num_tokens, self.num_prompt_tokens, 
         self.num_cached_tokens, self.block_table, self.intermediate_block_tokens, self.current_denoising_step) = state'''

    def __getstate__(self):
        # Simplified for multiprocessing; customize as needed
        return (self.seq_id, self.status, self.token_ids, self.num_tokens, self.num_prompt_tokens,
                self.num_cached_tokens, self.block_table, self.intermediate_block_tokens, self.current_denoising_step,
                self.first_unmask_steps, self.block_first_unmask_steps, self.global_denoising_step, self.token_logprobs, self.block_first_token_logprobs,
                self.prefix_last_hidden, self.last_denoise_hidden, self.current_logits_source_hidden,
                self.noise_mode, self.random_token_high, self.deterministic_randtok, self.randtok_seed,
                self.allow_accepted_change, self.edit_threshold, self.block_accepted_mask, self.block_prompt_mask,
                self.record_replay_trace, self.replay_trace_full, self.replay_trace,
                self.max_steps_per_block, self.exclude_prompt_tail_from_stop,
                self.truncate_on_stop)

    def __setstate__(self, state):
        if len(state) == 28:
            state = (*state, 4, False, False)
        elif len(state) == 29:
            state = (*state, False, False)
        elif len(state) == 30:
            state = (*state, False)
        (self.seq_id, self.status, self.token_ids, self.num_tokens, self.num_prompt_tokens,
         self.num_cached_tokens, self.block_table, self.intermediate_block_tokens, self.current_denoising_step,
         self.first_unmask_steps, self.block_first_unmask_steps, self.global_denoising_step, self.token_logprobs, self.block_first_token_logprobs,
         self.prefix_last_hidden, self.last_denoise_hidden, self.current_logits_source_hidden,
         self.noise_mode, self.random_token_high, self.deterministic_randtok, self.randtok_seed,
         self.allow_accepted_change, self.edit_threshold, self.block_accepted_mask, self.block_prompt_mask,
         self.record_replay_trace, self.replay_trace_full, self.replay_trace,
         self.max_steps_per_block, self.exclude_prompt_tail_from_stop,
         self.truncate_on_stop) = state



