import pickle
import os
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from jetengine_ext.config import Config
from jetengine_ext.engine.sequence import Sequence, RunType, SequenceStatus
from jetengine_ext.models.sdar import SDARForCausalLM
from jetengine_ext.models.sdar_moe import SDARMoeForCausalLM
from jetengine_ext.utils.context import set_context, get_context, reset_context
from jetengine_ext.utils.loader import load_model


def normalized_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.to(torch.float32)
    p = torch.clamp(p, min=eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    vocab_len = p.size(-1)
    if vocab_len == 1:
        return torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)
    entropy = -(p * torch.log(p)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(vocab_len), dtype=p.dtype, device=p.device))
    entropy_norm = entropy / max_entropy
    return torch.clamp(entropy_norm, 0.0, 1.0)



class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        init_method = (
            config.distributed_init_method
            or os.environ.get("JETENGINE_DISTRIBUTED_INIT_METHOD")
            or os.environ.get("JETENGINE_DIST_INIT_METHOD")
            or "tcp://localhost:2333"
        )
        if not dist.is_initialized():
            dist.init_process_group("nccl", init_method, world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        if "sdar" in hf_config.model_type and "moe" in hf_config.model_type:
            self.model = SDARMoeForCausalLM(hf_config, mask_token_id=config.mask_token_id)
        elif "sdar" in hf_config.model_type:
            self.model = SDARForCausalLM(hf_config, mask_token_id=config.mask_token_id)
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")
        load_model(self.model, config.model)
        # Sampler is removed from here
        self.warmup_model()
        self.allocate_kv_cache()
        # CUDA graph capture for block diffusion is complex and omitted for this example
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            shm_name = "jetengineshm"
            if rank == 0:
                # Create (and clean up stale) shared memory
                try:
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**20)
                except FileExistsError:
                    try:
                        stale = SharedMemory(name=shm_name)
                        stale.close()
                        stale.unlink()
                    except FileNotFoundError:
                        pass
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len, self.config.mask_token_id) for _ in range(num_seqs)]
        self.run(seqs, RunType.PREFILL)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim + self.block_size * hf_config.hidden_size // self.world_size) * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        print(f"Allocated KV cache with {config.num_kvcache_blocks * self.block_size} tokens")
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        assert self.world_size == 1, "Initialization in sample.jetengine_ext.engine.block_manager.BlockManager"
        # self.p = torch.empty(config.num_kvcache_blocks, self.block_size, hf_config.vocab_size // self.world_size)
        self.residual_hidden_states = torch.empty(config.num_kvcache_blocks, self.block_size, hf_config.hidden_size // self.world_size)
        self.valid = torch.zeros(config.num_kvcache_blocks, self.block_size, dtype=torch.bool)
        # self.model.model.p = self.p
        self.model.model.residual_hidden_states = self.residual_hidden_states
        self.model.model.valid = self.valid

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len == 0: return None
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32).cuda()

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, cu_seqlens_q, slot_mapping, is_last_step = [], [], [0], [], []
        max_seqlen_q = 0
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids)
            positions.extend(range(seqlen))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen)
            is_last_step.append(False)
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            for i in range(seqlen):
                block_idx = i // self.block_size 
                block_offset = i % self.block_size 
                physical_block_id = seq.block_table[block_idx]
                slot = physical_block_id * self.block_size + block_offset
                slot_mapping.append(slot)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(positions, dtype=torch.int64).cuda()
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).cuda()
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).cuda()
        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q, 
            cu_seqlens_k=cu_seqlens_q, 
            max_seqlen_q=max_seqlen_q, 
            max_seqlen_k=max_seqlen_q, 
            slot_mapping=slot_mapping, 
            is_last_denoise_step=is_last_step, # <-- Pass the new flag
            block_length=self.config.block_length,
            use_self_conditioning=self.config.use_self_conditioning,
            self_conditioning_mask_mode=self.config.self_conditioning_mask_mode,
            prefill_causal=self.config.prefill_causal,
        )
        return input_ids, positions

    def prepare_denoise(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cached_lens = []
        
        for seq in seqs:
            # The query is the current intermediate block
            q_tokens = seq.intermediate_block_tokens
            q_len = len(q_tokens)
            
            # The context (key/value) is the confirmed part of the sequence
            k_len = len(seq)
            
            input_ids.extend(q_tokens)
            # Positions are global
            positions.extend(range(k_len, k_len + q_len))
            cached_lens.append(k_len)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(positions, dtype=torch.int64).cuda()
        cached_lens = torch.tensor(cached_lens, dtype=torch.int32).cuda()
        block_tables = self.prepare_block_tables(seqs)
        
        set_context(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            block_tables=block_tables,
            block_length=self.config.block_length,
            seqs=seqs,
            use_self_conditioning=self.config.use_self_conditioning,
            self_conditioning_mask_mode=self.config.self_conditioning_mask_mode,
            prefill_causal=self.config.prefill_causal,
            commit_causal=bool(
                self.config.commit_causal
                and all(seq.status == SequenceStatus.SAVING for seq in seqs)
            ),
        )
        
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        hidden_states = self.model(input_ids, positions)
        if get_context().run_type == RunType.PREFILL:
            return hidden_states
        return self.model.compute_logits(hidden_states)

    def run(self, seqs: list[Sequence], run_type: RunType) -> torch.Tensor:
        if run_type == RunType.PREFILL:
            input_ids, positions = self.prepare_prefill(seqs)
        elif run_type == RunType.DENOISE:
            input_ids, positions = self.prepare_denoise(seqs)
        else:
            return None

        logits = self.run_model(input_ids, positions)
        reset_context()
        return logits if self.rank == 0 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 256)
        max_global_bs = max_bs * self.config.block_length
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_global_bs, dtype=torch.int64)
        positions = torch.zeros(max_global_bs, dtype=torch.int64)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_global_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                run_type=RunType.DENOISE,
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                block_length=self.config.block_length,
                use_self_conditioning=self.config.use_self_conditioning,
                self_conditioning_mask_mode=self.config.self_conditioning_mask_mode,
                prefill_causal=self.config.prefill_causal,
            )
            global_bs = bs * self.config.block_length
            outputs[:global_bs] = self.model(input_ids[:global_bs], positions[:global_bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:global_bs] = self.model(input_ids[:global_bs], positions[:global_bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
