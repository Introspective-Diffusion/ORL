"""CPU tests for independent SDAR proposal and confidence temperatures."""

from __future__ import annotations

import torch

from sglang.srt.dllm.algorithm.sdar_prefix_kv import SDARPrefixKV


def _algorithm(sampling_temperature: float) -> SDARPrefixKV:
    algorithm = object.__new__(SDARPrefixKV)
    algorithm.temperature_mode = "fixed"
    algorithm.temperature = sampling_temperature
    algorithm.temperature_min = sampling_temperature
    algorithm.temperature_max = sampling_temperature
    algorithm.top_k = 0
    algorithm.top_p = 1.0
    algorithm.confidence_temperature = 1.0
    algorithm._adaptive_temp_state_by_req = {}
    algorithm._adaptive_confidence_temp_state_by_req = {}
    return algorithm


def test_t0_proposals_are_argmax_for_both_confidence_temperatures():
    algorithm = _algorithm(0.0)
    logits = torch.tensor(
        [[0.1, 2.0, -1.0], [3.0, 2.9, 2.8]], dtype=torch.float32
    )
    proposal, _, _ = algorithm._sample_from_logits(logits, step=1)
    confidence_t1 = algorithm._confidence_probs(
        logits, temperature_override=1.0
    )
    confidence_t08 = algorithm._confidence_probs(
        logits, temperature_override=0.8
    )
    assert torch.equal(proposal, logits.argmax(dim=-1))
    assert not torch.allclose(confidence_t1, confidence_t08)


def test_t1_stochastic_proposals_do_not_depend_on_confidence_temperature():
    algorithm = _algorithm(1.0)
    logits = torch.tensor([[0.2, 0.1, 0.0]] * 64)

    torch.manual_seed(1234)
    algorithm._confidence_probs(logits, temperature_override=1.0)
    proposal_t1, _, _ = algorithm._sample_from_logits(logits, step=1)

    torch.manual_seed(1234)
    algorithm._confidence_probs(logits, temperature_override=0.8)
    proposal_t08, _, _ = algorithm._sample_from_logits(logits, step=1)

    assert torch.equal(proposal_t1, proposal_t08)
    assert not torch.equal(proposal_t1, logits.argmax(dim=-1))


def test_confidence_metrics_record_sampling_temperature_and_masks():
    metrics = {
        "effective_confidence_temperature": 0.9,
    }
    completed = (
        SDARPrefixKV._complete_adaptive_confidence_temperature_metrics(
            metrics,
            sampling_temperature=0.0,
            prompt_mask=torch.tensor([True, False, False, False]),
            accepted_mask=torch.tensor([True, True, False, True]),
            new_accept_mask=torch.tensor([False, True, False, False]),
            edit_mask=torch.tensor([False, False, False, True]),
        )
    )
    assert completed is not None
    assert completed["sampling_temperature"] == 0.0
    assert completed["accepted_count"] == 2
    assert completed["new_accept_count"] == 1
    assert completed["edit_count"] == 1


def test_confidence_and_proposal_controller_state_clear_independently():
    algorithm = _algorithm(0.0)
    algorithm._adaptive_temp_state_by_req[7] = {"proposal": 1.0}
    algorithm._adaptive_confidence_temp_state_by_req[7] = {
        "confidence": 1.0
    }

    algorithm._clear_adaptive_confidence_temp_state(7)
    assert 7 in algorithm._adaptive_temp_state_by_req
    assert 7 not in algorithm._adaptive_confidence_temp_state_by_req

    algorithm._clear_adaptive_temp_state(7)
    assert 7 not in algorithm._adaptive_temp_state_by_req
