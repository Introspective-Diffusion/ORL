import math

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.ppo.entropy_gated_rkl_probe import (
    ENTROPY_GATED_RKL_REPORT_SLOTS,
    accumulate_entropy_gated_rkl_report_,
    compute_entropy_gated_ce_rkl_terms,
    compute_entropy_gated_rkl_probe_stats,
    entropy_gated_rkl_blocks_response_level_ce,
    entropy_gated_rkl_parity_terms,
    entropy_gated_rkl_probe_metrics,
    merge_entropy_gated_rkl_probe_stats_,
    validate_entropy_gated_rkl_composition,
)


def test_high_entropy_branch_matches_hard_label_ce_gradient():
    student = torch.tensor([[0.2, -0.1, 0.3, 0.0]], requires_grad=True)
    teacher = torch.zeros_like(student)
    labels = torch.tensor([2])
    terms = compute_entropy_gated_ce_rkl_terms(
        student,
        teacher,
        labels,
        entropy_threshold=0.1,
        rkl_cap=5.0,
    )
    terms["loss"].sum().backward()
    hybrid_grad = student.grad.detach().clone()

    reference = student.detach().clone().requires_grad_(True)
    F.cross_entropy(reference, labels).backward()
    assert not bool(terms["low_entropy_mask"].item())
    assert torch.allclose(hybrid_grad, reference.grad, atol=1e-7, rtol=1e-6)


def test_low_entropy_identical_distribution_has_zero_rkl_and_gradient():
    student = torch.tensor([[12.0, -12.0]], requires_grad=True)
    teacher = student.detach().clone()
    terms = compute_entropy_gated_ce_rkl_terms(
        student,
        teacher,
        torch.tensor([1]),
        entropy_threshold=0.1,
        rkl_cap=5.0,
    )
    terms["loss"].sum().backward()
    assert bool(terms["low_entropy_mask"].item())
    assert terms["loss"].item() == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(student.grad, torch.zeros_like(student.grad), atol=1e-7)


def test_low_entropy_large_rkl_is_hard_capped():
    student = torch.tensor([[20.0, -20.0]], requires_grad=True)
    teacher = torch.tensor([[-20.0, 20.0]])
    terms = compute_entropy_gated_ce_rkl_terms(
        student,
        teacher,
        torch.tensor([0]),
        entropy_threshold=0.1,
        rkl_cap=5.0,
    )
    terms["loss"].sum().backward()
    assert terms["rkl_raw"].item() == pytest.approx(40.0, rel=1e-5)
    assert terms["rkl_used"].item() == 5.0
    assert bool(terms["rkl_capped_mask"].item())
    assert torch.allclose(student.grad, torch.zeros_like(student.grad), atol=1e-7)


def test_uniform_teacher_routes_to_ce_and_identical_rkl_is_zero():
    logits = torch.zeros((3, 4), dtype=torch.float32)
    stats = compute_entropy_gated_rkl_probe_stats(
        logits,
        logits,
        entropy_thresholds=(0.1,),
    )
    metrics = entropy_gated_rkl_probe_metrics(
        stats,
        entropy_thresholds=(0.1,),
    )
    assert metrics["entropy_gate_rkl_probe/teacher_entropy_mean"] == pytest.approx(1.0)
    assert metrics["entropy_gate_rkl_probe/rkl_all_mean"] == pytest.approx(0.0)
    assert metrics["entropy_gate_rkl_probe/threshold_0p100_rkl_tokens"] == 0.0


def test_disjoint_low_entropy_modes_expose_large_reverse_kl():
    student = torch.tensor([[20.0, -20.0]], dtype=torch.float32)
    teacher = torch.tensor([[-20.0, 20.0]], dtype=torch.float32)
    stats = compute_entropy_gated_rkl_probe_stats(
        student,
        teacher,
        entropy_thresholds=(0.1,),
    )
    metrics = entropy_gated_rkl_probe_metrics(
        stats,
        entropy_thresholds=(0.1,),
    )
    assert metrics["entropy_gate_rkl_probe/teacher_entropy_mean"] < 1e-4
    assert metrics["entropy_gate_rkl_probe/threshold_0p100_rkl_tokens"] == 1.0
    assert metrics["entropy_gate_rkl_probe/threshold_0p100_rkl_mean"] == pytest.approx(
        40.0, rel=1e-5
    )
    assert metrics["entropy_gate_rkl_probe/threshold_0p100_rkl_max"] == pytest.approx(
        40.0, rel=1e-5
    )
    assert metrics[
        "entropy_gate_rkl_probe/threshold_0p100_top1_mismatch_frac"
    ] == 1.0
    assert metrics["entropy_gate_rkl_probe/threshold_0p100_rkl_gt_20_frac"] == 1.0


def test_absolute_normalized_threshold_has_expected_effective_support():
    # Uniform over three tokens has H/log(V)=1 for V=3, so it is above 0.1.
    teacher = torch.zeros((1, 3), dtype=torch.float32)
    student = teacher.clone()
    stats = compute_entropy_gated_rkl_probe_stats(
        student,
        teacher,
        entropy_thresholds=(0.1, 1.0),
    )
    assert stats["threshold_count"].tolist() == [0.0, 1.0]
    assert math.exp(0.1 * math.log(151936)) == pytest.approx(3.297, rel=1e-3)


def _reference_non_cad_terms(
    student_logits, teacher_logits, labels, *, entropy_threshold, rkl_cap, logit_clip=50.0
):
    # Op-for-op copy of the pre-CAD hybrid loss for bit-identity checks.
    vocab_size = int(student_logits.shape[-1])
    student = torch.nan_to_num(
        student_logits.float(), nan=0.0, posinf=logit_clip, neginf=-logit_clip
    ).clamp(min=-logit_clip, max=logit_clip)
    teacher = torch.nan_to_num(
        teacher_logits.detach().float(), nan=0.0, posinf=logit_clip, neginf=-logit_clip
    ).clamp(min=-logit_clip, max=logit_clip)
    student_logp = F.log_softmax(student, dim=-1)
    teacher_logp = F.log_softmax(teacher, dim=-1)
    student_prob = student_logp.exp()
    teacher_prob = teacher_logp.exp()
    teacher_entropy = -(teacher_prob * teacher_logp).sum(dim=-1) / math.log(vocab_size)
    low_entropy_mask = teacher_entropy.detach().le(float(entropy_threshold))
    ce = F.cross_entropy(student, labels.long(), reduction="none")
    rkl_raw = (student_prob * (student_logp - teacher_logp)).sum(dim=-1)
    rkl_used = rkl_raw.clamp(min=0.0, max=float(rkl_cap))
    loss = torch.where(low_entropy_mask, rkl_used, ce)
    return {
        "loss": loss,
        "ce": ce,
        "rkl_raw": rkl_raw,
        "rkl_used": rkl_used,
        "low_entropy_mask": low_entropy_mask,
    }


def _manual_softmax(row):
    peak = max(row)
    exps = [math.exp(value - peak) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def test_cad_mode_disabled_is_bit_identical_to_reference():
    torch.manual_seed(20260822)
    student = torch.randn(16, 8)
    teacher = torch.randn(16, 8)
    # Sharpen half the teacher rows so both routing branches are populated.
    teacher[::2] *= 30.0
    labels = torch.randint(0, 8, (16,))
    terms = compute_entropy_gated_ce_rkl_terms(
        student,
        teacher,
        labels,
        entropy_threshold=0.1,
        rkl_cap=5.0,
        cad_mode=False,
        fkl_coef=0.7,
        cj_power=2.0,
    )
    reference = _reference_non_cad_terms(
        student, teacher, labels, entropy_threshold=0.1, rkl_cap=5.0
    )
    assert bool(terms["low_entropy_mask"].any())
    assert bool(terms["high_entropy_mask"].any())
    for key in ("loss", "ce", "rkl_raw", "rkl_used"):
        assert torch.equal(terms[key], reference[key])
    assert torch.equal(terms["low_entropy_mask"], reference["low_entropy_mask"])
    assert "fkl" not in terms and "cj" not in terms


def test_cad_mode_matches_hand_computed_tiny_example():
    student = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0], [1.0, 2.0, 0.5, -0.5]], requires_grad=True
    )
    teacher = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0, 2])
    fkl_coef, cj_power = 0.7, 2.0
    terms = compute_entropy_gated_ce_rkl_terms(
        student,
        teacher,
        labels,
        entropy_threshold=0.1,
        rkl_cap=5.0,
        cad_mode=True,
        fkl_coef=fkl_coef,
        cj_power=cj_power,
    )
    assert terms["low_entropy_mask"].tolist() == [True, False]

    q = _manual_softmax([10.0, 0.0, 0.0, 0.0])
    p = _manual_softmax([2.0, 1.0, 0.0, -1.0])
    rkl = sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q))
    rkl_used = min(max(rkl, 0.0), 5.0)
    fkl = sum(qi * (math.log(qi) - math.log(pi)) for pi, qi in zip(p, q))
    cj = max(q) ** cj_power
    assert rkl_used < 5.0
    assert terms["fkl"][0].item() == pytest.approx(fkl, abs=1e-5)
    assert terms["cj"][0].item() == pytest.approx(cj, abs=1e-5)
    assert terms["rkl_used"][0].item() == pytest.approx(rkl_used, abs=1e-5)
    assert terms["loss"][0].item() == pytest.approx(
        fkl_coef * fkl + cj * rkl_used, abs=1e-5
    )

    p_high = _manual_softmax([1.0, 2.0, 0.5, -0.5])
    assert terms["loss"][1].item() == pytest.approx(-math.log(p_high[2]), abs=1e-5)


def test_cad_mode_still_hard_caps_rkl():
    student = torch.tensor([[20.0, -20.0]], requires_grad=True)
    teacher = torch.tensor([[-20.0, 20.0]])
    fkl_coef = 0.5
    terms = compute_entropy_gated_ce_rkl_terms(
        student,
        teacher,
        torch.tensor([0]),
        entropy_threshold=0.1,
        rkl_cap=5.0,
        cad_mode=True,
        fkl_coef=fkl_coef,
        cj_power=1.0,
    )
    assert bool(terms["low_entropy_mask"].item())
    assert terms["rkl_raw"].item() == pytest.approx(40.0, rel=1e-5)
    assert terms["rkl_used"].item() == 5.0
    assert bool(terms["rkl_capped_mask"].item())
    assert terms["fkl"].item() == pytest.approx(40.0, abs=1e-4)
    assert terms["cj"].item() == pytest.approx(1.0, abs=1e-6)
    assert terms["loss"].item() == pytest.approx(
        fkl_coef * terms["fkl"].item() + terms["cj"].item() * 5.0, abs=1e-5
    )


def test_cad_mode_leaves_high_entropy_tokens_unchanged():
    torch.manual_seed(20260823)
    student = torch.randn(12, 6)
    teacher = torch.randn(12, 6)
    teacher[::3] *= 30.0
    labels = torch.randint(0, 6, (12,))
    kwargs = dict(entropy_threshold=0.1, rkl_cap=5.0)
    baseline = compute_entropy_gated_ce_rkl_terms(
        student, teacher, labels, cad_mode=False, **kwargs
    )
    cad = compute_entropy_gated_ce_rkl_terms(
        student, teacher, labels, cad_mode=True, fkl_coef=0.7, cj_power=2.0, **kwargs
    )
    high = baseline["high_entropy_mask"]
    assert bool(high.any()) and bool((~high).any())
    assert torch.equal(cad["high_entropy_mask"], high)
    assert torch.equal(cad["loss"][high], baseline["loss"][high])
    assert torch.equal(cad["ce"], baseline["ce"])
    assert not torch.equal(cad["loss"][~high], baseline["loss"][~high])


def test_cad_mode_rejects_invalid_coefficients():
    student = torch.zeros((1, 2))
    teacher = torch.zeros((1, 2))
    labels = torch.tensor([0])
    with pytest.raises(ValueError, match="fkl_coef"):
        compute_entropy_gated_ce_rkl_terms(
            student, teacher, labels,
            entropy_threshold=0.1, rkl_cap=5.0, cad_mode=True, fkl_coef=-1.0,
        )
    with pytest.raises(ValueError, match="cj_power"):
        compute_entropy_gated_ce_rkl_terms(
            student, teacher, labels,
            entropy_threshold=0.1, rkl_cap=5.0, cad_mode=True, cj_power=float("nan"),
        )


def test_aec_composition_guard_default_raises_and_opt_in_passes():
    # Mirror dp_actor's update_policy parsing of the guard-relevant config.
    entropy_gated_rkl_config = {"enable": True}
    base_kwargs = dict(
        negative_action_pg_update=False,
        signed_trace_ce_update=False,
        causal_wrong_ce_update=False,
        block_tpf_enable=False,
        response_tpf_ce_scale_enable=False,
        selected_vocab_size=0,
        entropy_coef=0.0,
    )
    with pytest.raises(ValueError, match="without AEC"):
        validate_entropy_gated_rkl_composition(
            adj_ent_enable=True,
            allow_aec_composition=bool(
                entropy_gated_rkl_config.get("allow_aec_composition", False)
            ),
            **base_kwargs,
        )

    entropy_gated_rkl_config["allow_aec_composition"] = True
    validate_entropy_gated_rkl_composition(
        adj_ent_enable=True,
        allow_aec_composition=bool(
            entropy_gated_rkl_config.get("allow_aec_composition", False)
        ),
        **base_kwargs,
    )

    # Without AEC the guard passes regardless of the flag.
    validate_entropy_gated_rkl_composition(
        adj_ent_enable=False, allow_aec_composition=False, **base_kwargs
    )

    # The flag does not unlock any other incompatible objective.
    with pytest.raises(ValueError, match="entropy_gated_rkl requires"):
        validate_entropy_gated_rkl_composition(
            adj_ent_enable=True,
            allow_aec_composition=True,
            **{**base_kwargs, "signed_trace_ce_update": True},
        )


def test_weighted_report_and_parity_match_hand_computed_sums():
    # 2 responses with unequal lengths and unequal per-token weights:
    # response A = tokens 0-2 (w=1/3 each), response B = tokens 3-4 (w=0.7).
    student = torch.tensor(
        [
            [2.0, 1.0, 0.0, -1.0],
            [0.5, -0.5, 1.5, 0.0],
            [1.0, 1.0, -2.0, 0.3],
            [3.0, -1.0, 0.2, 0.0],
            [-0.5, 2.0, 0.0, 1.0],
        ]
    )
    teacher = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 12.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 2, 1, 3, 2])
    weights = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.7, 0.7])
    for cad_mode in (False, True):
        kwargs = dict(entropy_threshold=0.1, rkl_cap=5.0, cad_mode=cad_mode)
        if cad_mode:
            kwargs.update(fkl_coef=0.7, cj_power=2.0)
        terms = compute_entropy_gated_ce_rkl_terms(
            student, teacher, labels, **kwargs
        )
        assert terms["low_entropy_mask"].tolist() == [
            True, False, True, False, True,
        ]
        assert bool(terms["rkl_capped_mask"].any())

        # Accumulate in two chunks, mirroring the streaming chunk loop.
        report = torch.zeros(
            ENTROPY_GATED_RKL_REPORT_SLOTS, dtype=torch.float64
        )
        for chunk in (slice(0, 3), slice(3, 5)):
            chunk_terms = compute_entropy_gated_ce_rkl_terms(
                student[chunk], teacher[chunk], labels[chunk], **kwargs
            )
            accumulate_entropy_gated_rkl_report_(
                report,
                chunk_terms,
                cad_mode=cad_mode,
                token_weights=weights[chunk],
            )

        loss_values = terms["loss"].tolist()
        weight_values = weights.tolist()
        expected_weighted = sum(
            w * l for w, l in zip(weight_values, loss_values)
        )
        expected_weight_sum = sum(weight_values)
        parity = entropy_gated_rkl_parity_terms(
            report, cad_mode=cad_mode, weighted=True
        )
        assert parity["total_count"] == 5.0
        assert parity["weight_sum"] == pytest.approx(
            expected_weight_sum, abs=1e-9
        )
        assert parity["numerator"] == pytest.approx(expected_weighted, abs=1e-6)
        assert parity["denominator"] == pytest.approx(
            expected_weight_sum, abs=1e-9
        )
        assert parity["loss"] == pytest.approx(
            expected_weighted / expected_weight_sum, abs=1e-6
        )

        # The same report still recomposes the unweighted semantics.
        unweighted = entropy_gated_rkl_parity_terms(
            report, cad_mode=cad_mode, weighted=False
        )
        loss_sum = sum(loss_values)
        assert unweighted["denominator"] == 5.0
        assert unweighted["numerator"] == pytest.approx(loss_sum, abs=1e-6)
        if not cad_mode:
            high = terms["high_entropy_mask"]
            decomposed = float(
                terms["ce"][high].sum() + terms["rkl_used"][~high].sum()
            )
            assert unweighted["numerator"] == pytest.approx(
                decomposed, abs=1e-6
            )

        # Unweighted accumulation fills the w=1 equivalents (shared schema).
        report_unweighted = torch.zeros(
            ENTROPY_GATED_RKL_REPORT_SLOTS, dtype=torch.float64
        )
        accumulate_entropy_gated_rkl_report_(
            report_unweighted, terms, cad_mode=cad_mode, token_weights=None
        )
        assert report_unweighted[9].item() == pytest.approx(
            report_unweighted[6].item(), abs=1e-9
        )
        assert report_unweighted[10].item() == 5.0


def test_response_level_ce_guard_term_default_blocks_and_flag_unlocks():
    entropy_gated_rkl_config = {"enable": True}
    assert entropy_gated_rkl_blocks_response_level_ce(
        entropy_gated_rkl_enable=True,
        allow_aec_composition=bool(
            entropy_gated_rkl_config.get("allow_aec_composition", False)
        ),
    )
    entropy_gated_rkl_config["allow_aec_composition"] = True
    assert not entropy_gated_rkl_blocks_response_level_ce(
        entropy_gated_rkl_enable=True,
        allow_aec_composition=bool(
            entropy_gated_rkl_config.get("allow_aec_composition", False)
        ),
    )
    assert not entropy_gated_rkl_blocks_response_level_ce(
        entropy_gated_rkl_enable=False,
        allow_aec_composition=False,
    )


def test_merge_sums_histograms_and_takes_maximum():
    student_a = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    teacher_a = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    student_b = torch.tensor([[20.0, -20.0]], dtype=torch.float32)
    teacher_b = torch.tensor([[-20.0, 20.0]], dtype=torch.float32)
    first = compute_entropy_gated_rkl_probe_stats(
        student_a,
        teacher_a,
        eligible_count=2,
        entropy_thresholds=(0.6,),
    )
    second = compute_entropy_gated_rkl_probe_stats(
        student_b,
        teacher_b,
        entropy_thresholds=(0.6,),
    )
    merged = merge_entropy_gated_rkl_probe_stats_(None, first)
    merge_entropy_gated_rkl_probe_stats_(merged, second)
    assert merged["base"][0].item() == 3.0
    assert merged["base"][1].item() == 2.0
    assert merged["base"][2].item() == 1.0
    assert merged["threshold_count"].item() == 2.0
    assert merged["threshold_rkl_hist"].sum().item() == 2.0
    assert merged["threshold_rkl_max"].item() == pytest.approx(40.0, rel=1e-5)
