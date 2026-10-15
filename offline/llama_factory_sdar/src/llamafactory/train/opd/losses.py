import torch
import torch.nn.functional as F


def topk_distillation_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    mode: str = "opdlm_forward_kl",
) -> torch.Tensor:
    """Compute token-wise sparse distillation loss on the teacher top-k support.

    Returns an unreduced loss with shape `student_logits.shape[:-1]`.

    Modes:
      - opdlm_forward_kl: OPDLM/Nemotron-style sparse forward KL using the
        teacher's original top-k log-probability mass, without top-k
        renormalization.
      - forward_kl: renormalized KL on the teacher top-k support.
      - jsd: renormalized Jensen-Shannon divergence on the teacher top-k support.
    """

    if teacher_topk_ids.shape[:-1] != student_logits.shape[:-1]:
        raise ValueError("teacher_topk_ids must match student_logits without vocab dim plus top-k.")
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError("teacher_topk_ids and teacher_topk_logprobs must have the same shape.")
    if mode not in {"opdlm_forward_kl", "forward_kl", "jsd"}:
        raise ValueError(f"Unsupported distillation mode: {mode}")

    selected_student_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_ids)
    if mode == "opdlm_forward_kl":
        student_log_probs_full = F.log_softmax(student_logits.float(), dim=-1)
        selected_student_log_probs = torch.gather(student_log_probs_full, dim=-1, index=teacher_topk_ids)
        finite_teacher = torch.isfinite(teacher_topk_logprobs)
        terms = torch.where(
            finite_teacher,
            teacher_topk_logprobs.float().exp() * (teacher_topk_logprobs.float() - selected_student_log_probs),
            torch.zeros_like(teacher_topk_logprobs.float()),
        )
        return terms.sum(dim=-1)

    student_log_probs = F.log_softmax(selected_student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_topk_logprobs.float(), dim=-1)

    teacher_probs = teacher_log_probs.exp()
    if mode == "forward_kl":
        return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    student_probs = student_log_probs.exp()
    mixture_probs = 0.5 * (teacher_probs + student_probs)
    mixture_log_probs = mixture_probs.clamp_min(torch.finfo(mixture_probs.dtype).tiny).log()
    teacher_kl = (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)
    student_kl = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)
    return 0.5 * (teacher_kl + student_kl)
