"""Deterministic rollout-quality rules independent of answer correctness."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Sequence


_LEXICAL_RE = re.compile(r"[A-Za-z]+|[0-9]+")


@dataclass(frozen=True)
class RuleQualityVerdict:
    good: bool
    reasons: tuple[str, ...]


def normalize_finish_reason(value: Any) -> str | None:
    """Return a lowercase finish-reason type from SGLang-style metadata."""

    current = value
    if isinstance(current, Mapping):
        current = current.get("finish_reason", current)
    if isinstance(current, Mapping):
        current = current.get("type", current.get("value"))
    enum_value = getattr(current, "value", None)
    if enum_value is not None:
        current = enum_value
    if current is None:
        return None
    text = str(current).strip().lower()
    if not text:
        return None
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _has_repeated_nonspace_character_run(
    text: str,
    *,
    max_run: int,
) -> bool:
    if max_run <= 0:
        return False
    previous: str | None = None
    run = 0
    for character in text:
        if character.isspace():
            continue
        if character.isalpha():
            previous = None
            run = 0
            continue
        if character == previous:
            run += 1
        else:
            previous = character
            run = 1
        if run >= max_run:
            return True
    return False


def _lexical_words(text: str) -> list[str]:
    return [word.lower() for word in _LEXICAL_RE.findall(text)]


def _count_answer_markers(text: str) -> int:
    count = 0
    for match in re.finditer(r"(?<!#)####(?!#)", text or ""):
        line_start = (text or "").rfind("\n", 0, match.start()) + 1
        prefix = (text or "")[line_start : match.start()]
        suffix = (text or "")[match.end() :].split("\n", 1)[0].strip()
        if not prefix.strip() and suffix and not re.match(
            r"(?:\\?boxed\b|\$|[-+]?(?:\d|\.\d))",
            suffix,
            flags=re.IGNORECASE,
        ):
            # Markdown H4 headings such as ``#### Verification`` are not GSM8K
            # answer delimiters. Quoted/instructional markers remain countable.
            continue
        count += 1
    return count


def _nonoverlapping_ngram_count(
    words: Sequence[str],
    ngram: tuple[str, ...],
) -> int:
    width = len(ngram)
    count = 0
    index = 0
    while index + width <= len(words):
        if tuple(words[index : index + width]) == ngram:
            count += 1
            index += width
        else:
            index += 1
    return count


def _has_repeated_ngram(
    words: Sequence[str],
    *,
    ngram_size: int,
    max_occurrences: int,
) -> bool:
    if ngram_size <= 0 or len(words) < ngram_size * (max_occurrences + 1):
        return False
    candidates = Counter(
        tuple(words[index : index + ngram_size])
        for index in range(len(words) - ngram_size + 1)
    )
    return any(
        raw_count > max_occurrences
        and _nonoverlapping_ngram_count(words, ngram) > max_occurrences
        for ngram, raw_count in candidates.items()
    )


def _reasoning_segments(text: str, min_words: int) -> list[tuple[str, ...]]:
    raw_segments = re.split(r"\n+|(?<=[.!?])\s+", text)
    return [
        tuple(words)
        for segment in raw_segments
        if len(words := _lexical_words(segment)) >= min_words
    ]


def _has_near_duplicate_segment_cycle(
    segments: Sequence[tuple[str, ...]],
    *,
    similarity: float,
    occurrences: int,
) -> bool:
    for anchor_index, anchor in enumerate(segments):
        matches = 1
        anchor_text = " ".join(anchor)
        for candidate in segments[anchor_index + 1 :]:
            length_ratio = min(len(anchor), len(candidate)) / max(
                len(anchor), len(candidate)
            )
            if length_ratio < similarity:
                continue
            ratio = SequenceMatcher(
                None,
                anchor_text,
                " ".join(candidate),
                autojunk=False,
            ).ratio()
            if ratio >= similarity:
                matches += 1
                if matches >= occurrences:
                    return True
    return False


def judge_rollout_quality_rules(
    text: str,
    *,
    max_word_run: int = 3,
    min_unique_word_ratio: float = 0.16,
    max_answer_markers: int = 1,
    long_ngram_size: int = 12,
    max_long_ngram_occurrences: int = 1,
    short_ngram_size: int = 8,
    max_short_ngram_occurrences: int = 2,
    min_segment_words: int = 10,
    near_duplicate_similarity: float = 0.9,
    near_duplicate_occurrences: int = 3,
    max_boxed_markers: int = 3,
    max_repeated_character_run: int = 32,
    max_whitespace_run: int = 64,
    tail_window_chars: int = 512,
    min_tail_nonspace_chars: int = 128,
    min_tail_unique_character_ratio: float = 0.02,
) -> RuleQualityVerdict:
    """Reject empty, looping, duplicated, or answer-cycling rollouts."""

    reasons: list[str] = []
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return RuleQualityVerdict(False, ("empty",))

    if (
        re.search(r"([A-Za-z]{4,})(?=\1)", compact, flags=re.IGNORECASE)
        or re.search(
            r"([A-Za-z]{3,})(?=\1\1)",
            compact,
            flags=re.IGNORECASE,
        )
    ):
        reasons.append("concatenated_word")

    words = _lexical_words(compact)
    run = 1
    for previous, current in zip(words, words[1:], strict=False):
        if current == previous:
            run += 1
            if run >= max_word_run:
                reasons.append("word_run")
                break
        else:
            run = 1

    if len(words) >= 16:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < min_unique_word_ratio:
            reasons.append("low_lexical_diversity")

    if _count_answer_markers(text or "") > max_answer_markers:
        reasons.append("answer_cycle")

    if len(re.findall(r"\\?boxed", text or "", flags=re.IGNORECASE)) > max_boxed_markers:
        reasons.append("boxed_cycle")

    if max_whitespace_run > 0 and re.search(
        rf"\s{{{max_whitespace_run},}}",
        text or "",
    ):
        reasons.append("whitespace_flood")

    if _has_repeated_nonspace_character_run(
        text or "",
        max_run=max_repeated_character_run,
    ):
        reasons.append("character_flood")

    if (
        tail_window_chars > 0
        and min_tail_nonspace_chars > 0
        and min_tail_unique_character_ratio >= 0.0
    ):
        tail = (text or "")[-tail_window_chars:]
        tail_nonspace = [character for character in tail if not character.isspace()]
        if (
            len(tail_nonspace) >= min_tail_nonspace_chars
            and len(set(tail_nonspace)) / len(tail_nonspace)
            < min_tail_unique_character_ratio
        ):
            reasons.append("low_diversity_tail")

    if _has_repeated_ngram(
        words,
        ngram_size=long_ngram_size,
        max_occurrences=max_long_ngram_occurrences,
    ):
        reasons.append("repeated_long_ngram")
    elif _has_repeated_ngram(
        words,
        ngram_size=short_ngram_size,
        max_occurrences=max_short_ngram_occurrences,
    ):
        reasons.append("repeated_short_ngram")

    segments = _reasoning_segments(text, min_segment_words)
    segment_counts = Counter(segments)
    if any(count >= 2 for count in segment_counts.values()):
        reasons.append("duplicate_reasoning_segment")
    elif _has_near_duplicate_segment_cycle(
        segments,
        similarity=near_duplicate_similarity,
        occurrences=near_duplicate_occurrences,
    ):
        reasons.append("near_duplicate_reasoning_cycle")

    return RuleQualityVerdict(not reasons, tuple(dict.fromkeys(reasons)))


__all__ = [
    "RuleQualityVerdict",
    "judge_rollout_quality_rules",
    "normalize_finish_reason",
]
