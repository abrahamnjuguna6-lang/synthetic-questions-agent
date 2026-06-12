"""Rule-based quality checks for generated question batches.

Phase 1 (this module) — fast, no LLM:
  - count_check      : exact number of questions required
  - format_check     : each question ends with '?'
  - uniqueness_check : pairwise TF-IDF cosine similarity < threshold
  - relevance_check  : at least one topic keyword appears per question

Phase 2 (LLM scoring) is handled in agents/nodes.py using the
QUALITY_SCORING_PROMPT template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


def count_check(questions: list[dict], expected: int) -> ValidationResult:
    if len(questions) == expected:
        return ValidationResult(passed=True)
    return ValidationResult(
        passed=False,
        issues=[f"Expected {expected} questions, got {len(questions)}."],
    )


def format_check(questions: list[dict]) -> ValidationResult:
    issues = []
    for i, q in enumerate(questions):
        # Stem check
        text = q.get("question", "").strip()
        if not text.endswith("?"):
            issues.append(f"Question {i + 1} stem does not end with '?': {text[:80]!r}")

        # Options check — must be a dict with keys A, B, C, D
        options = q.get("options")
        if options is None:
            issues.append(f"Question {i + 1} is missing 'options'.")
        elif not isinstance(options, dict):
            issues.append(f"Question {i + 1} 'options' must be a dict, got {type(options).__name__}.")
        else:
            missing = [k for k in ("A", "B", "C", "D") if k not in options]
            if missing:
                issues.append(
                    f"Question {i + 1} 'options' is missing keys: {', '.join(missing)}."
                )
    return ValidationResult(passed=len(issues) == 0, issues=issues)


def uniqueness_check(
    questions: list[dict], similarity_threshold: float = 0.85
) -> ValidationResult:
    """Flag near-duplicate questions using TF-IDF cosine similarity."""
    texts = [q.get("question", "") for q in questions]
    if len(texts) < 2:
        return ValidationResult(passed=True)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

        issues = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                score = sim_matrix[i, j]
                if score >= similarity_threshold:
                    issues.append(
                        f"Questions {i + 1} and {j + 1} are too similar "
                        f"(similarity={score:.2f}): {texts[i][:60]!r}"
                    )
        return ValidationResult(passed=len(issues) == 0, issues=issues)

    except ImportError:
        # scikit-learn not available — skip this check
        return ValidationResult(passed=True)


_GENERIC_FALLBACK_WORDS = {"unknown", "topic", "general", "knowledge", "question"}


def relevance_check(
    questions: list[dict], topic: str, subtopics: list[str]
) -> ValidationResult:
    """Each question should contain at least one keyword from topic/subtopics.

    Skips gracefully when the topic resolved to a generic fallback (e.g.
    'Unknown Topic', 'General Knowledge') so a failed JSON parse in topic
    analysis doesn't cascade into a hard validation failure.
    """
    keywords = _extract_keywords(topic) | {
        kw for st in subtopics for kw in _extract_keywords(st)
    }

    # Remove generic/fallback words — they appear in every English sentence
    meaningful = keywords - _GENERIC_FALLBACK_WORDS
    if not meaningful:
        # Not enough signal to do relevance checking — skip rather than fail
        return ValidationResult(passed=True)

    issues = []
    for i, q in enumerate(questions):
        text = q.get("question", "").lower()
        if not any(kw in text for kw in meaningful):
            issues.append(
                f"Question {i + 1} appears unrelated to the topic: "
                f"{q.get('question', '')[:80]!r}"
            )
    return ValidationResult(passed=len(issues) == 0, issues=issues)


def run_all_checks(
    questions: list[dict],
    expected_count: int,
    topic: str,
    subtopics: list[str],
    similarity_threshold: float = 0.85,
) -> ValidationResult:
    """Run all rule-based checks and aggregate results.

    Note: relevance_check is intentionally excluded here. Clinical and domain
    questions rarely contain the literal topic name in the question stem, so
    keyword matching produces too many false positives. Relevance is evaluated
    by the LLM quality scoring phase instead.
    """
    all_issues: list[str] = []

    for check_fn, args in [
        (count_check, (questions, expected_count)),
        (format_check, (questions,)),
        (uniqueness_check, (questions, similarity_threshold)),
    ]:
        result = check_fn(*args)
        all_issues.extend(result.issues)

    return ValidationResult(passed=len(all_issues) == 0, issues=all_issues)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> set[str]:
    """Lower-case words ≥4 chars from a phrase — used for relevance matching."""
    return {w.lower() for w in re.findall(r"\b\w{4,}\b", text)}
