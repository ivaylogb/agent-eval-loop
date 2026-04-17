"""Calibration: measure agreement between LLM judges and human annotators.

A judge is only trustworthy if it correlates with human judgment.
This module provides tools to:
1. Compare judge verdicts against human-annotated ground truth
2. Compute inter-rater agreement (Cohen's kappa)
3. Identify categories where judges are miscalibrated
4. Track calibration over time as rubrics are refined
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_eval_loop.models import EvalCategory, JudgeVerdict


@dataclass
class HumanAnnotation:
    """A human annotator's verdict on a conversation."""
    conversation_id: str
    category: EvalCategory
    passed: bool
    score: float
    annotator_id: str
    reasoning: str = ""


@dataclass
class CalibrationResult:
    """Agreement metrics between a judge and human annotators."""
    category: EvalCategory
    cohens_kappa: float  # -1 to 1, >0.6 is substantial agreement
    accuracy: float  # simple agreement rate
    false_positive_rate: float  # judge says pass, human says fail
    false_negative_rate: float  # judge says fail, human says pass
    score_correlation: float  # Pearson correlation of scores
    n_samples: int

    @property
    def is_calibrated(self) -> bool:
        """A judge is considered calibrated if kappa > 0.6."""
        return self.cohens_kappa > 0.6


@dataclass
class CalibrationReport:
    """Full calibration report across all categories."""
    results: dict[EvalCategory, CalibrationResult] = field(default_factory=dict)

    @property
    def all_calibrated(self) -> bool:
        return all(r.is_calibrated for r in self.results.values())

    @property
    def weakest_category(self) -> EvalCategory | None:
        if not self.results:
            return None
        return min(self.results, key=lambda c: self.results[c].cohens_kappa)

    def summary(self) -> str:
        lines = ["Calibration Report", "=" * 50]
        for cat, result in self.results.items():
            status = "✓" if result.is_calibrated else "✗"
            lines.append(
                f"{status} {cat.value}: κ={result.cohens_kappa:.3f}, "
                f"acc={result.accuracy:.1%}, n={result.n_samples}"
            )
        return "\n".join(lines)


def compute_calibration(
    judge_verdicts: list[JudgeVerdict],
    human_annotations: list[HumanAnnotation],
    conversation_ids: list[str],
) -> CalibrationReport:
    """Compute calibration metrics between judge verdicts and human annotations.

    Pairs up judge verdicts with human annotations by (conversation_id, category)
    and computes agreement metrics.
    """
    # Index human annotations
    human_index: dict[tuple[str, EvalCategory], HumanAnnotation] = {}
    for ann in human_annotations:
        human_index[(ann.conversation_id, ann.category)] = ann

    # Index judge verdicts
    judge_index: dict[tuple[str, EvalCategory], JudgeVerdict] = {}
    for conv_id, verdict in zip(conversation_ids, judge_verdicts):
        judge_index[(conv_id, verdict.category)] = verdict

    # Group by category
    from collections import defaultdict
    pairs_by_category: dict[EvalCategory, list[tuple[bool, bool, float, float]]] = defaultdict(list)

    for key in human_index:
        if key in judge_index:
            h = human_index[key]
            j = judge_index[key]
            pairs_by_category[h.category].append(
                (j.passed, h.passed, j.score, h.score)
            )

    report = CalibrationReport()
    for category, pairs in pairs_by_category.items():
        if len(pairs) < 5:
            continue  # Not enough samples

        judge_pass = [p[0] for p in pairs]
        human_pass = [p[1] for p in pairs]
        judge_scores = [p[2] for p in pairs]
        human_scores = [p[3] for p in pairs]

        report.results[category] = CalibrationResult(
            category=category,
            cohens_kappa=_cohens_kappa(judge_pass, human_pass),
            accuracy=_accuracy(judge_pass, human_pass),
            false_positive_rate=_false_positive_rate(judge_pass, human_pass),
            false_negative_rate=_false_negative_rate(judge_pass, human_pass),
            score_correlation=_pearson_correlation(judge_scores, human_scores),
            n_samples=len(pairs),
        )

    return report


def _cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Compute Cohen's kappa for two binary raters."""
    n = len(a)
    if n == 0:
        return 0.0

    agree = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agree / n  # observed agreement

    # Expected agreement by chance
    a_pos = sum(a) / n
    b_pos = sum(b) / n
    p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def _accuracy(predicted: list[bool], actual: list[bool]) -> float:
    if not predicted:
        return 0.0
    return sum(1 for p, a in zip(predicted, actual) if p == a) / len(predicted)


def _false_positive_rate(predicted: list[bool], actual: list[bool]) -> float:
    negatives = [(p, a) for p, a in zip(predicted, actual) if not a]
    if not negatives:
        return 0.0
    return sum(1 for p, _ in negatives if p) / len(negatives)


def _false_negative_rate(predicted: list[bool], actual: list[bool]) -> float:
    positives = [(p, a) for p, a in zip(predicted, actual) if a]
    if not positives:
        return 0.0
    return sum(1 for p, _ in positives if not p) / len(positives)


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)
