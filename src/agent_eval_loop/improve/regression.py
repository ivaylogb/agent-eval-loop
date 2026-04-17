"""Regression testing: verify improvements don't break passing cases.

Before accepting any improvement candidate, we re-run the full eval
suite to confirm that previously passing conversations still pass.
A candidate that improves one category but regresses another is rejected.

Best practice: accept ONLY candidates with zero regressions. A change
that fixes 5 failures but introduces 2 regressions is not an improvement —
it's a lateral move with hidden risk. This is strict by design. If the
optimizer can't find a fix that doesn't regress, the failure pattern
needs a human to look at it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_eval_loop.models import ConversationEval, EvalCategory


@dataclass
class RegressionResult:
    """Result of regression testing an improvement candidate."""
    passed: bool
    regressions: list[Regression]
    improvements: list[Improvement]
    unchanged: int
    summary: str


@dataclass
class Regression:
    """A specific regression: a conversation that was passing now fails."""
    conversation_id: str
    category: EvalCategory
    score_before: float
    score_after: float


@dataclass
class Improvement:
    """A specific improvement: a conversation that was failing now passes."""
    conversation_id: str
    category: EvalCategory
    score_before: float
    score_after: float


def check_regression(
    baseline_results: list[ConversationEval],
    candidate_results: list[ConversationEval],
    regression_tolerance: float = 0.05,
) -> RegressionResult:
    """Compare candidate eval results against a baseline.

    A regression is defined as: a conversation that passed in baseline
    now fails in candidate, OR a score drop greater than regression_tolerance.

    Args:
        baseline_results: Eval results from the current agent version
        candidate_results: Eval results from the candidate version
        regression_tolerance: Maximum acceptable score drop per conversation
    """
    # Index by conversation ID
    baseline_index = {r.conversation_id: r for r in baseline_results}
    candidate_index = {r.conversation_id: r for r in candidate_results}

    regressions = []
    improvements = []
    unchanged = 0

    # Check each conversation that appears in both sets
    common_ids = set(baseline_index) & set(candidate_index)

    for conv_id in common_ids:
        base = baseline_index[conv_id]
        cand = candidate_index[conv_id]

        # Compare per-category verdicts
        base_verdicts = {v.category: v for v in base.verdicts}
        cand_verdicts = {v.category: v for v in cand.verdicts}

        conv_changed = False
        for category in base_verdicts:
            if category not in cand_verdicts:
                continue

            bv = base_verdicts[category]
            cv = cand_verdicts[category]

            score_delta = cv.score - bv.score

            if bv.passed and not cv.passed:
                # Hard regression: was passing, now failing
                regressions.append(Regression(
                    conversation_id=conv_id,
                    category=category,
                    score_before=bv.score,
                    score_after=cv.score,
                ))
                conv_changed = True
            elif score_delta < -regression_tolerance:
                # Soft regression: significant score drop
                regressions.append(Regression(
                    conversation_id=conv_id,
                    category=category,
                    score_before=bv.score,
                    score_after=cv.score,
                ))
                conv_changed = True
            elif not bv.passed and cv.passed:
                # Improvement
                improvements.append(Improvement(
                    conversation_id=conv_id,
                    category=category,
                    score_before=bv.score,
                    score_after=cv.score,
                ))
                conv_changed = True

        if not conv_changed:
            unchanged += 1

    # Regression test passes if there are zero regressions
    passed = len(regressions) == 0

    summary_parts = [
        f"{'PASSED' if passed else 'FAILED'}: ",
        f"{len(improvements)} improvements, ",
        f"{len(regressions)} regressions, ",
        f"{unchanged} unchanged",
    ]

    return RegressionResult(
        passed=passed,
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
        summary="".join(summary_parts),
    )
