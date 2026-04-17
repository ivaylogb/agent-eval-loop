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

from agent_eval_loop.models import Conversation, ConversationEval, EvalCategory


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
    baseline_conversations: list[Conversation],
    baseline_results: list[ConversationEval],
    candidate_conversations: list[Conversation],
    candidate_results: list[ConversationEval],
    regression_tolerance: float = 0.05,
) -> RegressionResult:
    """Compare candidate eval results against a baseline.

    A regression is defined as: a conversation that passed in baseline
    now fails in candidate, OR a score drop greater than regression_tolerance.

    Conversations are matched across runs by (persona_id, scenario_id), not
    by conversation_id — each simulation run produces fresh UUIDs, so keying
    by ID would yield an empty intersection and the gate would never fire.

    Args:
        baseline_conversations: Conversations that produced baseline_results
        baseline_results: Eval results from the current agent version
        candidate_conversations: Conversations that produced candidate_results
        candidate_results: Eval results from the candidate version
        regression_tolerance: Maximum acceptable score drop per conversation
    """
    baseline_index = _index_by_persona_scenario(baseline_conversations, baseline_results)
    candidate_index = _index_by_persona_scenario(candidate_conversations, candidate_results)

    regressions = []
    improvements = []
    unchanged = 0

    common_keys = set(baseline_index) & set(candidate_index)

    for key in common_keys:
        base_conv_id, base = baseline_index[key]
        cand_conv_id, cand = candidate_index[key]

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
                regressions.append(Regression(
                    conversation_id=cand_conv_id,
                    category=category,
                    score_before=bv.score,
                    score_after=cv.score,
                ))
                conv_changed = True
            elif score_delta < -regression_tolerance:
                regressions.append(Regression(
                    conversation_id=cand_conv_id,
                    category=category,
                    score_before=bv.score,
                    score_after=cv.score,
                ))
                conv_changed = True
            elif not bv.passed and cv.passed:
                improvements.append(Improvement(
                    conversation_id=cand_conv_id,
                    category=category,
                    score_before=bv.score,
                    score_after=cv.score,
                ))
                conv_changed = True

        if not conv_changed:
            unchanged += 1

    # Regression test passes only when (1) we actually compared something, and
    # (2) nothing regressed. An empty intersection means we have no evidence
    # the candidate is safe — fail closed rather than endorse a no-op sample.
    passed = len(regressions) == 0 and len(common_keys) > 0

    summary_parts = [
        f"{'PASSED' if passed else 'FAILED'}: ",
        f"{len(improvements)} improvements, ",
        f"{len(regressions)} regressions, ",
        f"{unchanged} unchanged",
    ]
    if not common_keys:
        summary_parts.append(
            " (no baseline/candidate overlap — cannot verify safety)"
        )

    return RegressionResult(
        passed=passed,
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
        summary="".join(summary_parts),
    )


def _index_by_persona_scenario(
    conversations: list[Conversation],
    results: list[ConversationEval],
) -> dict[tuple[str, str], tuple[str, ConversationEval]]:
    """Index eval results by (persona_id, scenario_id) via conversation lookup.

    Returns {(persona_id, scenario_id): (conversation_id, eval)}. If the same
    (persona, scenario) pair appears more than once (e.g. repeated trials),
    the last occurrence wins — callers that want repeats should pre-group.
    """
    conv_by_id = {c.id: c for c in conversations}
    index: dict[tuple[str, str], tuple[str, ConversationEval]] = {}
    for result in results:
        conv = conv_by_id.get(result.conversation_id)
        if conv is None:
            continue
        index[(conv.persona_id, conv.scenario_id)] = (result.conversation_id, result)
    return index
