"""Optimizer: propose and validate improvements to agent components.

Takes failure patterns from the analyzer and proposes targeted edits.
Each proposal includes the full revised content so the loop can apply
it, run regression tests, and decide whether to keep it.
"""

from __future__ import annotations

from pathlib import Path

import anthropic

from agent_eval_loop.agent.config import load_config
from agent_eval_loop.models import (
    AgentConfig,
    ComponentType,
    ComponentVersion,
    FailurePattern,
    ImprovementCandidate,
)


class Optimizer:
    """Propose improvements to agent components based on failure patterns."""

    def __init__(
        self,
        agent_config: AgentConfig,
        model: str = "claude-sonnet-4-20250514",
        client: anthropic.Anthropic | None = None,
    ):
        self.agent_config = agent_config
        self.model = model
        self.client = client or anthropic.Anthropic()

    def propose_improvements(
        self,
        failure_patterns: list[FailurePattern],
        max_proposals: int = 3,
    ) -> list[ImprovementCandidate]:
        """Generate improvement proposals for the highest-frequency failures."""
        candidates = []
        sorted_patterns = sorted(failure_patterns, key=lambda p: p.frequency, reverse=True)

        for pattern in sorted_patterns[:max_proposals]:
            candidate = self._propose_for_pattern(pattern)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _propose_for_pattern(self, pattern: FailurePattern) -> ImprovementCandidate | None:
        """Use an LLM to propose a specific edit for a failure pattern.

        Returns a candidate with the full proposed_content included.
        """
        component = self.agent_config.components.get(pattern.component)
        if not component or not component.content:
            return None

        prompt = f"""You are an expert prompt engineer improving an AI agent.

The agent has a recurring failure pattern:
- Category: {pattern.category.value}
- Component: {pattern.component.value}
- Frequency: {pattern.frequency} occurrences
- Description: {pattern.description}
- Suggested direction: {pattern.suggested_fix}

Here is the current version of the {pattern.component.value} component:

<current_component>
{component.content}
</current_component>

Propose a revised version that addresses the failure pattern.
Requirements:
1. Fix the identified issue
2. Do NOT remove or weaken any existing correct behavior
3. Be as minimal as possible — change only what's needed
4. Add a brief comment at the top: "# Changed: [what changed and why]"

Return ONLY the revised component text, nothing else."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        proposed_content = ""
        for block in response.content:
            if block.type == "text":
                proposed_content += block.text

        proposed_content = proposed_content.strip()
        if not proposed_content:
            return None

        old_version = component.version
        new_version = _increment_version(old_version)

        return ImprovementCandidate(
            component=pattern.component,
            original_version=old_version,
            proposed_version=new_version,
            proposed_content=proposed_content,
            change_description=f"Fix {pattern.category.value}: {pattern.description[:100]}",
            target_failure_pattern=pattern.description,
        )

    def apply_candidate(
        self,
        candidate: ImprovementCandidate,
        output_dir: str | Path,
    ) -> AgentConfig:
        """Apply a candidate improvement and return a new agent config.

        Writes the new component version to disk. Returns a fresh config
        pointing to the updated component. Does NOT modify the original.
        """
        output_dir = Path(output_dir)
        comp_type = candidate.component

        # Write new component file
        comp_dir = output_dir / comp_type.value
        comp_dir.mkdir(parents=True, exist_ok=True)
        new_path = comp_dir / f"{candidate.proposed_version}.md"
        new_path.write_text(candidate.proposed_content)

        # Build new config with updated component
        new_components = dict(self.agent_config.components)
        new_components[comp_type] = ComponentVersion(
            component_type=comp_type,
            path=str(new_path),
            version=candidate.proposed_version,
            content=candidate.proposed_content,
        )

        return AgentConfig(
            name=f"{self.agent_config.name}_{candidate.proposed_version}",
            description=f"Candidate: {candidate.change_description}",
            components=new_components,
            model=self.agent_config.model,
            max_tokens=self.agent_config.max_tokens,
            temperature=self.agent_config.temperature,
        )


def _increment_version(version: str) -> str:
    """Simple version incrementing: v1 → v2, v1.2 → v1.3."""
    if version.startswith("v") and version[1:].isdigit():
        return f"v{int(version[1:]) + 1}"
    parts = version.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"{parts[0]}.{int(parts[1]) + 1}"
    return f"{version}_improved"
