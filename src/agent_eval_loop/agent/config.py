"""Agent configuration: load versioned components from disk into a runnable agent.

An agent configuration is a manifest (YAML file) that points to specific
versions of each context component. This decouples the agent's behavior
from a monolithic prompt string — you can update one routine, one tool
description, or one set of instructions without touching the others.

The build_system_prompt function assembles these components into the final
context window content, in a deliberate order:
    instructions → routines → tools_usage → tool_descriptions → macros

This ordering matters: instructions establish role and constraints first,
routines define the procedure, tools_usage sets orchestration rules,
tool descriptions provide schemas, and macros supply compliance templates.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_eval_loop.models import AgentConfig, ComponentType, ComponentVersion


def load_config(config_path: str | Path) -> AgentConfig:
    """Load an agent configuration from a YAML manifest.

    The manifest points to specific versions of each component.
    This function resolves those pointers and loads the actual content.
    """
    config_path = Path(config_path)
    base_dir = config_path.parent

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    components: dict[ComponentType, ComponentVersion] = {}

    for comp_key, comp_def in raw.get("components", {}).items():
        comp_type = ComponentType(comp_key)

        if isinstance(comp_def, str):
            comp_path = comp_def
            version = Path(comp_def).stem
        else:
            comp_path = comp_def["path"]
            version = comp_def.get("version", Path(comp_path).stem)

        full_path = base_dir / comp_path
        content = _load_component_content(full_path)

        components[comp_type] = ComponentVersion(
            component_type=comp_type,
            path=comp_path,
            version=version,
            content=content,
        )

    return AgentConfig(
        name=raw.get("name", config_path.stem),
        description=raw.get("description", ""),
        components=components,
        model=raw.get("model", "claude-sonnet-4-20250514"),
        max_tokens=raw.get("max_tokens", 1024),
        temperature=raw.get("temperature", 0.0),
    )


def _load_component_content(path: Path) -> str:
    """Load component content from a file."""
    if not path.exists():
        raise FileNotFoundError(f"Component file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".md", ".txt"):
        return path.read_text()
    elif suffix in (".yaml", ".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        return yaml.dump(data, default_flow_style=False)
    elif suffix == ".json":
        import json

        with open(path) as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    else:
        return path.read_text()


def build_system_prompt(config: AgentConfig) -> str:
    """Assemble the full system prompt from loaded components.

    Order matters: instructions → routines → tool descriptions.
    Components with no loaded content are skipped with a warning.
    """
    sections = []

    component_order = [
        (ComponentType.INSTRUCTIONS, "instructions"),
        (ComponentType.ROUTINES, "routines"),
        (ComponentType.TOOLS_USAGE, "tools_usage"),
        (ComponentType.TOOLS, "tool_descriptions"),
        (ComponentType.MACROS, "macros"),
    ]

    for comp_type, xml_tag in component_order:
        comp = config.components.get(comp_type)
        if comp is None:
            continue
        if not comp.content:
            # Component exists in config but content wasn't loaded
            continue
        sections.append(f"<{xml_tag}>\n{comp.content}\n</{xml_tag}>")

    return "\n\n".join(sections)
