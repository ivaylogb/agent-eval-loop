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
from typing import Any

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

    tool_schemas: list[dict[str, Any]] = []

    for comp_key, comp_def in raw.get("components", {}).items():
        comp_type = ComponentType(comp_key)

        if isinstance(comp_def, str):
            comp_path = comp_def
            version = Path(comp_def).stem
        else:
            comp_path = comp_def["path"]
            version = comp_def.get("version", Path(comp_path).stem)

        full_path = (base_dir / comp_path).resolve()
        content = _load_component_content(full_path)

        # Store the resolved absolute path so downstream writers (optimizer,
        # loop checkpoint) can reference the original component from any
        # working directory without losing track of the source file.
        components[comp_type] = ComponentVersion(
            component_type=comp_type,
            path=str(full_path),
            version=version,
            content=content,
        )

        # When the tools component is a YAML schema file, also parse it into
        # API tool definitions. Markdown/text tool components are assumed to
        # be narrative-only and contribute no API-level schemas.
        if comp_type is ComponentType.TOOLS and full_path.suffix.lower() in (".yaml", ".yml"):
            with open(full_path) as f:
                tools_raw = yaml.safe_load(f)
            tool_schemas = yaml_tools_to_api_schemas(tools_raw)

    return AgentConfig(
        name=raw.get("name", config_path.stem),
        description=raw.get("description", ""),
        components=components,
        model=raw.get("model", "claude-sonnet-4-20250514"),
        max_tokens=raw.get("max_tokens", 1024),
        temperature=raw.get("temperature", 0.0),
        tool_schemas=tool_schemas,
    )


def yaml_tools_to_api_schemas(data: Any) -> list[dict[str, Any]]:
    """Convert a parsed tools YAML into Anthropic API tool definitions.

    Expects the shape used by the customer_support example:
        tools:
          - name: lookup_order
            description: "..."
            input_schema: {...}
            output: "..."           # optional; folded into description
            errors: [...]           # optional; folded into description

    Narrative fields (``output``, ``errors``) are appended to ``description``
    so the API-level tool carries the full guidance the system prompt prose
    also contains. Only ``name``, ``description``, and ``input_schema`` survive
    on the wire — those are what the API accepts.
    """
    if not isinstance(data, dict):
        return []
    tools = data.get("tools") or []
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        description = (tool.get("description") or "").strip()
        output = tool.get("output")
        if output:
            description = f"{description}\n\nOutput: {str(output).strip()}"
        errors = tool.get("errors")
        if errors:
            lines = []
            for err in errors:
                if isinstance(err, dict):
                    code = err.get("code", "")
                    desc = err.get("description", "")
                    action = err.get("action", "")
                    lines.append(f"- {code}: {desc} — {action}")
            if lines:
                description = description + "\n\nErrors:\n" + "\n".join(lines)
        schemas.append({
            "name": name,
            "description": description,
            "input_schema": tool.get("input_schema") or {"type": "object", "properties": {}},
        })
    return schemas


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


def write_config_yaml(config: AgentConfig, path: str | Path) -> Path:
    """Serialize an AgentConfig to a YAML manifest that ``load_config`` can read.

    The newly-written component (if any) is emitted with a path relative to
    the manifest's directory when it already lives under that directory;
    unchanged components retain their absolute paths so the manifest stays
    self-contained regardless of where it's written.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    components_out: dict[str, dict[str, str]] = {}
    for comp_type, comp in config.components.items():
        comp_path = Path(comp.path)
        try:
            rel = comp_path.resolve().relative_to(path.parent)
            serialized_path = str(rel)
        except ValueError:
            serialized_path = str(comp_path)
        components_out[comp_type.value] = {
            "path": serialized_path,
            "version": comp.version,
        }

    doc = {
        "name": config.name,
        "description": config.description,
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "components": components_out,
    }

    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return path


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
