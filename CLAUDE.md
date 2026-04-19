# CLAUDE.md — Agent Eval Loop

## What this is
A framework for systematically improving AI agents through a simulate → evaluate → improve loop. Treats agent quality as an engineering problem with measurable iteration cycles.

## Architecture
- `src/agent_eval_loop/models.py` — All shared Pydantic data models. Start here.
- `src/agent_eval_loop/agent/` — Agent framework: versioned components, config loader, runner, scratchpad
- `src/agent_eval_loop/simulate/` — Conversation generation: personas, scenarios, multi-turn simulator
- `src/agent_eval_loop/evaluate/` — LLM-as-judge evaluation: judges with rubrics, scorer, calibration
- `src/agent_eval_loop/improve/` — Improvement engine: failure analysis, prompt optimization, regression testing
- `src/agent_eval_loop/loop.py` — Main orchestrator tying it all together
- `examples/customer_support/` — Fully worked example with all 5 component types
- `docs/best-practices.md` — Production best practices (context engineering, tool design, evals)

## Key design decisions
- **Everything is synchronous.** Batch framework, not real-time. Anthropic sync client throughout.
- **Generator uses AgentRunner.** Simulation goes through the same code path as production — tools execute during conversations.
- **Judges enforce pass_threshold, not the LLM.** LLM returns 0-1 score; rubric threshold determines pass/fail.
- **Reasoning before score in judge prompts.** The judge explains its thinking first, then commits to a number. This improves calibration.
- **Improve stage actually applies changes.** Optimizer writes new component versions to disk, loop runs regression tests, only promotes candidates with zero regressions.
- **ImprovementCandidate stores proposed_content.** Revised text lives in the model, not lost between steps.
- **Shared Anthropic client.** One HTTP connection across judges, generator, optimizer.
- **5 component types.** Instructions, Routines, Tools, Tools_Usage (orchestration rules), Macros (compliance templates).
- **Schema precedence: config wins silently.** In `AgentRunner._build_tool_definitions`, tool schemas parsed from the YAML `tools` component are consumed first. A handler's `tool_schema` attribute only fills in gaps for names the config didn't cover. If you're using `agent-tool-kit` tools with richer `description`/`when_not_to_use` text than what's in the YAML config, remove the corresponding entry from the `tools` component (or drop the component entirely) — otherwise the handler's schema is ignored without warning. See the inline comment in `agent/runner.py` for the exact decision point.

## Best practices encoded in the codebase
- **Investigate first, answer second** — example instructions enforce data retrieval before any text response
- **Tool descriptions include "when NOT to use"** — explicit scope restrictions prevent tool misrouting
- **Explicit over implicit** — tools_usage component encodes sequencing rules models won't infer
- **Fat tools over prompt orchestration** — wrap multi-step workflows in deterministic code
- **Regression testing gates every change** — zero regressions required for promotion

## Commands
```bash
pip install -e .                    # Install in dev mode
pip install -e ".[dev]"             # With dev dependencies
python -m agent_eval_loop --config examples/customer_support/config.yaml --scenarios examples/customer_support/scenarios.yaml
python examples/customer_support/run.py  # Run the example directly
ruff check src/                     # Lint
pytest tests/                       # Test
```

## Conventions
- Python 3.11+, type hints everywhere
- Pydantic v2 for all serializable data models
- Anthropic SDK (sync client) for all LLM calls
- Rich for CLI output
- YAML for config, JSONL for data output
- UTC timestamps via `datetime.now(timezone.utc)`

## What to build next
- Mock tool handlers for the customer support example (so conversations work end-to-end without a real backend)
- Tests for each module (start with models, then config loader, then judges)
- A `--dry-run` mode that shows what would happen without making API calls
- Parallel conversation generation (threading or asyncio)
- Export eval results to Braintrust format
- Few-shot examples in judge rubrics for the hardest eval categories
- A second example domain to prove the framework is domain-agnostic
