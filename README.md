# agent-eval-loop

A framework for systematically improving AI agents through simulated conversations, calibrated evaluation, and automated prompt optimization.

## The Problem

Most teams build agents and then manually tweak prompts when things break. This works until it doesn't — you fix one failure and introduce two regressions, you can't tell if a change actually helped or you just got lucky on a few test cases, and your "evaluation" is someone on the team running five conversations and saying "looks good."

This repo implements a disciplined alternative: a **simulate → evaluate → improve** loop that treats agent quality as an engineering problem, not a vibes problem.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                  OFFLINE ITERATION                   │
│           (run N times before deploying)             │
│                                                      │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐      │
│   │ SIMULATE  │→ │ EVALUATE  │→ │  IMPROVE  │──┐   │
│   │           │  │           │  │           │  │   │
│   │ Generate  │  │ Score w/  │  │ Propose   │  │   │
│   │ realistic │  │ calibrated│  │ targeted  │  │   │
│   │ convos    │  │ LLM judges│  │ fixes     │  │   │
│   └───────────┘  └───────────┘  └───────────┘  │   │
│        ↑                                        │   │
│        └────────────────────────────────────────┘   │
│                                                      │
│   Exit when eval scores converge                     │
└──────────────────────┬──────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │  HUMAN REVIEW   │
              │     GATE        │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  A/B TEST IN    │
              │  PRODUCTION     │
              └─────────────────┘
```

### Stage 1: Simulate

Generate thousands of realistic multi-turn conversations using synthetic personas. Each persona has a defined expertise level, communication style, emotional state, and goal. Conversations are seeded from known failure cases and production patterns.

### Stage 2: Evaluate

Score each conversation with calibrated LLM-as-judge evaluators. Multiple eval categories run independently — tool selection accuracy, response correctness, tone appropriateness, routine adherence, escalation decisions. Judges use structured rubrics and are calibrated against human annotations.

### Stage 3: Improve

Analyze failure patterns from evaluation. Trace each failure to a specific agent component. Propose targeted fixes — either manually or via automated prompt optimization. Run regression tests to verify fixes don't break passing cases. The improved agent re-enters the loop.

## The Agent Architecture

The agent under test is built from **independently versioned context components**:

| Component | What It Contains | Example |
|-----------|-----------------|---------|
| **Instructions** | Role, constraints, guardrails | "You are a support agent. Never share account numbers." |
| **Routines** | Step-by-step procedures (SOPs) | "1. Look up order. 2. Check status. 3. Share tracking..." |
| **Tool Descriptions** | Schemas, usage guidance, scope restrictions | `lookup_order(order_id: str) -> OrderStatus` |
| **Tools Usage** | Orchestration rules for tool calling | "Call lookup_order FIRST on every conversation, before any text response" |
| **Macros** | Pre-written templates for compliance-sensitive scenarios | Refund policy language, legal disclaimers |
| **Scratchpad** | Cross-turn working memory | Current order context, customer sentiment |

A new agent configuration is a **manifest** pointing to specific versions of each component:

```yaml
name: "customer_support_v3"
components:
  instructions: "instructions/v2.md"
  routines: "routines/order_tracking/v3.md"
  tools: "tools/v1.yaml"
  tools_usage: "tools_usage/v1.md"
```

Every change is traceable. When v3 beats v2 by 12 points, you know exactly which component changed and why.

## Quick Start

```bash
pip install -e .

# Run the full loop on the example domain
python -m agent_eval_loop --config examples/customer_support/config.yaml --scenarios examples/customer_support/scenarios.yaml --iterations 3

# Run the example directly
python examples/customer_support/run.py
```

## Project Structure

```
agent-eval-loop/
├── docs/
│   └── best-practices.md     # Production lessons → reusable principles
├── src/agent_eval_loop/
│   ├── agent/                # Core agent framework
│   │   ├── config.py         # Configuration loader + prompt assembly
│   │   ├── runner.py         # Agent execution engine (tool loop)
│   │   └── scratchpad.py     # Working memory
│   ├── simulate/             # Conversation generation
│   │   ├── personas.py       # Synthetic persona definitions
│   │   ├── scenarios.py      # Seed scenarios + failure cases
│   │   └── generator.py      # Multi-turn simulator (uses AgentRunner)
│   ├── evaluate/             # Evaluation framework
│   │   ├── judges.py         # LLM judge definitions (reasoning-first prompts)
│   │   ├── scorer.py         # Score aggregation + reporting
│   │   └── calibration.py    # Judge-human agreement (Cohen's kappa)
│   ├── improve/              # Improvement engine
│   │   ├── analyzer.py       # Failure pattern detection
│   │   ├── optimizer.py      # LLM-powered prompt optimization
│   │   └── regression.py     # Regression testing gate
│   └── loop.py               # Full cycle orchestrator
├── examples/
│   └── customer_support/     # Worked example with all component types
└── pyproject.toml
```

## Key Design Decisions

**Why versioned components instead of monolithic prompts?**
When your entire agent prompt is one big string, every change is a gamble. Versioned components let you make surgical changes — update one routine, test it, and know exactly what moved.

**Why LLM-as-judge instead of just human eval?**
Humans don't scale. You can't have a human review 5,000 simulated conversations every iteration. LLM judges scale — but they need calibration against human judgment. We measure inter-rater agreement and only trust judges above a correlation threshold.

**Why simulate before production?**
Production exposes edge cases — but slowly, expensively, and at the cost of real user experience. Simulation generates thousands of adversarial conversations before any real user is affected.

**Why loop offline before deploying?**
A single pass might improve one thing and regress another. Multiple offline iterations until scores converge gives confidence the improvement is stable, not a fluke.

## Extending to Your Domain

The framework is domain-agnostic. To adapt it:

1. **Define your agent components** — instructions, routines, tool descriptions, and tools_usage rules
2. **Create personas** relevant to your domain (expertise levels, communication styles, emotional states)
3. **Write scenarios** covering happy paths, error handling, edge cases, and adversarial inputs
4. **Define eval categories** and write judge rubrics for each
5. **Configure and run the loop**

See `examples/customer_support/` for a complete worked example with all component types.

## Best Practices

See [`docs/best-practices.md`](docs/best-practices.md) for production lessons on context engineering, tool design, evaluation, and operational patterns — distilled from building agents at scale.

Key principles covered:
- **Investigate first, answer second** — always retrieve data before responding
- **Tool descriptions are prompts** — include "when NOT to use" scope restrictions
- **Reasoning before score** — judge prompts produce better calibration when reasoning comes first
- **Fat tools over prompt orchestration** — wrap multi-step workflows in deterministic code
- **Regression testing gates every change** — never deploy without verifying zero regressions

## License

MIT
