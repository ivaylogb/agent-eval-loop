# Best Practices for Production AI Agents

Lessons learned from building and iterating on production agents serving millions of customers. These principles are domain-agnostic — they apply whether you're building customer support, code review, data analysis, or any other agent type.

---

## Context Engineering

### Modular prompt architecture

Production agents should be assembled from independently versioned components, not a single monolithic prompt string. When your entire prompt is one file, every change is a gamble — you can't isolate what broke, and you can't test one piece without re-testing everything.

Components we've found useful:

| Component | Purpose | Why it's separate |
|-----------|---------|-------------------|
| **Instructions** | Role, constraints, guardrails | Changes rarely; sets the foundation |
| **Routines** | Step-by-step procedures | Changes per workflow; the most frequently iterated piece |
| **Tool descriptions** | Schemas + usage guidance | Changes when tools change; shared across routines |
| **Tools usage** | Orchestration rules for tool calling | Encodes sequencing logic that models won't infer |
| **Macros** | Pre-written templates for sensitive scenarios | Compliance-controlled; changes require legal review |
| **Scratchpad** | Cross-turn working memory | Runtime-only; not versioned |

Each component is a file with a version identifier. A new agent configuration is a manifest pointing to specific versions. When v3 outperforms v2, you know exactly which component changed and can reason about why.

### Investigate first, answer second

This is the single most impactful behavioral rule we've found across all agent types. The pattern: call all relevant data tools on the first turn, before generating any text response. The agent's first message to the customer should include specific information that proves the agent knows their situation.

Why: generic greetings and "how can I help you?" messages provide zero value when the customer already stated their problem. Retrieving data first also prevents hallucination — the agent responds from real data, not from guessing.

Implementation: encode this as an explicit rule in the instructions and tools_usage components. Do not rely on the model to infer this behavior — newer models that follow instructions strictly will NOT make implicit tool calls unless explicitly told to.

### Explicit over implicit

Models that follow instructions more strictly (a general trend as models improve) will not infer patterns that earlier models would. If your agent relies on the model "just knowing" to call tools before responding, or to check order status before attempting a cancellation, it will break on the next model upgrade.

Every behavioral expectation should be an explicit instruction. If you find yourself saying "the model should know to do X," write it down.

---

## Tool Design

### Fat tools over prompt orchestration

When a task requires chaining multiple API calls (check status → verify eligibility → perform action → confirm), wrap the entire chain into a single composite tool. The LLM makes one tool call; the deterministic logic lives in code.

Benefits:
- Lower latency (one round-trip instead of 3-5 sequential ones)
- Eliminates sequential-dependency errors (the model can't forget a step)
- Easier to test deterministically
- The LLM can't mess up the ordering

### Tool descriptions are prompts

The schema and docstring of each tool are part of the agent's context window. Treat them with the same rigor as any other prompt component:
- Natural-language description of what the tool does
- Typed input schema with descriptions for each parameter
- Expected output format
- **When NOT to use it** — explicit scope restrictions

The "when NOT to use" section is the most commonly missing piece. Without it, models will call the most semantically similar tool regardless of whether it's correct. Example: a knowledge base search tool will get called for order-specific questions unless you explicitly say "Do NOT use this tool for order status, delivery tracking, or account-specific information."

### Minimize output surface area

Tools should return only the fields the LLM needs to reason about. A delivery status tool returns carrier, status, and estimated delivery — not the full shipment manifest with internal tracking IDs, warehouse codes, and API metadata.

Why: every unnecessary field consumes context window space, increases the chance the model fixates on irrelevant details, and risks leaking internal information to the customer.

### Structured error handling

Tools return typed error objects, not exceptions. Each error includes:
- An error category (not_found, timeout, invalid_input, unauthorized)
- A human-readable message
- A retryable flag
- Suggested action for the agent

This lets the LLM reason about failures: retry on transient errors, ask the customer for missing information, or escalate to a human. A Python traceback in the context window is worse than useless — it wastes tokens and the model can't act on it.

### Idempotency

Every action tool must be safely re-callable without side effects. If the model calls `create_return` twice because it didn't see the first response (timeout, context window truncation), the second call should return the existing return, not create a duplicate.

Enforce through idempotency keys derived from the session ID and action parameters.

---

## Evaluation

### The eval-driven development loop

Agent quality is a function of iteration velocity, not model capability. A mediocre model with fast iteration cycles will outperform a frontier model with slow iteration cycles.

The core loop:
1. **Simulate** — generate realistic multi-turn conversations with synthetic personas
2. **Evaluate** — score with calibrated LLM judges across multiple dimensions
3. **Improve** — trace failures to specific components, propose targeted fixes
4. **Repeat** — run N iterations offline before any change reaches production

The most important property of this loop: offline eval improvements must correlate with production metrics. If your eval says version B is better than version A, it should actually be better in production. Validate this correlation for each new domain — don't assume it transfers.

### Building a judge (the two-phase approach)

**Phase 1 (1 week):** Self-label 50 representative conversations. Write a binary LLM judge prompt. Run the judge across multiple models at temperature 0. Target ≥80% accuracy against your labels.

**Phase 2 (1 month):** Expand to 300+ samples with 3 domain expert labelers per sample. Measure inter-rater reliability via Cohen's kappa (target ≥0.7). Optimize the judge prompt using automated methods if manual iteration plateaus.

### Judge prompt design

- **Ask for reasoning BEFORE the score.** When the model explains its thinking first, the subsequent score is more calibrated. If you ask for the score first, the reasoning becomes post-hoc rationalization.
- **Use binary pass/fail labels** for clear signal. A 0-1 float score provides granularity, but the pass/fail threshold should be enforced by the rubric, not left to the model.
- **Include few-shot examples** in rubrics for difficult categories. Examples anchor calibration far more effectively than additional criteria text.
- **One judge per dimension.** Don't ask a single judge to evaluate tone, accuracy, and completeness simultaneously — the scores blur together. Separate judges with focused rubrics produce sharper signal.

### Regression testing gates every change

Never deploy an improvement without verifying it doesn't break previously passing cases. A change that fixes 5 failures but introduces 2 regressions is not an improvement — it's a lateral move with hidden risk.

Run the full eval suite on every candidate. Accept only if regressions = 0.

---

## Operational Patterns

### Temperature 0 in production

Set temperature to 0.0 for production agents. Non-zero temperature introduces non-determinism that makes debugging impossible and evals unreliable. If your agent behaves differently on the same input, you can't trust your eval scores.

### Model upgrades are not free

Newer models are not strictly better in every dimension. They may follow instructions more strictly (breaking implicit patterns), handle tool calls differently, or produce different output formats. Every model upgrade should be treated as a new agent version that goes through the full eval loop.

### The human review gate

The improvement loop runs offline. Only after eval scores converge does a candidate proceed to human review. The human reviewer checks:
- Do the changes make sense? (Sanity check on automated optimization)
- Are there qualitative issues the eval didn't catch?
- Is the agent's personality/brand voice preserved?

Then the candidate goes to A/B testing in production. The loop automates iteration; humans gate deployment.
