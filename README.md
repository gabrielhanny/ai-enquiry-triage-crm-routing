# AI Enquiry Triage & CRM Routing System

A small demo prototype showing how inbound business enquiries can be
triaged with an LLM, validated deterministically, routed conditionally, and
handled safely with human approval before any consequential action. Built
for an AI systems reasoning assessment — see [CLAUDE.md](CLAUDE.md) for the
full specification this implementation follows.

## Workflow

```text
Normalize -> LLM Triage + Extraction -> Deterministic Validation
    -> Archive                                              [junk]
    -> Draft Clarification -> Human Approval                [incomplete / low confidence]
    -> Manual Review / Recovery -> Human Approval           [LLM / API failure]
    -> Duplicate Check -> Mock CRM Action -> Draft Response -> Human Approval   [sales / support]
-> Audit Log   (always, exactly once per enquiry)
```

Orchestrated with LangGraph because the workflow has explicit state,
conditional branches, an approval gate, and a failure/recovery path.

An LLM/API failure is a system fault, not a data-completeness problem, and
is treated as a distinct **Manual Review / Recovery** path rather than an
"incomplete enquiry": `validate_node` checks for a failed LLM call before it
ever evaluates field completeness, and the audit row for a failed call
carries an `error` value so it is never conflated with a genuine
missing-fields case. It currently reuses the same clarification +
human-approval mechanism to reach a person, since both paths share the goal
of "stop and get a human," but the two are decided independently.

## Design principles

- **The LLM never decides anything consequential.** It classifies, extracts
  fields, flags likely-missing information, and drafts text — that's it.
  It has no authority over permissions, CRM identity, duplicate decisions,
  approval, or audit records.
- **The app re-derives the truth itself.** [app/validation.py](app/validation.py)
  computes required fields and a confidence threshold independently and does
  **not** trust the LLM's own `missing_information` claim (see
  `test_llm_missing_information_claim_is_not_authoritative` in
  [tests/test_validation.py](tests/test_validation.py)).
- **Safe by default.** No real email, CRM, or messaging integration exists.
  The "human approval" step ([app/approval.py](app/approval.py)) prints the
  draft and auto-approves — a stand-in for a real reviewer — and is the only
  place a real system would need a human click before anything leaves this
  prototype. Nothing is ever actually sent.
- **Failure handling is explicit.** If the LLM call fails after retries, the
  enquiry is routed to a Manual Review / Recovery path — never treated as an
  incomplete enquiry — so a human reviews it directly rather than the system
  guessing a category. The failure is recorded distinctly in the audit log
  (`error` field), separate from missing-fields cases.
- **Cost/latency awareness.** Uses `gpt-4o-mini` by default, `temperature=0`
  for the classification call, and capped `max_tokens`. Tests never call the
  real API — the LLM boundary (`app/llm.py`) is stubbed with `monkeypatch`.

## Model and tool choices

- **GPT-4o-mini** — triage classification, field extraction, and drafting
  are bounded tasks (three fixed categories, a handful of fields, short
  business emails). A small, fast, cheap model handles this reliably; a
  larger model isn't warranted at this scope.
- **LangGraph** — the workflow has explicit state, several conditional
  branches, an approval gate, and a failure/recovery path. Modeling this as
  a graph keeps the branching explicit instead of ad hoc if/else logic.
- **Pydantic** — validates and type-checks all LLM output against a fixed
  schema (`TriageResult`) before it's used anywhere downstream, so malformed
  output fails fast instead of propagating silently.
- **SQLite** — a single-file, zero-setup store, sufficient for a local audit
  trail in a demo; no separate database server needed.
- **Mock CRM adapter** — an in-memory stand-in reproducing the shape of real
  CRM behavior (lead/ticket creation, duplicate detection) without
  integrating a production CRM, consistent with the safety constraints.

## Security and permissions

- All enquiry data is synthetic (`app/scenarios.py`) — no real customer data
  is processed.
- The OpenAI API key is loaded from a gitignored `.env` file and is never
  hardcoded or committed.
- No credentials or secrets are stored in source code or written to the
  audit trail.
- In production, the key would come from a secrets manager with
  least-privilege access scoped to the triage service — not an `.env` file.
- CRM writes and external communications are gated entirely by application
  code (validation, routing, approval) — the LLM has no authority to
  trigger either.

## Deliberate automation boundary

This system deliberately does not autonomously send consequential external
communications or make binding commercial commitments. Every drafted
response or clarification must pass through the human approval gate
(`app/approval.py`) before it could be considered "sent" in a real
deployment. In this prototype that gate is a dry-run stub, but the boundary
is intentional: LLM output is a draft, never an action.

## Project structure

```
app/
  models.py       Pydantic models (RawEnquiry, TriageResult, ValidationResult, ...)
  llm.py          OpenAI calls: structured triage/extraction + drafting
  validation.py   Deterministic required-field checks, confidence threshold, routing
  crm.py          Mock CRM (in-memory): lead/ticket creation, duplicate detection
  audit.py        SQLite audit trail
  approval.py     Human-in-the-loop approval gate (dry-run auto-approve)
  graph.py        LangGraph state + node wiring
  scenarios.py    Four synthetic demo enquiries
demo.py           Runs the four scenarios end-to-end
tests/            pytest suite (models, validation, CRM, audit, full graph)
conftest.py       Isolates the audit DB and mock CRM state per test
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

Requires an `.env` file in the project root with:

```
OPENAI_API_KEY=sk-...
```

## Run the demo

```bash
python demo.py
```

Runs the four locked scenarios (complete sales, incomplete sales, support,
junk), plus a bonus re-submission of the first enquiry to show duplicate
detection. Each run prints its route, classification, CRM action, and
approval outcome, and appends one row to `data/audit.db`.

## Run the tests

```bash
pytest
```

Tests cover Pydantic model validation, deterministic validation/routing
rules, mock CRM behavior (including duplicate detection), the audit trail,
and the full compiled graph (with the LLM stubbed, including a simulated
LLM-failure path). No test calls the real OpenAI API.

## Out of scope

Per [CLAUDE.md](CLAUDE.md): no auth, user accounts, billing, web UI,
deployment, public API, real CRM/email/messaging integration, multi-agent
architecture, or multiple LLM providers. This is a workflow-architecture
demo, not a product.
