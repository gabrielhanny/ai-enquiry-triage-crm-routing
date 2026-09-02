# AI Enquiry Triage & CRM Routing System

A demo prototype showing how inbound business enquiries can be triaged with
an LLM, validated deterministically, matched against CRM records, routed to
the right owner, and drafted a response — all gated by human approval before
anything consequential happens. Built for an AI systems reasoning
assessment; see [CLAUDE.md](CLAUDE.md) for the original specification.

This README covers both phases:
- **Test 1** — the core workflow, proven with four clean synthetic scenarios (`demo.py`).
- **Test 2** — the same workflow adapted to ingest a messier, more realistic dataset (`run_dataset.py`).

## 1. Test 2 objective

Test 1 proved the workflow shape (triage → validate → route → approve →
audit) on clean, hand-written scenarios. Test 2 tests whether that same
workflow holds up against messier, more realistic input: a fictional dataset
of a staff directory, CRM export, 12 inbound emails (some with attachments),
and referenced documents — with typos, missing fields, contradictory
figures, and enquiries that don't cleanly split into "sales" or "support".
The goal is to extend, not replace, the Test 1 architecture to handle
ingestion, richer classification, CRM matching, conflict detection, and
staff routing, while keeping every consequential decision deterministic and
human-approved.

## 2. Architecture and data flow

```text
data/staff_directory.json, crm.csv, emails.json, documents/*.txt
        │  (app/ingest.py — loads verbatim, no cleanup)
        ▼
RawEnquiry (email body + attachment text combined as one block of evidence)
        │
        ▼
Normalize → LLM Triage + Extraction → Deterministic Validation (+ category
safeguards) → recommend_owner
        │
        ├─ archive                                                  [junk]
        ├─ clarify → human_approval → apply_crm_action (no-op)       [incomplete / low confidence]
        ├─ clarify → human_approval → apply_crm_action (no-op)       [LLM/API failure — Manual Review/Recovery]
        └─ analyze_crm (read-only) → draft_response → human_approval
               → apply_crm_action (mutates only here)
               [sales / support / technical / operations / infrastructure / other]
        │
        ▼
audit_log (SQLite — always exactly one row per enquiry)
```

`analyze_crm` is where Test 2's matching work concentrates, and it is
strictly **read-only**: it scores the enquiry against every candidate CRM
record (both the static `crm.csv` rows and records already created earlier
in the same run), decides a match status and a *recommended* action, and
flags any factual conflicts against the best match — all in
`app/crm_match.py`, all deterministic. Nothing is written to the CRM at this
stage; the recommendation is only realized by `apply_crm_action`, which runs
after `human_approval` and does nothing if approval was withheld. This
ordering (analyze → recommend → draft → approve → mutate → audit) was
tightened in a follow-up code review — see "What changed from Test 1" below.

Orchestration is still LangGraph, still a single graph with explicit state
and conditional edges — no multi-agent architecture, no new orchestration
layer.

## 3. What changed from Test 1, and why

The Test 1 graph, models, validation rules, mock CRM, audit trail, and
approval gate are **unchanged in behavior** — all 19 original tests still
pass without modification. Test 2 is additive:

| Addition | File | Why |
|---|---|---|
| Fixture ingestion | `app/ingest.py` | Load the four dataset files verbatim; parse messy `from` headers; merge attachment text into the enquiry body as combined evidence |
| More categories | `app/models.py` (`EnquiryCategory`) | `technical`, `operations`, `infrastructure`, `other` added alongside `sales`/`support`/`junk` — the real dataset doesn't fit a two-category split |
| Richer extraction | `app/models.py` (`TriageResult`) | Added `contact_name`, `phone`, `location` as optional fields the LLM may extract |
| CRM matching | `app/crm_match.py` (new) | Multi-signal, confidence-bucketed matching against real CRM data, instead of Test 1's exact-email-only duplicate check |
| Conflict detection | `app/crm_match.py` | Real fixture data contains contradictions (e.g. a corrected phone number) that must be preserved, not merged away |
| Owner recommendation | `app/staff.py` (new) | The dataset includes a staff directory; Test 1 had no concept of ownership |
| `recommend_owner` node | `app/graph.py` | One new node, wired in as a no-op when no staff directory is supplied — Test 1 callers are unaffected |
| `analyze_crm` / `apply_crm_action` split | `app/graph.py` | Read-only matching/recommendation is separated from the actual CRM mutation, which now only runs after `human_approval` (see below) |
| Richer audit `details` | `app/graph.py` (`audit_node`) | Same `audit_log` table/schema — new context (match status, conflicts, owner, recommended vs. applied action, source email id) is packed into the existing `details` JSON column, so `app/audit.py` needed zero changes |
| CLI inspection | `run_dataset.py` (new) | Table / per-enquiry detail / JSON dump over all 12 enquiries |

`decide_route` was simplified from an explicit sales/support if-else to
`return triage.category.value` — this returns identical strings for the
original two categories, so no Test 1 behavior changed, but it now
generalizes to every category without a growing if-chain.

**Follow-up fix (post code review):** the first version of Test 2 performed
the CRM write (`duplicate_check` node) *before* `human_approval`. A code
review flagged that as a genuine architectural gap — the mock CRM is only
local/in-memory, but the ordering would carry over unchanged if a real CRM
were ever plugged in later. The fix split that one node into two:

- `analyze_crm` (read-only) — matching, conflict detection, and a
  *recommended* action, computed before approval.
- `apply_crm_action` (the only place CRM state is ever mutated) — runs
  after `human_approval`, and does nothing if approval was withheld.

The same review also flagged that E010 (a customer correcting their own
phone number) was being classified `infrastructure` by the LLM — a category
reserved for reports about the business's *own* internal systems. Rather
than trust a prompt tweak alone, `app/validation.py::apply_category_safeguards`
adds a small deterministic backstop: if the LLM ever labels something
`infrastructure` and the enquiry text generically matches contact-correction
phrasing ("correcting my number", "please use this email", etc. — not
anything specific to E010's wording), it's reclassified to `operations`
before routing or owner-recommendation ever see it.

## 4. AI vs deterministic responsibilities

Unchanged principle from Test 1, extended to the new decisions Test 2 adds:

| Decision | Owner |
|---|---|
| Category, extracted fields, drafted text | LLM (`app/llm.py`) — advisory only |
| Required fields, confidence threshold, routing | Deterministic (`app/validation.py`) |
| CRM match status, match score, conflict detection | Deterministic (`app/crm_match.py`) — the LLM never decides "is this the same customer" |
| Which staff member owns an enquiry | Deterministic keyword match against `staff_directory.json` (`app/staff.py`) — the LLM never assigns owners |
| CRM writes, approval, audit | Deterministic, unchanged from Test 1 |

The LLM's own `missing_information` guess and any category/field it
proposes are never trusted directly — `validation.py` re-derives what's
actually required, and `crm_match.py`/`staff.py` re-derive matches and
ownership from the loaded fixtures, not from anything the LLM asserts about
them.

## 5. Handling messy data and uncertainty

- **Never invent missing facts.** The system prompt explicitly forbids
  guessing a company name from an email domain or filling in an unstated
  phone number. Optional fields stay `null` when the text doesn't state them
  (e.g. E001 never says "Hume Logistics" in the body — `company_name` comes
  back null, and matching still succeeds on the email address alone).
- **Preserve conflicts instead of merging them away.** `crm_match.detect_conflicts`
  only fires when both the enquiry and the matched CRM record have a
  non-empty value that disagrees (e.g. E010: phone `0411 999 120` on file vs
  `0411 999 102` in the new email) — both values are kept in the `Conflict`
  record, not overwritten. A blank CRM field being filled in is treated as
  new information, not a conflict.
- **Ambiguous CRM data is surfaced, not resolved.** E002 matches `C001` and
  `C002` — two CRM rows that look like the same company recorded twice with
  a typo — with identical scores. `match_enquiry` reports both candidates
  and adds an explicit note rather than silently picking one.
- **Attachments are evidence, not truth.** Attachment text is appended to
  the enquiry body under a clear `--- Attachment: filename ---` marker and
  the system prompt tells the LLM to weigh it with the same scrutiny as the
  email body — not to treat it as more authoritative just because it's a
  "document".

**On the `confidence` number specifically:** `TriageResult.confidence` is
the LLM's own self-reported estimate — it is **not** a calibrated
probability, and the code does not treat it as one. In practice,
gpt-4o-mini reports `0.90` for nearly every enquiry in this dataset,
including ones it got wrong (E010's `infrastructure` mislabel still came
back at 0.90). That's expected LLM behavior — self-reported confidence
tends to cluster on round numbers rather than reflecting fine-grained
calibration — so the system deliberately does not lean on small differences
in that number. `CONFIDENCE_THRESHOLD = 0.6` is used only as a coarse "did
the model even claim to be confident" gate; the actual uncertainty signals
that drive decisions are all separately computed and deterministic:
`ValidationResult.missing_fields` (what's actually absent), `MatchResult.status`
and `.notes` (how sure the CRM match is, and whether the reference data
itself looks ambiguous), and `Conflict` records (concrete factual
disagreements). Treat those as the real uncertainty signals; treat
`confidence` as a rough, unverified hint from the model that made the
classification.

## 6. CRM matching approach

`app/crm_match.py` scores every candidate record (loaded `crm.csv` rows plus
records already created this run) against five independent signals:

| Signal | Weight | Note |
|---|---|---|
| Exact email match | overrides to 0.9 | Treated as a near-certain identity signal on its own |
| Email domain match | +0.45 | Same organisation, different mailbox — plausible but not certain alone |
| Phone match (normalized) | +0.35 | |
| Company name similarity (suffix-stripped, fuzzy ratio) | +0.25 to +0.5 | |
| Contact name similarity | +0.2 | |
| Location overlap | +0.1 | |

Scores are capped at 1.0 and bucketed: `>=0.75` → `likely_match`,
`>=0.35` → `possible_match`, otherwise `no_match`. A `likely_match` attaches
to the existing record; a `possible_match` is flagged for review without
being treated as a duplicate; `no_match` falls back to Test 1's plain
create-new-record path. The thresholds are deliberately conservative — a
domain match alone (E010) lands as `possible_match`, not `likely_match`,
because it genuinely isn't certain.

## 7. Security and permission boundaries

- All data is synthetic and local — `staff_directory.json`, `crm.csv`,
  `emails.json`, and `documents/*.txt` are fixture files checked into the
  repo, not a connection to any real system.
- No real CRM, email, or messaging integration exists anywhere in the code.
- The OpenAI API key is loaded from a gitignored `.env` and never
  hardcoded; no credentials appear in fixtures, source, or audit records.
- CRM matching, conflict detection, staff routing, and CRM writes are all
  deterministic application code — the LLM has no authority to decide who
  owns an enquiry, whether a match is real, or whether to write to the CRM.
- In production, the API key would live in a secrets manager with
  least-privilege scope, not an `.env` file.

## 8. Human approval boundary

Every drafted response or clarification — regardless of category, match
status, or owner — passes through `app/approval.py` before it could be
considered "sent". This prototype's approval gate is a dry-run stub that
prints the draft and auto-approves (it has no reject path of its own — a
real deployment would need one), but the ordering it enforces is real:

- `analyze_crm` only *reads* CRM state and computes a recommendation —
  it never mutates anything.
- `apply_crm_action` — the only node that creates a lead/ticket or attaches
  an enquiry to an existing record — runs strictly after `human_approval`,
  and does nothing at all if approval was withheld.

This is enforced by graph structure, not convention: `apply_crm_action`
reads `state["approval"].approved` and returns a `not_applied` action
without touching the mock CRM if it's `False`. Tests
(`test_no_crm_mutation_occurs_before_approval`,
`test_rejected_approval_prevents_new_record_creation`,
`test_rejected_approval_prevents_attach_to_existing_record`) assert this
directly by spying on `crm.all_records()` at the moment approval is
requested, and by rejecting approval and checking no record is created or
linked.

## 9. Failure handling

If the LLM call fails after retries, the enquiry is routed to a **Manual
Review / Recovery** path — not treated as an incomplete enquiry.
`validate_node` checks for a failed LLM call before it ever evaluates field
completeness, and the failure is recorded distinctly in the audit `details`
(`error` field), separate from a genuine missing-fields case. It currently
reuses the same clarification + human-approval mechanism to reach a person,
since both paths share the goal of "stop and get a human," but the two are
decided independently and never conflated in the audit trail.

## 10. Cost/latency considerations

- Still `gpt-4o-mini`, `temperature=0` for classification, capped
  `max_tokens` — unchanged from Test 1.
- All matching, conflict detection, and owner recommendation are pure
  Python with no additional LLM calls — Test 2 adds zero LLM round-trips
  per enquiry over Test 1.
- `run_dataset.py --id E00X` currently reprocesses all 12 enquiries (12 LLM
  calls) to look up one result, rather than caching the last run — fine for
  a 12-enquiry demo, but the first thing to fix before this scaled up (see
  below).

## 11. Known weaknesses

- `--id` reprocesses the whole dataset instead of caching/reusing the last
  run's results.
- CRM matching weights were hand-tuned against this specific dataset; they
  aren't calibrated against a larger, more varied sample.
- The Manual Review / Recovery path is implemented via the same graph node
  as ordinary clarification — they're decided independently and logged
  distinctly, but they don't yet have visually distinct handling in
  `run_dataset.py`'s table output.
- Owner recommendation is keyword-matched against free-text `owns`
  descriptions — brittle if the staff directory's wording changes
  materially.
- No persistence of match/conflict history across separate runs of
  `run_dataset.py` (each run starts the in-memory mock CRM fresh, so
  cross-run duplicate detection only works within a single process).
- The `infrastructure` category safeguard is a single regex-based heuristic
  for "contact correction" phrasing. It catches the pattern this dataset
  exercises, but a differently-worded correction (or a genuinely internal
  report that happens to mention "update" and "email") could still slip
  through in either direction — it's a backstop, not a general-purpose
  classifier fix.
- `apply_crm_action`'s `attach()` call is a no-op for a match against a
  static `crm.csv` row (there's no local mutable copy of external CRM
  records to update) — approved "attach" decisions against real CRM rows
  aren't recorded anywhere except the audit trail. Fine for a mock/local
  prototype; a real integration would need an actual write path there.

## 12. What would be improved with another day

- Cache a dataset run so `--id` and `--json` don't re-trigger 12 LLM calls
  each time.
- Persist the mock CRM's in-run records to SQLite alongside the audit log,
  so duplicate detection works across separate `run_dataset.py` invocations.
- Add a small confidence-calibration pass (e.g. cross-checking a handful of
  match decisions by hand) to validate the CRM-matching thresholds.
- Surface the CRM's own likely-duplicate rows (like C001/C002) as a
  standalone data-quality report, not just as a note attached to whichever
  enquiry happens to match both.

## 13. Setup and run instructions

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

Requires an `.env` file in the project root with:

```
OPENAI_API_KEY=sk-...
```

**Test 1 — four clean synthetic scenarios:**

```bash
python demo.py
```

**Test 2 — the full 12-enquiry dataset:**

```bash
python run_dataset.py             # table of all 12 enquiries
python run_dataset.py --id E010   # full detail for one enquiry
python run_dataset.py --json      # structured JSON dump of all results
```

**Tests:**

```bash
pytest
```

60 tests total: the original 19 Test 1 tests (unmodified, still passing
against the current graph) plus 41 Test 2 tests covering ingestion, CRM
matching/conflicts, staff routing, the expanded graph
(technical/operations/infrastructure routing, likely/possible match
handling, conflict preservation, owner recommendation, audit content), the
`infrastructure`-misclassification safeguard, and — most importantly — that
no CRM mutation ever happens before approval, and that a rejected approval
blocks both new-record creation and attaching to an existing record. No test
calls the real OpenAI API — `app.llm` is stubbed with `monkeypatch`
throughout.

## Project structure

```
data/
  staff_directory.json   Fixture: 4 staff members and what they own
  crm.csv                Fixture: 5 existing (fictional) CRM records
  emails.json             Fixture: 12 inbound enquiries
  documents/               Fixture: 3 referenced attachments
  audit.db                  Runtime SQLite audit trail (gitignored)
app/
  models.py       Pydantic models — enquiry, triage, CRM, matching, staff
  ingest.py       Loads the fixture dataset verbatim; email → RawEnquiry
  llm.py          OpenAI calls: structured triage/extraction + drafting
  validation.py   Deterministic required-field checks, confidence, routing
  crm.py          Mock CRM (in-memory): lead/ticket creation, attach, exact-dup check
  crm_match.py    Multi-signal CRM matching + conflict detection (read-only)
  staff.py        Deterministic owner recommendation from staff_directory.json
  audit.py        SQLite audit trail
  approval.py     Human-in-the-loop approval gate (dry-run auto-approve)
  graph.py        LangGraph state + node wiring (analyze_crm is read-only;
                  apply_crm_action is the only node that mutates the CRM,
                  and only runs after human_approval)
  scenarios.py    Four Test 1 synthetic demo enquiries
demo.py            Test 1: runs the four scenarios end-to-end
run_dataset.py     Test 2: runs the 12-enquiry dataset + CLI inspection
tests/             pytest suite (Test 1 + Test 2, 60 tests)
conftest.py        Isolates the audit DB and mock CRM state per test
```

## Out of scope

Per [CLAUDE.md](CLAUDE.md): no auth, user accounts, billing, web UI,
deployment, public API, real CRM/email/messaging integration, multi-agent
architecture, or multiple LLM providers. This is a workflow-architecture
demo, not a product.
