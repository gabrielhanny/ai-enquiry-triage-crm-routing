# AI Enquiry Triage & CRM Routing System

## Project Context

This is a small demo prototype for an AI systems reasoning assessment and a potential portfolio project.

The system demonstrates how multi-channel business enquiries can be intelligently triaged, structured, validated, routed toward CRM workflows, and safely handled with human approval.

The project is intentionally a **demo prototype**, not a production SaaS.

## Core Workflow

```text
Email / Web Form / Messaging
        ↓
Normalize Enquiry
        ↓
LLM Triage + Structured Extraction
        ↓
Deterministic Validation
        ↓
Conditional Routing
   ├── Incomplete → Draft Clarification → Human Approval
   ├── Junk → Archive
   └── Sales / Support → Duplicate Check
                              ↓
                         Mock CRM Action
                              ↓
                         Draft Response
                              ↓
                       Human Approval
                              ↓
                          Audit Log
```

## Technology

* Python
* LangGraph
* OpenAI API
* Pydantic
* python-dotenv
* SQLite
* pytest

Use LangGraph because the workflow contains explicit state, conditional routing, approval gates, and failure/recovery paths.

Do not introduce additional frameworks unless clearly necessary.

## LLM Responsibilities

The LLM may:

* classify enquiries
* extract structured information
* identify likely missing information
* draft responses

The LLM must NOT be the source of truth for:

* permissions
* CRM identity
* duplicate decisions
* approval status
* consequential actions
* audit records

## Deterministic Responsibilities

Application code must control:

* schema validation
* required fields
* confidence thresholds
* routing rules
* duplicate handling
* CRM operations
* human approval
* audit logging
* retries / failure handling
* safe execution

## Safety

This prototype must never:

* send real emails
* modify a real CRM
* access real customer data
* expose API keys
* perform consequential external actions autonomously

Use synthetic data and mock/local adapters.

The default behavior should be safe / dry-run.

The OpenAI API key is loaded from `.env` and must never be hardcoded or committed.

## Scope

Build only what is necessary to demonstrate the core workflow.

Do NOT build:

* SaaS functionality
* authentication
* user accounts
* billing
* web UI
* deployment
* public API
* real CRM integration
* real email integration
* real messaging integration
* multi-agent architecture
* multiple LLM providers
* unnecessary abstractions

## Demo Scenarios

The prototype should demonstrate:

1. Complete sales enquiry
2. Incomplete sales enquiry
3. Support enquiry
4. Junk enquiry

Each scenario should exercise an appropriate conditional path.

## Implementation Priority

Priority order:

1. Data models
2. LangGraph state
3. Core workflow
4. Conditional routing
5. Mock CRM
6. Audit trail
7. Tests
8. README/documentation

Keep implementation small, readable, and easy to explain in a technical interview.

## Assessment Principle

The goal is not to demonstrate how much code can be produced.

The goal is to demonstrate:

* sound AI system architecture
* appropriate use of LLMs
* deterministic controls around LLM behavior
* reliable workflow orchestration
* human-in-the-loop judgement
* failure handling
* security awareness
* cost and latency awareness

Prefer the simplest design that convincingly demonstrates these principles.
