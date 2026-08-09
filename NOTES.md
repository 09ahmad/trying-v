# Candidate Information

**Email**: shkhahmad64@gmail.com

---

# Valura AI Arena — Solution Architecture & Design Notes

## Executive Summary

This repository contains a production-ready, multi-agent HTTP service built on the **Agno framework** for the Valura AI Take-Home Assessment. The service acts as an intelligent wealth desk, answering complex financial, KYC, market data, and portfolio questions while strictly enforcing data privacy, scope locks, and regulatory compliance.

---

## Key Architecture & Design Choices

### 1. Multi-Agent Ecosystem (Agno)

The solution implements a 7-agent team using the Agno framework:

- **`router` (AgentRouter)**: Always runs first. Uses a rule-based classifier for microsecond dispatch to specialists, avoiding waste of model call budget. Routes advice and cross-client requests directly to `compliance`.
- **`book_qa` (BookAgent)**: Multi-agent financial book specialist using Agno. Computes all balances, cash transactions, holdings, and rebalance drifts deterministically in Python to prevent LLM hallucination, then formats the output naturally via `valura-fast` or `valura-deep`.
- **`kyc_profile` (KYCProfileAgent)**: Manages customer identity and KYC profile lookups. All sensitive identifiers pass through a single, un-bypassable masking function.
- **`notes_desk` (NotesDeskAgent)**: Reads unstructured relationship notes and transaction memos. Employs injection resistance to treat all note text as data rather than instructions.
- **`market_desk` (MarketDeskAgent)**: Handles instrument prices, sectors, return calculations, and news items. Enforces strict coverage-gap abstentions for instruments outside `covered_symbols`.
- **`compliance` (ComplianceAgent)**: Enforces regulatory compliance and scope boundaries. Distinctly handles out-of-scope cross-client access and personalized investment advice.
- **`verifier` (VerifierAgent)**: Post-processes every drafted response before it leaves the service. Validates schema invariants, verifies confidence scores, deduplicates citations, and performs a final cross-client safety sweep.

### 2. Data Layer & Scope Enforcement (`takehome_service/data.py`)

- **Data Loading**: `DataLoader` reads `client_book.json` and `market_data.json` **once at startup**. Per-request disk reads are completely eliminated.
- **Scope Lock**: Scope boundaries are enforced in Python at the data access layer. Every database accessor requires a `client_id` parameter and filters data strictly for that client.
- **Masking System**: A single shared function `mask_sensitive(val)` converts bank account numbers, PANs, and identity numbers into the canonical `****XXXX` format (4 stars followed by the last 4 characters).
- **Month-Start Price Semantics**: Market prices are indexed by instrument symbol and month-start close dates. Lookups use the most recent close on or before the requested target date.

### 3. Safety, Refusals & Failure Handling

- **Advice vs Arithmetic Refusal**: Advice requests (e.g., "should I buy AAPL") trigger a policy refusal (`refused=True`), whereas factual drift calculations ("current vs target allocation") return exact mathematical answers (`refused=False`).
- **Blackout Resiliency**: Handles gateway `429` rate limits with exponential backoff and Retry-After headers. For quota-exhausted blackouts (`insufficient_quota`), the client immediately raises `BlackoutError`, returning `abstained=True` and setting the `upstream_issue` flag.
- **Prompt Injection Defense**: Notes Desk flags instruction-like patterns in note bodies, ensuring adversarial texts in relationship notes cannot hijack system instructions.

---

## Local Validation Loop

The solution was continuously validated using the offline assessment harness against practice data:

```bash
# 1. Start LLM Gateway (stub mode)
python gateway/llm_gateway.py

# 2. Start Service
PORT=8080 python -m uvicorn app:app --host 0.0.0.0 --port 8080

# 3. Run Assessment
python harness/run_assessment.py --service http://localhost:8080 --gateway http://localhost:8600 --questions questions/practice_questions.jsonl --out runs/latest

# 4. Score Trajectory
python harness/score.py --key harness/practice_key.json --leakmap harness/practice_leakmap.json --transcript runs/latest/transcript.jsonl --usage runs/latest/gateway_usage.json --roster runs/latest/roster.json
```

Automated PyTest suite covers all core modules:
```bash
pytest tests/ -v
```
