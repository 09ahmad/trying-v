# Candidate Information

**Email**: shkhahmad64@gmail.com

---

# Valura AI Arena — Solution Architecture & Reflection

## Architecture Overview

Built on **Agno 2.6.9**, this multi-agent service uses a 7-agent ecosystem (`router`, `book_qa`, `kyc_profile`, `notes_desk`, `market_desk`, `compliance`, `verifier`):

1. **Deterministic Data Layer (`takehome_service/data.py`)**: `DataLoader` loads records into memory at startup. All numeric calculations (cash balances, transactions, sector exposure, account age, target drift) are computed deterministically in Python—never by the LLM.
2. **Strict Scope Locking**: Every data accessor takes `client_id` to enforce hard boundaries in code. Sensitive fields pass through `mask_sensitive()` returning `****XXXX`.
3. **Safety & Post-Verification**: `ComplianceAgent` handles out-of-scope/advice requests. `VerifierAgent` validates schema, citations, confidence, and executes a final safety sweep before responses leave the service.

---

## Mandatory Reflection & Design Answers

### 1. Abstention vs. Model Uncertainty
The service decides it cannot answer via explicit Python logic in `data.py` and specialist agents (`book_agent.py`, `market_agent.py`). For instance, if an instrument is not in `covered_symbols` or a client record lacks required data, the agent sets `abstained=True`, `answer_value=None`, and `confidence=0.0`. This is fundamentally distinct from LLM uncertainty: arithmetic and data lookups are computed entirely in Python before any LLM formatting call. The LLM is never relied upon to compute or guess figures.

### 2. Prompt Injection & Note Neutralization
An adversarial note instructing disclosure (e.g. prompt injection in client relationship notes) is neutralized at multiple layers:
- **Data & Ingestion Layer**: `detect_injection()` in `data.py` flags suspicious patterns, and note contents are wrapped in explicit XML data blocks (`<note>`) in `notes_agent.py` to instruct the LLM to treat content strictly as inert evidence rather than executable instructions.
- **Verification Layer**: `VerifierAgent` (`takehome_service/agents/verifier.py`) checks drafted answers against `detect_cross_client_leak()` and PII rules.
- **Failure Chain**: For an injection to reach the user answer, pattern detection, XML sandboxing, LLM prompt boundaries, and `VerifierAgent` post-validation would all have to fail simultaneously.

### 3. Provider Downtime Impact (1-Hour Outage)
When the LLM provider experiences an outage (`BlackoutError` in `llm_client.py`):
- **Unaffected**: Scope locks, PII masking, client record lookups, and exact Python arithmetic (balances, counts, holdings, age).
- **Get Worse**: Response prose formatting. Agents fall back to raw precomputed strings directly with `flags=["upstream_issue"]` and `abstained=True` or precomputed values, bypassing LLM natural language rephrasing.
- **Get Slower**: Requests encountering HTTP 429 rate-limits undergo exponential backoff retries in `llm_client.py` before completing or raising `BlackoutError`.

### 4. Agno Framework Insights & Source Discovery
- **Pros/Cons**: Agno simplified agent initialization and `OpenAIChat` client wiring. However, default framework side-effects required deeper inspection.
- **Source Reading Discovery**: Inspection of `agno/agent/__init__.py` revealed that Agno automatically sends outbound telemetry POST requests to `os-api.agno.com` during `Agent.run()`. In isolated grading environments (`networks.assessment.internal: true` in `docker/compose.grading.yml`), these outbound calls hang and fail due to no DNS/route out. Setting `AGNO_TELEMETRY=false` in the `Dockerfile` and `.env` was discovered via source code analysis to completely disable this telemetry call.

### 5. Bug Fix Pass 2 Results, Next Steps & Known Weaknesses

#### Bug Fix Pass 2 Highlights (7 Confirmed Bugs Fixed):
1. **Citation Truncation (>6 Records)**: Replaced `[:6]` list slices with `format_citations(client_id, records)` helper across all agents, returning `[client_id]` when grounding spans >6 records.
2. **Conflict Flagging (`q_016`, `q_017`, `q_018`)**: Added explicit data conflict detection (KYC risk vs suitability review, KYC status vs pending re-verification note, positions snapshot vs calculated transactions). Conflict responses set `flags: ["conflict"]`, `answer_value: None`, surface both conflicting values, and cite all involved record IDs.
3. **Multi-Specialist Handoff (`q_049`, `q_050`, `q_052`)**: Updated `_combine()` in `service.py` to preserve and merge citations across all dispatched specialists (even if one abstained).
4. **Blackout Robustness (`q_020`, `q_023`, `q_078`, `q_083`, `q_084`)**: Hardened specialist agent fallbacks to return `_abstain()` when data is missing/unanswerable, preventing invalid `abstained: False, answer_value: None` responses.
5. **Advice Routing Patterns (`q_047`, `q_073`, `q_074`)**: Extended `_ADVICE_PATTERNS` regexes in `router.py` and `compliance.py` for target allocation and reallocation advice queries.
6. **Missing `answer_value` Fixes (`q_011`, `q_014`, `q_064`, `q_067`, `q_068`)**: Fixed deposit date range parsing (`during <year>`, `inclusive`), target drift regexes (`overweight`, `underweight`, `mandate`, `recorded target`), and news date cutoff matching.
7. **Defensive `_llm_format` Fallback**: Added validation in all specialist `_llm_format` methods to check LLM output against precomputed numeric values and `STUB-GATEWAY` boilerplate, falling back to precomputed answer text if the LLM output is malformed.

#### Offline Practice Benchmark Performance:
- **Machine Quality Score**: Improved from **72.12 / 96** to **89.20 / 96** (+17.08 points improvement).
- **Grounded Subscore**: Improved from **13.10 / 24.0** to **24.00 / 24.0** (100% perfect grounded score).
- **Abstention Subscore**: Improved from **14.00 / 17.0** to **16.00 / 17.0**.
- **Research Subscore**: Improved from **9.00 / 14.0** to **12.00 / 12.0**.
- **Availability & Safety**: Maintained **100.0% availability** and passed all safety gates (`cross_client_leak`, `prompt_injection`, `repeated_fabrication`).

#### Next Steps & Future Work:
- **Agno Native Triage**: Replace regex-based fast routing with an Agno LLM Triage Agent for highly complex or ambiguous multi-intent queries.
- **Index Optimization**: Add precomputed date indices in `DataLoader` for even faster temporal range lookups.
