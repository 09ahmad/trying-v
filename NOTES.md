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

### 5. Next Steps & Known Weaknesses
- **Known Weakness**: The top-level `AgentRouter` relies on regex keyword matching rather than Agno's native `Team` delegation or LLM routing primitives. While microsecond-fast and budget-efficient, regex dispatch can fall through on unexpected phrasing (as addressed in Fix 3).
- **Future Improvements**: Implement a hybrid router combining regex fast-path dispatch with an Agno LLM triage agent for ambiguous queries, expand unit tests for multi-intent questions, and optimize temporal index lookups.
