**Candidate email:** shkhahmad64@gmail.com

# NOTES

## 1. How to build and run it

```bash
# Local (no Docker)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python gateway/llm_gateway.py          # terminal 1 — stub LLM, no key needed
BOOK_PATH=data/client_book.json MARKET_PATH=data/market_data.json \
  LLM_BASE_URL=http://localhost:8600/v1 LLM_API_KEY=test \
  AGNO_TELEMETRY=false uvicorn app:app --host 0.0.0.0 --port 8080  # terminal 2
pytest tests/ -v                       # tests

# Docker (grading topology — network-isolated, matches real grading exactly)
docker compose -f docker/compose.grading.yml up --build

# Offline scoring
python harness/run_assessment.py --service http://localhost:8080 \
  --gateway http://localhost:8600 --questions questions/practice_questions.jsonl \
  --out runs/latest
python harness/score.py --key harness/practice_key.json \
  --leakmap harness/practice_leakmap.json \
  --transcript runs/latest/transcript.jsonl \
  --usage runs/latest/gateway_usage.json --roster runs/latest/roster.json
```

## 2. Architecture

FastAPI app (`app.py`) wrapping seven Agno agents: `router`, `book_qa`, `kyc_profile`,
`notes_desk`, `market_desk`, `compliance`, and `verifier`. The `router` is a pure-Python
regex classifier — no LLM call — that maps each question to one or more specialists and
decides whether to escalate to `valura-deep`. Every specialist reads exclusively through
`DataLoader` (`data.py`), scoped to the request's `client_id`. All arithmetic (balances,
counts, drift, sector exposure) is computed deterministically in Python; the LLM only
rephrases an already-committed answer into natural language. `compliance` intercepts advice
and cross-client requests before any specialist runs. `verifier` re-checks citations and
scans for cross-client leakage before the response leaves the service, downgrading to an
abstain rather than shipping an unverified answer.

## 3. Decisions made rather than derived, and open questions

- **Regex router, not an Agno `Team`.** Keeps dispatch fast, cheap, and fully testable
  without a live model. Each specialist is a real `agno.agent.Agent`; the orchestration layer
  above them is not. I'd want to clarify how strictly "genuinely Agno" is interpreted.
- **Symbol → `market_desk` always.** Any recognised ticker routes to `market_desk` even when
  the answer doesn't need a price (e.g. "date of first AAPL purchase"), biasing toward richer
  context at some routing-precision cost.
- **`_fallback()` unconditionally abstains.** Even if the LLM returns something plausible,
  the fallback always sets `abstained=True`. A free-form LLM value can't be validated against
  the schema without risking hallucination; the right fix is always to extend a dispatch regex,
  not to trust unvalidated output.
- **Citation truncation.** Answers backed by more than 6 records cite `client_id` instead,
  per the brief, via a shared `format_citations()` helper.

## 4. Required questions

**How does the service decide it cannot answer, and how is that different from the model being unsure?**
Abstention is decided entirely in Python before any LLM call: the specialist checks explicit
conditions (symbol not in covered set, field absent, no matching records) and sets
`abstained=True, answer_value=None`. The LLM never gets a vote — it only rephrases what the
code has already committed to. "The model wasn't sure" is not a possible failure mode here.

**A note instructs you to disclose something. At which layer is that neutralised?**
Three layers: `detect_injection()` in `data.py` flags suspicious note content before it is
used; note text is sent to the LLM inside an explicit inert data block, not as an instruction;
`VerifierAgent` scans the drafted answer for leakage and disclosure patterns before it ships.
All three would have to fail on the same question simultaneously for an injected instruction
to reach the user.

**Provider down for an hour — which answers get worse, slower, or unaffected?**
*Unaffected:* all Python-computed values (balances, counts, drift, masking, scope locks) —
none depend on the LLM. *Slower:* questions in the transient rate-limit band, which retry
with exponential backoff before succeeding. *Worse (phrasing only, not correctness):* during
a full blackout `_llm_format` catches the connection error and falls back to the raw
precomputed string; the number is never wrong, only less polished, and the response carries
`flags: ["upstream_issue"]`.

**What did Agno make easy, hard, and what required reading the source?**
Easy: wiring `OpenAIChat` to the stub gateway — pointing `base_url` at the local server just
worked. Hard: there is no built-in way to force a specialist to abstain from within a Team;
the per-specialist abstain logic had to live in each agent's own code. Source-required: by
default `Agent.run()` POSTs telemetry to `os-api.agno.com`. I found this only by reading
`agno/agent/__init__.py` after noticing the outbound call in logs. It matters directly here
because the grading network sets `internal: true` (no external routes), so every agent call
would have silently failed to phone home and added latency. `AGNO_TELEMETRY=false` in the
Dockerfile fixes it.

## 5. What I would do next, and what I know is weak

**Next:** replace the regex router with a genuine Agno `Team`/delegation graph so
orchestration is handled by the framework; add date indices to `DataLoader` for faster
temporal queries; fix the multi-agent combine step to prefer a non-abstained specialist answer
over an abstained one (currently costs q_084).

**Weak:** the regex router is the least Agno-native part and the one I'd most want to
revisit; free-text answer quality has only been validated against the stub (93.1/96 on the
practice set machine score — `judged_quality` requires a live upstream key not available
locally).