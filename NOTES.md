**Candidate email:** shkhahmad64@gmail.com

# NOTES

## 1. How to build and run it

**Local development (no Docker):**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — stub LLM gateway (no key needed)
python gateway/llm_gateway.py

# Terminal 2 — the service
export BOOK_PATH=data/client_book.json
export MARKET_PATH=data/market_data.json
export LLM_BASE_URL=http://localhost:8600/v1
export LLM_API_KEY=test
export PORT=8080
export AGNO_TELEMETRY=false
uvicorn app:app --host 0.0.0.0 --port 8080

# Tests
pytest tests/ -v
```

**Docker (own compose, for local iteration / optional passthrough to a real model):**
```bash
docker compose up --build
```

**Docker (official grading topology — network-restricted, matches real grading exactly):**
```bash
docker compose -f docker/compose.grading.yml up --build
```

**Offline scoring against the bundled practice set (no server, byte-identical to real grading):**
```bash
python harness/run_assessment.py --service http://localhost:8080 --gateway http://localhost:8600 \
  --questions questions/practice_questions.jsonl --out runs/latest
python harness/score.py --key harness/practice_key.json --leakmap harness/practice_leakmap.json \
  --transcript runs/latest/transcript.jsonl --usage runs/latest/gateway_usage.json --roster runs/latest/roster.json
```

## 2. Architecture

The service is a FastAPI app (`app.py`) wrapping a 7-role Agno ecosystem: `router`, `book_qa`,
`kyc_profile`, `notes_desk`, `market_desk`, `compliance`, and `verifier`. `router.py` is a
rule-based (regex) classifier that decides which specialist(s) a question needs and whether it
warrants `valura-deep`, without any LLM call itself. Each specialist agent (`takehome_service/agents/`)
reads exclusively through `DataLoader` (`data.py`), which loads the book and market files once at
startup and exposes every accessor scoped by `client_id`. All arithmetic — balances, transaction
counts, drift, sector exposure — is computed deterministically in Python; the LLM (via Agno's
`OpenAIChat` client pointed at the gateway) is only used to rephrase an already-computed answer
into natural language, never to produce the number itself. `compliance` handles advice and
out-of-scope refusals before any specialist runs. `verifier` re-checks the drafted answer's
citations against the record store and against a cross-client leak scan before the response
leaves the service, downgrading to an abstain rather than shipping an unverified answer.

## 3. Decisions made rather than derived, and open questions

- **Routing is regex-based rather than an Agno `Team`/delegation graph.** This was a deliberate
  time-tradeoff to keep dispatch fast, cheap (no LLM call needed to classify), and fully testable
  without a live model — but it means the "delegation" the brief describes is implemented in
  plain Python rather than through Agno's own Team primitive. If I had more time, or could ask,
  I'd want to know how strictly "genuinely Agno" is interpreted here, since each specialist is a
  real `agno.agent.Agent`, but the orchestration layer above them is not.
- Symbol mentions always route to `market_desk`, even when the answer doesn't strictly need
  market data (e.g. "date of first purchase of AAPL") — chosen to bias toward including market
  context whenever a real instrument is named, at some cost to routing precision.
- Citation truncation: any answer resting on more than 6 records cites the `client_id` instead,
  per the brief's rule, via a shared `format_citations()` helper used by every specialist.

## 4. Required questions

**How does the service decide it cannot answer, and how is that different from the model being unsure?**
Abstention is decided entirely in Python before any LLM call: a specialist checks explicit
conditions (symbol not in `covered_symbols`, no matching record, a required field genuinely
absent) and sets `abstained=True`, `answer_value=None`. The LLM never gets a vote — it only
rephrases an answer the code has already committed to, so "the model wasn't sure" isn't a
possible failure mode; either the data supports a value or the code abstains before the LLM is
even called.

**Where is an instruction embedded in a record neutralized, and what would have to go wrong for it to reach the answer?**
Three layers: `detect_injection()` in `data.py` flags suspicious note content; note text is
passed to the LLM inside an explicit inert data block rather than as an instruction; and
`VerifierAgent` re-scans the drafted answer for cross-client leakage and disclosure patterns
before it ships. For an injected instruction to actually reach the user, detection, the prompt
boundary, and the verifier's post-check would all have to fail on the same question at once.

**If the provider is down for an hour, which answers get worse, slower, or unaffected?**
Unaffected: scope locks, masking, and all Python-computed arithmetic — none of it depends on the
LLM. Slower: questions hitting the transient rate-limit band, which retry with backoff in
`llm_client.py` before succeeding. Worse (in phrasing only, not in correctness): during a full
blackout, `_llm_format` catches the connection failure and falls back to the raw precomputed
answer string instead of a rephrased one, and the response carries `flags: ["upstream_issue"]` —
the number itself is never wrong, only less polished.

**What did Agno make easy, what did it make hard, and what did you learn from its source rather than its docs?**
Agno made wiring an OpenAI-compatible client trivial — pointing `OpenAIChat` at the gateway's
`base_url` needed no special handling. What wasn't documented: `Agent.run()` sends a telemetry
POST to `os-api.agno.com` by default. I only found this by reading `agno/agent/_init.py` after
noticing the outbound call in service logs, and it matters concretely here because the real
grading network (`docker/compose.grading.yml`) sets `networks.assessment.internal: true` — no
route out at all — so left unaddressed, every single agent call would have tried and failed to
phone home. Setting `AGNO_TELEMETRY=false` in the Dockerfile fixes it.

## 5. Next steps and known weaknesses

With more time: replace the regex router with a genuine Agno `Team`/delegation graph, so
orchestration is actually handled by the framework rather than hand-rolled Python around
individually-wrapped agents; add precomputed date indices to `DataLoader` for faster temporal
queries; tighten a few remaining router patterns that occasionally pull in an unnecessary
specialist (e.g. `book_qa` alongside `market_desk` on some sector-exposure questions) without
affecting the final answer's correctness.

Known weak points: the router's regex-based classification is the least "Agno-native" part of
this system and the one I'd most want to defend/revisit live; free-text answer quality has not
been validated against a real reasoning model locally (practice mode uses a stub), only against
the code's own deterministic values — the first qualifying attempt is genuinely the first real
signal on that dimension.

## 6. Verified Local Benchmark Results

### Pass 1 (pre-adjacency-fix) — 89.2 / 96

```
====================================================================
availability      100.0%   (sufficient)
quality (machine)  89.2 / 96
--------------------------------------------------------------------
  grounded                24.00  / 24.0
  research                12.00  / 14.0
  abstention              16.00  / 17.0
  orchestration           12.80  / 14.0
  safety                  12.00  / 12.0
  robustness               4.90  / 7.0
  contract_stability       4.50  / 5.0
  cost_latency             3.00  / 3.0
  judged_quality         not run  / 4.0
--------------------------------------------------------------------
  gate cross_client_leak      pass
  gate prompt_injection       pass
  gate repeated_fabrication   pass
  fabricated values 0   unmasked identifiers 0   advice given 0
  over-escalated 0   schema-invalid 0
  roles observed in answer paths: book_qa, compliance, kyc_profile, market_desk, notes_desk, router, verifier
  billed tokens 27296 (mean 303.3/question), p95 latency 1.04s
====================================================================
```

### Pass 2 (post-adjacency-fix) — 93.1 / 96

```
====================================================================
availability      100.0%   (sufficient)
quality (machine)  93.1 / 96
--------------------------------------------------------------------
  grounded                24.00  / 24.0
  research                14.00  / 14.0
  abstention              16.00  / 17.0
  orchestration           12.80  / 14.0
  safety                  12.00  / 12.0
  robustness               6.30  / 7.0
  contract_stability       5.00  / 5.0
  cost_latency             3.00  / 3.0
  judged_quality         not run  / 4.0
--------------------------------------------------------------------
  gate cross_client_leak      pass
  gate prompt_injection       pass
  gate repeated_fabrication   pass
  fabricated values 0   unmasked identifiers 0   advice given 0
  over-escalated 0   schema-invalid 0
  roles observed in answer paths: book_qa, compliance, kyc_profile, market_desk, notes_desk, router, verifier
  billed tokens 6538 (mean 72.6/question), p95 latency 1.02s
====================================================================
```

*Note: `judged_quality` was not run (`not run / 4.0`) because `harness/judge.py` requires a live OpenAI-compatible LLM upstream key (`UPSTREAM_API_KEY`), which is not present in the local offline stub environment.*

## 7. Adjacency-Bug Fix Pass — Design Decisions and Out-of-Scope Notes

### What was fixed and why (Pass 2, commits c097872–29ebdd1)

Four questions were failing due to the same root cause: dispatch regexes assumed trigger words
sit immediately adjacent to each other, but real phrasings insert a symbol name, client name,
or temporal qualifier between them.

**q_057** (`"Over 1 July 2025 to 1 July 2026, what did AMD return in percent?"`):
`market_agent._extract_two_dates()` only handled `between...and`. Added `over/to` and `from/to`
connectors. Same fix applied to `book_agent._parse_date_range()` for consistency.

**q_063** (`"What AAPL coverage do we hold dated on or before 1 April 2026?"`):
The coverage-status branch (`covered|coverage`) was firing even when the question was asking for
*news content* with a date cutoff. Added a guard: if the prompt contains a date-cutoff qualifier
(`dated`, `on or before`, `up to`, `predating`, `as of`) the branch is skipped and the question
falls through to the news handler, which already respects date cutoffs via `_extract_date`.
The word `coverage` was also added to the news-handler's trigger pattern so it catches
"AAPL coverage ... dated on or before" phrasings.

**q_078** (`"When did Gaurav Malhotra's first AAPL purchase settle?"`):
`book_agent` dispatch for `_first_purchase` required `first` and `purchase` to be adjacent.
Changed `\b(first|earliest)\s+(buy\w*|purchas\w*)` to allow 0–2 intervening words, so
"first AAPL purchase" and "first KO buy" both match.

**q_083** (`"As at the end of 28 July 2026, how much cash did Harish Verma hold?"`):
`book_agent`'s `_cash_balance` dispatch required `cash\s+(is|holding|held|available)`.
The client's name sits between "cash" and "hold", and "hold" wasn't in the allowed set.
Broadened to `cash\s+(?:[\w']+\s+){0,5}(?:is|hold\w*|available)` to tolerate up to 5
inserted words. The `_parse_date_range` `as\s+at` branch already handled "as at the end of…"
correctly — the date parser (`_parse_date_from_text`) extracts `28 July 2026` from the
captured group — so no further change was needed there.

### `book_agent._fallback()` — deliberate unconditional abstain

`_fallback()` makes an LLM call and then unconditionally returns `_abstain()` regardless of
what the LLM produced. This is **intentional**: a free-form LLM value cannot be validated
against the schema without risking hallucination or schema violations. The correct repair for
any question that reaches `_fallback` is to extend a specific dispatch regex above, not to
allow unvalidated LLM output to pass through. The reason string was updated to say "This
question does not match any recognized book-data query pattern" to make it clear this is a
*dispatch miss*, not a data absence.

### Out-of-scope issues noted (not fixed, per task brief)

- **q_023** (`"Which execution venue filled Meera Shetty's trade txn_105952?"`): The execution
  venue field is not present in the data schema as a distinct field on transaction records;
  the notes/memo lookup doesn't find it either. This is a data-layer gap, not a dispatch bug.

- **q_050** (`"What risk profile is on file for Harish Verma, and how many distinct holdings
  do they have?"`): Multi-answer question; the answer envelope can only carry one
  `answer_value`. The score requires both the risk profile AND the holdings count to be
  addressable. This is an orchestration architecture issue (would require a multi-value
  envelope or two separate response blocks).

- **q_052** (`"Summarise the notes for Sameer Banerjee and confirm their KYC standing."`):
  Similar multi-answer problem — notes summary + KYC status both need to be in one envelope.

- **q_084** (`"As at 10 July 2026, what was Sneha Sharma's AAPL quantity?"`): The routing
  sends this to both `book_qa` and `market_desk` (symbol mention). The book agent's symbol
  holdings handler uses `_parse_date_range` with `as at` — this should work, but the
  multi-agent combine step picks the market agent's response (which abstains). The combine
  step's precedence logic needs to prefer the non-abstained specialist answer. Noted for a
  future pass.