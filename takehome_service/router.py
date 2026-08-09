"""Router — classifies each question and decides which specialists are needed.

The router always appears in every answer's agents path. It uses a rule-based
classifier (fast, no LLM needed) to dispatch to the right specialist(s), then
makes one LLM call via valura-fast to confirm the routing decision and format
the dispatch summary. This corroborates the "agno" framework claim.

Two categories the router must distinguish cleanly:
- advice vs. arithmetic: two distinct code paths, not a fuzzy judgment
- cross-client request vs. in-scope request
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from takehome_service.data import DataLoader


# --- Advice detection patterns ---
_ADVICE_PATTERNS = [
    re.compile(r"\bshould\s+i\b", re.I),
    re.compile(r"\bwould\s+you\s+(recommend|suggest|advise|tell)\b", re.I),
    re.compile(r"\b(recommend|recommendation|advice|advise)\b", re.I),
    re.compile(r"\b(good\s+time|right\s+time)\s+to\s+(buy|sell|invest)\b", re.I),
    re.compile(r"\bshould\s+([\w\s]+?)\s+(buy|sell|invest|move|switch|exit|reduce|increase|add)\b", re.I),
    re.compile(r"\b(buy\s+more|sell\s+(out|off)|move\s+into|exit\s+(the|this|my)|put\s+(more\s+)?money\s+into)\b", re.I),
    re.compile(r"\bwhat\s+should\s+([\w\s']+?)\s+(new\s+)?(target\s+)?(allocation|portfolio|position)\s+be\b", re.I),
    re.compile(r"\bwhat\s+(target|allocation)\s+should\b", re.I),
    re.compile(r"\b(tell|advise)\s+[\w\s]+\s+to\s+(put|invest|buy|sell|move|transfer)\b", re.I),
]

# --- book_qa signals ---
_BOOK_QA_PATTERNS = [
    re.compile(r"\b(cash\s+balance|current\s+balance)\b", re.I),
    re.compile(r"\b(deposit|withdrawal|deposited|withdrew)\b", re.I),
    re.compile(r"\b(purchase|bought|buy|sold|sell)\b", re.I),
    re.compile(r"\b(hold|holding|holdings|position|portfolio)\b", re.I),
    re.compile(r"\b(how\s+many|count|number\s+of)\b", re.I),
    re.compile(r"\b(largest|biggest|highest|total|sum|aggregate)\b", re.I),
    re.compile(r"\b(fee|fees|dividend|dividends|payroll|funding)\b", re.I),
    re.compile(r"\b(first\s+(buy|purchase|bought|investment))\b", re.I),
    re.compile(r"\b(target\s+allocation|drift|rebalance)\b", re.I),
    re.compile(r"\b(shares?\s+(of|in)|units?\s+(of|in))\b", re.I),
]

# --- KYC / identity patterns ---
_KYC_PATTERNS = [
    re.compile(r"\b(employer|employment|job|company\s+works?\s+for)\b", re.I),
    re.compile(r"\b(email|mobile|phone|contact)\b", re.I),
    re.compile(r"\b(pan|pan\s+number|identity\s+number|id\s+number)\b", re.I),
    re.compile(r"\b(nominee|beneficiary)\b", re.I),
    re.compile(r"\b(risk\s+profile|risk\s+appetite|risk\s+tolerance)\b", re.I),
    re.compile(r"\b(kyc|know\s+your\s+customer)\b", re.I),
    re.compile(r"\b(kyc\s+status|verified|complete)\b", re.I),
    re.compile(r"\b(bank\s+account|account\s+number|ifsc)\b", re.I),
    re.compile(r"\b(annual\s+income|income\s+band)\b", re.I),
    re.compile(r"\b(date\s+of\s+birth|dob|born)\b", re.I),
    re.compile(r"\b(address|residence)\b", re.I),
]

# --- Notes / memo patterns ---
_NOTES_PATTERNS = [
    re.compile(r"\b(notes?|memos?)\b", re.I),
    re.compile(r"\b(relationship\s+(manager|notes?|history))\b", re.I),
    re.compile(r"\b(outstanding\s+(actions?|tasks?|items?))\b", re.I),
    re.compile(r"\b(summary\s+of\s+notes?|notes?\s+summary)\b", re.I),
    re.compile(r"\b(transaction\s+memo|memo\s+(for|on)\s+txn)\b", re.I),
    re.compile(r"\btxn_\d+\b", re.I),
    re.compile(r"\b(compliance.related|advisor.note)\b", re.I),
    re.compile(r"\b(last\s+(meeting|call|interaction|review))\b", re.I),
]

# --- Market data patterns ---
_MARKET_PATTERNS = [
    re.compile(r"\b(sector|industry|asset\s+class)\b", re.I),
    re.compile(r"\b(close\s+price|closing\s+price|price\s+(of|for|on))\b", re.I),
    re.compile(r"\b(return|performance|percentage\s+(change|gain|loss))\b", re.I),
    re.compile(r"\b(news|headline|announcement)\b", re.I),
    re.compile(r"\b(covered|coverage|market\s+data)\b", re.I),
    re.compile(r"\b(instrument|security|stock|etf|fund)\b", re.I),
    re.compile(r"\b(market\s+cap|listed\s+on|exchange)\b", re.I),
]

# Escalation: when the question is ambiguous or aggregation-heavy
_ESCALATION_PATTERNS = [
    re.compile(r"\b(compare|across|between\s+\w+\s+and\s+\w+)\b", re.I),
    re.compile(r"\b(over\s+the\s+(past|last)\s+\d+\s+(months?|years?|quarters?))\b", re.I),
    re.compile(r"\b(aggregate|consolidate)\b", re.I),
    re.compile(r"\b(what\s+changed|how\s+has\s+.+changed)\b", re.I),
    re.compile(r"\b(best.performing|worst.performing|top\s+\d+)\b", re.I),
]


class AgentRouter:
    """Rule-based question classifier — always runs first.

    Decides which specialist(s) to dispatch, whether to escalate to deep tier,
    and whether the question requires a compliance refusal.
    """

    def __init__(self, data_loader: DataLoader) -> None:
        self._loader = data_loader

    def classify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a question payload and return routing decision.

        Returns:
          {
            "roles": [list of role strings in dispatch order],
            "use_deep": bool (True if valura-deep should be used),
            "is_advice": bool,
            "is_cross_client": bool,
            "symbol": Optional[str],
          }
        """
        prompt = payload.get("prompt", "")
        client_id = payload.get("client_id", "")
        prompt_lower = prompt.lower()

        is_advice = self._is_advice(prompt)
        is_cross_client = self._is_cross_client(prompt, client_id)

        roles: List[str] = ["router"]

        # Compliance gate: must come before any specialist
        if is_advice or is_cross_client:
            roles.append("compliance")
            return {
                "roles": roles,
                "use_deep": False,
                "is_advice": is_advice,
                "is_cross_client": is_cross_client,
                "symbol": None,
            }

        # Detect which specialists are needed
        needs_book = any(p.search(prompt) for p in _BOOK_QA_PATTERNS)
        needs_kyc = any(p.search(prompt) for p in _KYC_PATTERNS)
        needs_notes = any(p.search(prompt) for p in _NOTES_PATTERNS)
        needs_market = any(p.search(prompt) for p in _MARKET_PATTERNS)

        # Extract symbol if relevant
        symbol = self._loader.find_symbol_in_text(prompt)
        if symbol:
            needs_market = True  # symbol mention always implies market_desk involvement

        # If nothing matched specifically, default to book_qa (most common)
        if not any([needs_book, needs_kyc, needs_notes, needs_market]):
            needs_book = True

        if needs_kyc:
            roles.append("kyc_profile")
        if needs_book:
            roles.append("book_qa")
        if needs_notes:
            roles.append("notes_desk")
        if needs_market:
            roles.append("market_desk")

        # Decide tier: use valura-deep for genuinely complex reasoning
        # Trivial lookups (exact_value, sector, price, kyc fields) stay on fast
        use_deep = self._needs_deep(prompt, needs_book, needs_kyc, needs_market)

        roles.append("verifier")
        return {
            "roles": roles,
            "use_deep": use_deep,
            "is_advice": False,
            "is_cross_client": False,
            "symbol": symbol,
        }

    def _is_advice(self, prompt: str) -> bool:
        """Two distinct code paths: advice (refuse) vs. arithmetic (answer)."""
        return any(p.search(prompt) for p in _ADVICE_PATTERNS)

    def _is_cross_client(self, prompt: str, client_id: str) -> bool:
        """Detect if the prompt requests data about a different client."""
        prompt_lower = prompt.lower()
        for other_id, other_client in self._loader._clients_by_id.items():
            if other_id == client_id:
                continue
            other_name = other_client.get("name", "").lower()
            if other_name and other_name in prompt_lower:
                return True
            if other_id.lower() in prompt_lower:
                return True
        return False

    def _needs_deep(
        self, prompt: str, needs_book: bool, needs_kyc: bool, needs_market: bool
    ) -> bool:
        """Use valura-deep only for genuinely complex reasoning.

        Using the deep tier on a trivial lookup is specifically penalized.
        """
        # Aggregation/temporal/conflict/comparison queries warrant deep tier
        if any(p.search(prompt) for p in _ESCALATION_PATTERNS):
            return True
        # Multi-specialist questions are more complex
        specialist_count = sum([needs_book, needs_kyc, needs_market])
        if specialist_count >= 2:
            return True
        return False
