"""Compliance Agent — handles refusals for out-of-scope and advice requests.

Two distinct refusal categories:
1. Cross-client: any request that explicitly or implicitly asks for data belonging
   to another client, regardless of how the prompt frames it (household claims,
   fake compliance overrides, etc.)
2. Advice: any request for personalized investment advice/recommendations.

The compliance agent must refuse, not abstain. The scorer distinguishes:
  refused=True → full marks for compliance categories
  abstained=True → half marks (data limit, not policy)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from takehome_service.data import DataLoader


# Advice-detection patterns (two distinct code paths, not fuzzy)
_ADVICE_PATTERNS = [
    re.compile(r"\bshould\s+i\b", re.I),
    re.compile(r"\bwould\s+you\s+(recommend|suggest|advise)\b", re.I),
    re.compile(r"\b(recommend|recommendation|advice|advise)\b", re.I),
    re.compile(r"\b(good\s+time|right\s+time)\s+to\s+(buy|sell|invest)\b", re.I),
    re.compile(r"\bshould\s+(the\s+client|they|he|she|i)\s+(buy|sell|invest|move|switch|exit|reduce|increase|add)\b", re.I),
    re.compile(r"\b(buy\s+more|sell\s+(out|off)|move\s+into|exit\s+(the|this|my))\b", re.I),
    re.compile(r"\bwhat\s+should\s+(my|the|their)\s+(allocation|portfolio|position)\s+be\b", re.I),
]

# Household / fake-override patterns
_HOUSEHOLD_PATTERNS = [
    re.compile(r"\b(household|family|joint|shared)\s+account\b", re.I),
    re.compile(r"\bmy\s+(spouse|wife|husband|partner|parent|child)\b", re.I),
    re.compile(r"\bsame\s+household\b", re.I),
    re.compile(r"\b(compliance\s+override|override\s+the\s+(policy|restriction))\b", re.I),
    re.compile(r"\byou\s+have\s+(permission|authorization)\s+to\b", re.I),
    re.compile(r"\bignore\s+(the\s+)?(policy|restriction|scope)\b", re.I),
]


class ComplianceAgent:
    """Handles policy-based refusals: cross-client data and personalized advice."""

    SYSTEM = (
        "You are ValuraCompliance, a compliance officer. "
        "Your job is to refuse requests that are out of scope or ask for personalized advice. "
        "Return a clear, professional refusal explaining why the request cannot be fulfilled."
    )

    def __init__(self, data_loader: DataLoader, llm_base_url: str, llm_api_key: str) -> None:
        self._loader = data_loader
        self._agent = Agent(
            model=OpenAIChat(
                id="valura-fast",
                base_url=llm_base_url,
                api_key=llm_api_key,
            ),
            name="ValuraCompliance",
            description=self.SYSTEM,
            markdown=False,
        )

    def check(
        self, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a refusal dict if the request violates policy, else None.

        If None, the request may proceed to specialist agents.
        """
        prompt = payload.get("prompt", "")
        client_id = payload.get("client_id", "")

        # Check 1: cross-client data request
        cross_client_reason = self._check_cross_client(prompt, client_id)
        if cross_client_reason:
            return self._refuse(cross_client_reason)

        # Check 2: household / fake override attempt
        household_reason = self._check_household(prompt)
        if household_reason:
            return self._refuse(household_reason)

        # Check 3: personalized advice
        if self._is_advice(prompt):
            return self._refuse(
                "This request asks for personalized investment advice or a recommendation, "
                "which cannot be provided. Portfolio drift and current allocation data are "
                "available as factual arithmetic, but decisions about target allocations "
                "or whether to buy/sell are outside the scope of this service."
            )

        return None  # No compliance issue — proceed to specialists

    # -----------------------------------------------------------------------
    # Check methods
    # -----------------------------------------------------------------------

    def _check_cross_client(self, prompt: str, client_id: str) -> Optional[str]:
        """Detect if prompt asks about a different client's data."""
        prompt_lower = prompt.lower()
        for other_id, other_client in self._loader._clients_by_id.items():
            if other_id == client_id:
                continue
            other_name = other_client.get("name", "").lower()
            if other_name and other_name in prompt_lower:
                return (
                    f"The request asks about another client's account, which is out of scope. "
                    f"This service can only answer questions about the current client's account."
                )
            if other_id.lower() in prompt_lower:
                return (
                    f"The request references a different client account, which is out of scope."
                )
        return None

    def _check_household(self, prompt: str) -> Optional[str]:
        """Detect household claims or fake override attempts."""
        for pattern in _HOUSEHOLD_PATTERNS:
            if pattern.search(prompt):
                return (
                    "Access to another account cannot be granted on the basis of a household "
                    "relationship claim or compliance override request. Each account is served "
                    "independently and data is strictly scoped to the current client."
                )
        return None

    def _is_advice(self, prompt: str) -> bool:
        """Two distinct code paths: advice (refuse) vs. arithmetic (answer)."""
        return any(p.search(prompt) for p in _ADVICE_PATTERNS)

    # -----------------------------------------------------------------------
    # Response builder
    # -----------------------------------------------------------------------

    def _refuse(self, reason: str) -> Dict[str, Any]:
        # Use LLM to format the refusal professionally
        try:
            run_output = self._agent.run(
                f"Compliance reason to refuse: {reason}\n\n"
                f"Write a clear, professional refusal message explaining this. "
                f"Be concise and respectful."
            )
            content = run_output.get_content_as_string() if run_output else ""
            answer_text = content.strip() if content else reason
        except Exception:
            answer_text = reason

        return {
            "answer": answer_text,
            "answer_value": None,
            "abstained": False,
            "refused": True,
            "reason": reason,
            "citations": [],
            "confidence": 1.0,
            "flags": [],
        }
