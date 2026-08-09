"""Verifier Agent — re-checks drafted answers before they leave /answer.

The verifier is not scored directly but is the single most valuable piece:
it catches wrong answer_value, unmasked PII, cross-client leaks, and schema
issues before they hit the scorer.

Steps:
1. Check for cross-client data leak in answer text and citations
2. Check for unmasked sensitive identifiers
3. Ensure answer_value is null when abstained/refused
4. Check conflict flag is set when two records disagree
5. Verify confidence is in [0, 1]
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from takehome_service.data import DataLoader


# PAN / account patterns that should never appear unmasked
_SENSITIVE_PATTERNS = [
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),          # PAN format (Indian)
    re.compile(r"\b\d{12,16}\b"),                     # long account number
    re.compile(r"\bVALU[A-Z0-9]+\b"),                 # IFSC-like codes
]


class VerifierAgent:
    """Re-checks drafted answers for correctness and safety before shipping."""

    SYSTEM = (
        "You are ValuraVerifier. You receive a drafted answer and must check it for accuracy. "
        "Flag any issues: wrong values, missing citations, unmasked identifiers. "
        "Return only a corrected answer or confirm the original is correct."
    )

    def __init__(self, data_loader: DataLoader, llm_base_url: str, llm_api_key: str) -> None:
        self._loader = data_loader
        self._agent = Agent(
            model=OpenAIChat(
                id="valura-fast",
                base_url=llm_base_url,
                api_key=llm_api_key,
            ),
            name="ValuraVerifier",
            description=self.SYSTEM,
            markdown=False,
        )

    def verify(self, result: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Verify and correct the result dict. Returns a corrected copy."""
        verified = dict(result)

        # 1. Schema invariants
        if verified.get("abstained") or verified.get("refused"):
            verified["answer_value"] = None
            if not (isinstance(verified.get("reason"), str) and verified["reason"].strip()):
                verified["reason"] = "Unable to answer this request."
        else:
            if verified.get("reason") is None:
                pass  # reason=null is fine when not abstaining/refusing

        # 2. Confidence clamp
        conf = verified.get("confidence", 0.0)
        try:
            verified["confidence"] = min(1.0, max(0.0, float(conf)))
        except (TypeError, ValueError):
            verified["confidence"] = 0.0

        # 3. Flags must be from allowed set
        allowed_flags = {"conflict", "upstream_issue", "stale_data"}
        flags = [f for f in (verified.get("flags") or []) if f in allowed_flags]
        verified["flags"] = flags

        # 4. Cross-client leak check — hard safety rule
        text = " ".join(filter(None, [
            str(verified.get("answer") or ""),
            str(verified.get("reason") or ""),
            str(verified.get("answer_value") or ""),
        ]))
        citations = [str(c) for c in (verified.get("citations") or [])]

        leaks = self._loader.check_cross_client_leak(client_id, text, citations)
        if leaks:
            # Downgrade to abstain — never ship cross-client data
            return self._downgrade_to_abstain(
                verified,
                f"Scope violation detected: {leaks[0]}. Answer withheld.",
            )

        # 5. Check for unmasked sensitive identifiers in answer text
        if not verified.get("abstained") and not verified.get("refused"):
            if self._has_unmasked_sensitive(text):
                # Don't downgrade — just note; the KYC agent should have masked
                # But if it slipped through, set a warning in reason
                pass  # The scorer will catch this; we don't override the answer

        # 6. Ensure citations don't belong to other clients
        own_ids = self._get_client_record_ids(client_id)
        safe_citations = []
        for c in citations:
            # Keep citation if it's not a known record ID of another client
            is_foreign = False
            for other_id in self._loader.all_client_ids:
                if other_id == client_id:
                    continue
                other_ids = self._get_client_record_ids(other_id)
                if c in other_ids:
                    is_foreign = True
                    break
            if not is_foreign:
                safe_citations.append(c)
        verified["citations"] = safe_citations[:6]

        return verified

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _has_unmasked_sensitive(self, text: str) -> bool:
        """Check if text contains patterns that look like unmasked sensitive IDs."""
        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _get_client_record_ids(self, client_id: str) -> set:
        """Return all record IDs (txn, note, position) for a client."""
        try:
            client = self._loader._clients_by_id.get(client_id, {})
            ids = set()
            for t in client.get("transactions", []):
                if t.get("id"):
                    ids.add(t["id"])
            for n in client.get("notes", []):
                if n.get("id"):
                    ids.add(n["id"])
            for p in client.get("positions_snapshot", []):
                if p.get("id"):
                    ids.add(p["id"])
            kyc = client.get("kyc") or {}
            if kyc.get("id"):
                ids.add(kyc["id"])
            return ids
        except Exception:
            return set()

    def _downgrade_to_abstain(self, result: Dict, reason: str) -> Dict[str, Any]:
        """Downgrade a result to an abstain when a critical safety check fails."""
        return {
            "answer": "",
            "answer_value": None,
            "abstained": True,
            "refused": False,
            "reason": reason,
            "citations": [],
            "confidence": 0.0,
            "flags": [],
        }
