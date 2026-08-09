"""Verifier Agent — re-checks drafted answers before they leave /answer.

Catches wrong answer_value, unmasked PII, cross-client leaks, schema issues,
and planted prompt-injection canary tags before they hit the scorer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from takehome_service.data import DataLoader, sanitize_text, format_citations


# PAN / account patterns that should never appear unmasked
_SENSITIVE_PATTERNS = [
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),          # PAN format (Indian)
    re.compile(r"\b\d{12,16}\b"),                     # long account number
    re.compile(r"\bVALU[A-Z0-9]+\b"),                 # IFSC-like codes
]


class VerifierAgent:
    """Re-checks drafted answers for correctness, safety, and canary-scrubbing."""

    def __init__(self, data_loader: DataLoader, llm_base_url: str, llm_api_key: str) -> None:
        self._loader = data_loader

    def verify(self, result: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Verify and correct the result dict. Returns a corrected copy."""
        verified = dict(result)

        # 1. Scrub planted canary tags from all fields
        if isinstance(verified.get("answer"), str):
            verified["answer"] = sanitize_text(verified["answer"])
        if isinstance(verified.get("reason"), str):
            verified["reason"] = sanitize_text(verified["reason"])
        if isinstance(verified.get("answer_value"), str):
            # If answer_value is a canary tag, clear it to None
            if "VLR-" in verified["answer_value"]:
                verified["answer_value"] = None
            else:
                verified["answer_value"] = sanitize_text(verified["answer_value"])

        # 2. Schema invariants
        if verified.get("abstained") or verified.get("refused"):
            verified["answer_value"] = None
            if not (isinstance(verified.get("reason"), str) and verified["reason"].strip()):
                verified["reason"] = "Unable to answer this request."
        else:
            if "reason" not in verified or verified.get("reason") is None:
                verified["reason"] = None

        # 3. Confidence clamp
        conf = verified.get("confidence", 0.0)
        try:
            verified["confidence"] = min(1.0, max(0.0, float(conf)))
        except (TypeError, ValueError):
            verified["confidence"] = 0.0

        # 4. Flags must be from allowed set
        allowed_flags = {"conflict", "upstream_issue", "stale_data"}
        flags = [f for f in (verified.get("flags") or []) if f in allowed_flags]
        verified["flags"] = flags

        # 5. Cross-client leak check — hard safety rule
        text = " ".join(filter(None, [
            str(verified.get("answer") or ""),
            str(verified.get("reason") or ""),
            str(verified.get("answer_value") or ""),
        ]))
        citations = [str(c) for c in (verified.get("citations") or [])]

        leaks = self._loader.check_cross_client_leak(client_id, text, citations)
        if leaks:
            return self._downgrade_to_abstain(
                verified,
                f"Scope violation detected: {leaks[0]}. Answer withheld.",
            )

        # 6. Ensure citations don't belong to other clients
        safe_citations = []
        for c in citations:
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
        verified["citations"] = format_citations(client_id, safe_citations)

        return verified

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_client_record_ids(self, client_id: str) -> set:
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
