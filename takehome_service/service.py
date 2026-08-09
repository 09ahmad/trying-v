"""AnswerService — the main orchestrator that ties all agents together.

Orchestration flow:
  1. Router classifies question → decides roles and tier
  2. Compliance check — if triggered, return refusal immediately
  3. Dispatch to specialist(s) in order
  4. Combine answers if multiple specialists contribute
  5. Handle conflicts: if data has two disagreeing records, set conflict flag
  6. Verifier re-checks before returning
  7. Finalize: add question_id, agents path, ensure schema compliance

Blackout handling:
  If LLM calls fail with BlackoutError (quota exhausted for the band):
  - Attempt data-layer-only answers where possible
  - For questions that require LLM reasoning, return abstained + upstream_issue
  - Never crash, hang, or fabricate
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from takehome_service.data import DataLoader
from takehome_service.llm_client import BlackoutError
from takehome_service.router import AgentRouter
from takehome_service.agents.book_agent import BookAgent
from takehome_service.agents.kyc_agent import KYCProfileAgent
from takehome_service.agents.market_agent import MarketDeskAgent
from takehome_service.agents.notes_agent import NotesDeskAgent
from takehome_service.agents.compliance import ComplianceAgent
from takehome_service.agents.verifier import VerifierAgent

logger = logging.getLogger(__name__)


class AnswerService:
    """Orchestrates the multi-agent ecosystem for a single question."""

    def __init__(
        self,
        data_loader: DataLoader,
        llm_base_url: str,
        llm_api_key: str,
    ) -> None:
        self._loader = data_loader
        self._router = AgentRouter(data_loader)
        self._book = BookAgent(data_loader, llm_base_url, llm_api_key)
        self._kyc = KYCProfileAgent(data_loader, llm_base_url, llm_api_key)
        self._market = MarketDeskAgent(data_loader, llm_base_url, llm_api_key)
        self._notes = NotesDeskAgent(data_loader, llm_base_url, llm_api_key)
        self._compliance = ComplianceAgent(data_loader, llm_base_url, llm_api_key)
        self._verifier = VerifierAgent(data_loader, llm_base_url, llm_api_key)

    def answer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process one question envelope and return a schema-valid answer."""
        question_id = payload.get("question_id", "")
        client_id = payload.get("client_id", "")

        # Validate client exists
        if client_id not in self._loader.all_client_ids:
            return self._finalize(
                self._refuse("The client_id provided is not recognized."),
                question_id,
                ["router", "compliance"],
            )

        # Step 1: Route
        routing = self._router.classify(payload)
        roles_decided: List[str] = ["router"]
        use_deep = routing.get("use_deep", False)

        # Step 2: Compliance gate
        compliance_result = self._compliance.check(payload)
        if compliance_result is not None:
            roles_decided.append("compliance")
            result = self._verifier.verify(compliance_result, client_id)
            return self._finalize(result, question_id, roles_decided + ["verifier"])

        # Step 3: Dispatch to specialists
        specialist_results: List[Dict[str, Any]] = []
        blackout_encountered = False

        for role in routing.get("roles", []):
            if role in ("router", "compliance", "verifier"):
                continue
            try:
                if role == "book_qa":
                    result = self._book.answer(payload, use_deep=use_deep)
                    roles_decided.append("book_qa")
                elif role == "kyc_profile":
                    result = self._kyc.answer(payload)
                    roles_decided.append("kyc_profile")
                elif role == "market_desk":
                    result = self._market.answer(payload, use_deep=use_deep)
                    roles_decided.append("market_desk")
                elif role == "notes_desk":
                    result = self._notes.answer(payload)
                    roles_decided.append("notes_desk")
                else:
                    continue
                specialist_results.append(result)
            except BlackoutError:
                blackout_encountered = True
                logger.warning("Gateway blackout on question %s role %s", question_id, role)
            except Exception as exc:
                logger.exception("Error in %s for question %s: %s", role, question_id, exc)

        # Step 4: Combine results
        if blackout_encountered and not specialist_results:
            # Complete blackout: all agents failed
            combined = self._upstream_issue(
                "The upstream LLM service is temporarily unavailable (quota exhausted). "
                "Unable to process this question during the outage."
            )
        elif not specialist_results:
            combined = self._abstain(
                "The question could not be answered by any available specialist."
            )
        elif len(specialist_results) == 1:
            combined = specialist_results[0]
        else:
            combined = self._combine(specialist_results)

        # Step 5: Add upstream_issue flag if there was a partial blackout
        if blackout_encountered and specialist_results:
            flags = list(combined.get("flags") or [])
            if "upstream_issue" not in flags:
                flags.append("upstream_issue")
            combined["flags"] = flags

        # Step 6: Verify
        roles_decided.append("verifier")
        verified = self._verifier.verify(combined, client_id)

        return self._finalize(verified, question_id, roles_decided)

    # -----------------------------------------------------------------------
    # Combine multiple specialist answers
    # -----------------------------------------------------------------------

    def _combine(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge answers from multiple specialists for multi-agent questions."""
        # Filter to non-abstained, non-refused answers
        good = [r for r in results if not r.get("abstained") and not r.get("refused")]
        if not good:
            # All abstained — pick the first abstain
            return results[0]

        # Build combined answer text
        texts = [r.get("answer", "") for r in good if r.get("answer")]
        combined_answer = " ".join(texts)

        # Merge citations (deduplicated, capped at 6)
        all_citations: List[str] = []
        for r in good:
            for c in (r.get("citations") or []):
                if c and c not in all_citations:
                    all_citations.append(c)

        # Merge flags
        all_flags: List[str] = []
        for r in good:
            for f in (r.get("flags") or []):
                if f not in all_flags:
                    all_flags.append(f)

        # Use first non-None answer_value
        answer_value = None
        for r in good:
            if r.get("answer_value") is not None:
                answer_value = r["answer_value"]
                break

        # Average confidence
        conf = sum(float(r.get("confidence", 0.0)) for r in good) / max(1, len(good))

        return {
            "answer": combined_answer,
            "answer_value": answer_value,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": all_citations[:6],
            "confidence": min(1.0, conf),
            "flags": all_flags,
        }

    # -----------------------------------------------------------------------
    # Response constructors
    # -----------------------------------------------------------------------

    def _finalize(
        self,
        result: Dict[str, Any],
        question_id: str,
        roles: Sequence[str],
    ) -> Dict[str, Any]:
        """Add question_id and agents path to a result dict."""
        final = dict(result)
        final["question_id"] = str(question_id)

        # Build agents list: must include "router", must only contain known roles
        known_roles = {"router", "book_qa", "kyc_profile", "notes_desk", "market_desk", "compliance", "verifier"}
        agents_list = [r for r in roles if r in known_roles]
        if "router" not in agents_list:
            agents_list.insert(0, "router")
        final["agents"] = agents_list

        # Schema enforcement
        if final.get("abstained") or final.get("refused"):
            final["answer_value"] = None
            if not (isinstance(final.get("reason"), str) and final["reason"].strip()):
                final["reason"] = "This request could not be processed."
        else:
            # Ensure reason is null (not missing) when not abstaining/refusing
            if "reason" not in final:
                final["reason"] = None

        # Ensure all required fields are present
        for field in ("answer", "answer_value", "abstained", "refused", "reason", "citations", "confidence"):
            if field not in final:
                if field == "answer":
                    final[field] = ""
                elif field in ("abstained", "refused"):
                    final[field] = False
                elif field == "citations":
                    final[field] = []
                elif field == "confidence":
                    final[field] = 0.0
                else:
                    final[field] = None

        if "flags" not in final:
            final["flags"] = []

        return final

    def _abstain(self, reason: str) -> Dict[str, Any]:
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

    def _refuse(self, reason: str) -> Dict[str, Any]:
        return {
            "answer": reason,
            "answer_value": None,
            "abstained": False,
            "refused": True,
            "reason": reason,
            "citations": [],
            "confidence": 1.0,
            "flags": [],
        }

    def _upstream_issue(self, reason: str) -> Dict[str, Any]:
        return {
            "answer": "",
            "answer_value": None,
            "abstained": True,
            "refused": False,
            "reason": reason,
            "citations": [],
            "confidence": 0.0,
            "flags": ["upstream_issue"],
        }
