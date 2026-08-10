"""Reference service implementation connecting harness/reference_client.py to takehome_service.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from takehome_service.data import DataLoader
from takehome_service.roster import get_roster
from takehome_service.service import AnswerService


class Reference:
    """Reference implementation that wraps AnswerService for the client protocol."""

    def __init__(
        self,
        book: Any,
        llm_base_url: str,
        api_key: str,
        market: Optional[Any] = None,
    ) -> None:
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self.data_loader = DataLoader(book, market)
        self.service = AnswerService(
            data_loader=self.data_loader,
            llm_base_url=llm_base_url,
            llm_api_key=api_key,
        )

    def roster(self) -> Dict[str, Any]:
        """Return the agent roster declaration."""
        return get_roster()

    def answer(self, question_id: str, client_id: str, prompt: str) -> Dict[str, Any]:
        """Answer a single question envelope using AnswerService."""
        payload = {
            "question_id": question_id,
            "client_id": client_id,
            "prompt": prompt,
        }
        return self.service.answer(payload)
