"""Unit tests for Fix 4: Blackout error handling & blackout robustness."""
from __future__ import annotations

from unittest.mock import MagicMock
from takehome_service.data import DataLoader
from takehome_service.service import AnswerService
from takehome_service.llm_client import BlackoutError


def test_blackout_error_returns_schema_valid_response(monkeypatch):
    """When a specialist raises BlackoutError, service must return schema-valid response with upstream_issue."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    service = AnswerService(loader, "http://localhost:8600/v1", "test")

    # Mock book agent answer to raise BlackoutError
    def mock_book_answer(*args, **kwargs):
        raise BlackoutError("Quota exhausted")

    monkeypatch.setattr(service._book, "answer", mock_book_answer)

    payload = {
        "question_id": "test_blackout_q",
        "prompt": "What is the cash balance for Sneha Sharma?",
        "client_id": "cli_1014",
    }

    resp = service.answer(payload)
    assert resp["question_id"] == "test_blackout_q"
    assert resp["abstained"] is True or "upstream_issue" in resp["flags"]
    assert resp["answer_value"] is None
    assert isinstance(resp["agents"], list)
