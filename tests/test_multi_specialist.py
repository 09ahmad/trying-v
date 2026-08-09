"""Regression test for Fix 3: Multi-specialist citation handoff."""
from __future__ import annotations

from takehome_service.data import DataLoader
from takehome_service.service import AnswerService


def test_multi_specialist_combines_citations():
    """Synthetic multi-specialist scenario must preserve citations from both specialists."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    service = AnswerService(loader, "http://localhost:8600/v1", "test")

    spec1 = {
        "answer": "Notes summary: Client inquired about fees.",
        "answer_value": None,
        "abstained": False,
        "refused": False,
        "reason": None,
        "citations": ["note_5001", "note_5002"],
        "confidence": 0.9,
        "flags": [],
    }
    spec2 = {
        "answer": "The cash balance is 5000.00 USD.",
        "answer_value": "5000.00",
        "abstained": False,
        "refused": False,
        "reason": None,
        "citations": ["cli_1001"],
        "confidence": 0.85,
        "flags": [],
    }

    combined = service._combine([spec1, spec2], client_id="cli_1001")
    assert "note_5001" in combined["citations"] and "note_5002" in combined["citations"], "Must contain note citations"
    assert "cli_1001" in combined["citations"], "Must contain client cash citation"
