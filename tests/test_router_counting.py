"""Tests for router classification of counting questions (q_064 and related)."""
from __future__ import annotations

from takehome_service.data import DataLoader
from takehome_service.router import AgentRouter
from takehome_service.service import AnswerService


def test_q064_router_classification():
    """q_064 prompt must route to market_desk and NOT book_qa."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    router = AgentRouter(loader)

    payload = {
        "question_id": "q_064",
        "client_id": "cli_1005",
        "prompt": "Brief me on GOOG using only what was published by 26 November 2025, and say how many items that is.",
    }

    result = router.classify(payload)
    roles = result["roles"]

    assert "market_desk" in roles, f"roles missing market_desk: {roles}"
    assert "book_qa" not in roles, f"roles should not contain book_qa: {roles}"


def test_q064_answer_service_end_to_end():
    """q_064 AnswerService end-to-end response must exclude book_qa from agents, have value '2', and cite news_2009/news_2010."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    service = AnswerService(loader, "http://localhost:8600/v1", "test-key")

    payload = {
        "question_id": "q_064",
        "client_id": "cli_1005",
        "prompt": "Brief me on GOOG using only what was published by 26 November 2025, and say how many items that is.",
        "chaos": "off",
    }

    ans = service.answer(payload)

    assert "book_qa" not in ans["agents"], f"agents list should not include book_qa: {ans['agents']}"
    assert "market_desk" in ans["agents"], f"agents list must include market_desk: {ans['agents']}"
    assert ans["answer_value"] == "2", f"expected answer_value '2', got {ans.get('answer_value')!r}"
    assert set(ans["citations"]) == {"news_2009", "news_2010"}, f"expected citations news_2009 and news_2010, got {ans.get('citations')}"
