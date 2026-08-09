"""Unit tests for Fix 7: Defensive _llm_format fallback against malformed or STUB output."""
from __future__ import annotations

from unittest.mock import MagicMock
from takehome_service.data import DataLoader
from takehome_service.agents.book_agent import BookAgent


def test_llm_format_falls_back_on_garbled_output(monkeypatch):
    """When LLM returns garbled text missing numeric values or STUB-GATEWAY text, _llm_format falls back to precomputed_answer."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    agent = BookAgent(loader, "http://localhost:8600/v1", "test")

    # Mock agent.run to return garbled output
    mock_run = MagicMock()
    mock_run.get_content_as_string.return_value = "STUB-GATEWAY acknowledgement..."
    monkeypatch.setattr(agent._fast_agent, "run", lambda msg: mock_run)

    answer_text = agent._llm_format(
        precomputed_answer="The cash balance is 21337.17 USD.",
        data_summary="21337.17 USD",
        original_prompt="What is the cash balance?",
        use_deep=False,
    )

    assert answer_text == "The cash balance is 21337.17 USD."
