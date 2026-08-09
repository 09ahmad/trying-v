"""Regression tests for book_agent dispatch regex fixes (q_005, q_077, etc.)."""
from __future__ import annotations

import json
from pathlib import Path
from takehome_service.data import DataLoader
from takehome_service.agents.book_agent import BookAgent


def test_q005_sell_transactions_regex():
    """q_005 ('sell transactions') must match sell count handler and return non-null answer_value '8'."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    agent = BookAgent(loader, "http://localhost:8600/v1", "test")
    
    payload = {
        "prompt": "How many sell transactions did Sneha Sharma make in January 2025?",
        "client_id": "cli_1014",
    }
    resp = agent.answer(payload)
    assert not resp["abstained"], f"q_005 should not abstain, got reason: {resp.get('reason')}"
    assert resp["answer_value"] == "8", f"Expected '8', got {resp['answer_value']}"


def test_q077_plural_dividends_regex():
    """q_077 ('dividends', plural) must match dividend handler and return non-null answer_value '89.55'."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    agent = BookAgent(loader, "http://localhost:8600/v1", "test")
    
    payload = {
        "prompt": "Total up Sneha Sharma's MSFT dividends received in 2025, net of tax.",
        "client_id": "cli_1014",
    }
    resp = agent.answer(payload)
    assert not resp["abstained"], f"q_077 should not abstain, got reason: {resp.get('reason')}"
    assert resp["answer_value"] == "89.55", f"Expected '89.55', got {resp['answer_value']}"


def test_practice_key_regression_q005_q077():
    """Assert against practice_key.json values directly."""
    key_path = Path("harness/practice_key.json")
    assert key_path.exists(), "harness/practice_key.json must exist"
    
    with open(key_path) as f:
        key_data = json.load(f)
        
    questions_map = key_data.get("questions", {})
    
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    agent = BookAgent(loader, "http://localhost:8600/v1", "test")
    
    for qid in ["q_005", "q_077"]:
        expected_val = questions_map[qid]["expected"]["value"]
        if qid == "q_005":
            prompt = "How many sell transactions did Sneha Sharma make in January 2025?"
            cid = "cli_1014"
        else:
            prompt = "Total up Sneha Sharma's MSFT dividends received in 2025, net of tax."
            cid = "cli_1014"
            
        resp = agent.answer({"prompt": prompt, "client_id": cid})
        assert resp["answer_value"] == expected_val, f"Question {qid} failed: expected {expected_val}, got {resp['answer_value']}"
