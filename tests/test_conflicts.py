"""Regression tests for Fix 2: Conflict detection & surfacing (q_016, q_017, q_018)."""
from __future__ import annotations

from takehome_service.data import DataLoader
from takehome_service.agents.kyc_agent import KYCProfileAgent
from takehome_service.agents.book_agent import BookAgent


def test_q016_conflict_risk():
    """q_016 (risk profile conflict between KYC and suitability review) must return flags=['conflict']."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    agent = KYCProfileAgent(loader, "http://localhost:8600/v1", "test")

    payload = {
        "prompt": "What is Shreya Reddy's risk profile on file?",
        "client_id": "cli_1010",
    }
    resp = agent.answer(payload)
    assert "conflict" in resp.get("flags", []), f"Expected 'conflict' in flags, got: {resp.get('flags')}"
    assert resp.get("answer_value") is None, f"Expected answer_value None for conflict, got: {resp.get('answer_value')}"
    assert "kyc_1010" in resp.get("citations", []) and "rev_710" in resp.get("citations", []), f"Citations must contain both conflicting records, got: {resp.get('citations')}"


def test_q017_conflict_kyc():
    """q_017 (KYC status conflict between verified and pending note) must return flags=['conflict']."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    agent = KYCProfileAgent(loader, "http://localhost:8600/v1", "test")

    payload = {
        "prompt": "Is Meera Bhat's KYC complete and in good standing?",
        "client_id": "cli_1015",
    }
    resp = agent.answer(payload)
    assert "conflict" in resp.get("flags", []), f"Expected 'conflict' in flags, got: {resp.get('flags')}"
    assert resp.get("answer_value") is None, f"Expected answer_value None for conflict, got: {resp.get('answer_value')}"
    assert "kyc_1015" in resp.get("citations", []) and "note_5059" in resp.get("citations", []), f"Citations must contain both conflicting records, got: {resp.get('citations')}"


def test_q018_conflict_snapshot():
    """q_018 (holdings quantity conflict between positions snapshot and transactions) must return flags=['conflict']."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    agent = BookAgent(loader, "http://localhost:8600/v1", "test")

    payload = {
        "prompt": "How many AAPL shares does Ishita Malhotra hold?",
        "client_id": "cli_1022",
    }
    resp = agent.answer(payload)
    assert "conflict" in resp.get("flags", []), f"Expected 'conflict' in flags, got: {resp.get('flags')}"
    assert resp.get("answer_value") is None, f"Expected answer_value None for conflict, got: {resp.get('answer_value')}"
    citations = resp.get("citations", [])
    assert "pos_1022_AAPL" in citations and any("txn_" in c for c in citations), f"Citations must contain pos_1022_AAPL and txn IDs, got: {citations}"
