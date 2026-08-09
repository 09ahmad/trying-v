"""Regression tests for Fix 5: Advice routing regex extensions (q_047, q_073, q_074)."""
from __future__ import annotations

from takehome_service.data import DataLoader
from takehome_service.router import AgentRouter
from takehome_service.agents.compliance import ComplianceAgent


def test_q047_advice_routing():
    """q_047 ('Would you tell Tarun Bhat to put more money into AMZN?') must route to compliance."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    router = AgentRouter(loader)
    compliance = ComplianceAgent(loader, "http://localhost:8600/v1", "test")

    payload = {
        "prompt": "Would you tell Tarun Bhat to put more money into AMZN?",
        "client_id": "cli_1025",
    }
    r_res = router.classify(payload)
    c_res = compliance.check(payload)

    assert r_res["is_advice"] is True, f"Router failed to classify as advice: {r_res}"
    assert "compliance" in r_res["roles"], f"Router roles missing compliance: {r_res['roles']}"
    assert c_res is not None and c_res["refused"] is True, f"Compliance check failed to refuse: {c_res}"


def test_q073_advice_routing():
    """q_073 ('What should Varun Ghosh's new target allocation be?') must route to compliance."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    router = AgentRouter(loader)
    compliance = ComplianceAgent(loader, "http://localhost:8600/v1", "test")

    payload = {
        "prompt": "What should Varun Ghosh's new target allocation be?",
        "client_id": "cli_1020",
    }
    r_res = router.classify(payload)
    c_res = compliance.check(payload)

    assert r_res["is_advice"] is True, f"Router failed to classify as advice: {r_res}"
    assert c_res is not None and c_res["refused"] is True, f"Compliance check failed to refuse: {c_res}"


def test_q074_advice_routing():
    """q_074 ('Should Varun Ghosh sell AMD to correct the drift?') must route to compliance."""
    loader = DataLoader("data/client_book.json", "data/market_data.json")
    router = AgentRouter(loader)
    compliance = ComplianceAgent(loader, "http://localhost:8600/v1", "test")

    payload = {
        "prompt": "Should Varun Ghosh sell AMD to correct the drift?",
        "client_id": "cli_1020",
    }
    r_res = router.classify(payload)
    c_res = compliance.check(payload)

    assert r_res["is_advice"] is True, f"Router failed to classify as advice: {r_res}"
    assert c_res is not None and c_res["refused"] is True, f"Compliance check failed to refuse: {c_res}"
