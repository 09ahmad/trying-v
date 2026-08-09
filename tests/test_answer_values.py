"""Regression tests for Fix 6: missing answer_values (q_011, q_014, q_064, q_067, q_068)."""
from __future__ import annotations

import json
from takehome_service.data import DataLoader
from takehome_service.service import AnswerService


def test_answer_values_regression():
    """Verify concrete answer_values for q_011, q_014, q_064, q_067, q_068 match practice key."""
    key = json.load(open("harness/practice_key.json"))
    questions = key["questions"]

    loader = DataLoader("data/client_book.json", "data/market_data.json")
    service = AnswerService(loader, "http://localhost:8600/v1", "test")

    mapping = {
        "q_011": ("cli_1014", "How much did Sneha Sharma deposit in total between 27 January 2025 and 27 July 2026 inclusive?"),
        "q_014": ("cli_1019", "How much did Ritika Sharma deposit in total during 2025?"),
        "q_064": ("cli_1005", "Brief me on GOOG using only what was published by 26 November 2025, and say how many items that is."),
        "q_067": ("cli_1014", "How overweight or underweight is Sneha Sharma in MSFT against the mandate, in percentage points?"),
        "q_068": ("cli_1007", "Against the recorded target, where does Pooja Sharma's AMD weight stand, in percentage points?"),
    }

    for qid, (cid, prompt) in mapping.items():
        expected_val = questions[qid]["expected"]["value"]
        resp = service.answer({"question_id": qid, "prompt": prompt, "client_id": cid})
        val = resp.get("answer_value")
        assert val == expected_val, f"Question {qid} failed: expected {expected_val!r}, got {val!r}"
