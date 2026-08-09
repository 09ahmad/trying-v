"""Unit tests for Fix 1: Citation formatting rule (>6 records -> [client_id])."""
from __future__ import annotations

from takehome_service.data import format_citations


def test_format_citations_under_or_equal_six():
    """<= 6 record IDs should be returned unchanged (deduplicated)."""
    records = ["txn_101", "txn_102", "txn_103", "txn_104", "txn_105", "txn_106"]
    res = format_citations("cli_1001", records)
    assert res == records, f"Expected {records}, got {res}"

    # Deduplication test
    records_with_dup = ["txn_101", "txn_102", "txn_101", "txn_103"]
    res_dup = format_citations("cli_1001", records_with_dup)
    assert res_dup == ["txn_101", "txn_102", "txn_103"]


def test_format_citations_over_six():
    """> 6 record IDs must return [client_id]."""
    records = ["txn_101", "txn_102", "txn_103", "txn_104", "txn_105", "txn_106", "txn_107"]
    res = format_citations("cli_1001", records)
    assert res == ["cli_1001"], f"Expected ['cli_1001'], got {res}"
