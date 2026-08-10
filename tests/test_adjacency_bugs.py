"""Regression tests for dispatch-regex word-adjacency bugs.

Covers the four targeted questions fixed in this pass:
  q_057 — market_agent._extract_two_dates: 'over X to Y' phrasing
  q_063 — market_agent: coverage branch fires on date-cutoff news queries
  q_078 — book_agent._first_purchase: symbol inserted between 'first' and 'purchase'
  q_083 — book_agent._cash_balance: client name inserted between 'cash' and 'hold'

Also includes non-regression checks for the questions most at risk from loosening.
"""
from __future__ import annotations

import pytest
from takehome_service.data import DataLoader
from takehome_service.agents.book_agent import BookAgent
from takehome_service.agents.market_agent import MarketDeskAgent


@pytest.fixture(scope="module")
def loader():
    return DataLoader("data/client_book.json", "data/market_data.json")


@pytest.fixture(scope="module")
def book(loader):
    return BookAgent(loader, "http://localhost:8600/v1", "test")


@pytest.fixture(scope="module")
def market(loader):
    return MarketDeskAgent(loader, "http://localhost:8600/v1", "test")


# ---------------------------------------------------------------------------
# q_057 — 'over X to Y' phrasing must resolve a two-date range
# ---------------------------------------------------------------------------

class TestQ057OverToConnector:
    def test_over_to_returns_non_null_value(self, market):
        """q_057: 'Over 1 July 2025 to 1 July 2026' must produce a numeric return value."""
        resp = market.answer({
            "prompt": "Over 1 July 2025 to 1 July 2026, what did AMD return in percent?",
            "client_id": "cli_1017",
        })
        assert not resp["abstained"], f"q_057 must not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None, "q_057 answer_value must not be None"
        # Should be a numeric string (possibly with sign)
        try:
            float(resp["answer_value"])
        except (TypeError, ValueError):
            pytest.fail(f"q_057 answer_value should be numeric, got: {resp['answer_value']!r}")

    def test_over_to_matches_between_and_result(self, market):
        """q_057 and q_056 ask the same question with different phrasings — values must match."""
        resp_between = market.answer({
            "prompt": "What was AMD's percentage return between 1 July 2025 and 1 July 2026?",
            "client_id": "cli_1008",
        })
        resp_over = market.answer({
            "prompt": "Over 1 July 2025 to 1 July 2026, what did AMD return in percent?",
            "client_id": "cli_1017",
        })
        assert resp_between["answer_value"] is not None
        assert resp_over["answer_value"] is not None
        assert resp_between["answer_value"] == resp_over["answer_value"], (
            f"Phrasing mismatch: between={resp_between['answer_value']!r} "
            f"over={resp_over['answer_value']!r}"
        )

    def test_from_to_connector_also_works(self, market):
        """'from X to Y' phrasing should also resolve correctly."""
        resp = market.answer({
            "prompt": "From 1 July 2025 to 1 July 2026, AMD's percentage return?",
            "client_id": "cli_1008",
        })
        assert not resp["abstained"], f"from/to phrasing should not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None


# ---------------------------------------------------------------------------
# q_063 — coverage+date-cutoff must NOT return 'covered' as answer_value
# ---------------------------------------------------------------------------

class TestQ063CoverageContentVsStatus:
    def test_coverage_dated_on_or_before_returns_news_count(self, market):
        """q_063: 'AAPL coverage do we hold dated on or before 1 April 2026' must
        return a news count, NOT the literal string 'covered'."""
        resp = market.answer({
            "prompt": "What AAPL coverage do we hold dated on or before 1 April 2026? Give the count and the substance.",
            "client_id": "cli_1003",
        })
        assert not resp["abstained"], f"q_063 must not abstain: {resp.get('reason')}"
        assert resp["answer_value"] != "covered", (
            "q_063 must not return literal 'covered'; it should return the news count"
        )
        # answer_value should be a numeric count of news items
        if resp["answer_value"] is not None:
            try:
                int(resp["answer_value"])
            except ValueError:
                pytest.fail(
                    f"q_063 answer_value should be an integer news count, got: {resp['answer_value']!r}"
                )

    def test_plain_coverage_status_still_works(self, market):
        """A genuine 'is X covered?' question must still hit the coverage-status branch."""
        resp = market.answer({
            "prompt": "Is AAPL covered in our market dataset?",
            "client_id": "cli_1018",
        })
        assert not resp["abstained"], "coverage-status question should not abstain"
        assert resp["answer_value"] == "covered", (
            f"Plain coverage-status question should return 'covered', got {resp['answer_value']!r}"
        )

    def test_q062_news_items_unchanged(self, market):
        """q_062 (news items up to 1 April 2026) must not regress after coverage-branch fix."""
        resp = market.answer({
            "prompt": "How many news items are on file for AAPL up to 1 April 2026, and what do they cover?",
            "client_id": "cli_1018",
        })
        assert not resp["abstained"], f"q_062 should not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None


# ---------------------------------------------------------------------------
# q_078 — symbol inserted between 'first' and 'purchase'
# ---------------------------------------------------------------------------

class TestQ078FirstSymbolPurchase:
    def test_first_aapl_purchase_not_abstained(self, book):
        """q_078: 'first AAPL purchase' must route to _first_purchase, not _fallback."""
        resp = book.answer({
            "prompt": "When did Gaurav Malhotra's first AAPL purchase settle?",
            "client_id": "cli_1001",
        })
        assert not resp["abstained"], f"q_078 must not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None, "q_078 answer_value must not be None"
        # Must be a date string ISO or similar
        assert len(resp["answer_value"]) >= 8, f"answer_value looks too short: {resp['answer_value']!r}"

    def test_first_purchase_no_symbol_still_works(self, book):
        """q_004 style: 'first buy KO' — KO between but regex should still match."""
        resp = book.answer({
            "prompt": "On what date did Sameer Ghosh first buy KO?",
            "client_id": "cli_1023",
        })
        assert not resp["abstained"], f"first buy KO should not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None

    def test_buy_count_not_affected_for_adjacent_phrasing(self, book):
        """q_006: 'purchases' without 'first' must still route to count handler."""
        resp = book.answer({
            "prompt": "How many purchases did Harish Verma make in July 2024?",
            "client_id": "cli_1024",
        })
        assert not resp["abstained"], f"buy count should not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None


# ---------------------------------------------------------------------------
# q_083 — client name inserted between 'cash' and 'hold'
# ---------------------------------------------------------------------------

class TestQ083CashNameHold:
    def test_cash_name_hold_phrasing_not_abstained(self, book):
        """q_083: 'how much cash did Harish Verma hold' must route to _cash_balance."""
        resp = book.answer({
            "prompt": "As at the end of 28 July 2026, how much cash did Harish Verma hold?",
            "client_id": "cli_1024",
        })
        assert not resp["abstained"], f"q_083 must not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None, "q_083 answer_value must not be None"
        try:
            float(resp["answer_value"])
        except (TypeError, ValueError):
            pytest.fail(f"q_083 answer_value should be numeric, got: {resp['answer_value']!r}")

    def test_q009_cash_balance_as_at_still_works(self, book):
        """q_009: 'cash balance as at 28 July 2026' must still work after broadening."""
        resp = book.answer({
            "prompt": "What was Sneha Sharma's cash balance as at 28 July 2026?",
            "client_id": "cli_1014",
        })
        assert not resp["abstained"], f"q_009 should not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None

    def test_q001_current_cash_balance_still_works(self, book):
        """q_001: plain 'current cash balance' must still work."""
        resp = book.answer({
            "prompt": "What is the current cash balance on Sneha Sharma's account?",
            "client_id": "cli_1014",
        })
        assert not resp["abstained"], f"q_001 should not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None

    def test_q041_cash_position_still_works(self, book):
        """q_041: 'cash position' phrasing must still work."""
        resp = book.answer({
            "prompt": "Pull up the cash position on Sneha Sharma's account.",
            "client_id": "cli_1014",
        })
        assert not resp["abstained"], f"q_041 should not abstain: {resp.get('reason')}"
        assert resp["answer_value"] is not None


# ---------------------------------------------------------------------------
# _fallback reason honesty
# ---------------------------------------------------------------------------

class TestFallbackReason:
    def test_fallback_abstain_reason_is_descriptive(self, book):
        """_fallback must abstain with a reason that indicates dispatch-miss, not data absence."""
        # Send a prompt that deliberately doesn't match any dispatch pattern
        resp = book.answer({
            "prompt": "Zyxwvutsrqponm this matches nothing in the dispatch table.",
            "client_id": "cli_1014",
        })
        assert resp["abstained"], "Unmatched prompt should abstain via _fallback"
        assert "pattern" in resp.get("reason", "").lower() or "recognized" in resp.get("reason", "").lower(), (
            f"Fallback reason should mention dispatch pattern, got: {resp.get('reason')!r}"
        )
