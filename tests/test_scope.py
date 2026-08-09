"""Tests for cross-client scope filtering.

The scope lock is enforced in the data layer. Any fact, figure, name, or record
id belonging to another client must never appear in a response.
"""
import pytest
from takehome_service.data import DataLoader


BOOK = "data/client_book.json"
MARKET = "data/market_data.json"


@pytest.fixture(scope="module")
def loader():
    return DataLoader(BOOK, MARKET)


class TestScopeFiltering:
    def test_get_client_by_id(self, loader):
        """Each client is accessible by their own id."""
        for cid in list(loader.all_client_ids)[:3]:
            client = loader._require_client(cid)
            assert client is not None
            assert client.get("id") == cid

    def test_client_not_found_raises(self, loader):
        """Requesting a non-existent client raises DataAccessError."""
        from takehome_service.data import DataAccessError
        with pytest.raises(DataAccessError):
            loader._require_client("cli_99999_fake")

    def test_transactions_scoped_to_client(self, loader):
        """Transactions returned belong only to the requested client."""
        all_ids = list(loader.all_client_ids)
        cid = all_ids[0]
        txns = loader.get_transactions(cid)
        # Ensure no transaction from another client appears
        other_ids = set(all_ids) - {cid}
        other_client_txn_ids = set()
        for other_id in other_ids:
            other_client = loader._clients_by_id[other_id]
            for t in other_client.get("transactions", []):
                if t.get("id"):
                    other_client_txn_ids.add(t["id"])
        txn_ids = {t.get("id") for t in txns}
        assert txn_ids.isdisjoint(other_client_txn_ids), (
            "Transactions from another client appeared in results"
        )

    def test_notes_scoped_to_client(self, loader):
        """Notes returned belong only to the requested client."""
        all_ids = list(loader.all_client_ids)
        cid = all_ids[0]
        notes = loader.get_notes(cid)
        note_ids = {n.get("id") for n in notes}
        # Check against another client's notes
        other = all_ids[1]
        other_note_ids = {n.get("id") for n in loader._clients_by_id[other].get("notes", [])}
        assert note_ids.isdisjoint(other_note_ids)

    def test_cross_client_leak_detection(self, loader):
        """check_cross_client_leak detects when another client's name appears."""
        all_ids = list(loader.all_client_ids)
        cid = all_ids[0]
        other_cid = all_ids[1]
        other_name = loader._clients_by_id[other_cid].get("name", "")
        if not other_name:
            pytest.skip("No name for other client")
        leaks = loader.check_cross_client_leak(cid, f"The balance for {other_name} is 100", [])
        assert len(leaks) > 0

    def test_no_leak_for_own_name(self, loader):
        """check_cross_client_leak does not flag the client's own name."""
        all_ids = list(loader.all_client_ids)
        cid = all_ids[0]
        own_name = loader._clients_by_id[cid].get("name", "")
        leaks = loader.check_cross_client_leak(cid, f"Hello {own_name}", [])
        assert len(leaks) == 0

    def test_foreign_citation_detected(self, loader):
        """A citation belonging to another client is flagged as a leak."""
        all_ids = list(loader.all_client_ids)
        cid = all_ids[0]
        other_cid = all_ids[1]
        other_txns = loader._clients_by_id[other_cid].get("transactions", [])
        if not other_txns:
            pytest.skip("No transactions for other client")
        foreign_txn_id = other_txns[0].get("id")
        leaks = loader.check_cross_client_leak(cid, "answer text", [foreign_txn_id])
        assert len(leaks) > 0

    def test_as_at_date_filtering(self, loader):
        """Transactions after a target date are excluded with on_or_before filter."""
        from datetime import datetime
        all_ids = list(loader.all_client_ids)
        cid = all_ids[0]
        # Get all transactions
        all_txns = loader.get_transactions(cid)
        # Filter to before 2024-01-01
        cutoff = datetime(2024, 1, 1)
        early_txns = loader.get_transactions(cid, on_or_before=cutoff)
        # Every returned transaction must be on or before cutoff
        for t in early_txns:
            txn_date = loader.parse_date(t.get("date", ""))
            assert txn_date is None or txn_date <= cutoff, (
                f"Transaction {t.get('id')} dated {t.get('date')} leaked past cutoff {cutoff}"
            )
        # Earlier set should be <= all transactions
        assert len(early_txns) <= len(all_txns)
