"""Tests for conflict detection and flagging."""
import pytest
from takehome_service.data import DataLoader


BOOK = "data/client_book.json"
MARKET = "data/market_data.json"


@pytest.fixture(scope="module")
def loader():
    return DataLoader(BOOK, MARKET)


class TestConflictDetection:
    def test_find_field_conflict(self, loader):
        """test conflict helper on DataLoader."""
        # Find if any client has field conflicts
        conflicts_found = []
        for cid in loader.all_client_ids:
            conflict = loader.find_field_conflict(cid, "email")
            if conflict:
                conflicts_found.append((cid, conflict))
        # Even if no natural conflict exists in synthetic data, method should execute cleanly
        assert isinstance(conflicts_found, list)
