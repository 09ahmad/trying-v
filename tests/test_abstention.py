"""Tests for coverage-gap abstention and data limit abstentions."""
import pytest
from takehome_service.data import DataLoader
from takehome_service.agents.market_agent import MarketDeskAgent


BOOK = "data/client_book.json"
MARKET = "data/market_data.json"


@pytest.fixture(scope="module")
def loader():
    return DataLoader(BOOK, MARKET)


@pytest.fixture(scope="module")
def market_agent(loader):
    return MarketDeskAgent(loader, "http://localhost:8600/v1", "test")


class TestAbstention:
    def test_covered_symbol(self, loader):
        """AAPL is in covered symbols."""
        assert loader.is_covered("AAPL")

    def test_uncovered_symbol_abstains(self, market_agent):
        """Uncovered symbol query must return abstained=True."""
        result = market_agent.answer({"prompt": "What is the price of FAKESYMBOL?"})
        assert result["abstained"] is True
        assert result["refused"] is False
        assert result["answer_value"] is None
        assert result["reason"] is not None

    def test_coverage_check_uncovered(self, market_agent):
        """Asking if an uncovered symbol is covered returns abstained=True with clear reason."""
        result = market_agent.answer({"prompt": "Is UNCOVERED123 covered in market data?"})
        assert result["abstained"] is True
        assert result["answer_value"] is None
        assert result["reason"] is not None
