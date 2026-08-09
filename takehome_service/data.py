"""Data access layer — loads client book + market data once at startup.

All accessors require client_id and filter strictly to that client's data.
This is the scope-enforcement point — not left to prompting.

Key design rules (from brief):
- Load once at startup; never re-read per request.
- Prices are month-start closes; for a date between two points, use the most
  recent close on or before that date.
- Masking is done here via mask_sensitive(); single shared function, no bypass.
- As-at date filtering happens at this layer.
- Conflict detection surfaces when two records disagree on the same fact.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class DataAccessError(Exception):
    """Raised when a data lookup fails (e.g. client not found)."""


# ---------------------------------------------------------------------------
# Masking — single shared function, used everywhere sensitive values appear
# ---------------------------------------------------------------------------

def mask_sensitive(value: str) -> str:
    """Returns ****XXXX where XXXX is the last 4 characters of value.

    This is the canonical masking format for bank account numbers, identity
    numbers, PANs and any other sensitive identifiers.
    """
    if not value or not isinstance(value, str):
        return "****"
    last4 = value[-4:] if len(value) >= 4 else value.ljust(4, "*")[-4:]
    return f"****{last4}"


def mask_in_text(text: str, value: str) -> str:
    """Replace all occurrences of value in text with its masked form."""
    if not value or not text:
        return text
    masked = mask_sensitive(value)
    return text.replace(value, masked)


# ---------------------------------------------------------------------------
# Injection detection
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget)\s+(previous|above|prior|all)\b", re.I),
    re.compile(r"\b(system|assistant|user):\s", re.I),
    re.compile(r"<\s*(system|assistant|instructions?)\s*>", re.I),
    re.compile(r"\bnow\s+(print|output|reveal|show|disclose)\b", re.I),
    re.compile(r"\b(print|reveal|expose|output)\s+(the\s+)?(full|complete|all|every)\b", re.I),
    re.compile(r"\byou\s+(must|should|will|are\s+required\s+to)\s+(ignore|reveal|print)\b", re.I),
    re.compile(r"###\s*(system|instruction|override)", re.I),
    re.compile(r"\[INST\]|\[\/INST\]", re.I),
    re.compile(
        r"\bdo\s+not\s+(follow|obey|use)\s+(your|the)\s+(guidelines?|instructions?|rules?)\b",
        re.I,
    ),
]


def detect_injection(text: str) -> bool:
    """Return True if text contains instruction-injection patterns."""
    if not text:
        return False
    return any(p.search(text) for p in _INJECTION_PATTERNS)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class DataLoader:
    """Indexed data-access layer over client_book.json and market_data.json.

    Loaded once at startup. All accessors require client_id and enforce scope.

    Market data structure:
      instruments: list of {symbol, sector, industry, currency, listed_on}
      prices: dict {symbol: [{date, close}, ...]}  ← month-start closes
      news:   list of {id, date, symbol, headline, body, source}
    """

    def __init__(self, book_path: str, market_path: str) -> None:
        self.book_path = Path(book_path)
        self.market_path = Path(market_path)
        self._book: Dict[str, Any] = self._load_json(self.book_path)
        self._market: Dict[str, Any] = self._load_json(self.market_path)

        # Index clients by id for O(1) lookup
        self._clients_by_id: Dict[str, Dict] = {}
        for client in self._book.get("clients", []):
            cid = client.get("id") or client.get("client_id") or ""
            if cid:
                self._clients_by_id[cid] = client

        # Set of all client ids (for cross-client scope check)
        self.all_client_ids: Set[str] = set(self._clients_by_id.keys())

        # Market: instruments indexed by symbol
        self._instruments_by_symbol: Dict[str, Dict] = {
            inst["symbol"]: inst
            for inst in self._market.get("instruments", [])
            if inst.get("symbol")
        }

        # Market: prices is a dict {symbol: [{date, close}, ...]}
        raw_prices = self._market.get("prices", {})
        if isinstance(raw_prices, dict):
            self._prices_by_symbol: Dict[str, List[Dict]] = raw_prices
        else:
            # Fallback: list of {symbol, date, close}
            self._prices_by_symbol = {}
            for p in raw_prices:
                sym = p.get("symbol")
                if sym:
                    self._prices_by_symbol.setdefault(sym, []).append(p)

        # Market: news is a list of {id, date, symbol, headline, body, source}
        self._news_by_symbol: Dict[str, List[Dict]] = {}
        for n in self._market.get("news", []):
            sym = n.get("symbol")
            if sym:
                self._news_by_symbol.setdefault(sym, []).append(n)

        # covered_symbols is authoritative — subset of all instruments
        self.covered_symbols: Set[str] = set(
            self._market.get("meta", {}).get("covered_symbols", [])
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise DataAccessError(f"Data file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _require_client(self, client_id: str) -> Dict[str, Any]:
        """Return client dict or raise DataAccessError."""
        client = self._clients_by_id.get(client_id)
        if client is None:
            raise DataAccessError(f"Client not found: {client_id!r}")
        return client

    def parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse ISO date string to datetime, returning None on failure."""
        if not date_text:
            return None
        try:
            return datetime.fromisoformat(str(date_text).strip()[:10])
        except ValueError:
            return None

    def _parse_decimal(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", "").replace("$", "").strip())
        except ValueError:
            return 0.0

    # -----------------------------------------------------------------------
    # Client identity / KYC
    # -----------------------------------------------------------------------

    def get_client_identity(self, client_id: str) -> Dict[str, Any]:
        """Return top-level identity fields for a client."""
        client = self._require_client(client_id)
        return {
            "id": client.get("id", ""),
            "name": client.get("name", ""),
            "email": client.get("email", ""),
            "phone": client.get("phone", client.get("mobile", "")),
            "mobile": client.get("mobile", client.get("phone", "")),
        }

    def get_kyc(self, client_id: str) -> Dict[str, Any]:
        """Return the KYC record for client_id."""
        client = self._require_client(client_id)
        kyc = client.get("kyc") or {}
        return dict(kyc)

    def get_accounts(self, client_id: str) -> List[Dict[str, Any]]:
        """Return all accounts for client_id."""
        client = self._require_client(client_id)
        return list(client.get("accounts", []))

    # -----------------------------------------------------------------------
    # Transactions — with as-at date filtering at data layer
    # -----------------------------------------------------------------------

    def get_transactions(
        self,
        client_id: str,
        on_or_before: Optional[datetime] = None,
        txn_type: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return transactions for client_id, filtered by date/type/symbol.

        on_or_before: filter out records after this date (as-at semantics).
        txn_type: filter to a specific transaction type.
        symbol: filter to a specific instrument symbol.
        """
        client = self._require_client(client_id)
        results = []
        for txn in client.get("transactions", []):
            if on_or_before is not None:
                txn_date = self.parse_date(txn.get("date", ""))
                if txn_date is None or txn_date > on_or_before:
                    continue
            if txn_type is not None and txn.get("type") != txn_type:
                continue
            if symbol is not None and txn.get("symbol") != symbol:
                continue
            results.append(txn)
        return results

    # -----------------------------------------------------------------------
    # Positions / holdings
    # -----------------------------------------------------------------------

    def get_positions_snapshot(self, client_id: str) -> List[Dict[str, Any]]:
        """Return the current positions snapshot for client_id."""
        client = self._require_client(client_id)
        return list(client.get("positions_snapshot", []))

    def compute_cash_balance(self, client_id: str) -> Tuple[float, List[str]]:
        """Compute cash balance from transaction history. Returns (balance, cited_ids)."""
        txns = self.get_transactions(client_id)
        balance = 0.0
        cited: List[str] = []
        for txn in txns:
            ttype = txn.get("type", "")
            amount = self._parse_decimal(txn.get("amount_usd") or txn.get("amount_inr") or 0)
            net = self._parse_decimal(txn.get("net_usd") or txn.get("gross_usd") or 0)
            txn_id = txn.get("id", "")
            if ttype == "deposit":
                balance += amount
                cited.append(txn_id)
            elif ttype in ("withdrawal", "fee"):
                balance -= amount
                cited.append(txn_id)
            elif ttype == "buy":
                balance -= net
                cited.append(txn_id)
            elif ttype == "sell":
                balance += net
                cited.append(txn_id)
        return balance, cited

    def compute_holdings(self, client_id: str, symbol: str) -> Tuple[float, List[str]]:
        """Compute net holdings of symbol by summing buys - sells. Returns (qty, cited_ids)."""
        txns = self.get_transactions(client_id, symbol=symbol)
        qty = 0.0
        cited: List[str] = []
        for txn in txns:
            q = self._parse_decimal(txn.get("quantity") or 0)
            if txn.get("type") == "buy":
                qty += q
            elif txn.get("type") == "sell":
                qty -= q
            cited.append(txn.get("id", ""))
        return qty, cited

    def compute_target_drift(
        self, client_id: str
    ) -> Optional[Tuple[float, str, List[str]]]:
        """Compute drift of current allocation vs agreed target.

        Returns (drift_percentage, description, cited_ids) or None if no target on file.
        This is arithmetic from data, not advice.
        """
        client = self._require_client(client_id)
        target = (
            client.get("target_allocation")
            or (client.get("investment_policy") or {}).get("target_allocation")
            or (client.get("suitability_reviews") or [{}])[-1].get("target_allocation")
            if client.get("suitability_reviews")
            else None
        )
        if not target:
            return None

        snapshot = self.get_positions_snapshot(client_id)
        if not snapshot:
            return None

        # Compute current allocation by market value
        total_mv = sum(
            self._parse_decimal(pos.get("market_value_usd") or pos.get("market_value") or pos.get("value") or 0)
            for pos in snapshot
        )
        if total_mv <= 0:
            return None

        # Compare equity vs target (simplification: equity vs target_equity)
        target_equity_pct = self._parse_decimal(
            target.get("equity") or target.get("stocks") or 0
        )
        current_equity_mv = sum(
            self._parse_decimal(pos.get("market_value_usd") or pos.get("market_value") or pos.get("value") or 0)
            for pos in snapshot
            if pos.get("asset_class", "").lower() in ("equity", "stock", "stocks")
        )
        current_equity_pct = (current_equity_mv / total_mv) * 100 if total_mv > 0 else 0.0
        drift = current_equity_pct - target_equity_pct
        cited = [pos.get("id", "") for pos in snapshot[:6] if pos.get("id")]
        desc = (
            f"Current equity allocation is {current_equity_pct:.1f}%, "
            f"target is {target_equity_pct:.1f}%, "
            f"drift is {drift:+.1f} percentage points."
        )
        return drift, desc, cited

    # -----------------------------------------------------------------------
    # Notes
    # -----------------------------------------------------------------------

    def get_notes(self, client_id: str) -> List[Dict[str, Any]]:
        """Return all notes for client_id, with injection flag added."""
        client = self._require_client(client_id)
        notes = []
        for note in client.get("notes", []):
            note_copy = dict(note)
            text = str(note.get("text", "") or note.get("body", "") or "")
            note_copy["_injection_detected"] = detect_injection(text)
            notes.append(note_copy)
        return notes

    def get_transaction_memo(self, client_id: str, txn_id: str) -> Optional[str]:
        """Return the memo/description for a specific transaction."""
        txns = self.get_transactions(client_id)
        for txn in txns:
            if txn.get("id") == txn_id:
                return txn.get("description") or txn.get("memo") or txn.get("note") or ""
        return None

    # -----------------------------------------------------------------------
    # Market data
    # -----------------------------------------------------------------------

    def is_covered(self, symbol: str) -> bool:
        return symbol in self.covered_symbols

    def get_instrument(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return instrument record for symbol, or None if not covered."""
        if not self.is_covered(symbol):
            return None
        return self._instruments_by_symbol.get(symbol)

    def get_price_on_or_before(self, symbol: str, target: datetime) -> Optional[Dict[str, Any]]:
        """Return the most recent month-start close on or before target date.

        Prices are month-start closes (not daily). For a date between two points,
        uses the most recent close on or before that date.
        """
        if not self.is_covered(symbol):
            return None
        history = self._prices_by_symbol.get(symbol, [])
        valid: List[Tuple[datetime, Dict]] = []
        for p in history:
            d = self.parse_date(p.get("date", ""))
            if d and d <= target:
                valid.append((d, p))
        if not valid:
            return None
        _, price = max(valid, key=lambda x: x[0])
        return price

    def get_price_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Return full price history for a covered symbol."""
        if not self.is_covered(symbol):
            return []
        return list(self._prices_by_symbol.get(symbol, []))

    def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Return news items for a covered symbol."""
        if not self.is_covered(symbol):
            return []
        return list(self._news_by_symbol.get(symbol, []))

    def get_client_covered_symbols(self, client_id: str) -> List[str]:
        """Return symbols held by client that are also covered in market data."""
        snapshot = self.get_positions_snapshot(client_id)
        held = {pos.get("symbol") for pos in snapshot if pos.get("symbol")}
        return sorted(held & self.covered_symbols)

    # -----------------------------------------------------------------------
    # Conflict detection
    # -----------------------------------------------------------------------

    def find_field_conflict(
        self, client_id: str, field_name: str
    ) -> Optional[Tuple[Any, Any, str, str]]:
        """Look for conflicting values of field_name across accounts/records.

        Returns (value_a, value_b, record_id_a, record_id_b) if conflict found,
        else None.
        """
        client = self._require_client(client_id)
        candidates: List[Tuple[Any, str]] = []

        if field_name in client:
            candidates.append((client[field_name], client.get("id", "client")))

        kyc = client.get("kyc") or {}
        if field_name in kyc:
            candidates.append((kyc[field_name], kyc.get("id", "kyc")))

        for acc in client.get("accounts", []):
            if field_name in acc:
                candidates.append((acc[field_name], acc.get("id", "account")))

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                val_a, id_a = candidates[i]
                val_b, id_b = candidates[j]
                if str(val_a).strip() != str(val_b).strip():
                    return val_a, val_b, id_a, id_b
        return None

    # -----------------------------------------------------------------------
    # Scope-safety check: ensure no cross-client data in response text
    # -----------------------------------------------------------------------

    def check_cross_client_leak(
        self, client_id: str, text: str, citations: List[str]
    ) -> List[str]:
        """Return list of leak descriptions if other clients' data appears in text/citations."""
        leaks: List[str] = []
        for other_id, other_client in self._clients_by_id.items():
            if other_id == client_id:
                continue
            other_name = other_client.get("name", "")
            if other_name and other_name in text:
                leaks.append(f"response contains name of client {other_id}")
            other_txn_ids = {t.get("id") for t in other_client.get("transactions", [])}
            other_note_ids = {n.get("id") for n in other_client.get("notes", [])}
            other_pos_ids = {p.get("id") for p in other_client.get("positions_snapshot", [])}
            other_ids = (other_txn_ids | other_note_ids | other_pos_ids) - {None}
            for c in citations:
                if c in other_ids:
                    leaks.append(f"citation {c!r} belongs to client {other_id}")
        return leaks

    # -----------------------------------------------------------------------
    # Symbol extraction
    # -----------------------------------------------------------------------

    def find_symbol_in_text(self, text: str) -> Optional[str]:
        """Extract a market symbol mentioned in text."""
        all_symbols = set(self._instruments_by_symbol.keys())
        words = set(re.findall(r"\b[A-Z]{2,6}\b", text))
        for sym in words:
            if sym in self.covered_symbols:
                return sym
        for sym in words:
            if sym in all_symbols:
                return sym
        return None
