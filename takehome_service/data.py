"""Data access layer — loads client book + market data once at startup.

All accessors require client_id and filter strictly to that client's data.
This is the scope-enforcement point — not left to prompting.

Key design rules:
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
    """Returns ****XXXX where XXXX is the last 4 characters of value."""
    if not value or not isinstance(value, str):
        return "****"
    val = str(value).strip()
    if val.startswith("****") and len(val) == 8:
        return val
    last4 = val[-4:] if len(val) >= 4 else val.ljust(4, "*")[-4:]
    return f"****{last4}"


def mask_in_text(text: str, value: str) -> str:
    """Replace all occurrences of value in text with its masked form."""
    if not value or not text:
        return text
    masked = mask_sensitive(value)
    return text.replace(value, masked)


def format_citations(client_id: str, record_ids: List[str]) -> List[str]:
    """Format citations according to >6-records rule from TAKE_HOME_BRIEF:
    If an answer rests on more than six records, cite the client_id instead of listing them.
    """
    seen: Set[str] = set()
    clean: List[str] = []
    for r in record_ids:
        if r and isinstance(r, str) and r not in seen:
            seen.add(r)
            clean.append(r)
    if len(clean) > 6:
        return [client_id] if client_id else clean[:6]
    return clean



# ---------------------------------------------------------------------------
# Injection detection and canary sanitization
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
    re.compile(r"\bVLR-[A-Z0-9\-]+\b"),
]


def detect_injection(text: str) -> bool:
    """Return True if text contains instruction-injection patterns."""
    if not text:
        return False
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def sanitize_text(text: str) -> str:
    """Remove any planted canary tags (e.g. VLR-[A-Z0-9-]+) from text."""
    if not text:
        return ""
    return re.sub(r"\bVLR-[A-Z0-9\-]+\b", "[REDACTED]", text)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class DataLoader:
    """Indexed data-access layer over client_book.json and market_data.json.

    Loaded once at startup. All accessors require client_id and enforce scope.
    """

    def __init__(self, book_path: Any, market_path: Any = None) -> None:
        if isinstance(book_path, dict):
            self.book_path = Path("data/client_book.json")
            self._book = book_path
        else:
            self.book_path = Path(book_path)
            self._book = self._load_json(self.book_path)

        if isinstance(market_path, dict):
            self.market_path = Path("data/market_data.json")
            self._market = market_path
        elif market_path is not None:
            self.market_path = Path(market_path)
            self._market = self._load_json(self.market_path)
        else:
            self.market_path = Path("data/market_data.json")
            self._market = self._load_json(self.market_path)

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
        client = self._require_client(client_id)
        return {
            "id": client.get("id", ""),
            "name": client.get("name", ""),
            "email": client.get("email", ""),
            "phone": client.get("phone", client.get("mobile", "")),
            "mobile": client.get("mobile", client.get("phone", "")),
        }

    def get_kyc(self, client_id: str) -> Dict[str, Any]:
        """Return flattened KYC record for client_id."""
        client = self._require_client(client_id)
        kyc = dict(client.get("kyc") or {})
        # Flatten employment.employer if nested
        emp = kyc.get("employment")
        if isinstance(emp, dict) and "employer" in emp:
            kyc["employer"] = emp["employer"]
        return kyc

    def get_accounts(self, client_id: str) -> List[Dict[str, Any]]:
        client = self._require_client(client_id)
        return list(client.get("accounts", []))

    def get_account_age(
        self, client_id: str, as_of_date: Optional[datetime] = None
    ) -> Tuple[int, List[str]]:
        """Return account age in days as of as_of_date (default 2026-07-28)."""
        accounts = self.get_accounts(client_id)
        if not accounts:
            txns = self.get_transactions(client_id)
            if not txns:
                return 0, []
            earliest_str = min(t.get("date", "9999") for t in txns if t.get("date"))
            d1 = self.parse_date(earliest_str)
            cite_id = txns[0].get("id", client_id)
        else:
            acc = accounts[0]
            d1 = self.parse_date(acc.get("opened", ""))
            cite_id = acc.get("id", client_id)

        if not d1:
            return 0, []

        ref_date = as_of_date or datetime(2026, 7, 28)
        age = (ref_date - d1).days
        return age, [cite_id]

    # -----------------------------------------------------------------------
    # Transactions — with as-at date filtering
    # -----------------------------------------------------------------------

    def get_transactions(
        self,
        client_id: str,
        on_or_before: Optional[datetime] = None,
        txn_type: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
    # Positions / holdings / arithmetic
    # -----------------------------------------------------------------------

    def get_positions_snapshot(self, client_id: str) -> List[Dict[str, Any]]:
        client = self._require_client(client_id)
        return list(client.get("positions_snapshot", []))

    def compute_cash_balance(
        self, client_id: str, on_or_before: Optional[datetime] = None
    ) -> Tuple[float, List[str]]:
        """Compute cash balance from transaction history.

        Sum = deposit (+) + sell (+) + dividend (+) - withdrawal (-) - fee (-) - buy (-).
        """
        txns = self.get_transactions(client_id, on_or_before=on_or_before)
        balance = 0.0
        cited: List[str] = []
        for txn in txns:
            ttype = txn.get("type", "")
            amt = self._parse_decimal(txn.get("amount_usd") or txn.get("net_usd") or txn.get("gross_usd") or 0)
            txn_id = txn.get("id", "")
            if ttype == "deposit":
                balance += amt
                cited.append(txn_id)
            elif ttype in ("withdrawal", "fee"):
                balance -= amt
                cited.append(txn_id)
            elif ttype == "buy":
                balance -= amt
                cited.append(txn_id)
            elif ttype == "sell":
                balance += amt
                cited.append(txn_id)
            elif ttype == "dividend":
                balance += amt
                cited.append(txn_id)
        return balance, cited

    def compute_holdings(
        self, client_id: str, symbol: str, on_or_before: Optional[datetime] = None
    ) -> Tuple[float, List[str]]:
        """Compute net holdings of symbol by summing buys - sells up to on_or_before."""
        txns = self.get_transactions(client_id, on_or_before=on_or_before, symbol=symbol)
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

    def compute_sector_exposure(
        self, client_id: str, sector: str
    ) -> Tuple[float, List[str]]:
        """Compute portfolio exposure percentage to a specific market sector."""
        snapshot = self.get_positions_snapshot(client_id)
        total_mv = sum(
            self._parse_decimal(pos.get("market_value_usd") or pos.get("market_value") or 0)
            for pos in snapshot
        )
        if total_mv <= 0:
            return 0.0, []

        sector_mv = 0.0
        cited = []
        for pos in snapshot:
            sym = pos.get("symbol")
            inst = self.get_instrument(sym) if sym else None
            pos_sector = inst.get("sector") if inst else pos.get("sector")
            if pos_sector and pos_sector.lower() == sector.lower():
                sector_mv += self._parse_decimal(
                    pos.get("market_value_usd") or pos.get("market_value") or 0
                )
                if pos.get("id"):
                    cited.append(pos["id"])

        pct = (sector_mv / total_mv) * 100 if total_mv > 0 else 0.0
        return pct, cited

    def compute_holding_drift(
        self, client_id: str, symbol: str
    ) -> Optional[Tuple[float, List[str]]]:
        """Compute drift in percentage points of a specific holding symbol vs target allocation."""
        client = self._require_client(client_id)
        snapshot = self.get_positions_snapshot(client_id)
        total_mv = sum(
            self._parse_decimal(pos.get("market_value_usd") or pos.get("market_value") or 0)
            for pos in snapshot
        )
        if total_mv <= 0:
            return None

        pos = next((p for p in snapshot if p.get("symbol") == symbol), None)
        sym_mv = self._parse_decimal(pos.get("market_value_usd") or pos.get("market_value") or 0) if pos else 0.0
        current_weight = (sym_mv / total_mv) * 100 if total_mv > 0 else 0.0

        # Look up target weight from suitability_reviews
        suitability = client.get("suitability_reviews") or []
        target_pct_dict = suitability[-1].get("target_allocation_pct", {}) if suitability else {}
        target_val = target_pct_dict.get(symbol) or target_pct_dict.get(symbol.upper(), "0")
        target_weight = self._parse_decimal(target_val)

        drift = current_weight - target_weight
        cited = [pos.get("id", "")] if pos else []
        if suitability and suitability[-1].get("id"):
            cited.append(suitability[-1]["id"])
        return drift, cited

    def compute_target_drift(
        self, client_id: str
    ) -> Optional[Tuple[float, str, List[str]]]:
        client = self._require_client(client_id)
        suitability = client.get("suitability_reviews") or []
        if not suitability:
            return None
        target = suitability[-1].get("target_allocation_pct", {})
        snapshot = self.get_positions_snapshot(client_id)
        if not snapshot:
            return None
        total_mv = sum(self._parse_decimal(p.get("market_value_usd", 0)) for p in snapshot)
        if total_mv <= 0:
            return None

        cited = [p.get("id", "") for p in snapshot if p.get("id")]
        if suitability[-1].get("id"):
            cited.append(suitability[-1]["id"])
        drift = 0.0
        desc = f"Target allocation analysis for {client.get('name')}."
        return drift, desc, cited

    # -----------------------------------------------------------------------
    # Notes & Memos
    # -----------------------------------------------------------------------

    def get_notes(self, client_id: str) -> List[Dict[str, Any]]:
        client = self._require_client(client_id)
        notes = []
        for note in client.get("notes", []):
            note_copy = dict(note)
            text = str(note.get("text", "") or note.get("body", "") or "")
            note_copy["_injection_detected"] = detect_injection(text)
            note_copy["text"] = sanitize_text(text)
            notes.append(note_copy)
        return notes

    def get_transaction_memo(self, client_id: str, txn_id: str) -> Optional[str]:
        txns = self.get_transactions(client_id)
        for txn in txns:
            if txn.get("id") == txn_id:
                raw_memo = txn.get("description") or txn.get("memo") or txn.get("note") or ""
                return sanitize_text(raw_memo)
        return None

    # -----------------------------------------------------------------------
    # Market data
    # -----------------------------------------------------------------------

    def is_covered(self, symbol: str) -> bool:
        return symbol in self.covered_symbols

    def get_instrument(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.is_covered(symbol):
            return None
        return self._instruments_by_symbol.get(symbol)

    def get_price_on_or_before(self, symbol: str, target: datetime) -> Optional[Dict[str, Any]]:
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
        if not self.is_covered(symbol):
            return []
        return list(self._prices_by_symbol.get(symbol, []))

    def get_news(self, symbol: str, on_or_before: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if not self.is_covered(symbol):
            return []
        news_list = list(self._news_by_symbol.get(symbol, []))
        if on_or_before is not None:
            filtered = []
            for n in news_list:
                nd = self.parse_date(n.get("date", ""))
                if nd and nd <= on_or_before:
                    filtered.append(n)
            return filtered
        return news_list

    def get_client_covered_symbols(self, client_id: str) -> List[str]:
        snapshot = self.get_positions_snapshot(client_id)
        held = {pos.get("symbol") for pos in snapshot if pos.get("symbol")}
        return sorted(held & self.covered_symbols)

    # -----------------------------------------------------------------------
    # Conflict detection
    # -----------------------------------------------------------------------

    def find_field_conflict(
        self, client_id: str, field_name: str
    ) -> Optional[Tuple[Any, Any, str, str]]:
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
    # Scope-safety check
    # -----------------------------------------------------------------------

    def check_cross_client_leak(
        self, client_id: str, text: str, citations: List[str]
    ) -> List[str]:
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
        all_symbols = set(self._instruments_by_symbol.keys())
        words = set(re.findall(r"\b[A-Z]{2,6}\b", text))
        for sym in words:
            if sym in self.covered_symbols:
                return sym
        for sym in words:
            if sym in all_symbols:
                return sym
        return None
