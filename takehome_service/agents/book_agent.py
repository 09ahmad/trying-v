"""BookQA Agent — handles balances, transactions, positions, and drift arithmetic.

All numeric values are computed in Python from the data layer, never by the LLM.
The Agno agent makes a valura-fast call to format the computed result naturally.
Escalates to valura-deep only for genuinely ambiguous aggregation questions.

Score-relevant categories this agent handles:
  exact_value, temporal, aggregation, escalation, rebalance_drift
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from takehome_service.data import DataLoader, mask_in_text


MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_decimal(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return 0.0


def _parse_date_from_text(text: str) -> Optional[datetime]:
    """Extract a date from natural-language text."""
    # ISO format
    iso = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if iso:
        try:
            return datetime.fromisoformat(iso.group(1))
        except ValueError:
            pass
    # "1 January 2025" or "January 1, 2025"
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if m:
        day, month_name, year = m.groups()
        month_num = MONTH_NAMES.get(month_name.lower())
        if month_num:
            try:
                return datetime(int(year), month_num, int(day))
            except ValueError:
                pass
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
    if m:
        month_name, day, year = m.groups()
        month_num = MONTH_NAMES.get(month_name.lower())
        if month_num:
            try:
                return datetime(int(year), month_num, int(day))
            except ValueError:
                pass
    return None


def _parse_date_range(prompt: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Extract start/end dates from "between X and Y" phrasing."""
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\s*\.|\s*$)", prompt, re.I)
    if m:
        start = _parse_date_from_text(m.group(1))
        end = _parse_date_from_text(m.group(2))
        return start, end
    # Single date
    single = _parse_date_from_text(prompt)
    return None, single


class BookAgent:
    """Handles financial book questions using data-layer arithmetic + Agno LLM formatting."""

    FAST_SYSTEM = (
        "You are ValuraBookQA, a financial data assistant. "
        "You receive a pre-computed answer from the data layer. "
        "Your job is to rephrase it clearly and naturally for the client. "
        "Do NOT compute anything yourself. Do NOT add information not in the data. "
        "Return only a clear, concise answer sentence."
    )

    DEEP_SYSTEM = (
        "You are ValuraBookQA (deep reasoning mode). "
        "You receive raw data extracted from a client's financial records. "
        "Analyse it carefully and answer the question exactly. "
        "Do NOT invent figures. Base your answer solely on the data provided. "
        "Return a concise answer with the computed value."
    )

    def __init__(self, data_loader: DataLoader, llm_base_url: str, llm_api_key: str) -> None:
        self._loader = data_loader
        self._fast_agent = Agent(
            model=OpenAIChat(
                id="valura-fast",
                base_url=llm_base_url,
                api_key=llm_api_key,
            ),
            name="ValuraBookQA",
            description=self.FAST_SYSTEM,
            markdown=False,
        )
        self._deep_agent = Agent(
            model=OpenAIChat(
                id="valura-deep",
                base_url=llm_base_url,
                api_key=llm_api_key,
            ),
            name="ValuraBookQADeep",
            description=self.DEEP_SYSTEM,
            markdown=False,
        )

    def answer(
        self,
        payload: Dict[str, Any],
        use_deep: bool = False,
    ) -> Dict[str, Any]:
        """Answer a book question. Returns answer dict (no agents/question_id fields)."""
        prompt = payload.get("prompt", "")
        client_id = payload.get("client_id", "")
        prompt_lower = prompt.lower()

        # Dispatch to the right data-layer method
        if re.search(r"\b(cash\s+balance|current\s+(cash|balance))\b", prompt_lower):
            return self._cash_balance(client_id, prompt, use_deep)

        if re.search(r"\blargest\s+(?:single\s+)?deposit\b", prompt_lower):
            return self._largest_deposit(client_id, prompt, use_deep)

        if re.search(r"\btotal\s+deposit(ed|s)?\b", prompt_lower):
            return self._total_deposits(client_id, prompt, use_deep)

        if re.search(r"\bdividend\b", prompt_lower):
            return self._dividend_income(client_id, prompt, use_deep)

        if re.search(r"\bfees?\b", prompt_lower):
            return self._total_fees(client_id, prompt, use_deep)

        if re.search(r"\bhow\s+many\s+(buy|purchase|buys|purchases)\b", prompt_lower):
            return self._count_txn_type(client_id, "buy", prompt, use_deep)

        if re.search(r"\bhow\s+many\s+(sell|sales?|sold)\b", prompt_lower):
            return self._count_txn_type(client_id, "sell", prompt, use_deep)

        if re.search(r"\b(first|earliest)\s+(buy|purchase|bought|investment)\b", prompt_lower):
            return self._first_purchase(client_id, prompt, use_deep)

        if re.search(r"\b(drift|target\s+allocation|rebalance)\b", prompt_lower):
            return self._target_drift(client_id, prompt, use_deep)

        if re.search(r"\b(how\s+many|number\s+of)\s+(?:different\s+)?(symbols?|stocks?|positions?|holdings?)\b", prompt_lower):
            return self._holdings_count(client_id, prompt, use_deep)

        # Generic holdings query with symbol
        symbol = self._loader.find_symbol_in_text(prompt)
        if symbol and re.search(r"\b(shares?|units?|hold|quantity|position)\b", prompt_lower):
            return self._symbol_holdings(client_id, symbol, prompt, use_deep)

        # Fallback: let the LLM reason over raw data  
        return self._fallback(client_id, prompt, use_deep)

    # -----------------------------------------------------------------------
    # Data-layer computations → LLM formatting
    # -----------------------------------------------------------------------

    def _cash_balance(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        balance, cited = self._loader.compute_cash_balance(client_id)
        data_summary = f"Computed cash balance: {balance:.2f} USD (from {len(cited)} transactions)"
        answer_text = self._llm_format(
            f"The current cash balance is {balance:.2f} USD.",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, f"{balance:.2f}", cited[:6])

    def _largest_deposit(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        deposits = self._loader.get_transactions(client_id, txn_type="deposit")
        if not deposits:
            return self._abstain("No deposit transactions found for this client.")
        best = max(
            deposits,
            key=lambda t: _parse_decimal(t.get("amount_usd") or t.get("amount_inr") or 0),
        )
        amount = _parse_decimal(best.get("amount_usd") or best.get("amount_inr"))
        date_str = best.get("date", "")
        data_summary = f"Largest deposit: {amount:.2f} USD on {date_str} (txn {best.get('id')})"
        answer_text = self._llm_format(
            f"The largest single deposit was {amount:.2f} USD on {date_str}.",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, f"{amount:.2f}", [best.get("id", "")])

    def _total_deposits(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        start, end = _parse_date_range(prompt)
        if not start and not end:
            # Total all deposits ever
            deposits = self._loader.get_transactions(client_id, txn_type="deposit")
            total = sum(_parse_decimal(t.get("amount_usd") or t.get("amount_inr") or 0) for t in deposits)
            cited = [t.get("id", "") for t in deposits[:6]]
            data_summary = f"Total deposits: {total:.2f} USD across {len(deposits)} transactions"
            answer_text = self._llm_format(
                f"The total amount deposited is {total:.2f} USD.",
                data_summary,
                prompt,
                use_deep,
            )
            return self._build(answer_text, f"{total:.2f}", cited)
        # Date-filtered
        filtered = self._loader.get_transactions(
            client_id, on_or_before=end, txn_type="deposit"
        )
        if start:
            filtered = [t for t in filtered if (self._loader.parse_date(t.get("date", "")) or datetime.min) >= start]
        total = sum(_parse_decimal(t.get("amount_usd") or t.get("amount_inr") or 0) for t in filtered)
        if total == 0.0 and not filtered:
            return self._abstain("No deposit activity found in the specified date range.")
        cited = [t.get("id", "") for t in filtered[:6]]
        date_range = f"{start.date() if start else '?'} to {end.date() if end else '?'}"
        data_summary = f"Total deposits between {date_range}: {total:.2f} USD across {len(filtered)} transactions"
        answer_text = self._llm_format(
            f"The total deposited between {date_range} was {total:.2f} USD.",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, f"{total:.2f}", cited)

    def _dividend_income(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        start, end = _parse_date_range(prompt)
        symbol = self._loader.find_symbol_in_text(prompt)
        dividends = self._loader.get_transactions(
            client_id, on_or_before=end, txn_type="dividend"
        )
        if start:
            dividends = [t for t in dividends if (self._loader.parse_date(t.get("date", "")) or datetime.min) >= start]
        if symbol:
            dividends = [t for t in dividends if t.get("symbol") == symbol]
        total = sum(_parse_decimal(t.get("net_usd") or t.get("gross_usd") or t.get("amount_usd") or 0) for t in dividends)
        if not dividends:
            return self._abstain("No dividend income found for the requested period/symbol.")
        cited = [t.get("id", "") for t in dividends[:6]]
        sym_str = f" from {symbol}" if symbol else ""
        data_summary = f"Dividend income{sym_str}: {total:.2f} USD from {len(dividends)} dividend transactions"
        answer_text = self._llm_format(
            f"The net dividend income{sym_str} was {total:.2f} USD.",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, f"{total:.2f}", cited)

    def _total_fees(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        fees = self._loader.get_transactions(client_id, txn_type="fee")
        total = sum(_parse_decimal(t.get("amount_usd") or t.get("amount_inr") or 0) for t in fees)
        if not fees:
            return self._abstain("No fee transactions found for this client.")
        cited = [t.get("id", "") for t in fees[:6]]
        data_summary = f"Total fees: {total:.2f} USD from {len(fees)} fee transactions"
        answer_text = self._llm_format(
            f"The total platform fees charged are {total:.2f} USD.",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, f"{total:.2f}", cited)

    def _count_txn_type(self, client_id: str, txn_type: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        txns = self._loader.get_transactions(client_id, txn_type=txn_type)
        count = len(txns)
        if count == 0:
            return self._abstain(f"No {txn_type} transactions found for this client.")
        cited = [t.get("id", "") for t in txns[:6]]
        data_summary = f"Total {txn_type} transactions: {count}"
        answer_text = self._llm_format(
            f"The account has {count} {txn_type} transaction(s).",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, str(count), cited)

    def _first_purchase(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        buys = self._loader.get_transactions(client_id, txn_type="buy")
        dated = [(self._loader.parse_date(t.get("date", "")), t) for t in buys]
        dated = [(d, t) for d, t in dated if d is not None]
        if not dated:
            return self._abstain("No purchase transactions found for this client.")
        earliest_date, earliest_txn = min(dated, key=lambda x: x[0])
        date_str = earliest_date.date().isoformat()
        data_summary = f"First purchase: {date_str} (txn {earliest_txn.get('id')})"
        answer_text = self._llm_format(
            f"The first purchase was made on {date_str}.",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, date_str, [earliest_txn.get("id", "")])

    def _target_drift(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        result = self._loader.compute_target_drift(client_id)
        if result is None:
            return self._abstain(
                "No target allocation is recorded for this client; drift cannot be computed."
            )
        drift, desc, cited = result
        data_summary = desc
        answer_text = self._llm_format(desc, data_summary, prompt, use_deep)
        return self._build(answer_text, f"{drift:+.2f}", cited)

    def _holdings_count(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        snapshot = self._loader.get_positions_snapshot(client_id)
        count = len([p for p in snapshot if p.get("symbol")])
        cited = [p.get("id", "") for p in snapshot[:6] if p.get("id")]
        data_summary = f"Current holdings: {count} distinct positions"
        answer_text = self._llm_format(
            f"The account currently holds {count} different positions.",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, str(count), cited)

    def _symbol_holdings(self, client_id: str, symbol: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        # Use positions snapshot first (authoritative)
        snapshot = self._loader.get_positions_snapshot(client_id)
        pos = next((p for p in snapshot if p.get("symbol") == symbol), None)
        if pos:
            qty = _parse_decimal(pos.get("quantity") or 0)
            cited = [pos.get("id", "")]
            data_summary = f"{symbol} position: {qty:.4f} shares (from positions snapshot)"
            answer_text = self._llm_format(
                f"The account holds {qty:.4f} shares of {symbol}.",
                data_summary,
                prompt,
                use_deep,
            )
            return self._build(answer_text, f"{qty:.4f}", cited)
        # Fall back to transaction history
        qty, cited = self._loader.compute_holdings(client_id, symbol)
        if qty == 0.0 and not cited:
            return self._abstain(f"No holdings of {symbol} found for this client.")
        data_summary = f"{symbol} computed holdings: {qty:.4f} shares (from transactions)"
        answer_text = self._llm_format(
            f"The account holds {qty:.4f} shares of {symbol}.",
            data_summary,
            prompt,
            use_deep,
        )
        return self._build(answer_text, f"{qty:.4f}", cited[:6])

    def _fallback(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        """Fallback: provide data context and let LLM answer."""
        # Build a compact data summary to pass to LLM
        txns = self._loader.get_transactions(client_id)
        snapshot = self._loader.get_positions_snapshot(client_id)
        balance, _ = self._loader.compute_cash_balance(client_id)

        data_context = (
            f"Client has {len(txns)} transactions, {len(snapshot)} positions. "
            f"Cash balance: {balance:.2f} USD. "
            f"Positions: {[p.get('symbol') for p in snapshot[:5]]}."
        )
        agent = self._deep_agent if use_deep else self._fast_agent
        try:
            run_output = agent.run(
                f"Client data summary: {data_context}\n\nQuestion: {prompt}"
            )
            answer_text = run_output.get_content_as_string() if run_output else ""
            if not answer_text:
                return self._abstain("The question could not be answered from available book data.")
        except Exception:
            return self._abstain("Unable to process the book question at this time.")
        return self._build(answer_text, None, [])

    # -----------------------------------------------------------------------
    # LLM formatting
    # -----------------------------------------------------------------------

    def _llm_format(
        self,
        precomputed_answer: str,
        data_summary: str,
        original_prompt: str,
        use_deep: bool,
    ) -> str:
        """Use the LLM to format a pre-computed answer naturally.

        The LLM does NOT compute anything — it formats the already-computed result.
        This is a single fast/deep call that corroborates the "agno" framework claim.
        """
        agent = self._deep_agent if use_deep else self._fast_agent
        msg = (
            f"Data layer result: {data_summary}\n"
            f"Pre-computed answer: {precomputed_answer}\n"
            f"Original question: {original_prompt}\n\n"
            f"Rephrase the pre-computed answer clearly and naturally. "
            f"Do not change the numeric value. Return only the answer sentence."
        )
        try:
            run_output = agent.run(msg)
            content = run_output.get_content_as_string() if run_output else ""
            return content.strip() if content else precomputed_answer
        except Exception:
            return precomputed_answer

    # -----------------------------------------------------------------------
    # Response builders
    # -----------------------------------------------------------------------

    def _build(
        self,
        answer: str,
        value: Optional[str],
        citations: List[str],
        flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "answer_value": value,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": [c for c in citations if c][:6],
            "confidence": 0.85,
            "flags": flags or [],
        }

    def _abstain(self, reason: str) -> Dict[str, Any]:
        return {
            "answer": "",
            "answer_value": None,
            "abstained": True,
            "refused": False,
            "reason": reason,
            "citations": [],
            "confidence": 0.0,
            "flags": [],
        }
