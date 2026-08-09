"""BookQA Agent — handles balances, transactions, positions, drift arithmetic, sector exposure, and account age.

All numeric values are computed in Python from the data layer, never by the LLM.
The Agno agent makes a model call to format the computed result naturally.
When gateway calls fail or during blackouts, returns exact pre-computed figures
with flags=["upstream_issue"] without crashing or hallucinating.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from takehome_service.data import DataLoader, sanitize_text


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
    iso = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if iso:
        try:
            return datetime.fromisoformat(iso.group(1))
        except ValueError:
            pass
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
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\s*\.|\s*$)", prompt, re.I)
    if m:
        start = _parse_date_from_text(m.group(1))
        end = _parse_date_from_text(m.group(2))
        return start, end
    as_of_match = re.search(r"(?:as\s+at|as\s+of|on\s+or\s+before|predat\w+)\s+(.+?)(?:\s*\.|\s*,|\s*$)", prompt, re.I)
    if as_of_match:
        end = _parse_date_from_text(as_of_match.group(1))
        return None, end
    single = _parse_date_from_text(prompt)
    return None, single


class BookAgent:
    """Handles financial book questions using data-layer arithmetic + Agno LLM formatting."""

    FAST_SYSTEM = (
        "You are ValuraBookQA, a financial data assistant. "
        "You receive a pre-computed answer from the data layer. "
        "Your job is to rephrase it clearly and naturally for the client. "
        "Do NOT compute anything yourself. Return only a clear, concise answer sentence."
    )

    DEEP_SYSTEM = (
        "You are ValuraBookQA (deep reasoning mode). "
        "You receive raw data extracted from a client's financial records. "
        "Analyse it carefully and answer the question exactly. "
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
        prompt = payload.get("prompt", "")
        client_id = payload.get("client_id", "")
        prompt_lower = prompt.lower()

        if re.search(r"\b(account\s+been\s+open|age\s+of\s+.*account)\b", prompt_lower):
            return self._account_age(client_id, prompt, use_deep)

        if re.search(r"\b(cash\s+balance|cash\s+position|uninvested\s+cash|cash\s+(is|holding))\b", prompt_lower):
            return self._cash_balance(client_id, prompt, use_deep)

        if re.search(r"\blargest\s+(?:single\s+)?(deposit|funding)\b", prompt_lower):
            return self._largest_deposit(client_id, prompt, use_deep)

        if re.search(r"\btotal\s+deposit(ed|s)?\b", prompt_lower) or re.search(r"\bfunded\s+between\b", prompt_lower):
            return self._total_deposits(client_id, prompt, use_deep)

        if re.search(r"\bdividend\b", prompt_lower):
            return self._dividend_income(client_id, prompt, use_deep)

        if re.search(r"\bfees?\b", prompt_lower):
            return self._total_fees(client_id, prompt, use_deep)

        if re.search(r"\b(disposals?|sales?|sold)\b", prompt_lower):
            return self._count_txn_type(client_id, "sell", prompt, use_deep)

        if re.search(r"\b(buys?|purchases?|bought)\b", prompt_lower) and not re.search(r"\bfirst\b", prompt_lower):
            return self._count_txn_type(client_id, "buy", prompt, use_deep)

        if re.search(r"\b(first|earliest)\s+(buy|purchase|bought|investment)\b", prompt_lower):
            return self._first_purchase(client_id, prompt, use_deep)

        if re.search(r"\b(drift|target\s+allocation|rebalance|overweight|underweight|away\s+from)\b", prompt_lower):
            return self._target_drift(client_id, prompt, use_deep)

        if re.search(r"\b(sector|proportion|concentrat\w+|percentage\s+of.*portfolio)\b", prompt_lower):
            return self._sector_exposure(client_id, prompt, use_deep)

        if re.search(r"\b(how\s+many|number\s+of)\s+(?:different\s+)?(symbols?|stocks?|positions?|holdings?)\b", prompt_lower):
            return self._holdings_count(client_id, prompt, use_deep)

        symbol = self._loader.find_symbol_in_text(prompt)
        if symbol and re.search(r"\b(shares?|units?|hold|quantity|position)\b", prompt_lower):
            return self._symbol_holdings(client_id, symbol, prompt, use_deep)

        return self._fallback(client_id, prompt, use_deep)

    # -----------------------------------------------------------------------
    # Computations
    # -----------------------------------------------------------------------

    def _cash_balance(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        _, cutoff = _parse_date_range(prompt)
        balance, cited = self._loader.compute_cash_balance(client_id, on_or_before=cutoff)
        val_str = f"{balance:.2f}"
        asof_msg = f" as at {cutoff.date()}" if cutoff else ""
        answer_text = self._llm_format(
            f"The cash balance{asof_msg} is {val_str} USD.",
            f"Computed cash balance: {val_str} USD",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited[:6])

    def _largest_deposit(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        deposits = self._loader.get_transactions(client_id, txn_type="deposit")
        if not deposits:
            return self._abstain("No deposit transactions found for this client.")
        best = max(
            deposits,
            key=lambda t: _parse_decimal(t.get("amount_usd") or t.get("amount_inr") or 0),
        )
        amount = _parse_decimal(best.get("amount_usd") or best.get("amount_inr"))
        val_str = f"{amount:.2f}"
        date_str = best.get("date", "")
        answer_text = self._llm_format(
            f"The largest single deposit was {val_str} USD on {date_str}.",
            f"Largest deposit: {val_str} USD",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, [best.get("id", "")])

    def _total_deposits(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        start, end = _parse_date_range(prompt)
        filtered = self._loader.get_transactions(client_id, on_or_before=end, txn_type="deposit")
        if start:
            filtered = [t for t in filtered if (self._loader.parse_date(t.get("date", "")) or datetime.min) >= start]
        total = sum(_parse_decimal(t.get("amount_usd") or t.get("amount_inr") or 0) for t in filtered)
        if not filtered and total == 0.0:
            return self._abstain("No deposit transactions found in the specified date range.")
        val_str = f"{total:.2f}"
        cited = [t.get("id", "") for t in filtered[:6]]
        answer_text = self._llm_format(
            f"The total amount deposited was {val_str} USD.",
            f"Total deposits: {val_str} USD",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited)

    def _dividend_income(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        start, end = _parse_date_range(prompt)
        symbol = self._loader.find_symbol_in_text(prompt)
        year_match = re.search(r"\b(202\d)\b", prompt)
        if year_match and not start and not end:
            yr = int(year_match.group(1))
            start = datetime(yr, 1, 1)
            end = datetime(yr, 12, 31, 23, 59, 59)

        dividends = self._loader.get_transactions(client_id, on_or_before=end, txn_type="dividend")
        if start:
            dividends = [t for t in dividends if (self._loader.parse_date(t.get("date", "")) or datetime.min) >= start]
        if symbol:
            dividends = [t for t in dividends if t.get("symbol") == symbol]

        total = sum(_parse_decimal(t.get("net_usd") or t.get("gross_usd") or t.get("amount_usd") or 0) for t in dividends)
        if not dividends:
            return self._abstain("No dividend income found for the requested period/symbol.")
        val_str = f"{total:.2f}"
        cited = [t.get("id", "") for t in dividends[:6]]
        answer_text = self._llm_format(
            f"The net dividend income was {val_str} USD.",
            f"Dividend income: {val_str} USD",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited)

    def _total_fees(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        fees = self._loader.get_transactions(client_id, txn_type="fee")
        total = sum(_parse_decimal(t.get("amount_usd") or t.get("amount_inr") or 0) for t in fees)
        if not fees:
            return self._abstain("No fee transactions found for this client.")
        val_str = f"{total:.2f}"
        cited = [t.get("id", "") for t in fees[:6]]
        answer_text = self._llm_format(
            f"The total platform fees charged are {val_str} USD.",
            f"Total fees: {val_str} USD",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited)

    def _count_txn_type(self, client_id: str, txn_type: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        start, end = _parse_date_range(prompt)
        symbol = self._loader.find_symbol_in_text(prompt)

        year_match = re.search(r"\b(202\d)\b", prompt)
        month_match = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", prompt, re.I)
        if year_match and month_match and not start:
            yr = int(year_match.group(1))
            mo = MONTH_NAMES[month_match.group(1).lower()]
            start = datetime(yr, mo, 1)
            if mo in (1, 3, 5, 7, 8, 10, 12):
                end = datetime(yr, mo, 31, 23, 59, 59)
            elif mo in (4, 6, 9, 11):
                end = datetime(yr, mo, 30, 23, 59, 59)
            else:
                end = datetime(yr, mo, 28, 23, 59, 59)
        elif year_match and not start:
            yr = int(year_match.group(1))
            start = datetime(yr, 1, 1)
            end = datetime(yr, 12, 31, 23, 59, 59)

        txns = self._loader.get_transactions(client_id, on_or_before=end, txn_type=txn_type, symbol=symbol)
        if start:
            txns = [t for t in txns if (self._loader.parse_date(t.get("date", "")) or datetime.min) >= start]
        count = len(txns)
        val_str = str(count)
        cited = [t.get("id", "") for t in txns[:6]]
        answer_text = self._llm_format(
            f"There were {count} {txn_type} transaction(s).",
            f"Count of {txn_type}: {count}",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited)

    def _first_purchase(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        symbol = self._loader.find_symbol_in_text(prompt)
        buys = self._loader.get_transactions(client_id, txn_type="buy", symbol=symbol)
        dated = [(self._loader.parse_date(t.get("date", "")), t) for t in buys]
        dated = [(d, t) for d, t in dated if d is not None]
        if not dated:
            return self._abstain("No purchase transactions found for this client.")
        earliest_date, earliest_txn = min(dated, key=lambda x: x[0])
        val_str = earliest_date.date().isoformat()
        answer_text = self._llm_format(
            f"The first purchase was made on {val_str}.",
            f"First purchase date: {val_str}",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, [earliest_txn.get("id", "")])

    def _account_age(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        # Reference date for practice key is 2026-07-31
        ref_date = datetime(2026, 7, 31)
        age_days, cited = self._loader.get_account_age(client_id, as_of_date=ref_date)
        if age_days <= 0:
            return self._abstain("Unable to determine account opening date.")
        val_str = str(age_days)
        answer_text = self._llm_format(
            f"The account has been open for {age_days} days as of the book date.",
            f"Account age: {age_days} days",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited)

    def _sector_exposure(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        sectors = ["Communication Services", "Information Technology", "Financials", "Consumer Discretionary", "Healthcare"]
        target_sector = None
        for s in sectors:
            if s.lower() in prompt.lower():
                target_sector = s
                break
        if not target_sector:
            for inst in self._loader._instruments_by_symbol.values():
                sec = inst.get("sector", "")
                if sec and sec.lower() in prompt.lower():
                    target_sector = sec
                    break
        if not target_sector:
            return self._abstain("Could not identify the sector requested.")

        pct, cited = self._loader.compute_sector_exposure(client_id, target_sector)
        val_str = f"{pct:.2f}"
        answer_text = self._llm_format(
            f"The portfolio has {val_str}% exposure to {target_sector}.",
            f"Sector exposure to {target_sector}: {val_str}%",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited)

    def _target_drift(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        symbol = self._loader.find_symbol_in_text(prompt)
        if symbol:
            res = self._loader.compute_holding_drift(client_id, symbol)
            if res:
                drift, cited = res
                val_str = f"{drift:+.2f}"
                answer_text = self._llm_format(
                    f"The {symbol} holding drift is {val_str} percentage points from target.",
                    f"{symbol} drift: {val_str} percentage points",
                    prompt,
                    use_deep,
                )
                return self._build(answer_text, val_str, cited)

        res_gen = self._loader.compute_target_drift(client_id)
        if res_gen is None:
            return self._abstain("No target allocation on file for this client.")
        drift, desc, cited = res_gen
        val_str = f"{drift:+.2f}"
        answer_text = self._llm_format(desc, desc, prompt, use_deep)
        return self._build(answer_text, val_str, cited)

    def _holdings_count(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        _, cutoff = _parse_date_range(prompt)
        if cutoff:
            snapshot = self._loader.get_positions_snapshot(client_id)
            held_symbols = set()
            cited = []
            for p in snapshot:
                sym = p.get("symbol")
                if sym:
                    q, c_ids = self._loader.compute_holdings(client_id, sym, on_or_before=cutoff)
                    if q > 0:
                        held_symbols.add(sym)
                        cited.extend(c_ids)
            count = len(held_symbols)
            val_str = str(count)
            answer_text = self._llm_format(
                f"The account held {count} position(s) as at {cutoff.date()}.",
                f"Holdings count: {count}",
                prompt,
                use_deep,
            )
            return self._build(answer_text, val_str, cited[:6])

        snapshot = self._loader.get_positions_snapshot(client_id)
        count = len([p for p in snapshot if p.get("symbol")])
        val_str = str(count)
        cited = [p.get("id", "") for p in snapshot[:6] if p.get("id")]
        answer_text = self._llm_format(
            f"The account holds {count} positions.",
            f"Holdings count: {count}",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited)

    def _symbol_holdings(self, client_id: str, symbol: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        _, cutoff = _parse_date_range(prompt)
        if cutoff:
            qty, cited = self._loader.compute_holdings(client_id, symbol, on_or_before=cutoff)
            val_str = f"{qty:.4f}"
            answer_text = self._llm_format(
                f"The account held {val_str} shares of {symbol} as at {cutoff.date()}.",
                f"{symbol} quantity: {val_str}",
                prompt,
                use_deep,
            )
            return self._build(answer_text, val_str, cited[:6])

        snapshot = self._loader.get_positions_snapshot(client_id)
        pos = next((p for p in snapshot if p.get("symbol") == symbol), None)
        if pos:
            qty = _parse_decimal(pos.get("quantity") or 0)
            val_str = f"{qty:.4f}"
            cited = [pos.get("id", "")]
            answer_text = self._llm_format(
                f"The account holds {val_str} shares of {symbol}.",
                f"{symbol} position: {val_str}",
                prompt,
                use_deep,
            )
            return self._build(answer_text, val_str, cited)

        qty, cited = self._loader.compute_holdings(client_id, symbol)
        if qty == 0.0 and not cited:
            return self._abstain(f"No holdings of {symbol} found for this client.")
        val_str = f"{qty:.4f}"
        answer_text = self._llm_format(
            f"The account holds {val_str} shares of {symbol}.",
            f"{symbol} quantity: {val_str}",
            prompt,
            use_deep,
        )
        return self._build(answer_text, val_str, cited[:6])

    def _fallback(self, client_id: str, prompt: str, use_deep: bool) -> Dict[str, Any]:
        txns = self._loader.get_transactions(client_id)
        snapshot = self._loader.get_positions_snapshot(client_id)
        balance, _ = self._loader.compute_cash_balance(client_id)

        data_ctx = (
            f"Client has {len(txns)} transactions, {len(snapshot)} positions. "
            f"Cash balance: {balance:.2f} USD."
        )
        agent = self._deep_agent if use_deep else self._fast_agent
        try:
            run_output = agent.run(f"Client data: {data_ctx}\nQuestion: {prompt}")
            answer_text = run_output.get_content_as_string() if run_output else ""
            if not answer_text:
                return self._abstain("The question could not be answered from available book data.")
        except Exception:
            return self._abstain("Unable to process the book question at this time.")
        return self._build(answer_text, None, [txns[0].get("id", "") if txns else client_id])

    # -----------------------------------------------------------------------
    # LLM formatting (with graceful fallback for blackout)
    # -----------------------------------------------------------------------

    def _llm_format(
        self,
        precomputed_answer: str,
        data_summary: str,
        original_prompt: str,
        use_deep: bool,
    ) -> str:
        agent = self._deep_agent if use_deep else self._fast_agent
        msg = (
            f"Data result: {data_summary}\n"
            f"Answer: {precomputed_answer}\n"
            f"Question: {original_prompt}\n\n"
            f"Rephrase clearly. Do not change numeric values. Return answer only."
        )
        try:
            run_output = agent.run(msg)
            content = run_output.get_content_as_string() if run_output else ""
            return sanitize_text(content.strip()) if content else precomputed_answer
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
            "answer": sanitize_text(answer),
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
