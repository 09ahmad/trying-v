"""MarketDesk Agent — instrument/sector/price-history/news lookups.

Critical rules:
- Explicitly abstains when a symbol is outside covered_symbols. A model will
  answer from parametric memory; this agent must NOT — uncovered means no price,
  sector, or news here, period.
- Distinguishes "current drift from target" (arithmetic → answer) from "what
  should the allocation be" (advice → refuse/compliance).
- Prices are month-start closes; for a date between two points, uses the most
  recent close on or before that date and states which date was used.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from takehome_service.data import DataLoader


class MarketDeskAgent:
    """Handles market data questions: prices, sectors, news, coverage, drift."""

    SYSTEM = (
        "You are ValuraMarket, a market data assistant. "
        "You can ONLY answer about instruments in the provided data. "
        "If an instrument is not in the data, you must say so explicitly. "
        "Prices are month-start closes. State which date's close you used. "
        "Return a clear, concise answer. Do not use your own knowledge of market prices."
    )

    def __init__(self, data_loader: DataLoader, llm_base_url: str, llm_api_key: str) -> None:
        self._loader = data_loader
        self._agent = Agent(
            model=OpenAIChat(
                id="valura-fast",
                base_url=llm_base_url,
                api_key=llm_api_key,
            ),
            name="ValuraMarket",
            description=self.SYSTEM,
            markdown=False,
        )

    def answer(self, payload: Dict[str, Any], use_deep: bool = False) -> Dict[str, Any]:
        """Answer a market data question."""
        prompt = payload.get("prompt", "")
        client_id = payload.get("client_id", "")
        prompt_lower = prompt.lower()

        # Extract symbol from prompt
        symbol = self._loader.find_symbol_in_text(prompt)

        # --- Coverage check: must come FIRST, before any other lookup ---
        if re.search(r"\b(covered|coverage|data\s+available|in\s+your\s+coverage)\b", prompt_lower):
            if symbol:
                if self._loader.is_covered(symbol):
                    return self._build(
                        f"{symbol} is covered in the market dataset.",
                        "covered",
                        [symbol],
                    )
                return self._abstain(
                    f"{symbol} is not in the covered symbols list. "
                    f"No price, sector, or news data is available for this instrument."
                )
            return self._abstain("No symbol was identified in the coverage question.")

        # --- Sector / industry ---
        if re.search(r"\b(sector|industry|asset\s+class)\b", prompt_lower) and symbol:
            return self._sector_answer(symbol, prompt)

        # --- Price / close ---
        if re.search(r"\b(close|closing|price|worth|value)\b", prompt_lower) and symbol:
            return self._price_answer(symbol, prompt, prompt_lower)

        # --- Return / performance ---
        if re.search(r"\b(return|performance|gain|loss|percentage\s+change|grew|fell|rose)\b", prompt_lower) and symbol:
            return self._return_answer(symbol, prompt, prompt_lower)

        # --- News ---
        if re.search(r"\b(news|headline|announcement|articles?|reports?)\b", prompt_lower):
            if symbol:
                return self._news_answer(symbol, prompt)
            # Client-scoped news: find their covered holdings
            covered = self._loader.get_client_covered_symbols(client_id)
            if covered:
                return self._multi_news_answer(covered, prompt)
            return self._abstain("No covered symbols found for this client to retrieve news for.")

        # --- Generic market question with symbol ---
        if symbol:
            if not self._loader.is_covered(symbol):
                return self._abstain(
                    f"{symbol} is not in the covered dataset. "
                    f"No price, sector, or news is available for this instrument."
                )
            return self._generic_market_answer(symbol, prompt)

        return self._abstain("No covered instrument symbol was identified in the market question.")

    # -----------------------------------------------------------------------
    # Specific answer methods
    # -----------------------------------------------------------------------

    def _sector_answer(self, symbol: str, prompt: str) -> Dict[str, Any]:
        """Answer a sector/industry question for a symbol."""
        if not self._loader.is_covered(symbol):
            return self._abstain(
                f"{symbol} is not in the covered dataset. "
                f"Sector and industry information is not available."
            )
        inst = self._loader.get_instrument(symbol)
        if not inst:
            return self._abstain(f"No instrument record found for {symbol}.")
        sector = inst.get("sector", "")
        industry = inst.get("industry", "")
        answer_value = sector
        precomputed = f"{symbol} belongs to the {sector} sector"
        if industry:
            precomputed += f" ({industry} industry)"
        precomputed += "."
        answer_text = self._llm_format(precomputed, prompt)
        return self._build(answer_text, answer_value, [symbol])

    def _price_answer(self, symbol: str, prompt: str, prompt_lower: str) -> Dict[str, Any]:
        """Answer a price/close question for a symbol at a given date."""
        if not self._loader.is_covered(symbol):
            return self._abstain(
                f"{symbol} is not in the covered dataset. "
                f"No price data is available for this instrument."
            )
        target_date = self._extract_date(prompt)
        if target_date is None:
            # Default to most recent price
            history = self._loader.get_price_history(symbol)
            if not history:
                return self._abstain(f"No price history available for {symbol}.")
            latest = max(history, key=lambda p: p.get("date", ""))
            close = float(latest.get("close", 0))
            date_used = latest.get("date", "")
            precomputed = f"{symbol} closed at {close:.2f} USD on {date_used} (most recent available)."
            answer_text = self._llm_format(precomputed, prompt)
            return self._build(answer_text, f"{close:.2f}", [symbol])

        price_rec = self._loader.get_price_on_or_before(symbol, target_date)
        if not price_rec:
            return self._abstain(
                f"No price data available for {symbol} on or before {target_date.date()}."
            )
        close = float(price_rec.get("close", 0))
        date_used = price_rec.get("date", "")
        precomputed = (
            f"{symbol} closed at {close:.2f} USD on {date_used}. "
            f"(Most recent month-start close on or before {target_date.date()}.)"
        )
        answer_text = self._llm_format(precomputed, prompt)
        return self._build(answer_text, f"{close:.2f}", [symbol])

    def _return_answer(self, symbol: str, prompt: str, prompt_lower: str) -> Dict[str, Any]:
        """Compute percentage return between two dates."""
        if not self._loader.is_covered(symbol):
            return self._abstain(
                f"{symbol} is not in the covered dataset. "
                f"No price/return data is available."
            )
        dates = self._extract_two_dates(prompt)
        if not dates:
            return self._abstain(
                "Could not parse the date range for the return calculation."
            )
        start_date, end_date = dates
        start_price_rec = self._loader.get_price_on_or_before(symbol, start_date)
        end_price_rec = self._loader.get_price_on_or_before(symbol, end_date)
        if not start_price_rec or not end_price_rec:
            return self._abstain(
                f"Insufficient price data for {symbol} in the requested range."
            )
        start_close = float(start_price_rec.get("close", 0))
        end_close = float(end_price_rec.get("close", 0))
        if start_close == 0:
            return self._abstain(f"Start price for {symbol} is zero; cannot compute return.")
        pct_return = ((end_close - start_close) / start_close) * 100
        start_date_used = start_price_rec.get("date", "")
        end_date_used = end_price_rec.get("date", "")
        precomputed = (
            f"{symbol} returned {pct_return:+.2f}% from {start_date_used} "
            f"({start_close:.2f} USD) to {end_date_used} ({end_close:.2f} USD)."
        )
        answer_text = self._llm_format(precomputed, prompt)
        return self._build(answer_text, f"{pct_return:.2f}", [symbol])

    def _news_answer(self, symbol: str, prompt: str) -> Dict[str, Any]:
        """Return news summary for a covered symbol."""
        if not self._loader.is_covered(symbol):
            return self._abstain(
                f"{symbol} is not in the covered dataset. "
                f"No news is available for this instrument."
            )
        news_items = self._loader.get_news(symbol)
        if not news_items:
            return self._abstain(f"No news items found for {symbol} in the dataset.")
        # Sort by date descending
        news_items = sorted(news_items, key=lambda n: n.get("date", ""), reverse=True)
        count = len(news_items)
        news_ids = [n.get("id", "") for n in news_items[:6] if n.get("id")]
        # Build context for LLM
        news_context = "\n".join(
            f"- [{n.get('date')}] {n.get('headline', '')}: {n.get('body', '')}"
            for n in news_items[:5]
        )
        precomputed = f"{symbol} has {count} news items in the dataset. Recent items:\n{news_context}"
        answer_text = self._llm_format(precomputed, prompt)
        return self._build(answer_text, str(count), news_ids)

    def _multi_news_answer(self, symbols: List[str], prompt: str) -> Dict[str, Any]:
        """Return news for multiple covered symbols held by client."""
        all_news = []
        news_ids = []
        for sym in symbols:
            items = self._loader.get_news(sym)
            for n in items:
                all_news.append(f"[{n.get('date')}] {sym}: {n.get('headline', '')}")
                if n.get("id"):
                    news_ids.append(n["id"])
        if not all_news:
            return self._abstain("No news items found for the client's covered holdings.")
        context = "\n".join(all_news[:10])
        answer_text = self._llm_format(
            f"Recent news for covered holdings:\n{context}", prompt
        )
        return self._build(answer_text, str(len(all_news)), news_ids[:6])

    def _generic_market_answer(self, symbol: str, prompt: str) -> Dict[str, Any]:
        """Generic market question for a covered symbol."""
        inst = self._loader.get_instrument(symbol)
        history = self._loader.get_price_history(symbol)
        news = self._loader.get_news(symbol)
        data_ctx = (
            f"Symbol: {symbol}\n"
            f"Instrument: {inst}\n"
            f"Price history ({len(history)} records, latest: "
            f"{sorted(history, key=lambda p: p.get('date',''))[-1] if history else 'none'})\n"
            f"News: {len(news)} items"
        )
        try:
            run_output = self._agent.run(
                f"Market data:\n{data_ctx}\n\nQuestion: {prompt}\n\n"
                f"Answer using only the data above. Do not use your own knowledge of prices."
            )
            content = run_output.get_content_as_string() if run_output else ""
            if content:
                return self._build(content.strip(), None, [symbol])
        except Exception:
            pass
        return self._abstain(f"Could not answer the market question for {symbol}.")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _extract_date(self, text: str) -> Optional[datetime]:
        """Extract a single target date from text."""
        iso = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if iso:
            try:
                return datetime.fromisoformat(iso.group(1))
            except ValueError:
                pass
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
        if m:
            day, mname, year = m.groups()
            mnum = month_names.get(mname.lower())
            if mnum:
                try:
                    return datetime(int(year), mnum, int(day))
                except ValueError:
                    pass
        m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
        if m:
            mname, day, year = m.groups()
            mnum = month_names.get(mname.lower())
            if mnum:
                try:
                    return datetime(int(year), mnum, int(day))
                except ValueError:
                    pass
        return None

    def _extract_two_dates(self, text: str) -> Optional[tuple]:
        """Extract start and end dates from 'between X and Y' or two ISO dates."""
        m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\s*\.|\s*$)", text, re.I)
        if m:
            d1 = self._extract_date(m.group(1))
            d2 = self._extract_date(m.group(2))
            if d1 and d2:
                return d1, d2
        # Two ISO dates
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        if len(dates) >= 2:
            try:
                return datetime.fromisoformat(dates[0]), datetime.fromisoformat(dates[1])
            except ValueError:
                pass
        return None

    def _llm_format(self, precomputed: str, original_prompt: str) -> str:
        try:
            run_output = self._agent.run(
                f"Data result: {precomputed}\n"
                f"Question: {original_prompt}\n\n"
                f"Rephrase the data result clearly. Do not change any values or add your own knowledge."
            )
            content = run_output.get_content_as_string() if run_output else ""
            return content.strip() if content else precomputed
        except Exception:
            return precomputed

    def _build(
        self, answer: str, value: Optional[str], citations: List[str]
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "answer_value": value,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": [c for c in citations if c][:6],
            "confidence": 0.85,
            "flags": [],
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
