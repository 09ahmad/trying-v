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

from takehome_service.data import DataLoader, sanitize_text, format_citations

# Common English words that might be uppercase in prompts
_EXCLUDED_WORDS = {
    "WHAT", "WITH", "THEN", "FROM", "THAT", "THIS", "HAVE", "WHEN",
    "THEY", "YOUR", "ALSO", "SOME", "HERE", "WERE", "WILL", "ONLY",
    "READ", "DATE", "PULL", "LOOK", "WORK", "GIVE", "TELL", "FIND",
    "CHECK", "SHOW", "TAKE", "MAKE", "LIST", "VIEW", "SAME", "INFO",
}


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
                max_retries=3,
            ),
            name="ValuraMarket",
            description=self.SYSTEM,
            markdown=False,
        )

    def answer(self, payload: Dict[str, Any], use_deep: bool = False) -> Dict[str, Any]:
        prompt = payload.get("prompt", "")
        client_id = payload.get("client_id", "")
        prompt_lower = prompt.lower()

        # 1. Detect any potential symbol in prompt
        symbol = self._loader.find_symbol_in_text(prompt)

        # If no covered symbol found, scan for any capital ticker candidate e.g. WMT, PFE
        if not symbol:
            words = re.findall(r"\b[A-Z]{2,5}\b", prompt)
            for w in words:
                if w not in _EXCLUDED_WORDS:
                    # Potential uncovered symbol mentioned!
                    if not self._loader.is_covered(w):
                        return self._abstain(
                            f"{w} is not in the covered symbols dataset. "
                            f"No market data, price, sector, or news is available."
                        )

        # 2. Mandatory coverage check for detected symbol
        if symbol and not self._loader.is_covered(symbol):
            return self._abstain(
                f"{symbol} is not in the covered symbols dataset. "
                f"No market data, price, sector, or news is available."
            )

        # 3. Coverage query — only for IS-X-COVERED status questions.
        # If the prompt asks for coverage/news *content* with a date cutoff
        # (e.g. "coverage do we hold dated on or before …", "coverage predat…",
        #  "coverage up to …"), skip this branch and fall through to the news
        # handler which already respects date cutoffs via _extract_date.
        _is_date_cutoff_query = bool(re.search(
            r"\b(dated|on\s+or\s+before|up\s+to|predat\w*|before\s+\d|as\s+of)\b",
            prompt_lower,
        ))
        if re.search(r"\b(covered|coverage|data\s+available|in\s+your\s+coverage)\b", prompt_lower) \
                and not _is_date_cutoff_query:
            if symbol:
                if self._loader.is_covered(symbol):
                    return self._build(
                        f"{symbol} is covered in the market dataset.",
                        "covered",
                        [symbol],
                    )
                return self._abstain(f"{symbol} is not in the covered symbols list.")
            return self._abstain("No symbol was identified in the coverage question.")

        # 4. Sector / industry
        if re.search(r"\b(sector|industry|asset\s+class)\b", prompt_lower) and symbol:
            return self._sector_answer(symbol, prompt)

        # 5. Price / close
        if re.search(r"\b(close|closing|price|worth|value)\b", prompt_lower) and symbol:
            return self._price_answer(symbol, prompt, prompt_lower)

        # 6. Return / performance
        if re.search(r"\b(return|performance|gain|loss|percentage\s+change|grew|fell|rose)\b", prompt_lower) and symbol:
            return self._return_answer(symbol, prompt, prompt_lower)

        # 7. News — also handles "coverage dated on or before X" queries (stepped
        # over the coverage-status branch above when _is_date_cutoff_query is True).
        if re.search(
            r"\b(news|headline|announcement|articles?|reports?|published|published\s+by|items?|coverage)\b",
            prompt_lower,
        ):
            if symbol:
                return self._news_answer(symbol, prompt)
            covered = self._loader.get_client_covered_symbols(client_id)
            if covered:
                return self._multi_news_answer(covered, prompt)
            return self._abstain("No covered symbols found for this client to retrieve news for.")

        # 8. Generic market question with symbol
        if symbol:
            return self._generic_market_answer(symbol, prompt)

        return self._abstain("No covered instrument symbol was identified in the market question.")

    # -----------------------------------------------------------------------
    # Specific answer methods
    # -----------------------------------------------------------------------

    def _sector_answer(self, symbol: str, prompt: str) -> Dict[str, Any]:
        if not self._loader.is_covered(symbol):
            return self._abstain(f"{symbol} is not in the covered dataset.")
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
        if not self._loader.is_covered(symbol):
            return self._abstain(f"{symbol} is not in the covered dataset.")

        target_date = self._extract_date(prompt)
        history = self._loader.get_price_history(symbol)
        if not history:
            return self._abstain(f"No price history available for {symbol}.")

        latest_date_str = max(p.get("date", "") for p in history)
        latest_date = self._loader.parse_date(latest_date_str)

        if target_date is not None:
            if latest_date and target_date > latest_date:
                return self._abstain(
                    f"Requested price date {target_date.date()} is beyond the dataset coverage period."
                )

            price_rec = self._loader.get_price_on_or_before(symbol, target_date)
            if not price_rec:
                return self._abstain(
                    f"No price data available for {symbol} on or before {target_date.date()}."
                )
            close = float(price_rec.get("close", 0))
            date_used = price_rec.get("date", "")
            precomputed = f"{symbol} closed at {close:.2f} USD on {date_used}."
            answer_text = self._llm_format(precomputed, prompt)
            return self._build(answer_text, f"{close:.2f}", [symbol])

        latest = max(history, key=lambda p: p.get("date", ""))
        close = float(latest.get("close", 0))
        date_used = latest.get("date", "")
        precomputed = f"{symbol} closed at {close:.2f} USD on {date_used}."
        answer_text = self._llm_format(precomputed, prompt)
        return self._build(answer_text, f"{close:.2f}", [symbol])

    def _return_answer(self, symbol: str, prompt: str, prompt_lower: str) -> Dict[str, Any]:
        if not self._loader.is_covered(symbol):
            return self._abstain(f"{symbol} is not in the covered dataset.")

        dates = self._extract_two_dates(prompt)
        if not dates:
            return self._abstain("Could not parse the date range for the return calculation.")

        start_date, end_date = dates
        start_price_rec = self._loader.get_price_on_or_before(symbol, start_date)
        end_price_rec = self._loader.get_price_on_or_before(symbol, end_date)
        if not start_price_rec or not end_price_rec:
            return self._abstain(f"Insufficient price data for {symbol} in the requested range.")

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
        if not self._loader.is_covered(symbol):
            return self._abstain(
                f"{symbol} is not in the covered dataset. No news is available."
            )

        cutoff = self._extract_date(prompt)
        news_items = self._loader.get_news(symbol, on_or_before=cutoff)
        if not news_items:
            return self._abstain(f"No news items found for {symbol} in the requested timeframe.")

        news_items = sorted(news_items, key=lambda n: n.get("date", ""), reverse=True)
        count = len(news_items)
        news_ids = [n.get("id", "") for n in news_items if n.get("id")]
        news_context = "\n".join(
            f"- [{n.get('date')}] {n.get('headline', '')}: {n.get('body', '')}"
            for n in news_items[:5]
        )
        precomputed = f"{symbol} has {count} news items. Recent items:\n{news_context}"
        answer_text = self._llm_format(precomputed, prompt)
        return self._build(answer_text, str(count), news_ids)

    def _multi_news_answer(self, symbols: List[str], prompt: str) -> Dict[str, Any]:
        cutoff = self._extract_date(prompt)
        all_news = []
        news_ids = []
        for sym in symbols:
            items = self._loader.get_news(sym, on_or_before=cutoff)
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
        return self._build(answer_text, str(len(all_news)), news_ids)

    def _generic_market_answer(self, symbol: str, prompt: str) -> Dict[str, Any]:
        inst = self._loader.get_instrument(symbol)
        history = self._loader.get_price_history(symbol)
        news = self._loader.get_news(symbol)
        data_ctx = (
            f"Symbol: {symbol}\n"
            f"Instrument: {inst}\n"
            f"Price history ({len(history)} records)\n"
            f"News: {len(news)} items"
        )
        try:
            run_output = self._agent.run(
                f"Market data:\n{data_ctx}\n\nQuestion: {prompt}\n\n"
                f"Answer using only the data above."
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
        # "between X and Y"
        m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\s*\.|\s*$)", text, re.I)
        if m:
            d1 = self._extract_date(m.group(1))
            d2 = self._extract_date(m.group(2))
            if d1 and d2:
                return d1, d2
        # "over X to Y" or "from X to Y"
        m2 = re.search(r"(?:over|from)\s+(.+?)\s+to\s+(.+?)(?:\s*\.|\s*$)", text, re.I)
        if m2:
            d1 = self._extract_date(m2.group(1))
            d2 = self._extract_date(m2.group(2))
            if d1 and d2:
                return d1, d2
        # fallback: two bare ISO dates in order
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
                f"Rephrase clearly. Do not change values or add your own knowledge."
            )
            content = run_output.get_content_as_string() if run_output else ""
            if not content or "STUB-GATEWAY" in content or "insufficient_quota" in content or "exceeded your current quota" in content or "rate limit" in content.lower():
                return sanitize_text(precomputed)
            numbers = re.findall(r"-?\d+(?:\.\d+)?", precomputed)
            if numbers:
                for num in numbers:
                    clean_num = num.lstrip("-")
                    if clean_num not in content and clean_num.replace(".", "") not in content.replace(",", "").replace(".", ""):
                        return sanitize_text(precomputed)
            return sanitize_text(content.strip())
        except Exception:
            return sanitize_text(precomputed)

    def _build(
        self, answer: str, value: Optional[str], citations: List[str], client_id: str = ""
    ) -> Dict[str, Any]:
        return {
            "answer": sanitize_text(answer),
            "answer_value": value,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": format_citations(client_id, citations),
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
