"""Service configuration — reads five required environment variables."""
from __future__ import annotations

import os
from typing import Optional


class Settings:
    def __init__(self) -> None:
        self.book_path: str = os.environ.get("BOOK_PATH", "data/client_book.json")
        self.market_path: str = os.environ.get("MARKET_PATH", "data/market_data.json")
        self.llm_base_url: str = os.environ.get("LLM_BASE_URL", "http://localhost:8600/v1")
        self.llm_api_key: str = os.environ.get("LLM_API_KEY", "test")
        self.port: int = int(os.environ.get("PORT", "8080"))
        self.loaded: bool = False
        self.data_loader: Optional[object] = None
        self.answer_service: Optional[object] = None

    def initialize(self) -> None:
        """Load data once at startup. Never call per-request."""
        from takehome_service.data import DataLoader
        from takehome_service.service import AnswerService

        self.data_loader = DataLoader(self.book_path, self.market_path)
        self.answer_service = AnswerService(
            data_loader=self.data_loader,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
        )
        self.loaded = True


settings = Settings()
