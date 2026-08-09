"""Agent roster — returned by GET /agents.

Matches schema/agents.schema.json exactly.
"""
from __future__ import annotations

from typing import Any, Dict, List


# The roster must declare all roles that will appear in answer `agents` arrays.
# model field must be exactly "valura-fast" or "valura-deep" — gateway rejects others.
ROSTER: Dict[str, Any] = {
    "framework": "agno",
    "framework_version": "2.6.9",
    "agents": [
        {
            "role": "router",
            "name": "ValuraRouter",
            "model": "valura-fast",
            "tools": ["classify_question", "dispatch_specialist"],
        },
        {
            "role": "book_qa",
            "name": "ValuraBookQA",
            "model": "valura-fast",
            "tools": ["get_cash_balance", "get_transactions", "get_holdings", "get_positions"],
        },
        {
            "role": "kyc_profile",
            "name": "ValuraKYCProfile",
            "model": "valura-fast",
            "tools": ["get_kyc_record", "get_identity", "mask_sensitive"],
        },
        {
            "role": "notes_desk",
            "name": "ValuraNotes",
            "model": "valura-fast",
            "tools": ["get_notes", "get_transaction_memo"],
        },
        {
            "role": "market_desk",
            "name": "ValuraMarket",
            "model": "valura-fast",
            "tools": ["get_price", "get_sector", "get_news", "check_coverage"],
        },
        {
            "role": "compliance",
            "name": "ValuraCompliance",
            "model": "valura-fast",
            "tools": ["check_scope", "check_advice"],
        },
        {
            "role": "verifier",
            "name": "ValuraVerifier",
            "model": "valura-fast",
            "tools": ["verify_answer", "check_citations"],
        },
    ],
}


def get_roster() -> Dict[str, Any]:
    return ROSTER
