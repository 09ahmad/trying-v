"""Tests for JSON Schema contract validation.

Validates responses against the real schema files (schema/answer.schema.json
and schema/agents.schema.json), not hand-transcribed guesses.
"""
import pytest
from takehome_service.validation import SchemaValidator, SchemaValidationError


@pytest.fixture(scope="module")
def answer_validator():
    return SchemaValidator("schema/answer.schema.json")


@pytest.fixture(scope="module")
def agents_validator():
    return SchemaValidator("schema/agents.schema.json")


class TestAnswerSchema:
    def _valid_answer(self, **overrides):
        base = {
            "question_id": "q_001",
            "answer": "The balance is 100 USD.",
            "answer_value": "100.00",
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": ["txn_1"],
            "confidence": 0.85,
            "flags": [],
            "agents": ["router", "book_qa"],
        }
        base.update(overrides)
        return base

    def test_valid_answer_passes(self, answer_validator):
        answer_validator.validate(self._valid_answer())

    def test_abstained_null_value(self, answer_validator):
        """When abstained, answer_value must be null."""
        answer = self._valid_answer(
            abstained=True,
            answer_value=None,
            answer="",
            reason="Cannot determine this.",
        )
        answer_validator.validate(answer)

    def test_refused_with_reason(self, answer_validator):
        """When refused, reason can be non-null."""
        answer = self._valid_answer(
            refused=True,
            answer_value=None,
            answer="Cannot provide advice.",
            reason="Policy prohibits personalized advice.",
        )
        answer_validator.validate(answer)

    def test_agents_must_include_router(self, answer_validator):
        """agents array must contain 'router' (contains constraint)."""
        answer = self._valid_answer(agents=["book_qa"])
        with pytest.raises(SchemaValidationError):
            answer_validator.validate(answer)

    def test_confidence_above_1_invalid(self, answer_validator):
        """confidence must be between 0 and 1."""
        answer = self._valid_answer(confidence=1.5)
        with pytest.raises(SchemaValidationError):
            answer_validator.validate(answer)

    def test_confidence_zero_valid(self, answer_validator):
        answer_validator.validate(self._valid_answer(confidence=0.0))

    def test_confidence_one_valid(self, answer_validator):
        answer_validator.validate(self._valid_answer(confidence=1.0))

    def test_flags_allowed_values(self, answer_validator):
        """Only allowed flag values."""
        answer = self._valid_answer(flags=["conflict", "upstream_issue"])
        answer_validator.validate(answer)

    def test_flags_invalid_value_rejected(self, answer_validator):
        answer_bad = self._valid_answer(flags=["unknown_flag"])
        with pytest.raises(SchemaValidationError):
            answer_validator.validate(answer_bad)

    def test_citations_list_of_strings(self, answer_validator):
        answer = self._valid_answer(citations=["txn_1", "txn_2"])
        answer_validator.validate(answer)

    def test_missing_required_fields(self, answer_validator):
        required = ("question_id", "answer", "abstained", "refused", "confidence", "citations")
        for field in required:
            base = self._valid_answer()
            del base[field]
            with pytest.raises(SchemaValidationError):
                answer_validator.validate(base)

    def test_answer_value_can_be_string_or_null(self, answer_validator):
        answer_validator.validate(self._valid_answer(answer_value="100.00"))
        answer_validator.validate(self._valid_answer(answer_value=None))


class TestAgentsSchema:
    def _valid_roster(self):
        return {
            "framework": "agno",
            "framework_version": "2.6.9",
            "agents": [
                {"role": "router", "name": "R", "model": "valura-fast", "tools": []},
                {"role": "book_qa", "name": "B", "model": "valura-fast", "tools": []},
                {"role": "kyc_profile", "name": "K", "model": "valura-fast", "tools": []},
                {"role": "notes_desk", "name": "N", "model": "valura-fast", "tools": []},
                {"role": "market_desk", "name": "M", "model": "valura-fast", "tools": []},
                {"role": "compliance", "name": "C", "model": "valura-fast", "tools": []},
                {"role": "verifier", "name": "V", "model": "valura-fast", "tools": []},
            ],
        }

    def test_valid_roster(self, agents_validator):
        agents_validator.validate(self._valid_roster())

    def test_framework_must_be_agno(self, agents_validator):
        roster = self._valid_roster()
        roster["framework"] = "not-agno"
        with pytest.raises(SchemaValidationError):
            agents_validator.validate(roster)

    def test_min_six_agents(self, agents_validator):
        roster = self._valid_roster()
        roster["agents"] = roster["agents"][:5]  # Only 5 agents
        with pytest.raises(SchemaValidationError):
            agents_validator.validate(roster)

    def test_model_must_be_valura_fast_or_deep(self, agents_validator):
        roster = self._valid_roster()
        roster["agents"][0]["model"] = "gpt-4o"  # Invalid
        with pytest.raises(SchemaValidationError):
            agents_validator.validate(roster)

    def test_role_must_be_known(self, agents_validator):
        roster = self._valid_roster()
        roster["agents"][0]["role"] = "unknown_role"
        with pytest.raises(SchemaValidationError):
            agents_validator.validate(roster)

    def test_our_actual_roster_is_valid(self, agents_validator):
        """Our actual production roster must pass schema validation."""
        from takehome_service.roster import get_roster
        agents_validator.validate(get_roster())
