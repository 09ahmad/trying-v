"""Tests for masking — the single shared mask_sensitive function.

Critical: every masked field must render as ****XXXX (4 stars + last 4 chars).
No bypass path exists. The scorer scans ALL text fields (answer, reason, citations)
for unmasked values.
"""
import pytest
from takehome_service.data import mask_sensitive, mask_in_text


class TestMaskSensitive:
    def test_bank_account_masked(self):
        acc = "99933311281536"
        result = mask_sensitive(acc)
        assert result == "****1536"
        assert "99933311281" not in result

    def test_pan_masked(self):
        pan = "QEFZP8716O"
        result = mask_sensitive(pan)
        assert result == "****716O"
        assert "QEFZP" not in result

    def test_short_value_padded(self):
        result = mask_sensitive("AB")
        assert result.startswith("****")
        assert len(result) == 8  # 4 stars + 4 chars

    def test_empty_string(self):
        result = mask_sensitive("")
        assert result == "****"

    def test_none_input(self):
        result = mask_sensitive(None)
        assert result == "****"

    def test_format_is_4stars_last4(self):
        """Every masked value must be exactly ****XXXX."""
        for value in ["1234567890", "ABCDE12345", "hello", "12"]:
            masked = mask_sensitive(value)
            assert masked.startswith("****"), f"Expected ****XXXX for {value!r}, got {masked!r}"
            assert len(masked) == 8, f"Expected length 8 for {value!r}, got {len(masked)}"

    def test_mask_in_text_replaces_all(self):
        text = "Account: 99933311281536, and also 99933311281536"
        result = mask_in_text(text, "99933311281536")
        assert "99933311281536" not in result
        assert "****1536" in result

    def test_mask_in_text_empty(self):
        assert mask_in_text("", "value") == ""
        assert mask_in_text("text", "") == "text"
