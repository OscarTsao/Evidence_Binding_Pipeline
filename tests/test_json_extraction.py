"""Tests for LLM JSON extraction utilities."""

from __future__ import annotations

import pytest

from final_sc_review.llm.base import (
    extract_json_from_response,
    JSONExtractionError,
)


class TestJSONExtraction:
    """Tests for extract_json_from_response function."""

    def test_simple_json(self):
        """Test extraction from simple JSON response."""
        response = '{"key": "value"}'
        result = extract_json_from_response(response)
        assert result == {"key": "value"}

    def test_json_with_markdown_json_block(self):
        """Test extraction from markdown ```json block."""
        response = '''Here is the result:
```json
{"has_evidence": true, "confidence": 0.9}
```
Done!'''
        result = extract_json_from_response(response)
        assert result == {"has_evidence": True, "confidence": 0.9}

    def test_json_with_generic_markdown_block(self):
        """Test extraction from generic markdown ``` block."""
        response = '''Result:
```
{"ranking": [1, 2, 3]}
```'''
        result = extract_json_from_response(response)
        assert result == {"ranking": [1, 2, 3]}

    def test_json_with_surrounding_text(self):
        """Test extraction with text before and after JSON."""
        response = '''Based on my analysis, here's the result:
{"has_suicidal_ideation": false, "severity": "none"}
This was determined using clinical guidelines.'''
        result = extract_json_from_response(response)
        assert result == {"has_suicidal_ideation": False, "severity": "none"}

    def test_nested_json(self):
        """Test extraction of nested JSON objects."""
        response = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = extract_json_from_response(response)
        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_missing_opening_brace(self):
        """Test error on missing opening brace."""
        response = '"key": "value"}'
        with pytest.raises(JSONExtractionError) as exc_info:
            extract_json_from_response(response)
        assert "No JSON object found" in str(exc_info.value)

    def test_missing_closing_brace(self):
        """Test error on missing closing brace."""
        response = '{"key": "value"'
        with pytest.raises(JSONExtractionError) as exc_info:
            extract_json_from_response(response)
        assert "No JSON object found" in str(exc_info.value)

    def test_invalid_json_syntax(self):
        """Test error on invalid JSON syntax."""
        response = '{"key": value}'  # Missing quotes around value
        with pytest.raises(JSONExtractionError) as exc_info:
            extract_json_from_response(response)
        assert "Invalid JSON" in str(exc_info.value)

    def test_malformed_markdown_block(self):
        """Test error on malformed markdown block."""
        response = '```json\n{"key": "value"}'  # Missing closing ```
        with pytest.raises(JSONExtractionError) as exc_info:
            extract_json_from_response(response)
        assert "Malformed markdown" in str(exc_info.value)

    def test_default_on_error(self):
        """Test default value returned on error."""
        response = 'no json here'
        default = {"fallback": True}
        result = extract_json_from_response(response, default=default)
        assert result == default

    def test_strict_mode_raises(self):
        """Test strict mode raises on any error."""
        response = 'no json here'
        with pytest.raises(JSONExtractionError):
            extract_json_from_response(response, strict=True)

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        response = '''

        {
            "key": "value"
        }

        '''
        result = extract_json_from_response(response)
        assert result == {"key": "value"}

    def test_unicode_json(self):
        """Test handling of unicode characters."""
        response = '{"text": "Hello, world!", "emoji": "test"}'
        result = extract_json_from_response(response)
        assert result == {"text": "Hello, world!", "emoji": "test"}

    def test_boolean_values(self):
        """Test handling of boolean values."""
        response = '{"true_val": true, "false_val": false, "null_val": null}'
        result = extract_json_from_response(response)
        assert result == {"true_val": True, "false_val": False, "null_val": None}
