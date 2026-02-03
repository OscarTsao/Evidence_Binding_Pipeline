"""Gemini API client for LLM integration experiments.

This module provides a clean interface to Gemini 1.5 Flash for:
- LLM reranking (post-P3)
- LLM verification (evidence correctness)

Uses the new google-genai package (google.generativeai is deprecated).
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper for Gemini API with retry logic, error handling, and automatic key rotation."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize Gemini client with support for multiple API keys.

        Args:
            model_name: Gemini model name (default: gemini-2.5-flash)
            api_key: Single API key (if None, reads from GEMINI_API_KEY env var)
            api_keys: List of API keys for rotation (overrides api_key if provided)
                     Can also read from GEMINI_API_KEYS env var (comma-separated)
            temperature: Sampling temperature (0.0 for deterministic)
            max_retries: Maximum retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configure API keys (support multiple for rotation)
        if api_keys:
            self.api_keys = api_keys
        elif os.getenv("GEMINI_API_KEYS"):
            # Support comma-separated list in env var
            self.api_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS").split(",")]
        elif api_key:
            self.api_keys = [api_key]
        elif os.getenv("GEMINI_API_KEY"):
            self.api_keys = [os.getenv("GEMINI_API_KEY")]
        else:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY or GEMINI_API_KEYS env var, "
                "or pass api_key/api_keys parameter."
            )

        # Track current key index and exhausted keys
        self.current_key_index = 0
        self.exhausted_keys = set()

        # Initialize client with first API key
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])

        logger.info(
            f"Initialized GeminiClient with model={model_name}, temp={temperature}, "
            f"{len(self.api_keys)} API key(s)"
        )

    def _rotate_api_key(self) -> bool:
        """Rotate to next available API key.

        Returns:
            True if successfully rotated to new key, False if all keys exhausted
        """
        # Mark current key as exhausted
        self.exhausted_keys.add(self.current_key_index)

        # Find next non-exhausted key
        for i in range(len(self.api_keys)):
            next_index = (self.current_key_index + 1 + i) % len(self.api_keys)
            if next_index not in self.exhausted_keys:
                self.current_key_index = next_index
                self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
                logger.info(
                    f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)} "
                    f"({len(self.exhausted_keys)} exhausted)"
                )
                return True

        # All keys exhausted
        logger.error(f"All {len(self.api_keys)} API keys exhausted")
        return False

    def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate JSON response from Gemini.

        Args:
            prompt: Input prompt
            schema: Optional JSON schema for validation

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If response is not valid JSON
            Exception: If API call fails after retries and all keys exhausted
        """
        for attempt in range(self.max_retries):
            try:
                # Generate response using new API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=8192,
                    ),
                )

                # Extract text
                text = response.text.strip()

                # Try to extract JSON from code blocks
                if "```json" in text:
                    start = text.find("```json") + 7
                    end = text.find("```", start)
                    text = text[start:end].strip()
                elif "```" in text:
                    start = text.find("```") + 3
                    end = text.find("```", start)
                    text = text[start:end].strip()

                # Parse JSON
                result = json.loads(text)

                # Validate schema if provided
                if schema:
                    self._validate_schema(result, schema)

                return result

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: JSON decode error: {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(f"Raw response: {text}")
                    raise ValueError(f"Failed to parse JSON after {self.max_retries} attempts")

            except Exception as e:
                error_str = str(e)

                # Check if this is a quota error (429 RESOURCE_EXHAUSTED)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    logger.warning(f"Quota exhausted for API key {self.current_key_index + 1}")

                    # Try to rotate to next key
                    if self._rotate_api_key():
                        logger.info("Retrying with new API key...")
                        continue  # Retry immediately with new key
                    else:
                        logger.error("All API keys exhausted")
                        raise Exception("All API keys have exhausted their quotas") from e

                # Other errors
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)

    def generate_text(self, prompt: str) -> str:
        """Generate text response from Gemini.

        Args:
            prompt: Input prompt

        Returns:
            Generated text

        Raises:
            Exception: If API call fails after retries and all keys exhausted
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=8192,
                    ),
                )
                return response.text.strip()

            except Exception as e:
                error_str = str(e)

                # Check if this is a quota error (429 RESOURCE_EXHAUSTED)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    logger.warning(f"Quota exhausted for API key {self.current_key_index + 1}")

                    # Try to rotate to next key
                    if self._rotate_api_key():
                        logger.info("Retrying with new API key...")
                        continue  # Retry immediately with new key
                    else:
                        logger.error("All API keys exhausted")
                        raise Exception("All API keys have exhausted their quotas") from e

                # Other errors
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)

    @staticmethod
    def _validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Basic schema validation (checks required keys and types)."""
        if "required" in schema:
            for key in schema["required"]:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")

        if "properties" in schema:
            for key, prop_schema in schema["properties"].items():
                if key in data:
                    expected_type = prop_schema.get("type")
                    actual_value = data[key]

                    # Type checking (basic)
                    type_map = {
                        "string": str,
                        "number": (int, float),
                        "integer": int,
                        "boolean": bool,
                        "array": list,
                        "object": dict,
                    }

                    if expected_type in type_map:
                        expected_python_type = type_map[expected_type]
                        if not isinstance(actual_value, expected_python_type):
                            raise ValueError(
                                f"Invalid type for {key}: expected {expected_type}, "
                                f"got {type(actual_value).__name__}"
                            )


def test_gemini_connection():
    """Test Gemini API connection."""
    try:
        client = GeminiClient()
        result = client.generate_json(
            prompt="""
            Return a JSON object with the following structure:
            {
                "status": "ok",
                "message": "Connection successful"
            }
            """,
            schema={
                "type": "object",
                "required": ["status", "message"],
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
        )
        logger.info(f"Connection test result: {result}")
        return True

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gemini_connection()
