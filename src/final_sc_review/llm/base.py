"""Base LLM utilities for evidence retrieval pipeline.

Features:
- Response caching for efficiency
- Timeout handling for API reliability
- Better error handling with fallbacks
- Shared JSON extraction from LLM responses
"""

import hashlib
import json
import logging
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class JSONExtractionError(Exception):
    """Raised when JSON extraction from LLM response fails."""
    pass


def extract_json_from_response(
    response: str,
    default: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Extract JSON object from LLM response with robust parsing.

    Handles:
    - Markdown code blocks (```json ... ``` or ``` ... ```)
    - Raw JSON in response text
    - Malformed responses with fallback

    Args:
        response: Raw LLM response text
        default: Default value to return on error (None = raise)
        strict: If True, raise on any extraction error

    Returns:
        Extracted JSON as dictionary

    Raises:
        JSONExtractionError: If extraction fails and no default provided
    """
    response = response.strip()

    try:
        # Try markdown JSON blocks first
        if "```json" in response:
            try:
                start = response.index("```json") + 7
                end = response.index("```", start)
                response = response[start:end].strip()
            except ValueError as e:
                raise JSONExtractionError(
                    f"Malformed markdown JSON block: missing closing ```"
                ) from e

        elif "```" in response:
            try:
                start = response.index("```") + 3
                end = response.index("```", start)
                response = response[start:end].strip()
            except ValueError as e:
                raise JSONExtractionError(
                    f"Malformed markdown code block: missing closing ```"
                ) from e

        # Find JSON object boundaries
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            json_str = response[start:end]
        except ValueError as e:
            raise JSONExtractionError(
                f"No JSON object found in response (missing {{ or }})"
            ) from e

        # Parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise JSONExtractionError(
                f"Invalid JSON at position {e.pos}: {e.msg}"
            ) from e

    except JSONExtractionError:
        if strict:
            raise
        if default is not None:
            logger.warning(f"JSON extraction failed, using default")
            return default
        raise

    except Exception as e:
        if strict:
            raise JSONExtractionError(f"Unexpected error: {e}") from e
        if default is not None:
            logger.warning(f"JSON extraction failed unexpectedly: {e}")
            return default
        raise JSONExtractionError(f"Unexpected error: {e}") from e


class LLMTimeoutError(Exception):
    """Raised when LLM generation exceeds timeout."""
    pass


@contextmanager
def timeout_context(seconds: int, error_message: str = "Operation timed out"):
    """Context manager for timeout handling.

    Args:
        seconds: Timeout in seconds (0 = no timeout)
        error_message: Error message for timeout

    Note:
        Only works on Unix systems. On Windows, timeout is ignored.
    """
    if seconds <= 0:
        yield
        return

    def signal_handler(signum, frame):
        raise LLMTimeoutError(error_message)

    # Try to use SIGALRM (Unix only)
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    except (AttributeError, ValueError):
        # SIGALRM not available (Windows) - no timeout
        yield


class LLMBase:
    """Base class for LLM-based components with caching and bias controls."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        cache_dir: Optional[Path] = None,
        load_in_4bit: bool = False,
        temperature: float = 0.0,  # Deterministic by default
        max_tokens: int = 512,
        timeout_seconds: int = 60,  # Timeout for generation
    ):
        """Initialize LLM base.

        Args:
            model_name: HuggingFace model ID
            device: Device to run on (cuda/cpu)
            cache_dir: Directory for response caching
            load_in_4bit: Use 4-bit quantization
            temperature: Sampling temperature (0 = greedy)
            max_tokens: Max output tokens
            timeout_seconds: Timeout for generation (0 = no timeout)
        """
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        
        # Setup cache
        if cache_dir is None:
            cache_dir = Path("outputs/llm_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading LLM: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with optional quantization
        model_kwargs = {"trust_remote_code": True}
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not load_in_4bit:
            self.model = self.model.to(device)
            
        self.model.eval()
        
        logger.info(f"Model loaded on {device}")

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response using shared utility.

        This method is provided for backward compatibility.
        Prefer using extract_json_from_response() directly for new code.
        """
        return extract_json_from_response(response)
        
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        key_str = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load response from cache if exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, encoding='utf-8') as f:
                data = json.load(f)
                return data["response"]
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"response": response}, f)
    
    def generate(
        self,
        prompt: str,
        use_cache: bool = True,
        timeout: Optional[int] = None,
        fallback_on_error: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response from prompt with timeout and error handling.

        Args:
            prompt: Input prompt
            use_cache: Whether to use caching
            timeout: Override timeout (None = use default, 0 = no timeout)
            fallback_on_error: Return this string on error instead of raising
            **kwargs: Additional generation parameters

        Returns:
            Generated text

        Raises:
            LLMTimeoutError: If generation exceeds timeout (unless fallback provided)
            RuntimeError: If generation fails (unless fallback provided)
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(prompt, **kwargs)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Determine timeout
        effective_timeout = timeout if timeout is not None else self.timeout_seconds

        try:
            with timeout_context(effective_timeout, f"LLM generation timed out after {effective_timeout}s"):
                # Generate
                messages = [{"role": "user", "content": prompt}]

                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                ).to(self.device)

                # Extract temperature from kwargs to avoid conflict
                temp = kwargs.pop('temperature', self.temperature)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=temp if temp > 0 else None,
                        do_sample=temp > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )

                # Decode only the new tokens
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                # Cache response
                if use_cache:
                    self._save_to_cache(cache_key, response)

                return response

        except LLMTimeoutError as e:
            logger.warning(f"LLM timeout: {e}")
            if fallback_on_error is not None:
                return fallback_on_error
            raise

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            if fallback_on_error is not None:
                return fallback_on_error
            raise RuntimeError(f"LLM generation failed: {e}") from e
    
    def generate_multiple(
        self,
        prompt: str,
        n: int = 3,
        **kwargs
    ) -> List[str]:
        """Generate multiple responses for self-consistency check.
        
        Args:
            prompt: Input prompt
            n: Number of responses to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of n responses
        """
        responses = []
        for i in range(n):
            # Use temperature > 0 for diversity
            temp = kwargs.pop('temperature', 0.7)
            response = self.generate(
                prompt,
                use_cache=False,  # Don't cache diverse samples
                temperature=temp,
                **kwargs
            )
            responses.append(response)
        return responses
