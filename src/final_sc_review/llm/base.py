"""Base LLM utilities for evidence retrieval pipeline."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


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
    ):
        """Initialize LLM base.
        
        Args:
            model_name: HuggingFace model ID
            device: Device to run on (cuda/cpu)
            cache_dir: Directory for response caching
            load_in_4bit: Use 4-bit quantization
            temperature: Sampling temperature (0 = greedy)
            max_tokens: Max output tokens
        """
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        
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
            with open(cache_file) as f:
                data = json.load(f)
                return data["response"]
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({"response": response}, f)
    
    def generate(
        self,
        prompt: str,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """Generate response from prompt.
        
        Args:
            prompt: Input prompt
            use_cache: Whether to use caching
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(prompt, **kwargs)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
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
