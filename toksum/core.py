"""
Core functionality for token counting across different LLM providers.
"""

import re
from typing import Dict, List, Optional, Union

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from .exceptions import UnsupportedModelError, TokenizationError


# Model mappings for different providers
OPENAI_MODELS = {
    "gpt-4": "cl100k_base",
    "gpt-4-0314": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-32k-0314": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-vision-preview": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",  # NEW
    "gpt-4-turbo-2024-04-09": "cl100k_base",  # NEW
    "gpt-4o": "cl100k_base",  # NEW
    "gpt-4o-2024-05-13": "cl100k_base",  # NEW
    "gpt-4o-mini": "cl100k_base",  # NEW
    "gpt-4o-mini-2024-07-18": "cl100k_base",  # NEW
    "gpt-4o-2024-08-06": "cl100k_base",  # ADDED
    "gpt-4o-2024-11-20": "cl100k_base",  # ADDED
    "gpt-4-1106-vision-preview": "cl100k_base",  # ADDED
    "gpt-4-turbo-2024-04-09": "cl100k_base",  # ADDED
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    "gpt-3.5-turbo-instruct": "cl100k_base",  # ADDED
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
}

ANTHROPIC_MODELS = {
    "claude-3-opus-20240229": "claude-3",
    "claude-3-sonnet-20240229": "claude-3",
    "claude-3-haiku-20240307": "claude-3",
    "claude-3.5-sonnet-20240620": "claude-3.5",  # NEW
    "claude-3.5-sonnet-20241022": "claude-3.5",  # NEW
    "claude-3.5-haiku-20241022": "claude-3.5",  # NEW
    "claude-3-5-sonnet-20240620": "claude-3.5",  # NEW (alternative naming)
    "claude-3-opus": "claude-3",  # ADDED (short name)
    "claude-3-sonnet": "claude-3",  # ADDED (short name)
    "claude-3-haiku": "claude-3",  # ADDED (short name)
    "claude-2.1": "claude-2",
    "claude-2.0": "claude-2",
    "claude-instant-1.2": "claude-instant",
    "claude-instant-1.1": "claude-instant",
    "claude-instant-1.0": "claude-instant",
    "claude-instant": "claude-instant",  # ADDED (short name)
}

# Google Models (using approximation similar to Claude)
GOOGLE_MODELS = {
    "gemini-pro": "gemini",  # NEW
    "gemini-pro-vision": "gemini",  # NEW
    "gemini-1.5-pro": "gemini-1.5",  # NEW
    "gemini-1.5-flash": "gemini-1.5",  # NEW
    "gemini-1.5-pro-latest": "gemini-1.5",  # ADDED
    "gemini-1.5-flash-latest": "gemini-1.5",  # ADDED
    "gemini-1.0-pro": "gemini",  # ADDED
    "gemini-1.0-pro-vision": "gemini",  # ADDED
    "gemini-ultra": "gemini-ultra",  # ADDED
}

# Meta Models (using approximation)
META_MODELS = {
    "llama-2-7b": "llama-2",  # NEW
    "llama-2-13b": "llama-2",  # NEW
    "llama-2-70b": "llama-2",  # NEW
    "llama-3-8b": "llama-3",  # ADDED
    "llama-3-70b": "llama-3",  # ADDED
    "llama-3.1-8b": "llama-3.1",  # ADDED
    "llama-3.1-70b": "llama-3.1",  # ADDED
    "llama-3.1-405b": "llama-3.1",  # ADDED
    "llama-3.2-1b": "llama-3.2",  # ADDED
    "llama-3.2-3b": "llama-3.2",  # ADDED
}

# Mistral Models (using approximation)
MISTRAL_MODELS = {
    "mistral-7b": "mistral",  # NEW
    "mistral-8x7b": "mistral",  # NEW
    "mistral-large": "mistral-large",  # ADDED
    "mistral-medium": "mistral-medium",  # ADDED
    "mistral-small": "mistral-small",  # ADDED
    "mistral-tiny": "mistral-tiny",  # ADDED
    "mixtral-8x7b": "mixtral",  # ADDED
    "mixtral-8x22b": "mixtral",  # ADDED
}

# Cohere Models (using approximation)
COHERE_MODELS = {
    "command": "cohere",  # NEW
    "command-light": "cohere",  # ADDED
    "command-nightly": "cohere",  # ADDED
    "command-r": "cohere-r",  # ADDED
    "command-r-plus": "cohere-r",  # ADDED
}


class TokenCounter:
    """
    A token counter for various LLM models.
    
    Supports OpenAI GPT models and Anthropic Claude models.
    """
    
    def __init__(self, model: str):
        """
        Initialize the TokenCounter with a specific model.
        
        Args:
            model: The model name (e.g., 'gpt-4', 'claude-3-opus-20240229')
        
        Raises:
            UnsupportedModelError: If the model is not supported
            TokenizationError: If required dependencies are missing
        """
        self.model = model.lower()
        self.provider = self._detect_provider()
        self._setup_tokenizer()
    
    def _detect_provider(self) -> str:
        """Detect which provider the model belongs to."""
        if self.model in OPENAI_MODELS:
            return "openai"
        elif self.model in ANTHROPIC_MODELS:
            return "anthropic"
        elif self.model in GOOGLE_MODELS:
            return "google"
        elif self.model in META_MODELS:
            return "meta"
        elif self.model in MISTRAL_MODELS:
            return "mistral"
        elif self.model in COHERE_MODELS:
            return "cohere"
        else:
            supported = (list(OPENAI_MODELS.keys()) + list(ANTHROPIC_MODELS.keys()) + 
                        list(GOOGLE_MODELS.keys()) + list(META_MODELS.keys()) + 
                        list(MISTRAL_MODELS.keys()) + list(COHERE_MODELS.keys()))
            raise UnsupportedModelError(self.model, supported)
    
    def _setup_tokenizer(self) -> None:
        """Setup the appropriate tokenizer for the model."""
        if self.provider == "openai":
            if tiktoken is None:
                raise TokenizationError(
                    "tiktoken is required for OpenAI models. Install with: pip install tiktoken",
                    model=self.model
                )
            
            encoding_name = OPENAI_MODELS[self.model]
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                raise TokenizationError(f"Failed to load tokenizer: {str(e)}", model=self.model)
        
        else:
            # For all other providers (Anthropic, Google, Meta, Mistral, Cohere), 
            # we'll use approximation since they don't provide public tokenizers
            self.tokenizer = None
    
    def count(self, text: str) -> int:
        """
        Count tokens in the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens
            
        Raises:
            TokenizationError: If tokenization fails
        """
        if not isinstance(text, str):
            raise TokenizationError("Input must be a string", model=self.model)
        
        try:
            if self.provider == "openai":
                return len(self.tokenizer.encode(text))
            else:
                # Use approximation for all other providers
                return self._approximate_tokens(text)
        except Exception as e:
            raise TokenizationError(str(e), model=self.model, text_preview=text)
    
    def _approximate_tokens(self, text: str) -> int:
        """
        Approximate token count for non-OpenAI models.
        
        This uses a general approximation algorithm that works reasonably well
        for most LLMs, with slight adjustments based on the provider.
        """
        if not text:
            return 0
        
        # Basic character-based approximation
        char_count = len(text)
        
        # Adjust for whitespace (spaces and newlines are often separate tokens)
        whitespace_count = len(re.findall(r'\s+', text))
        
        # Adjust for punctuation (often separate tokens)
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        
        # Provider-specific adjustments
        if self.provider == "anthropic":
            # Anthropic's guidance: ~4 characters = 1 token
            base_tokens = char_count / 4
            adjustment = (whitespace_count + punctuation_count) * 0.3
        elif self.provider == "google":
            # Gemini models tend to have similar tokenization to GPT
            base_tokens = char_count / 3.8
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "meta":
            # LLaMA models have slightly different tokenization
            base_tokens = char_count / 3.5
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "mistral":
            # Mistral models similar to GPT
            base_tokens = char_count / 3.7
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "cohere":
            # Cohere models
            base_tokens = char_count / 4.2
            adjustment = (whitespace_count + punctuation_count) * 0.3
        else:
            # Default approximation
            base_tokens = char_count / 4
            adjustment = (whitespace_count + punctuation_count) * 0.3
        
        return max(1, int(base_tokens + adjustment))
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens for a list of messages (chat format).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Total token count including message formatting overhead
        """
        if not isinstance(messages, list):
            raise TokenizationError("Messages must be a list", model=self.model)
        
        total_tokens = 0
        
        for message in messages:
            if not isinstance(message, dict) or 'content' not in message:
                raise TokenizationError("Each message must be a dict with 'content' key", model=self.model)
            
            # Count content tokens
            content_tokens = self.count(message['content'])
            total_tokens += content_tokens
            
            # Add overhead for message formatting
            if self.provider == "openai":
                # OpenAI adds ~4 tokens per message for formatting
                total_tokens += 4
                if 'role' in message:
                    total_tokens += self.count(message['role'])
            elif self.provider == "anthropic":
                # Claude has different formatting overhead
                total_tokens += 3
        
        # Add final assistant message overhead for OpenAI
        if self.provider == "openai":
            total_tokens += 2
        
        return total_tokens


def count_tokens(text: str, model: str) -> int:
    """
    Convenience function to count tokens for a given text and model.
    
    Args:
        text: The text to count tokens for
        model: The model name
        
    Returns:
        The number of tokens
    """
    counter = TokenCounter(model)
    return counter.count(text)


def get_supported_models() -> Dict[str, List[str]]:
    """
    Get a dictionary of supported models by provider.
    
    Returns:
        Dictionary with provider names as keys and lists of model names as values
    """
    return {
        "openai": list(OPENAI_MODELS.keys()),
        "anthropic": list(ANTHROPIC_MODELS.keys()),
        "google": list(GOOGLE_MODELS.keys()),
        "meta": list(META_MODELS.keys()),
        "mistral": list(MISTRAL_MODELS.keys()),
        "cohere": list(COHERE_MODELS.keys()),
    }


def estimate_cost(token_count: int, model: str, input_tokens: bool = True) -> float:
    """
    Estimate the cost for a given number of tokens and model.
    
    Args:
        token_count: Number of tokens
        model: Model name
        input_tokens: Whether these are input tokens (True) or output tokens (False)
        
    Returns:
        Estimated cost in USD
        
    Note:
        Prices are approximate and may change. Always check current pricing.
    """
    # Approximate pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3.5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3.5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3.5-haiku-20241022": {"input": 0.001, "output": 0.005},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
    }
    
    model_lower = model.lower()
    if model_lower not in pricing:
        return 0.0
    
    price_per_1k = pricing[model_lower]["input" if input_tokens else "output"]
    return (token_count / 1000) * price_per_1k
