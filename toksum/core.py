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
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
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
    "claude-2.1": "claude-2",
    "claude-2.0": "claude-2",
    "claude-instant-1.2": "claude-instant",
    "claude-instant-1.1": "claude-instant",
    "claude-instant-1.0": "claude-instant",
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
        else:
            supported = list(OPENAI_MODELS.keys()) + list(ANTHROPIC_MODELS.keys())
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
        
        elif self.provider == "anthropic":
            # For Anthropic models, we'll use a simple approximation
            # since they don't provide a public tokenizer
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
            elif self.provider == "anthropic":
                return self._approximate_claude_tokens(text)
        except Exception as e:
            raise TokenizationError(str(e), model=self.model, text_preview=text)
    
    def _approximate_claude_tokens(self, text: str) -> int:
        """
        Approximate token count for Claude models.
        
        This is based on Anthropic's guidance that ~4 characters = 1 token
        for English text, with adjustments for different text patterns.
        """
        if not text:
            return 0
        
        # Basic character-based approximation
        char_count = len(text)
        
        # Adjust for whitespace (spaces and newlines are often separate tokens)
        whitespace_count = len(re.findall(r'\s+', text))
        
        # Adjust for punctuation (often separate tokens)
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        
        # Rough approximation: 4 chars per token, but add extra for whitespace and punctuation
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
