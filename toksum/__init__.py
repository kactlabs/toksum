"""
toksum - A Python library for counting tokens in text for major LLMs.

This library provides token counting functionality for:
- OpenAI GPT models (GPT-3.5, GPT-4, etc.)
- Anthropic Claude models

Usage:
    from toksum import count_tokens, TokenCounter
    
    # Quick token counting
    token_count = count_tokens("Hello, world!", model="gpt-4")
    
    # Using TokenCounter class
    counter = TokenCounter("gpt-4")
    token_count = counter.count("Hello, world!")
"""

from .core import TokenCounter, count_tokens, get_supported_models, estimate_cost
from .exceptions import UnsupportedModelError, TokenizationError

__version__ = "0.6.0"
__author__ = "Raja CSP Raman"
__email__ = "raja.csp@gmail.com"

__all__ = [
    "TokenCounter",
    "count_tokens",
    "get_supported_models",
    "estimate_cost",
    "UnsupportedModelError",
    "TokenizationError",
]
