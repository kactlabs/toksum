"""
toksum - A Python library for counting tokens in text for major LLMs.

This library provides comprehensive token counting functionality for a wide range of 
Large Language Models (LLMs) from various providers including OpenAI, Anthropic, 
Google, Meta, Mistral, and many others.

Features:
    - Precise token counting for OpenAI models using tiktoken
    - Intelligent approximations for 100+ other LLM models
    - Support for chat message format token counting
    - Cost estimation for supported models
    - Case-insensitive model name matching
    - Comprehensive error handling

Supported Providers:
    - **OpenAI**: GPT-4, GPT-3.5, GPT-4o, O1 models, and more
    - **Anthropic**: Claude 3/3.5 (Opus, Sonnet, Haiku), Claude 2, Claude Instant
    - **Google**: Gemini Pro, Gemini 1.5, PaLM models
    - **Meta**: LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.2, LLaMA 3.3
    - **Mistral**: Mistral 7B, Mixtral, Mistral Large
    - **Cohere**: Command, Command-R, Command-R+
    - **xAI**: Grok models
    - **Alibaba**: Qwen models
    - **Baidu**: ERNIE models
    - **And 20+ other providers with 200+ models**

Basic Usage:
    .. code-block:: python

        from toksum import count_tokens, TokenCounter
        
        # Quick token counting
        token_count = count_tokens("Hello, world!", model="gpt-4")
        print(f"Token count: {token_count}")
        
        # Using TokenCounter class for multiple operations
        counter = TokenCounter("gpt-4")
        token_count = counter.count("Hello, world!")
        
        # Count tokens in chat messages
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        message_tokens = counter.count_messages(messages)

Advanced Usage:
    .. code-block:: python

        from toksum import get_supported_models, estimate_cost
        
        # Get all supported models
        models = get_supported_models()
        print(f"OpenAI models: {models['openai']}")
        
        # Estimate costs
        tokens = count_tokens("Your text here", "gpt-4")
        cost = estimate_cost(tokens, "gpt-4", input_tokens=True)
        print(f"Estimated cost: ${cost:.4f}")

Error Handling:
    The library provides specific exceptions for different error conditions:
    
    - :exc:`UnsupportedModelError`: When an unsupported model is specified
    - :exc:`TokenizationError`: When tokenization fails
    - :exc:`EmptyTextError`: When attempting to tokenize empty text
    - :exc:`InvalidTokenError`: When invalid tokens are encountered

Installation:
    .. code-block:: bash

        pip install toksum

    For OpenAI model support (recommended):
    
    .. code-block:: bash

        pip install toksum[openai]

Note:
    For precise token counting with OpenAI models, the ``tiktoken`` library is required.
    For other providers, the library uses intelligent approximation algorithms that
    provide reasonable accuracy for cost estimation and planning purposes.

Version: 0.6.0
Author: Raja CSP Raman
Email: raja.csp@gmail.com
"""

from .core import TokenCounter, count_tokens, get_supported_models, estimate_cost
from .exceptions import UnsupportedModelError, TokenizationError

__version__ = "1.1.0"
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
