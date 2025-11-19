"""
Example demonstrating the new exception handling features in toksum.
"""

from toksum import (
    TokenCounter, 
    count_tokens,
    ModelDeprecationError,
    RateLimitError,
    ConfigurationError,
    BatchProcessingError,
    UnsupportedModelError,
    TokenizationError
)
from typing import List, Dict, Any
import time


def demonstrate_model_deprecation():
    """Demonstrate model deprecation handling."""
    print("=== Model Deprecation Example ===")
    
    # Simulate checking for deprecated models
    deprecated_models = {
        "text-davinci-002": {
            "replacement": "gpt-3.5-turbo",
            "deprecation_date": "2023-01-04",
            "removal_date": "2024-01-04"
        }
    }
    
    model = "text-davinci-002"
    
    if model in deprecated_models:
        info = deprecated_models[model]
        try:
            # In strict mode, this would raise an exception
            raise ModelDeprecationError(
                model=model,
                replacement_model=info["replacement"],
                deprecation_date=info["deprecation_date"],
                removal_date=info["removal_date"]
            )
        except ModelDeprecationError as e:
            print(f"Warning: {e}")
            print(f"Replacement model: {e.replacement_model}")
            print(f"Deprecated on: {e.deprecation_date}")
            print(f"Will be removed on: {e.removal_date}")
    print()


def demonstrate_rate_limit_handling():
    """Demonstrate rate limit error handling."""
    print("=== Rate Limit Handling Example ===")
    
    try:
        # Simulate a rate limit scenario
        raise RateLimitError(
            message="Too many requests per minute",
            retry_after=60,
            provider="openai"
        )
    except RateLimitError as e:
        print(f"Rate limit hit: {e}")
        print(f"Provider: {e.provider}")
        print(f"Retry after: {e.retry_after} seconds")
        print("Implementing exponential backoff...")
    print()


def demonstrate_configuration_error():
    """Demonstrate configuration error handling."""
    print("=== Configuration Error Example ===")
    
    try:
        # Simulate a configuration issue
        raise ConfigurationError(
            message="Missing API key for OpenAI provider",
            config_key="OPENAI_API_KEY",
            suggested_fix="Set the OPENAI_API_KEY environment variable"
        )
    except ConfigurationError as e:
        print(f"Configuration issue: {e}")
        print(f"Config key: {e.config_key}")
        print(f"Suggested fix: {e.suggested_fix}")
    print()


def demonstrate_batch_processing():
    """Demonstrate batch processing with error handling."""
    print("=== Batch Processing Example ===")
    
    texts = [
        "Hello, world!",
        "",  # This will cause an error
        "This is a test.",
        None,  # This will also cause an error
        "Another test text."
    ]
    
    model = "gpt-4"
    results = []
    failed_indices = []
    
    for i, text in enumerate(texts):
        try:
            if text is None or text == "":
                raise TokenizationError("Invalid text input", model=model)
            
            tokens = count_tokens(text, model)
            results.append(tokens)
            print(f"Text {i+1}: {tokens} tokens")
            
        except (TokenizationError, UnsupportedModelError) as e:
            failed_indices.append(i)
            print(f"Text {i+1}: Failed - {e}")
    
    # If we had significant failures, we could raise a batch error
    if len(failed_indices) > len(texts) // 2:  # More than half failed
        try:
            raise BatchProcessingError(
                message="Too many items failed in batch",
                failed_items=failed_indices,
                total_items=len(texts)
            )
        except BatchProcessingError as e:
            print(f"\nBatch processing error: {e}")
            print(f"Failed items: {e.failed_items}")
            print(f"Total items: {e.total_items}")
    
    print()


def demonstrate_graceful_error_handling():
    """Demonstrate graceful error handling across the library."""
    print("=== Graceful Error Handling Example ===")
    
    test_cases = [
        ("gpt-4", "Hello, world!"),
        ("invalid-model", "Test text"),
        ("gpt-4", ""),
        ("claude-3-opus", None),
    ]
    
    for model, text in test_cases:
        try:
            if text is None:
                raise TokenizationError("Text cannot be None", model=model)
            
            tokens = count_tokens(text, model)
            print(f"✓ {model} with '{text}': {tokens} tokens")
            
        except UnsupportedModelError as e:
            print(f"✗ Unsupported model: {e.model}")
            
        except TokenizationError as e:
            print(f"✗ Tokenization failed: {e}")
            
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
    
    print()


def main():
    """Run all exception handling demonstrations."""
    print("toksum Exception Handling Examples\n")
    
    demonstrate_model_deprecation()
    demonstrate_rate_limit_handling()
    demonstrate_configuration_error()
    demonstrate_batch_processing()
    demonstrate_graceful_error_handling()
    
    print("=== Summary ===")
    print("New exception types added:")
    print("• ModelDeprecationError - Handle deprecated models gracefully")
    print("• RateLimitError - Manage API rate limits with retry information")
    print("• ConfigurationError - Clear configuration issue reporting")
    print("• BatchProcessingError - Handle batch operation failures")
    print("\nThese exceptions provide better error context and enable")
    print("more robust error handling in applications using toksum.")


if __name__ == "__main__":
    main()