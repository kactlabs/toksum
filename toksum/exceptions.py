"""
Custom exceptions for the toksum library.

This module defines a hierarchy of custom exceptions that provide detailed error
information for various failure modes in token counting operations. The exceptions
are designed to give users clear, actionable error messages with relevant context.

Exception Hierarchy:
    .. code-block:: text
    
        ToksumError (base)
        ├── UnsupportedModelError
        ├── ModelNotFoundError  
        ├── TokenizationError
        │   ├── InvalidTokenError
        │   └── EmptyTextError

The exception hierarchy allows for both specific error handling and general
error catching at different levels of granularity.

Usage Examples:
    Specific exception handling:

    .. code-block:: python

        from toksum import TokenCounter
        from toksum.exceptions import UnsupportedModelError, TokenizationError
        
        try:
            counter = TokenCounter("unknown-model")
            tokens = counter.count("Hello, world!")
        except UnsupportedModelError as e:
            print(f"Model not supported: {e.model}")
            print(f"Available models: {e.supported_models}")
        except TokenizationError as e:
            print(f"Tokenization failed: {e}")
            if e.model:
                print(f"Model: {e.model}")
            if e.text_preview:
                print(f"Text preview: {e.text_preview}")

    General error handling:

    .. code-block:: python

        from toksum import count_tokens
        from toksum.exceptions import ToksumError
        
        try:
            tokens = count_tokens("Hello!", "gpt-4")
        except ToksumError as e:
            print(f"Toksum error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

Error Context:
    All exceptions include relevant context information:
    
    - **Model information**: Which model caused the error
    - **Text previews**: Sample of problematic text (truncated for privacy)
    - **Supported alternatives**: List of valid options when applicable
    - **Detailed messages**: Clear descriptions of what went wrong

The exceptions are designed to be both human-readable and machine-parseable,
making them suitable for both debugging and automated error handling.
"""

from typing import List, Optional


class ToksumError(Exception):
    """
    Base exception class for all toksum library errors.
    
    This is the root exception class that all other toksum exceptions inherit from.
    It allows for catching any toksum-related error with a single exception type
    while still providing the ability to catch specific error types when needed.

    This exception should generally not be raised directly. Instead, use one of
    the more specific exception subclasses that provide additional context and
    error-specific information.

    Examples:
        Catching all toksum errors:

        .. code-block:: python

            from toksum import count_tokens
            from toksum.exceptions import ToksumError
            
            try:
                tokens = count_tokens("Hello!", "some-model")
            except ToksumError as e:
                print(f"Toksum error occurred: {e}")
                # Handle any toksum-related error
            except Exception as e:
                print(f"Unexpected error: {e}")
                # Handle non-toksum errors

        Catching specific errors with fallback:

        .. code-block:: python

            from toksum import TokenCounter
            from toksum.exceptions import UnsupportedModelError, TokenizationError, ToksumError
            
            try:
                counter = TokenCounter("gpt-4")
                tokens = counter.count("Hello!")
            except UnsupportedModelError:
                print("Model not supported")
            except TokenizationError:
                print("Tokenization failed")
            except ToksumError:
                print("Other toksum error")
            except Exception:
                print("Non-toksum error")

    Note:
        This base class provides a common interface for all toksum exceptions
        but does not add any additional functionality beyond the standard
        Python Exception class.
    """


class UnsupportedModelError(ToksumError):
    """
    Raised when an unsupported model is specified.
    
    This exception is raised when attempting to create a TokenCounter or use
    count_tokens() with a model name that is not recognized by the library.
    The exception includes the invalid model name and optionally a list of
    supported models for reference.

    Attributes:
        model (str): The unsupported model name that was specified
        supported_models (List[str]): List of supported model names (optional)

    The exception message automatically includes the unsupported model name
    and, if provided, a formatted list of supported alternatives.

    Examples:
        Basic usage with model validation:

        .. code-block:: python

            from toksum import TokenCounter
            from toksum.exceptions import UnsupportedModelError
            
            try:
                counter = TokenCounter("invalid-model-name")
            except UnsupportedModelError as e:
                print(f"Model '{e.model}' is not supported")
                if e.supported_models:
                    print("Supported models include:")
                    for model in e.supported_models[:5]:  # Show first 5
                        print(f"  - {model}")

        Handling with model suggestions:

        .. code-block:: python

            from toksum import get_supported_models
            from toksum.exceptions import UnsupportedModelError
            
            def safe_create_counter(model_name):
                try:
                    return TokenCounter(model_name)
                except UnsupportedModelError as e:
                    print(f"Error: {e}")
                    
                    # Suggest similar models
                    all_models = get_supported_models()
                    suggestions = []
                    for provider_models in all_models.values():
                        for model in provider_models:
                            if model_name.lower() in model.lower():
                                suggestions.append(model)
                    
                    if suggestions:
                        print("Did you mean one of these?")
                        for suggestion in suggestions[:3]:
                            print(f"  - {suggestion}")
                    
                    return None

        Programmatic error handling:

        .. code-block:: python

            def validate_model(model_name):
                try:
                    TokenCounter(model_name)
                    return True
                except UnsupportedModelError:
                    return False
            
            # Test multiple models
            test_models = ["gpt-4", "invalid-model", "claude-3-opus"]
            for model in test_models:
                if validate_model(model):
                    print(f"✓ {model} is supported")
                else:
                    print(f"✗ {model} is not supported")

    Common Causes:
        - **Typos in model names**: "gpt4" instead of "gpt-4"
        - **Incorrect casing**: "GPT-4" vs "gpt-4" (though toksum is case-insensitive)
        - **Outdated model names**: Using deprecated or renamed models
        - **Provider confusion**: Using model names from unsupported providers
        - **Version specificity**: Using overly specific version numbers

    Resolution:
        1. **Check spelling**: Verify the model name is spelled correctly
        2. **Use get_supported_models()**: Get the complete list of supported models
        3. **Check provider**: Ensure the model provider is supported
        4. **Update library**: Newer models may require a library update

    See Also:
        - :func:`get_supported_models`: Get all supported models by provider
        - :class:`TokenCounter`: Main class that raises this exception
        - :func:`count_tokens`: Convenience function that may raise this exception
    """

    def __init__(self, model: str, supported_models: Optional[List[str]] = None):
        """
        Initialize UnsupportedModelError with model name and optional supported models list.

        Args:
            model (str): The unsupported model name that was specified
            supported_models (Optional[List[str]]): List of supported model names
                                                   for user reference. If provided,
                                                   will be included in error message.
        """
        self.model = model
        self.supported_models = supported_models or []

        if self.supported_models:
            message = f"Model '{model}' is not supported. Supported models: {', '.join(self.supported_models)}"
        else:
            message = f"Model '{model}' is not supported."

        super().__init__(message)


class ModelNotFoundError(ToksumError):
    """
    Raised when a specified model is not found.
    
    This exception is raised in scenarios where a model name is recognized
    as potentially valid but cannot be located or accessed. This is distinct
    from UnsupportedModelError, which is raised for completely unrecognized
    model names.

    This exception is typically used for cases where:
    - A model exists but is temporarily unavailable
    - A model requires special access or authentication
    - A model has been deprecated or removed
    - There are network or service issues preventing model access

    Attributes:
        model (str): The model name that could not be found

    Examples:
        Basic error handling:

        .. code-block:: python

            from toksum.exceptions import ModelNotFoundError
            
            try:
                # Some operation that might fail to find a model
                result = some_model_operation("gpt-4")
            except ModelNotFoundError as e:
                print(f"Model not found: {e}")
                # Maybe try a fallback model or retry later

        Distinguishing from UnsupportedModelError:

        .. code-block:: python

            from toksum import TokenCounter
            from toksum.exceptions import UnsupportedModelError, ModelNotFoundError
            
            try:
                counter = TokenCounter("some-model")
            except UnsupportedModelError:
                print("Model is not supported by toksum")
            except ModelNotFoundError:
                print("Model is supported but cannot be found/accessed")

    Note:
        This exception is less commonly used in the current toksum implementation
        but is available for future extensions and specific use cases where
        model availability needs to be distinguished from model support.

    See Also:
        - :exc:`UnsupportedModelError`: For completely unsupported models
        - :class:`TokenCounter`: May raise this exception in certain scenarios
    """
    def __init__(self, model: str):
        """
        Initialize ModelNotFoundError with the model name.

        Args:
            model (str): The model name that could not be found
        """
        super().__init__(f"Model '{model}' not found.")


class TokenizationError(ToksumError):
    """
    Raised when tokenization fails for any reason.
    
    This is the primary exception for tokenization-related failures. It provides
    detailed context about what went wrong, including the model being used and
    a preview of the problematic text (truncated for privacy and readability).

    This exception covers a wide range of tokenization failures:
    - Invalid input types (non-string input)
    - Missing dependencies (e.g., tiktoken for OpenAI models)
    - Tokenizer initialization failures
    - Text processing errors
    - Encoding/decoding issues

    Attributes:
        model (Optional[str]): The model name being used when the error occurred
        text_preview (Optional[str]): A preview of the problematic text (truncated)

    The exception message includes the base error description and automatically
    appends model and text preview information when available.

    Examples:
        Basic error handling:

        .. code-block:: python

            from toksum import TokenCounter
            from toksum.exceptions import TokenizationError
            
            try:
                counter = TokenCounter("gpt-4")
                tokens = counter.count(123)  # Invalid input type
            except TokenizationError as e:
                print(f"Tokenization failed: {e}")
                if e.model:
                    print(f"Model: {e.model}")
                if e.text_preview:
                    print(f"Text preview: {e.text_preview}")

        Handling missing dependencies:

        .. code-block:: python

            try:
                counter = TokenCounter("gpt-4")  # Requires tiktoken
                tokens = counter.count("Hello!")
            except TokenizationError as e:
                if "tiktoken" in str(e):
                    print("Please install tiktoken: pip install tiktoken")
                else:
                    print(f"Tokenization error: {e}")

        Robust error handling with fallbacks:

        .. code-block:: python

            def safe_count_tokens(text, model, fallback_model=None):
                try:
                    counter = TokenCounter(model)
                    return counter.count(text)
                except TokenizationError as e:
                    print(f"Primary model failed: {e}")
                    
                    if fallback_model:
                        try:
                            print(f"Trying fallback model: {fallback_model}")
                            counter = TokenCounter(fallback_model)
                            return counter.count(text)
                        except TokenizationError:
                            print("Fallback model also failed")
                    
                    return None

        Input validation with detailed errors:

        .. code-block:: python

            def validate_and_count(text, model):
                try:
                    if not isinstance(text, str):
                        raise TokenizationError(
                            f"Input must be string, got {type(text).__name__}",
                            model=model
                        )
                    
                    counter = TokenCounter(model)
                    return counter.count(text)
                    
                except TokenizationError as e:
                    print(f"Validation failed: {e}")
                    return None

    Text Preview Handling:
        The text_preview attribute contains a truncated version of the problematic
        text to help with debugging while protecting privacy:
        
        - **Truncation**: Long text is truncated to ~50 characters + "..."
        - **Privacy**: Sensitive information is not fully exposed in error messages
        - **Debugging**: Provides enough context to identify the problematic content

    Common Causes:
        - **Wrong input type**: Passing int, list, dict instead of string
        - **None input**: Passing None instead of a string
        - **Missing dependencies**: tiktoken not installed for OpenAI models
        - **Encoding issues**: Text with problematic character encodings
        - **Memory issues**: Extremely large text causing processing failures

    Resolution:
        1. **Check input type**: Ensure text is a string
        2. **Install dependencies**: Install tiktoken for OpenAI models
        3. **Validate text**: Check for encoding issues or special characters
        4. **Check text size**: Very large texts may cause issues
        5. **Try different model**: Some models may handle edge cases better

    See Also:
        - :exc:`InvalidTokenError`: For specific token-related errors
        - :exc:`EmptyTextError`: For empty text specific errors
        - :class:`TokenCounter`: Main class that raises this exception
    """

    def __init__(self, message: str, model: Optional[str] = None, text_preview: Optional[str] = None):
        """
        Initialize TokenizationError with detailed context.

        Args:
            message (str): The base error message describing what went wrong
            model (Optional[str]): The model name being used when error occurred
            text_preview (Optional[str]): Preview of the problematic text.
                                        Will be truncated if longer than 50 characters.
        """
        self.model = model
        self.text_preview = text_preview

        full_message = f"Tokenization failed: {message}"
        if model:
            full_message += f" (model: {model})"
        if text_preview:
            preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
            full_message += f" (text preview: '{preview}')"

        super().__init__(full_message)


class InvalidTokenError(TokenizationError):
    """
    Raised when an invalid token is encountered during tokenization.
    
    This exception is a specialized form of TokenizationError that specifically
    handles cases where individual tokens are invalid or problematic. It provides
    additional context about the specific token that caused the issue.

    This exception might be raised when:
    - A token contains invalid characters or sequences
    - A token exceeds maximum length limits
    - A token has encoding issues
    - A token conflicts with model-specific restrictions

    Attributes:
        token (str): The specific invalid token that caused the error
        model (Optional[str]): The model name (inherited from TokenizationError)
        text_preview (Optional[str]): Preview of the text (inherited from TokenizationError)

    Examples:
        Handling invalid token errors:

        .. code-block:: python

            from toksum import TokenCounter
            from toksum.exceptions import InvalidTokenError, TokenizationError
            
            try:
                counter = TokenCounter("gpt-4")
                tokens = counter.count("Some text with problematic content")
            except InvalidTokenError as e:
                print(f"Invalid token encountered: '{e.token}'")
                print(f"Error details: {e}")
                # Maybe try preprocessing the text to remove problematic tokens
            except TokenizationError as e:
                print(f"General tokenization error: {e}")

        Token-specific error handling:

        .. code-block:: python

            def safe_tokenize_with_cleanup(text, model):
                try:
                    counter = TokenCounter(model)
                    return counter.count(text)
                except InvalidTokenError as e:
                    print(f"Removing problematic token: {e.token}")
                    # Remove or replace the problematic token
                    cleaned_text = text.replace(e.token, "")
                    return counter.count(cleaned_text)
                except TokenizationError:
                    print("Could not tokenize even after cleanup")
                    return None

    Note:
        This exception is not commonly raised in the current toksum implementation
        but is available for future enhancements and edge cases where specific
        token validation is needed.

    See Also:
        - :exc:`TokenizationError`: Parent class for general tokenization errors
        - :exc:`EmptyTextError`: For empty text specific cases
        - :class:`TokenCounter`: May raise this exception in specific scenarios
    """
    def __init__(self, token: str, message: str, model: Optional[str] = None, text_preview: Optional[str] = None):
        """
        Initialize InvalidTokenError with token and context information.

        Args:
            token (str): The specific invalid token that caused the error
            message (str): Description of why the token is invalid
            model (Optional[str]): The model name being used
            text_preview (Optional[str]): Preview of the problematic text
        """
        full_message = f"Invalid token '{token}': {message}"
        super().__init__(full_message, model, text_preview)

        
class EmptyTextError(TokenizationError):
    """
    Raised when attempting to tokenize empty text in contexts where it's not allowed.
    
    This exception is a specialized form of TokenizationError for cases where
    empty text is specifically problematic. Note that in most toksum operations,
    empty text is handled gracefully and returns 0 tokens. This exception is
    reserved for contexts where empty text indicates a logical error.

    This exception might be raised when:
    - Empty text is passed to functions that require non-empty content
    - Batch operations encounter unexpected empty strings
    - Validation functions detect empty content where it shouldn't be
    - Message content is empty in chat format validation

    Attributes:
        model (Optional[str]): The model name (inherited from TokenizationError)

    Examples:
        Handling empty text validation:

        .. code-block:: python

            from toksum.exceptions import EmptyTextError, TokenizationError
            
            def validate_content(text, model):
                try:
                    if not text.strip():  # Check for empty or whitespace-only
                        raise EmptyTextError(model=model)
                    
                    counter = TokenCounter(model)
                    return counter.count(text)
                    
                except EmptyTextError as e:
                    print("Error: Content cannot be empty")
                    return None
                except TokenizationError as e:
                    print(f"Tokenization error: {e}")
                    return None

        Batch processing with empty text handling:

        .. code-block:: python

            def process_text_batch(texts, model):
                results = []
                counter = TokenCounter(model)
                
                for i, text in enumerate(texts):
                    try:
                        if not text:
                            raise EmptyTextError(model=model)
                        
                        tokens = counter.count(text)
                        results.append(tokens)
                        
                    except EmptyTextError:
                        print(f"Skipping empty text at index {i}")
                        results.append(0)  # or None, depending on requirements
                    except TokenizationError as e:
                        print(f"Error processing text {i}: {e}")
                        results.append(None)
                
                return results

        Message validation:

        .. code-block:: python

            def validate_messages(messages):
                for i, msg in enumerate(messages):
                    if not msg.get("content", "").strip():
                        raise EmptyTextError(
                            f"Message {i} has empty content"
                        )

    Normal Empty Text Handling:
        In most toksum operations, empty text is handled normally:

        .. code-block:: python

            counter = TokenCounter("gpt-4")
            tokens = counter.count("")  # Returns 0, no exception
            
            messages = [
                {"role": "user", "content": ""},  # This might raise EmptyTextError
                {"role": "assistant", "content": "Hello!"}
            ]

    Use Cases:
        - **Content validation**: Ensuring required fields are not empty
        - **Batch processing**: Identifying problematic entries in datasets
        - **API validation**: Checking inputs before expensive operations
        - **Data quality**: Ensuring content meets minimum requirements

    Note:
        This exception is used sparingly in toksum. Most empty text cases
        are handled gracefully by returning 0 tokens. Use this exception
        only when empty text represents a logical error in your application.

    See Also:
        - :exc:`TokenizationError`: Parent class for general tokenization errors
        - :exc:`InvalidTokenError`: For specific token validation issues
        - :meth:`TokenCounter.count`: Handles empty text gracefully (returns 0)
        - :meth:`TokenCounter.count_messages`: May raise this for empty message content
    """
    def __init__(self, model: Optional[str] = None):
        """
        Initialize EmptyTextError with optional model context.

        Args:
            model (Optional[str]): The model name being used when the error occurred
        """
        super().__init__("Cannot tokenize empty text.", model)

