"""
Custom exceptions for the toksum library.
"""

from typing import List, Optional


class ToksumError(Exception):
    """Base exception class for toksum library."""
    pass


class UnsupportedModelError(ToksumError):
    """Raised when an unsupported model is specified."""

    def __init__(self, model: str, supported_models: Optional[List[str]] = None):
        self.model = model
        self.supported_models = supported_models or []

        if self.supported_models:
            message = f"Model '{model}' is not supported. Supported models: {', '.join(self.supported_models)}"
        else:
            message = f"Model '{model}' is not supported."

        super().__init__(message)


class ModelNotFoundError(ToksumError):
    """Raised when a specified model is not found."""
    def __init__(self, model: str):
        super().__init__(f"Model '{model}' not found.")


class TokenizationError(ToksumError):
    """Raised when tokenization fails."""

    def __init__(self, message: str, model: Optional[str] = None, text_preview: Optional[str] = None):
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
    """Raised when an invalid token is encountered during tokenization."""
    def __init__(self, token: str, message: str, model: Optional[str] = None, text_preview: Optional[str] = None):
        full_message = f"Invalid token '{token}': {message}"
        super().__init__(full_message, model, text_preview)

        
class EmptyTextError(TokenizationError):
    """Raised when attempting to tokenize empty text."""
    def __init__(self, model: Optional[str] = None):
        super().__init__("Cannot tokenize empty text.", model)


class ModelDeprecationError(ToksumError):
    """Raised when a deprecated model is used and strict mode is enabled."""
    
    def __init__(self, model: str, replacement_model: Optional[str] = None, 
                 deprecation_date: Optional[str] = None, removal_date: Optional[str] = None):
        self.model = model
        self.replacement_model = replacement_model
        self.deprecation_date = deprecation_date
        self.removal_date = removal_date
        
        message = f"Model '{model}' is deprecated"
        if deprecation_date:
            message += f" (deprecated on {deprecation_date})"
        if removal_date:
            message += f" and will be removed on {removal_date}"
        if replacement_model:
            message += f". Use '{replacement_model}' instead"
        message += "."
        
        super().__init__(message)


class RateLimitError(ToksumError):
    """Raised when rate limits are exceeded during tokenization operations."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, 
                 provider: Optional[str] = None):
        self.retry_after = retry_after
        self.provider = provider
        
        full_message = f"Rate limit exceeded: {message}"
        if provider:
            full_message += f" (provider: {provider})"
        if retry_after:
            full_message += f". Retry after {retry_after} seconds"
        
        super().__init__(full_message)


class ConfigurationError(ToksumError):
    """Raised when there's an issue with library configuration or setup."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 suggested_fix: Optional[str] = None):
        self.config_key = config_key
        self.suggested_fix = suggested_fix
        
        full_message = f"Configuration error: {message}"
        if config_key:
            full_message += f" (config key: {config_key})"
        if suggested_fix:
            full_message += f". Suggested fix: {suggested_fix}"
        
        super().__init__(full_message)


class BatchProcessingError(ToksumError):
    """Raised when batch processing operations fail."""
    
    def __init__(self, message: str, failed_items: Optional[List[int]] = None, 
                 total_items: Optional[int] = None):
        self.failed_items = failed_items or []
        self.total_items = total_items
        
        full_message = f"Batch processing failed: {message}"
        if failed_items and total_items:
            success_count = total_items - len(failed_items)
            full_message += f" ({success_count}/{total_items} items processed successfully)"
        elif failed_items:
            full_message += f" ({len(failed_items)} items failed)"
        
        super().__init__(full_message)

