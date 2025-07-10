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

