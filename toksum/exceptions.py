"""
Custom exceptions for the toksum library.
"""


class ToksumError(Exception):
    """Base exception class for toksum library."""
    pass


class UnsupportedModelError(ToksumError):
    """Raised when an unsupported model is specified."""
    
    def __init__(self, model: str, supported_models: list = None):
        self.model = model
        self.supported_models = supported_models or []
        
        if self.supported_models:
            message = f"Model '{model}' is not supported. Supported models: {', '.join(self.supported_models)}"
        else:
            message = f"Model '{model}' is not supported."
        
        super().__init__(message)


class TokenizationError(ToksumError):
    """Raised when tokenization fails."""
    
    def __init__(self, message: str, model: str = None, text_preview: str = None):
        self.model = model
        self.text_preview = text_preview
        
        full_message = f"Tokenization failed: {message}"
        if model:
            full_message += f" (model: {model})"
        if text_preview:
            preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
            full_message += f" (text preview: '{preview}')"
        
        super().__init__(full_message)
