import functools
from typing import List, Dict, Union, Any

# Define custom exceptions
class UnsupportedModelError(ValueError):
    """Raised when an unsupported model is specified."""
    pass

class TokenizationError(Exception):
    """Raised when tokenization fails."""
    pass

# Centralized model registry
# This dictionary will store information about supported models,
# including their tokenizers and pricing.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt-4": {
        "provider": "openai",
        "tokenizer": "cl100k_base",
        "pricing": {"input": 0.03, "output": 0.06}, # Example pricing per 1K tokens
        "context_window": 8192,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "tokenizer": "cl100k_base",
        "pricing": {"input": 0.0010, "output": 0.0020},
        "context_window": 4096,
    },
    "claude-3-opus-20240229": {
        "provider": "anthropic",
        "tokenizer": "approximate", # Anthropic uses an approximation
        "pricing": {"input": 0.15, "output": 0.75},
        "context_window": 200000,
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "tokenizer": "approximate",
        "pricing": {"input": 0.00035, "output": 0.0007},
        "context_window": 1000000,
    },
    # Add more models as needed
}

@functools.lru_cache(maxsize=128)
def _get_tokenizer(model_name: str) -> Any:
    """
    Retrieves the appropriate tokenizer for a given model.
    This function is cached to avoid repeated tokenizer loading.

    Args:
        model_name: The name of the model.

    Returns:
        The tokenizer object or a string indicating approximation.

    Raises:
        UnsupportedModelError: If the model is not found in the registry.
    """
    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info:
        raise UnsupportedModelError(f"Model '{model_name}' is not supported.")

    tokenizer_type = model_info["tokenizer"]

    if tokenizer_type == "cl100k_base":
        try:
            import tiktoken
            return tiktoken.get_encoding(tokenizer_type)
        except ImportError:
            raise ImportError("The 'tiktoken' library is required for OpenAI models. Please install it with 'pip install tiktoken'.")
    elif tokenizer_type == "approximate":
        return "approximate" # Placeholder for approximation logic
    else:
        raise TokenizationError(f"Unknown tokenizer type: {tokenizer_type} for model {model_name}")

def _count_openai_tokens(text: str, encoding: Any) -> int:
    """
    Counts tokens for OpenAI models using tiktoken.

    Args:
        text: The input text.
        encoding: The tiktoken encoding object.

    Returns:
        The number of tokens.
    """
    return len(encoding.encode(text))

def _count_anthropic_tokens(text: str) -> int:
    """
    Approximates token count for Anthropic models.
    Anthropic models typically use ~4 characters per token.

    Args:
        text: The input text.

    Returns:
        The approximate number of tokens.
    """
    return len(text) // 4 # Simple approximation

def _count_generic_tokens(text: str) -> int:
    """
    Approximates token count for other models.
    A common heuristic is word count or character count / 4.

    Args:
        text: The input text.

    Returns:
        The approximate number of tokens.
    """
    return len(text.split()) # Simple word count approximation

@functools.lru_cache(maxsize=1024)
def count_tokens(text: str, model: str) -> int:
    """
    Counts tokens in a given text for a specified LLM model.

    Args:
        text: The input text string.
        model: The name of the LLM model (e.g., "gpt-4", "claude-3-opus-20240229").

    Returns:
        The number of tokens as an integer.

    Raises:
        ValueError: If `text` is not a string or `model` is not a string.
        UnsupportedModelError: If the specified model is not supported.
        TokenizationError: If an error occurs during tokenization.
    """
    if not isinstance(text, str):
        raise ValueError("Input 'text' must be a string.")
    if not isinstance(model, str):
        raise ValueError("Input 'model' must be a string.")

    model_info = MODEL_REGISTRY.get(model)
    if not model_info:
        raise UnsupportedModelError(f"Model '{model}' is not supported.")

    provider = model_info["provider"]
    tokenizer = _get_tokenizer(model)

    try:
        if provider == "openai":
            return _count_openai_tokens(text, tokenizer)
        elif provider == "anthropic":
            return _count_anthropic_tokens(text)
        else:
            return _count_generic_tokens(text)
    except Exception as e:
        raise TokenizationError(f"Failed to tokenize text for model '{model}': {e}")

def get_supported_models() -> Dict[str, List[str]]:
    """
    Retrieves a dictionary of all supported models, grouped by provider.

    Returns:
        A dictionary where keys are provider names (str) and values are
        lists of model names (List[str]) supported by that provider.
    """
    supported_models: Dict[str, List[str]] = {}
    for model_name, info in MODEL_REGISTRY.items():
        provider = info["provider"]
        if provider not in supported_models:
            supported_models[provider] = []
        supported_models[provider].append(model_name)
    return supported_models

def estimate_cost(
    token_count: int, model: str, input_tokens: bool = True, currency: str = "USD"
) -> float:
    """
    Estimates the cost for a given number of tokens and model.

    Args:
        token_count: The number of tokens.
        model: The name of the LLM model.
        input_tokens: True if counting input tokens, False for output tokens.
        currency: The desired currency for cost estimation (e.g., "USD", "INR").

    Returns:
        The estimated cost as a float.

    Raises:
        ValueError: If `token_count` is not a non-negative integer, `model` is not a string,
                    or `currency` is not supported.
        UnsupportedModelError: If the specified model is not supported.
    """
    if not isinstance(token_count, int) or token_count < 0:
        raise ValueError("Input 'token_count' must be a non-negative integer.")
    if not isinstance(model, str):
        raise ValueError("Input 'model' must be a string.")
    if currency not in ["USD", "INR"]: # Extend with more currencies as needed
        raise ValueError(f"Unsupported currency: '{currency}'. Only 'USD' and 'INR' are supported.")

    model_info = MODEL_REGISTRY.get(model)
    if not model_info:
        raise UnsupportedModelError(f"Model '{model}' is not supported for cost estimation.")

    pricing = model_info["pricing"]
    cost_per_1k_tokens = pricing["input"] if input_tokens else pricing["output"]

    estimated_cost_usd = (token_count / 1000) * cost_per_1k_tokens

    if currency == "INR":
        # Example conversion rate, this should ideally be dynamic or configurable
        usd_to_inr_rate = 83.0
        return estimated_cost_usd * usd_to_inr_rate
    return estimated_cost_usd

class TokenCounter:
    """
    A class for counting tokens for a specific LLM model.

    Attributes:
        model: The name of the LLM model.
        _tokenizer: The tokenizer object or approximation indicator for the model.
    """

    def __init__(self, model: str):
        """
        Initializes the TokenCounter with a specific model.

        Args:
            model: The name of the LLM model.

        Raises:
            ValueError: If `model` is not a string.
            UnsupportedModelError: If the specified model is not supported.
        """
        if not isinstance(model, str):
            raise ValueError("Input 'model' must be a string.")
        if model not in MODEL_REGISTRY:
            raise UnsupportedModelError(f"Model '{model}' is not supported.")

        self.model: str = model
        self._tokenizer: Any = _get_tokenizer(model)

    @functools.lru_cache(maxsize=1024)
    def count(self, text: str) -> int:
        """
        Counts tokens in a given text using the initialized model's tokenizer.

        Args:
            text: The input text string.

        Returns:
            The number of tokens as an integer.

        Raises:
            ValueError: If `text` is not a string.
            TokenizationError: If an error occurs during tokenization.
        """
        if not isinstance(text, str):
            raise ValueError("Input 'text' must be a string.")

        model_info = MODEL_REGISTRY[self.model]
        provider = model_info["provider"]

        try:
            if provider == "openai":
                return _count_openai_tokens(text, self._tokenizer)
            elif provider == "anthropic":
                return _count_anthropic_tokens(text)
            else:
                return _count_generic_tokens(text)
        except Exception as e:
            raise TokenizationError(f"Failed to tokenize text for model '{self.model}': {e}")

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts tokens in a list of chat messages for the initialized model.
        This method approximates the token count for chat messages,
        including overhead for roles and message structure.

        Args:
            messages: A list of message dictionaries, each with "role" and "content" keys.
                      Example: [{"role": "user", "content": "Hello"}]

        Returns:
            The total number of tokens as an integer.

        Raises:
            ValueError: If `messages` is not a list or contains invalid message formats.
            TokenizationError: If an error occurs during tokenization.
        """
        if not isinstance(messages, list):
            raise ValueError("Input 'messages' must be a list of dictionaries.")

        total_tokens = 0
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("Each message must be a dictionary with 'role' and 'content' keys.")
            
            # Add tokens for content
            total_tokens += self.count(message["content"])
            
            # Add tokens for role and other overhead (approximation)
            # This is a simplified approximation; actual overhead varies by model and API
            total_tokens += self.count(message["role"]) + 3 # +3 for role, colon, and newline/separator

        # Add a small overhead for the overall conversation structure
        total_tokens += 3 # Example: for system message start, assistant response start, etc.
        return total_tokens
