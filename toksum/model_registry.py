"""
Model registry for token counting functionality.
Provides model support checking and cost information.
"""

from typing import Dict, List, Set
import re

# Import all model dictionaries from the main module
from .core import (
    OPENAI_MODELS, ANTHROPIC_MODELS, GOOGLE_MODELS, META_MODELS, MISTRAL_MODELS,
    COHERE_MODELS, ANTHROPIC_LEGACY_MODELS, OPENAI_LEGACY_MODELS, PERPLEXITY_MODELS,
    HUGGINGFACE_MODELS, AI21_MODELS, TOGETHER_MODELS, XAI_MODELS, ALIBABA_MODELS,
    BAIDU_MODELS, HUAWEI_MODELS, YANDEX_MODELS, STABILITY_MODELS, TII_MODELS,
    ELEUTHERAI_MODELS, MOSAICML_MODELS, REPLIT_MODELS, MINIMAX_MODELS, ALEPH_ALPHA_MODELS,
    DEEPSEEK_MODELS, TSINGHUA_MODELS, RWKV_MODELS, COMMUNITY_MODELS, ANTHROPIC_HAIKU_MODELS,
    OPENAI_O1_MODELS, ANTHROPIC_COMPUTER_USE_MODELS, GOOGLE_GEMINI_2_MODELS, META_LLAMA_33_MODELS,
    MISTRAL_LARGE_2_MODELS, DEEPSEEK_V3_MODELS, QWEN_25_MODELS, ANTHROPIC_CLAUDE_21_MODELS,
    OPENAI_VISION_MODELS, COHERE_COMMAND_R_PLUS_MODELS, ANTHROPIC_INSTANT_2_MODELS,
    GOOGLE_PALM_MODELS, MICROSOFT_MODELS, AMAZON_MODELS, NVIDIA_MODELS, IBM_MODELS,
    SALESFORCE_MODELS, BIGCODE_MODELS, ANTHROPIC_OPUS_MODELS, OPENAI_GPT4_TURBO_MODELS,
    ANTHROPIC_SONNET_MODELS, GOOGLE_GEMINI_PRO_MODELS, META_LLAMA2_CHAT_MODELS,
    META_LLAMA3_INSTRUCT_MODELS, MISTRAL_INSTRUCT_MODELS, OPENAI_EMBEDDING_MODELS,
    DATABRICKS_MODELS, VOYAGE_MODELS
)

class UnsupportedModelError(ValueError):
    """Exception raised when a model is not supported."""
    
    def __init__(self, model: str, supported_models: List[str] = None):
        self.model = model
        self.supported_models = supported_models or []
        message = f"Model '{model}' is not supported."
        if supported_models:
            message += f" Supported models: {', '.join(sorted(supported_models)[:10])}{'...' if len(supported_models) > 10 else ''}"
        super().__init__(message)


class TokenizationError(Exception):
    """Exception raised when tokenization fails."""
    
    def __init__(self, message: str, model: str = None, text_preview: str = None):
        self.model = model
        self.text_preview = text_preview
        full_message = message
        if model:
            full_message += f" (model: {model})"
        if text_preview:
            preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
            full_message += f" (text: '{preview}')"
        super().__init__(full_message)


def is_supported_model(model: str) -> bool:
    """
    Check if a model is supported by the token counter.
    
    Args:
        model: The model name to check
        
    Returns:
        True if the model is supported, False otherwise
    """
    if not model or not isinstance(model, str):
        return False
    
    model_lower = model.lower()
    
    # Check all model dictionaries
    all_models = [
        OPENAI_MODELS, ANTHROPIC_MODELS, GOOGLE_MODELS, META_MODELS, MISTRAL_MODELS,
        COHERE_MODELS, ANTHROPIC_LEGACY_MODELS, OPENAI_LEGACY_MODELS, PERPLEXITY_MODELS,
        HUGGINGFACE_MODELS, AI21_MODELS, TOGETHER_MODELS, XAI_MODELS, ALIBABA_MODELS,
        BAIDU_MODELS, HUAWEI_MODELS, YANDEX_MODELS, STABILITY_MODELS, TII_MODELS,
        ELEUTHERAI_MODELS, MOSAICML_MODELS, REPLIT_MODELS, MINIMAX_MODELS, ALEPH_ALPHA_MODELS,
        DEEPSEEK_MODELS, TSINGHUA_MODELS, RWKV_MODELS, COMMUNITY_MODELS, ANTHROPIC_HAIKU_MODELS,
        OPENAI_O1_MODELS, ANTHROPIC_COMPUTER_USE_MODELS, GOOGLE_GEMINI_2_MODELS, META_LLAMA_33_MODELS,
        MISTRAL_LARGE_2_MODELS, DEEPSEEK_V3_MODELS, QWEN_25_MODELS, ANTHROPIC_CLAUDE_21_MODELS,
        OPENAI_VISION_MODELS, COHERE_COMMAND_R_PLUS_MODELS, ANTHROPIC_INSTANT_2_MODELS,
        GOOGLE_PALM_MODELS, MICROSOFT_MODELS, AMAZON_MODELS, NVIDIA_MODELS, IBM_MODELS,
        SALESFORCE_MODELS, BIGCODE_MODELS, ANTHROPIC_OPUS_MODELS, OPENAI_GPT4_TURBO_MODELS,
        ANTHROPIC_SONNET_MODELS, GOOGLE_GEMINI_PRO_MODELS, META_LLAMA2_CHAT_MODELS,
        META_LLAMA3_INSTRUCT_MODELS, MISTRAL_INSTRUCT_MODELS, OPENAI_EMBEDDING_MODELS,
        DATABRICKS_MODELS, VOYAGE_MODELS
    ]
    
    for model_dict in all_models:
        if model_lower in {k.lower(): v for k, v in model_dict.items()}:
            return True
    
    return False


def get_supported_models() -> Dict[str, List[str]]:
    """
    Get a dictionary of supported models by provider.
    
    Returns:
        Dictionary with provider names as keys and lists of model names as values
    """
    return {
        "openai": (list(OPENAI_MODELS.keys()) + list(OPENAI_LEGACY_MODELS.keys()) + 
                  list(OPENAI_O1_MODELS.keys()) + list(OPENAI_VISION_MODELS.keys()) +
                  list(OPENAI_GPT4_TURBO_MODELS.keys()) + list(OPENAI_EMBEDDING_MODELS.keys())),
        "databricks": list(DATABRICKS_MODELS.keys()),
        "voyage": list(VOYAGE_MODELS.keys()),
        "anthropic": (list(ANTHROPIC_MODELS.keys()) + list(ANTHROPIC_LEGACY_MODELS.keys()) + 
                     list(ANTHROPIC_HAIKU_MODELS.keys()) + list(ANTHROPIC_COMPUTER_USE_MODELS.keys()) +
                     list(ANTHROPIC_CLAUDE_21_MODELS.keys()) + list(ANTHROPIC_INSTANT_2_MODELS.keys()) +
                     list(ANTHROPIC_OPUS_MODELS.keys()) + list(ANTHROPIC_SONNET_MODELS.keys())),
        "google": (list(GOOGLE_MODELS.keys()) + list(GOOGLE_GEMINI_2_MODELS.keys()) + 
                  list(GOOGLE_PALM_MODELS.keys()) + list(GOOGLE_GEMINI_PRO_MODELS.keys())),
        "meta": (list(META_MODELS.keys()) + list(META_LLAMA_33_MODELS.keys()) +
                list(META_LLAMA2_CHAT_MODELS.keys()) + list(META_LLAMA3_INSTRUCT_MODELS.keys())),
        "mistral": (list(MISTRAL_MODELS.keys()) + list(MISTRAL_LARGE_2_MODELS.keys()) +
                   list(MISTRAL_INSTRUCT_MODELS.keys())),
        "cohere": list(COHERE_MODELS.keys()) + list(COHERE_COMMAND_R_PLUS_MODELS.keys()),
        "perplexity": list(PERPLEXITY_MODELS.keys()),
        "huggingface": list(HUGGINGFACE_MODELS.keys()),
        "ai21": list(AI21_MODELS.keys()),
        "together": list(TOGETHER_MODELS.keys()),
        "xai": list(XAI_MODELS.keys()),
        "alibaba": list(ALIBABA_MODELS.keys()) + list(QWEN_25_MODELS.keys()),
        "baidu": list(BAIDU_MODELS.keys()),
        "huawei": list(HUAWEI_MODELS.keys()),
        "yandex": list(YANDEX_MODELS.keys()),
        "stability": list(STABILITY_MODELS.keys()),
        "tii": list(TII_MODELS.keys()),
        "eleutherai": list(ELEUTHERAI_MODELS.keys()),
        "mosaicml": list(MOSAICML_MODELS.keys()),
        "replit": list(REPLIT_MODELS.keys()),
        "minimax": list(MINIMAX_MODELS.keys()),
        "aleph_alpha": list(ALEPH_ALPHA_MODELS.keys()),
        "deepseek": list(DEEPSEEK_MODELS.keys()) + list(DEEPSEEK_V3_MODELS.keys()),
        "tsinghua": list(TSINGHUA_MODELS.keys()),
        "rwkv": list(RWKV_MODELS.keys()),
        "community": list(COMMUNITY_MODELS.keys()),
        "microsoft": list(MICROSOFT_MODELS.keys()),
        "amazon": list(AMAZON_MODELS.keys()),
        "nvidia": list(NVIDIA_MODELS.keys()),
        "ibm": list(IBM_MODELS.keys()),
        "salesforce": list(SALESFORCE_MODELS.keys()),
        "bigcode": list(BIGCODE_MODELS.keys()),
    }


def get_model_cost(model: str, token_type: str = "input") -> float:
    """
    Get the cost per token for a specific model.
    
    Args:
        model: The model name
        token_type: Either "input" or "output"
        
    Returns:
        Cost per token in USD, or 0.0 if model pricing is not available
        
    Raises:
        ValueError: If token_type is not "input" or "output"
    """
    if token_type not in ["input", "output"]:
        raise ValueError("token_type must be either 'input' or 'output'")
    
    model = model.lower()
    
    # Approximate pricing per 1K tokens (in USD)
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
        "dbrx-instruct": {"input": 0.001, "output": 0.002},
        "dbrx-base": {"input": 0.001, "output": 0.002},
        "dolly-v2-12b": {"input": 0.001, "output": 0.002},
        "dolly-v2-7b": {"input": 0.001, "output": 0.002},
        "dolly-v2-3b": {"input": 0.001, "output": 0.002},
        "voyage-2": {"input": 0.0001, "output": 0.0001},
        "voyage-large-2": {"input": 0.0001, "output": 0.0001},
        "voyage-code-2": {"input": 0.0001, "output": 0.0001},
        "voyage-finance-2": {"input": 0.0001, "output": 0.0001},
        "voyage-law-2": {"input": 0.0001, "output": 0.0001},
        "voyage-multilingual-2": {"input": 0.0001, "output": 0.0001},
    }
    
    # Try exact match first
    if model in pricing:
        return pricing[model][token_type]
    
    # Try pattern matching for similar models
    for pattern, costs in pricing.items():
        if re.match(pattern.replace("-", "[-_]?"), model):
            return costs[token_type]
    
    # Default pricing for unknown models
    return 0.0
