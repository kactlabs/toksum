"""
Core functionality for token counting across different LLM providers.
"""

import re
from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tiktoken
    from anthropic import Anthropic
else:
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
    "gpt-4o-2024-08-06": "cl100k_base",  # ADDED
    "gpt-4o-2024-11-20": "cl100k_base",  # ADDED
    "gpt-4-1106-vision-preview": "cl100k_base",  # ADDED
    "gpt-4-turbo-2024-04-09": "cl100k_base",  # ADDED
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    "gpt-3.5-turbo-instruct": "cl100k_base",  # ADDED
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
    "claude-3-opus": "claude-3",  # ADDED (short name)
    "claude-3-sonnet": "claude-3",  # ADDED (short name)
    "claude-3-haiku": "claude-3",  # ADDED (short name)
    "claude-2.1": "claude-2",
    "claude-2.0": "claude-2",
    "claude-instant-1.2": "claude-instant",
    "claude-instant-1.1": "claude-instant",
    "claude-instant-1.0": "claude-instant",
    "claude-instant": "claude-instant",  # ADDED (short name)
}

# Google Models (using approximation similar to Claude)
GOOGLE_MODELS = {
    "gemini-pro": "gemini",  # NEW
    "gemini-pro-vision": "gemini",  # NEW
    "gemini-1.5-pro": "gemini-1.5",  # NEW
    "gemini-1.5-flash": "gemini-1.5",  # NEW
    "gemini-1.5-pro-latest": "gemini-1.5",  # ADDED
    "gemini-1.5-flash-latest": "gemini-1.5",  # ADDED
    "gemini-1.0-pro": "gemini",  # ADDED
    "gemini-1.0-pro-vision": "gemini",  # ADDED
    "gemini-ultra": "gemini-ultra",  # ADDED
}

# Meta Models (using approximation)
META_MODELS = {
    "llama-2-7b": "llama-2",  # NEW
    "llama-2-13b": "llama-2",  # NEW
    "llama-2-70b": "llama-2",  # NEW
    "llama-3-8b": "llama-3",  # ADDED
    "llama-3-70b": "llama-3",  # ADDED
    "llama-3.1-8b": "llama-3.1",  # ADDED
    "llama-3.1-70b": "llama-3.1",  # ADDED
    "llama-3.1-405b": "llama-3.1",  # ADDED
    "llama-3.2-1b": "llama-3.2",  # ADDED
    "llama-3.2-3b": "llama-3.2",  # ADDED
}

# Mistral Models (using approximation)
MISTRAL_MODELS = {
    "mistral-7b": "mistral",  # NEW
    "mistral-8x7b": "mistral",  # NEW
    "mistral-large": "mistral-large",  # ADDED
    "mistral-medium": "mistral-medium",  # ADDED
    "mistral-small": "mistral-small",  # ADDED
    "mistral-tiny": "mistral-tiny",  # ADDED
    "mixtral-8x7b": "mixtral",  # ADDED
    "mixtral-8x22b": "mixtral",  # ADDED
}

# Cohere Models (using approximation)
COHERE_MODELS = {
    "command": "cohere",  # NEW
    "command-light": "cohere",  # ADDED
    "command-nightly": "cohere",  # ADDED
    "command-r": "cohere-r",  # ADDED
    "command-r-plus": "cohere-r",  # ADDED
    "command-r-08-2024": "cohere-r",  # ADDED
    "command-r-plus-08-2024": "cohere-r",  # ADDED
}

# Anthropic Legacy Models (using approximation)
ANTHROPIC_LEGACY_MODELS = {
    "claude-1": "claude-1",  # ADDED
    "claude-1.3": "claude-1",  # ADDED
    "claude-1.3-100k": "claude-1",  # ADDED
}

# OpenAI Legacy Models (additional variants)
OPENAI_LEGACY_MODELS = {
    "gpt-3": "r50k_base",  # ADDED
    "text-embedding-ada-002": "cl100k_base",  # ADDED
    "text-embedding-3-small": "cl100k_base",  # ADDED
    "text-embedding-3-large": "cl100k_base",  # ADDED
    "gpt-4-base": "cl100k_base",  # ADDED
    "gpt-3.5-turbo-instruct-0914": "cl100k_base",  # ADDED
}

# Perplexity Models (using approximation)
PERPLEXITY_MODELS = {
    "pplx-7b-online": "perplexity",  # ADDED
    "pplx-70b-online": "perplexity",  # ADDED
    "pplx-7b-chat": "perplexity",  # ADDED
    "pplx-70b-chat": "perplexity",  # ADDED
    "codellama-34b-instruct": "perplexity",  # ADDED
}

# Hugging Face Models (using approximation)
HUGGINGFACE_MODELS = {
    "microsoft/DialoGPT-medium": "huggingface",  # ADDED
    "microsoft/DialoGPT-large": "huggingface",  # ADDED
    "facebook/blenderbot-400M-distill": "huggingface",  # ADDED
    "facebook/blenderbot-1B-distill": "huggingface",  # ADDED
    "facebook/blenderbot-3B": "huggingface",  # ADDED
}

# AI21 Models (using approximation)
AI21_MODELS = {
    "j2-light": "ai21",  # ADDED
    "j2-mid": "ai21",  # ADDED
    "j2-ultra": "ai21",  # ADDED
    "j2-jumbo-instruct": "ai21",  # ADDED
}

# Together AI Models (using approximation)
TOGETHER_MODELS = {
    "togethercomputer/RedPajama-INCITE-Chat-3B-v1": "together",  # ADDED
    "togethercomputer/RedPajama-INCITE-Chat-7B-v1": "together",  # ADDED
    "NousResearch/Nous-Hermes-Llama2-13b": "together",  # ADDED
}

# xAI Models (using approximation)
XAI_MODELS = {
    "grok-1": "xai",  # NEW
    "grok-1.5": "xai",  # NEW
    "grok-2": "xai",  # NEW
    "grok-beta": "xai",  # NEW
}

# Alibaba Models (using approximation)
ALIBABA_MODELS = {
    "qwen-1.5-0.5b": "qwen",  # NEW
    "qwen-1.5-1.8b": "qwen",  # NEW
    "qwen-1.5-4b": "qwen",  # NEW
    "qwen-1.5-7b": "qwen",  # NEW
    "qwen-1.5-14b": "qwen",  # NEW
    "qwen-1.5-32b": "qwen",  # NEW
    "qwen-1.5-72b": "qwen",  # NEW
    "qwen-1.5-110b": "qwen",  # NEW
    "qwen-2-0.5b": "qwen-2",  # NEW
    "qwen-2-1.5b": "qwen-2",  # NEW
    "qwen-2-7b": "qwen-2",  # NEW
    "qwen-2-57b": "qwen-2",  # NEW
    "qwen-2-72b": "qwen-2",  # NEW
    "qwen-vl": "qwen-vl",  # NEW
    "qwen-vl-chat": "qwen-vl",  # NEW
    "qwen-vl-plus": "qwen-vl",  # NEW
}

# Baidu Models (using approximation)
BAIDU_MODELS = {
    "ernie-4.0": "ernie",  # NEW
    "ernie-3.5": "ernie",  # NEW
    "ernie-3.0": "ernie",  # NEW
    "ernie-speed": "ernie",  # NEW
    "ernie-lite": "ernie",  # NEW
    "ernie-tiny": "ernie",  # NEW
    "ernie-bot": "ernie",  # NEW
    "ernie-bot-4": "ernie",  # NEW
}

# Huawei Models (using approximation)
HUAWEI_MODELS = {
    "pangu-alpha-2.6b": "pangu",  # NEW
    "pangu-alpha-13b": "pangu",  # NEW
    "pangu-alpha-200b": "pangu",  # NEW
    "pangu-coder": "pangu",  # NEW
    "pangu-coder-15b": "pangu",  # NEW
}

# Yandex Models (using approximation)
YANDEX_MODELS = {
    "yalm-100b": "yalm",  # NEW
    "yalm-200b": "yalm",  # NEW
    "yagpt": "yalm",  # NEW
    "yagpt-2": "yalm",  # NEW
}

# Stability AI Models (using approximation)
STABILITY_MODELS = {
    "stablelm-alpha-3b": "stablelm",  # NEW
    "stablelm-alpha-7b": "stablelm",  # NEW
    "stablelm-base-alpha-3b": "stablelm",  # NEW
    "stablelm-base-alpha-7b": "stablelm",  # NEW
    "stablelm-tuned-alpha-3b": "stablelm",  # NEW
    "stablelm-tuned-alpha-7b": "stablelm",  # NEW
    "stablelm-zephyr-3b": "stablelm",  # NEW
}

# TII Models (using approximation)
TII_MODELS = {
    "falcon-7b": "falcon",  # NEW
    "falcon-7b-instruct": "falcon",  # NEW
    "falcon-40b": "falcon",  # NEW
    "falcon-40b-instruct": "falcon",  # NEW
    "falcon-180b": "falcon",  # NEW
    "falcon-180b-chat": "falcon",  # NEW
}

# EleutherAI Models (using approximation)
ELEUTHERAI_MODELS = {
    "gpt-neo-125m": "gpt-neo",  # NEW
    "gpt-neo-1.3b": "gpt-neo",  # NEW
    "gpt-neo-2.7b": "gpt-neo",  # NEW
    "gpt-neox-20b": "gpt-neox",  # NEW
    "pythia-70m": "pythia",  # NEW
    "pythia-160m": "pythia",  # NEW
    "pythia-410m": "pythia",  # NEW
    "pythia-1b": "pythia",  # NEW
    "pythia-1.4b": "pythia",  # NEW
    "pythia-2.8b": "pythia",  # NEW
    "pythia-6.9b": "pythia",  # NEW
    "pythia-12b": "pythia",  # NEW
}

# MosaicML/Databricks Models (using approximation)
MOSAICML_MODELS = {
    "mpt-7b": "mpt",  # NEW
    "mpt-7b-chat": "mpt",  # NEW
    "mpt-7b-instruct": "mpt",  # NEW
    "mpt-30b": "mpt",  # NEW
    "mpt-30b-chat": "mpt",  # NEW
    "mpt-30b-instruct": "mpt",  # NEW
    "dbrx": "dbrx",  # NEW
    "dbrx-instruct": "dbrx",  # NEW
}

# Replit Models (using approximation)
REPLIT_MODELS = {
    "replit-code-v1-3b": "replit",  # NEW
    "replit-code-v1.5-3b": "replit",  # NEW
    "replit-code-v2-3b": "replit",  # NEW
}

# MiniMax Models (using approximation)
MINIMAX_MODELS = {
    "abab5.5-chat": "minimax",  # NEW
    "abab5.5s-chat": "minimax",  # NEW
    "abab6-chat": "minimax",  # NEW
    "abab6.5-chat": "minimax",  # NEW
    "abab6.5s-chat": "minimax",  # NEW
}

# Aleph Alpha Models (using approximation)
ALEPH_ALPHA_MODELS = {
    "luminous-base": "luminous",  # NEW
    "luminous-extended": "luminous",  # NEW
    "luminous-supreme": "luminous",  # NEW
    "luminous-supreme-control": "luminous",  # NEW
}

# DeepSeek Models (using approximation)
DEEPSEEK_MODELS = {
    "deepseek-coder-1.3b": "deepseek",  # NEW
    "deepseek-coder-6.7b": "deepseek",  # NEW
    "deepseek-coder-33b": "deepseek",  # NEW
    "deepseek-coder-instruct": "deepseek",  # NEW
    "deepseek-vl-1.3b": "deepseek-vl",  # NEW
    "deepseek-vl-7b": "deepseek-vl",  # NEW
    "deepseek-llm-7b": "deepseek",  # NEW
    "deepseek-llm-67b": "deepseek",  # NEW
}

# Tsinghua KEG Lab Models (using approximation)
TSINGHUA_MODELS = {
    "chatglm-6b": "chatglm",  # NEW
    "chatglm2-6b": "chatglm",  # NEW
    "chatglm3-6b": "chatglm",  # NEW
    "glm-4": "chatglm",  # NEW
    "glm-4v": "chatglm",  # NEW
}

# RWKV Models (using approximation)
RWKV_MODELS = {
    "rwkv-4-169m": "rwkv",  # NEW
    "rwkv-4-430m": "rwkv",  # NEW
    "rwkv-4-1b5": "rwkv",  # NEW
    "rwkv-4-3b": "rwkv",  # NEW
    "rwkv-4-7b": "rwkv",  # NEW
    "rwkv-4-14b": "rwkv",  # NEW
    "rwkv-5-world": "rwkv",  # NEW
}

# Community Fine-tuned Models (using approximation)
COMMUNITY_MODELS = {
    "vicuna-7b": "vicuna",  # NEW
    "vicuna-13b": "vicuna",  # NEW
    "vicuna-33b": "vicuna",  # NEW
    "alpaca-7b": "alpaca",  # NEW
    "alpaca-13b": "alpaca",  # NEW
    "wizardlm-7b": "wizardlm",  # NEW
    "wizardlm-13b": "wizardlm",  # NEW
    "wizardlm-30b": "wizardlm",  # NEW
    "orca-mini-3b": "orca",  # NEW
    "orca-mini-7b": "orca",  # NEW
    "orca-mini-13b": "orca",  # NEW
    "zephyr-7b-alpha": "zephyr",  # NEW
    "zephyr-7b-beta": "zephyr",  # NEW
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
        self.tokenizer: Optional[Any] = None
        self.model = model.lower()
        self.provider = self._detect_provider()
        self._setup_tokenizer()
    
    def _detect_provider(self) -> str:
        """Detect which provider the model belongs to."""
        # Create lowercase versions of all model dictionaries for case-insensitive matching
        openai_models_lower = {k.lower(): v for k, v in OPENAI_MODELS.items()}
        openai_legacy_models_lower = {k.lower(): v for k, v in OPENAI_LEGACY_MODELS.items()}
        anthropic_models_lower = {k.lower(): v for k, v in ANTHROPIC_MODELS.items()}
        anthropic_legacy_models_lower = {k.lower(): v for k, v in ANTHROPIC_LEGACY_MODELS.items()}
        google_models_lower = {k.lower(): v for k, v in GOOGLE_MODELS.items()}
        meta_models_lower = {k.lower(): v for k, v in META_MODELS.items()}
        mistral_models_lower = {k.lower(): v for k, v in MISTRAL_MODELS.items()}
        cohere_models_lower = {k.lower(): v for k, v in COHERE_MODELS.items()}
        perplexity_models_lower = {k.lower(): v for k, v in PERPLEXITY_MODELS.items()}
        huggingface_models_lower = {k.lower(): v for k, v in HUGGINGFACE_MODELS.items()}
        ai21_models_lower = {k.lower(): v for k, v in AI21_MODELS.items()}
        together_models_lower = {k.lower(): v for k, v in TOGETHER_MODELS.items()}
        xai_models_lower = {k.lower(): v for k, v in XAI_MODELS.items()}
        alibaba_models_lower = {k.lower(): v for k, v in ALIBABA_MODELS.items()}
        baidu_models_lower = {k.lower(): v for k, v in BAIDU_MODELS.items()}
        huawei_models_lower = {k.lower(): v for k, v in HUAWEI_MODELS.items()}
        yandex_models_lower = {k.lower(): v for k, v in YANDEX_MODELS.items()}
        stability_models_lower = {k.lower(): v for k, v in STABILITY_MODELS.items()}
        tii_models_lower = {k.lower(): v for k, v in TII_MODELS.items()}
        eleutherai_models_lower = {k.lower(): v for k, v in ELEUTHERAI_MODELS.items()}
        mosaicml_models_lower = {k.lower(): v for k, v in MOSAICML_MODELS.items()}
        replit_models_lower = {k.lower(): v for k, v in REPLIT_MODELS.items()}
        minimax_models_lower = {k.lower(): v for k, v in MINIMAX_MODELS.items()}
        aleph_alpha_models_lower = {k.lower(): v for k, v in ALEPH_ALPHA_MODELS.items()}
        deepseek_models_lower = {k.lower(): v for k, v in DEEPSEEK_MODELS.items()}
        tsinghua_models_lower = {k.lower(): v for k, v in TSINGHUA_MODELS.items()}
        rwkv_models_lower = {k.lower(): v for k, v in RWKV_MODELS.items()}
        community_models_lower = {k.lower(): v for k, v in COMMUNITY_MODELS.items()}
        
        if self.model in openai_models_lower or self.model in openai_legacy_models_lower:
            return "openai"
        elif self.model in anthropic_models_lower or self.model in anthropic_legacy_models_lower:
            return "anthropic"
        elif self.model in google_models_lower:
            return "google"
        elif self.model in meta_models_lower:
            return "meta"
        elif self.model in mistral_models_lower:
            return "mistral"
        elif self.model in cohere_models_lower:
            return "cohere"
        elif self.model in perplexity_models_lower:
            return "perplexity"
        elif self.model in huggingface_models_lower:
            return "huggingface"
        elif self.model in ai21_models_lower:
            return "ai21"
        elif self.model in together_models_lower:
            return "together"
        elif self.model in xai_models_lower:
            return "xai"
        elif self.model in alibaba_models_lower:
            return "alibaba"
        elif self.model in baidu_models_lower:
            return "baidu"
        elif self.model in huawei_models_lower:
            return "huawei"
        elif self.model in yandex_models_lower:
            return "yandex"
        elif self.model in stability_models_lower:
            return "stability"
        elif self.model in tii_models_lower:
            return "tii"
        elif self.model in eleutherai_models_lower:
            return "eleutherai"
        elif self.model in mosaicml_models_lower:
            return "mosaicml"
        elif self.model in replit_models_lower:
            return "replit"
        elif self.model in minimax_models_lower:
            return "minimax"
        elif self.model in aleph_alpha_models_lower:
            return "aleph_alpha"
        elif self.model in deepseek_models_lower:
            return "deepseek"
        elif self.model in tsinghua_models_lower:
            return "tsinghua"
        elif self.model in rwkv_models_lower:
            return "rwkv"
        elif self.model in community_models_lower:
            return "community"
        else:
            supported = (list(OPENAI_MODELS.keys()) + list(OPENAI_LEGACY_MODELS.keys()) +
                        list(ANTHROPIC_MODELS.keys()) + list(ANTHROPIC_LEGACY_MODELS.keys()) +
                        list(GOOGLE_MODELS.keys()) + list(META_MODELS.keys()) + 
                        list(MISTRAL_MODELS.keys()) + list(COHERE_MODELS.keys()) +
                        list(PERPLEXITY_MODELS.keys()) + list(HUGGINGFACE_MODELS.keys()) +
                        list(AI21_MODELS.keys()) + list(TOGETHER_MODELS.keys()) +
                        list(XAI_MODELS.keys()) + list(ALIBABA_MODELS.keys()) +
                        list(BAIDU_MODELS.keys()) + list(HUAWEI_MODELS.keys()) +
                        list(YANDEX_MODELS.keys()) + list(STABILITY_MODELS.keys()) +
                        list(TII_MODELS.keys()) + list(ELEUTHERAI_MODELS.keys()) +
                        list(MOSAICML_MODELS.keys()) + list(REPLIT_MODELS.keys()) +
                        list(MINIMAX_MODELS.keys()) + list(ALEPH_ALPHA_MODELS.keys()) +
                        list(DEEPSEEK_MODELS.keys()) + list(TSINGHUA_MODELS.keys()) +
                        list(RWKV_MODELS.keys()) + list(COMMUNITY_MODELS.keys()))
            raise UnsupportedModelError(self.model, supported)
    
    def _setup_tokenizer(self) -> None:
        """Setup the appropriate tokenizer for the model."""
        if self.provider == "openai":
            if tiktoken is None:
                raise TokenizationError(
                    "tiktoken is required for OpenAI models. Install with: pip install tiktoken",
                    model=self.model
                )
            
            # Create lowercase versions for case-insensitive matching
            openai_models_lower = {k.lower(): v for k, v in OPENAI_MODELS.items()}
            openai_legacy_models_lower = {k.lower(): v for k, v in OPENAI_LEGACY_MODELS.items()}
            
            # Check both main and legacy OpenAI models
            if self.model in openai_models_lower:
                encoding_name = openai_models_lower[self.model]
            else:
                encoding_name = openai_legacy_models_lower[self.model]
            
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                raise TokenizationError(f"Failed to load tokenizer: {str(e)}", model=self.model)
        
        else:
            # For all other providers, we'll use approximation since they don't provide public tokenizers
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
                if self.tokenizer is None:
                    raise TokenizationError("Tokenizer not initialized", model=self.model)
                return len(self.tokenizer.encode(text))
            else:
                # Use approximation for all other providers
                return self._approximate_tokens(text)
        except Exception as e:
            raise TokenizationError(str(e), model=self.model, text_preview=text)
    
    def _approximate_tokens(self, text: str) -> int:
        """
        Approximate token count for non-OpenAI models.
        
        This uses a general approximation algorithm that works reasonably well
        for most LLMs, with slight adjustments based on the provider.
        """
        if not text:
            return 0
        
        # Basic character-based approximation
        char_count = len(text)
        
        # Adjust for whitespace (spaces and newlines are often separate tokens)
        whitespace_count = len(re.findall(r'\s+', text))
        
        # Adjust for punctuation (often separate tokens)
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        
        # Provider-specific adjustments
        if self.provider == "anthropic":
            # Anthropic's guidance: ~4 characters = 1 token
            base_tokens = char_count / 4
            adjustment = (whitespace_count + punctuation_count) * 0.3
        elif self.provider == "google":
            # Gemini models tend to have similar tokenization to GPT
            base_tokens = char_count / 3.8
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "meta":
            # LLaMA models have slightly different tokenization
            base_tokens = char_count / 3.5
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "mistral":
            # Mistral models similar to GPT
            base_tokens = char_count / 3.7
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "cohere":
            # Cohere models
            base_tokens = char_count / 4.2
            adjustment = (whitespace_count + punctuation_count) * 0.3
        elif self.provider == "perplexity":
            # Perplexity models similar to LLaMA
            base_tokens = char_count / 3.6
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "huggingface":
            # HuggingFace models vary, use conservative estimate
            base_tokens = char_count / 4.0
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "ai21":
            # AI21 models similar to GPT
            base_tokens = char_count / 3.8
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "together":
            # Together AI models vary, use conservative estimate
            base_tokens = char_count / 3.9
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "xai":
            # xAI Grok models similar to GPT
            base_tokens = char_count / 3.8
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "alibaba":
            # Alibaba Qwen models, Chinese-optimized
            base_tokens = char_count / 3.2
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "baidu":
            # Baidu Ernie models, Chinese-optimized
            base_tokens = char_count / 3.3
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "huawei":
            # Huawei PanGu models, Chinese-optimized
            base_tokens = char_count / 3.4
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "yandex":
            # Yandex YaLM models, Russian-optimized
            base_tokens = char_count / 3.6
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "stability":
            # Stability AI StableLM models
            base_tokens = char_count / 3.8
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "tii":
            # TII Falcon models
            base_tokens = char_count / 3.7
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "eleutherai":
            # EleutherAI models (GPT-Neo, GPT-NeoX, Pythia)
            base_tokens = char_count / 3.6
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "mosaicml":
            # MosaicML/Databricks models (MPT, DBRX)
            base_tokens = char_count / 3.7
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "replit":
            # Replit code models
            base_tokens = char_count / 3.5
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "minimax":
            # MiniMax Chinese models
            base_tokens = char_count / 3.3
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "aleph_alpha":
            # Aleph Alpha Luminous models
            base_tokens = char_count / 3.9
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "deepseek":
            # DeepSeek models
            base_tokens = char_count / 3.6
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "tsinghua":
            # Tsinghua ChatGLM models, Chinese-optimized
            base_tokens = char_count / 3.2
            adjustment = (whitespace_count + punctuation_count) * 0.2
        elif self.provider == "rwkv":
            # RWKV models
            base_tokens = char_count / 3.8
            adjustment = (whitespace_count + punctuation_count) * 0.25
        elif self.provider == "community":
            # Community fine-tuned models (Vicuna, Alpaca, etc.)
            base_tokens = char_count / 3.6
            adjustment = (whitespace_count + punctuation_count) * 0.2
        else:
            # Default approximation
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
        "openai": list(OPENAI_MODELS.keys()) + list(OPENAI_LEGACY_MODELS.keys()),
        "anthropic": list(ANTHROPIC_MODELS.keys()) + list(ANTHROPIC_LEGACY_MODELS.keys()),
        "google": list(GOOGLE_MODELS.keys()),
        "meta": list(META_MODELS.keys()),
        "mistral": list(MISTRAL_MODELS.keys()),
        "cohere": list(COHERE_MODELS.keys()),
        "perplexity": list(PERPLEXITY_MODELS.keys()),
        "huggingface": list(HUGGINGFACE_MODELS.keys()),
        "ai21": list(AI21_MODELS.keys()),
        "together": list(TOGETHER_MODELS.keys()),
        "xai": list(XAI_MODELS.keys()),
        "alibaba": list(ALIBABA_MODELS.keys()),
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
        "deepseek": list(DEEPSEEK_MODELS.keys()),
        "tsinghua": list(TSINGHUA_MODELS.keys()),
        "rwkv": list(RWKV_MODELS.keys()),
        "community": list(COMMUNITY_MODELS.keys()),
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
