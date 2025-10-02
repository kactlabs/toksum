"""
Core functionality for token counting across different LLM providers.
"""

from .model_registry import is_supported_model
from .exceptions import UnsupportedModelError, TokenizationError
import re
from typing import Dict, List, Optional, Literal, Union, TYPE_CHECKING, Any
from functools import lru_cache

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

# MosaicML Models (using approximation)
MOSAICML_MODELS = {
    "mpt-7b": "mpt",  # NEW
    "mpt-7b-chat": "mpt",  # NEW
    "mpt-7b-instruct": "mpt",  # NEW
    "mpt-30b": "mpt",  # NEW
    "mpt-30b-chat": "mpt",  # NEW
    "mpt-30b-instruct": "mpt",  # NEW
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

# Anthropic Claude 3.5 Haiku Models (using approximation)
ANTHROPIC_HAIKU_MODELS = {
    "claude-3.5-haiku-20241022": "claude-3.5-haiku",  # NEW
    "claude-3-5-haiku-20241022": "claude-3.5-haiku",  # NEW (alternative naming)
}

# OpenAI O1 Models (using approximation)
OPENAI_O1_MODELS = {
    "o1-preview": "o1",  # NEW
    "o1-mini": "o1",  # NEW
    "o1-preview-2024-09-12": "o1",  # NEW
    "o1-mini-2024-09-12": "o1",  # NEW
}

# Anthropic Computer Use Models (using approximation)
ANTHROPIC_COMPUTER_USE_MODELS = {
    "claude-3-5-sonnet-20241022": "claude-3.5-computer",  # NEW
    "claude-3.5-sonnet-computer-use": "claude-3.5-computer",  # NEW
}

# Google Gemini 2.0 Models (using approximation)
GOOGLE_GEMINI_2_MODELS = {
    "gemini-2.0-flash-exp": "gemini-2.0",  # NEW
    "gemini-2.0-flash": "gemini-2.0",  # NEW
    "gemini-exp-1206": "gemini-exp",  # NEW
    "gemini-exp-1121": "gemini-exp",  # NEW
}

# Meta Llama 3.3 Models (using approximation)
META_LLAMA_33_MODELS = {
    "llama-3.3-70b": "llama-3.3",  # NEW
    "llama-3.3-70b-instruct": "llama-3.3",  # NEW
}

# Mistral Large 2 Models (using approximation)
MISTRAL_LARGE_2_MODELS = {
    "mistral-large-2": "mistral-large-2",  # NEW
    "mistral-large-2407": "mistral-large-2",  # NEW
}

# DeepSeek V3 Models (using approximation)
DEEPSEEK_V3_MODELS = {
    "deepseek-v3": "deepseek-v3",  # NEW
    "deepseek-v3-base": "deepseek-v3",  # NEW
}

# Qwen 2.5 Models (using approximation)
QWEN_25_MODELS = {
    "qwen-2.5-72b": "qwen-2.5",  # NEW
    "qwen-2.5-32b": "qwen-2.5",  # NEW
    "qwen-2.5-14b": "qwen-2.5",  # NEW
    "qwen-2.5-7b": "qwen-2.5",  # NEW
}

# Anthropic Claude 2.1 Models (using approximation)
ANTHROPIC_CLAUDE_21_MODELS = {
    "claude-2.1-200k": "claude-2.1",  # NEW
    "claude-2.1-100k": "claude-2.1",  # NEW
}

# OpenAI GPT-4 Vision Models (using approximation)
OPENAI_VISION_MODELS = {
    "gpt-4-vision": "cl100k_base",  # NEW
    "gpt-4-vision-preview-0409": "cl100k_base",  # NEW
    "gpt-4-vision-preview-1106": "cl100k_base",  # NEW
}

# Cohere Command R+ Models (using approximation)
COHERE_COMMAND_R_PLUS_MODELS = {
    "command-r-plus-04-2024": "cohere-r-plus",  # NEW
    "command-r-plus-08-2024": "cohere-r-plus",  # NEW
}

# Anthropic Claude Instant 2 Models (using approximation)
ANTHROPIC_INSTANT_2_MODELS = {
    "claude-instant-2": "claude-instant-2",  # NEW
    "claude-instant-2.0": "claude-instant-2",  # NEW
}

# Google PaLM Models (using approximation)
GOOGLE_PALM_MODELS = {
    "palm-2": "palm",  # NEW
    "palm-2-chat": "palm",  # NEW
    "palm-2-codechat": "palm",  # NEW
}

# Microsoft Models (using approximation)
MICROSOFT_MODELS = {
    "phi-3-mini": "phi",  # NEW
    "phi-3-small": "phi",  # NEW
    "phi-3-medium": "phi",  # NEW
    "phi-3.5-mini": "phi",  # NEW
}

# Amazon Bedrock Models (using approximation)
AMAZON_MODELS = {
    "titan-text-express": "titan",  # NEW
    "titan-text-lite": "titan",  # NEW
    "titan-embed-text": "titan",  # NEW
}

# Nvidia Models (using approximation)
NVIDIA_MODELS = {
    "nemotron-4-340b": "nemotron",  # NEW
    "nemotron-3-8b": "nemotron",  # NEW
}

# IBM Models (using approximation)
IBM_MODELS = {
    "granite-13b-chat": "granite",  # NEW
    "granite-13b-instruct": "granite",  # NEW
    "granite-20b-code": "granite",  # NEW
}

# Salesforce Models (using approximation)
SALESFORCE_MODELS = {
    "codegen-16b": "codegen",  # NEW
    "codegen-6b": "codegen",  # NEW
    "codegen-2b": "codegen",  # NEW
}

# BigCode Models (using approximation)
BIGCODE_MODELS = {
    "starcoder": "starcoder",  # NEW
    "starcoder2-15b": "starcoder",  # NEW
    "starcoderbase": "starcoder",  # NEW
    "starcoder2-3b": "starcoder",  # ADDED
    "starcoder2-7b": "starcoder",  # ADDED
    "starcoder-plus": "starcoder",  # ADDED
    "starcoderbase-1b": "starcoder",  # ADDED
    "starcoderbase-3b": "starcoder",  # ADDED
    "starcoderbase-7b": "starcoder",  # ADDED
}

# Anthropic Claude 3 Opus Models (using approximation)
ANTHROPIC_OPUS_MODELS = {
    "claude-3-opus-20240229": "claude-3-opus",  # ADDED
    "claude-3-opus-latest": "claude-3-opus",  # ADDED
    "claude-3-opus": "claude-3-opus",  # ADDED
}

# OpenAI GPT-4 Turbo Models (using approximation)
OPENAI_GPT4_TURBO_MODELS = {
    "gpt-4-turbo-preview": "cl100k_base",  # ADDED
    "gpt-4-0125-preview": "cl100k_base",  # ADDED
    "gpt-4-1106-preview": "cl100k_base",  # ADDED
    "gpt-4-turbo-2024-04-09": "cl100k_base",  # ADDED
}

# Anthropic Claude 3 Sonnet Models (using approximation)
ANTHROPIC_SONNET_MODELS = {
    "claude-3-sonnet-20240229": "claude-3-sonnet",  # ADDED
    "claude-3-sonnet-latest": "claude-3-sonnet",  # ADDED
    "claude-3-sonnet": "claude-3-sonnet",  # ADDED
}

# Google Gemini Pro Models (using approximation)
GOOGLE_GEMINI_PRO_MODELS = {
    "gemini-pro": "gemini-pro",  # ADDED
    "gemini-pro-vision": "gemini-pro",  # ADDED
    "gemini-1.0-pro": "gemini-pro",  # ADDED
    "gemini-1.0-pro-001": "gemini-pro",  # ADDED
    "gemini-1.0-pro-latest": "gemini-pro",  # ADDED
    "gemini-1.0-pro-vision-latest": "gemini-pro",  # ADDED
}

# Meta Llama 2 Chat Models (using approximation)
META_LLAMA2_CHAT_MODELS = {
    "llama-2-7b-chat": "llama-2-chat",  # ADDED
    "llama-2-13b-chat": "llama-2-chat",  # ADDED
    "llama-2-70b-chat": "llama-2-chat",  # ADDED
    "llama-2-7b-chat-hf": "llama-2-chat",  # ADDED
    "llama-2-13b-chat-hf": "llama-2-chat",  # ADDED
    "llama-2-70b-chat-hf": "llama-2-chat",  # ADDED
}

# Meta Llama 3 Instruct Models (using approximation)
META_LLAMA3_INSTRUCT_MODELS = {
    "llama-3-8b-instruct": "llama-3-instruct",  # ADDED
    "llama-3-70b-instruct": "llama-3-instruct",  # ADDED
    "llama-3.1-8b-instruct": "llama-3-instruct",  # ADDED
    "llama-3.1-70b-instruct": "llama-3-instruct",  # ADDED
    "llama-3.1-405b-instruct": "llama-3-instruct",  # ADDED
    "llama-3.2-1b-instruct": "llama-3-instruct",  # ADDED
    "llama-3.2-3b-instruct": "llama-3-instruct",  # ADDED
}

# Mistral Instruct Models (using approximation)
MISTRAL_INSTRUCT_MODELS = {
    "mistral-7b-instruct": "mistral-instruct",  # ADDED
    "mistral-7b-instruct-v0.1": "mistral-instruct",  # ADDED
    "mistral-7b-instruct-v0.2": "mistral-instruct",  # ADDED
    "mistral-7b-instruct-v0.3": "mistral-instruct",  # ADDED
    "mixtral-8x7b-instruct": "mistral-instruct",  # ADDED
    "mixtral-8x22b-instruct": "mistral-instruct",  # ADDED
}

# OpenAI Embedding Models (using approximation)
OPENAI_EMBEDDING_MODELS = {
    "text-embedding-ada-002": "cl100k_base",  # ADDED
    "text-embedding-3-small": "cl100k_base",  # ADDED
    "text-embedding-3-large": "cl100k_base",  # ADDED
    "text-similarity-ada-001": "r50k_base",  # ADDED
    "text-similarity-babbage-001": "r50k_base",  # ADDED
    "text-similarity-curie-001": "r50k_base",  # ADDED
    "text-similarity-davinci-001": "r50k_base",  # ADDED
}

# Databricks Models
DATABRICKS_MODELS = {
    "dbrx": "databricks", # ADDED
    "dbrx-instruct": "databricks",
    "dbrx-base": "databricks",
    "dolly-v2-12b": "databricks",
    "dolly-v2-7b": "databricks",
    "dolly-v2-3b": "databricks",
}

# Voyage AI Models
VOYAGE_MODELS = {
    "voyage-2": "voyage",
    "voyage-large-2": "voyage",
    "voyage-code-2": "voyage",
    "voyage-finance-2": "voyage",
    "voyage-law-2": "voyage",
    "voyage-multilingual-2": "voyage",
}


class TokenCounter:
    """
A token counter for various Large Language Model (LLM) providers.

This class provides functionality to count tokens for different LLMs, including OpenAI, Anthropic, Google, Meta, and many others.  It supports both individual text strings and lists of messages (for chat-like interactions).  The token counting is precise for OpenAI models using the official tiktoken library, and provides reasonable approximations for other providers.

Example Usage:

# Count tokens for a single text string
counter = TokenCounter("gpt-4")
token_count = counter.count("This is a test string.")
print(f"Token count: {token_count}")

# Count tokens for a list of messages (chat format)
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "How can I help you?"},
]
token_count = counter.count_messages(messages)
print(f"Token count (messages): {token_count}")

# Estimate cost (requires model to be in pricing dictionary)
cost = estimate_cost(token_count, "gpt-4")
print(f"Estimated cost: ${cost:.4f}")

"""
    def __init__(self, model: str):
        """
        Initialize the TokenCounter with a specific model.
        
        Args:
            model: The model name (e.g., 'gpt-4', 'claude-3-opus-20240229')
        
        Raises:
            UnsupportedModelError: If the model is not supported
            TokenizationError:
