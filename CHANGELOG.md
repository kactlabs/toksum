# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2025-01-04

### Added
- 139 new models across 16 new providers, bringing total to 251+ models
- **New Providers:**
  - **xAI (4 models):** grok-1, grok-1.5, grok-2, grok-beta
  - **Alibaba (16 models):** qwen-1.5 series (0.5b to 110b), qwen-2 series, qwen-vl variants
  - **Baidu (8 models):** ernie-4.0, ernie-3.5, ernie-3.0, ernie-speed, ernie-lite, ernie-tiny, ernie-bot, ernie-bot-4
  - **Huawei (5 models):** pangu-alpha series (2.6b, 13b, 200b), pangu-coder variants
  - **Yandex (4 models):** yalm-100b, yalm-200b, yagpt, yagpt-2
  - **Stability AI (7 models):** stablelm-alpha, stablelm-base-alpha, stablelm-tuned-alpha, stablelm-zephyr variants
  - **TII (6 models):** falcon series (7b, 40b, 180b) with instruct and chat variants
  - **EleutherAI (12 models):** gpt-neo series, gpt-neox-20b, pythia series (70m to 12b)
  - **MosaicML/Databricks (8 models):** mpt series (7b, 30b) with chat/instruct variants, dbrx models
  - **Replit (3 models):** replit-code series (v1, v1.5, v2)
  - **MiniMax (5 models):** abab series (5.5 to 6.5) chat models
  - **Aleph Alpha (4 models):** luminous series (base, extended, supreme, supreme-control)
  - **DeepSeek (8 models):** deepseek-coder series, deepseek-vl series, deepseek-llm series
  - **Tsinghua KEG Lab (5 models):** chatglm series (6b variants), glm-4, glm-4v
  - **RWKV (7 models):** rwkv-4 series (169m to 14b), rwkv-5-world
  - **Community Fine-tuned (13 models):** vicuna, alpaca, wizardlm, orca-mini, zephyr variants

### Enhanced
- Provider-specific tokenization approximations for all new providers
- Optimized approximations for Chinese models (Alibaba, Baidu, Huawei, MiniMax, Tsinghua)
- Optimized approximations for Russian models (Yandex)
- Specialized approximations for code models (Replit, DeepSeek-Coder, Huawei PanGu-Coder)
- Enhanced model detection and validation across 26 total providers
- Comprehensive case-insensitive model name matching

### Changed
- Expanded to 26 total providers
- Total model support increased from 112 to 251+ models
- Updated comprehensive provider categorization and documentation

## [0.6.0] - 2025-01-04

### Fixed
- **Type Safety Improvements:** Fixed all mypy type errors for better code quality and IDE support
- **Exception Handling:** Added proper Optional type annotations for function parameters with None defaults
- **Import Handling:** Improved conditional imports using TYPE_CHECKING to avoid mypy issues with optional dependencies
- **Runtime Safety:** Added runtime checks to ensure tokenizer is properly initialized before use

### Enhanced
- Full mypy compliance with `--ignore-missing-imports` flag
- Better type hints throughout the codebase
- Improved developer experience with proper type checking support

### Technical Changes
- Updated `toksum/exceptions.py` with proper `Optional[List[str]]` and `Optional[str]` type annotations
- Restructured imports in `toksum/core.py` using `TYPE_CHECKING` pattern
- Added explicit tokenizer null checks before method calls
- Enhanced type safety without breaking existing functionality

## [0.5.0] - 2025-01-04

### Added
- 28 additional models across 4 new providers, bringing total to 112 models
- **New Providers:**
  - **Perplexity (5 models):** pplx-7b-online, pplx-70b-online, pplx-7b-chat, pplx-70b-chat, codellama-34b-instruct
  - **Hugging Face (5 models):** microsoft/DialoGPT-medium, microsoft/DialoGPT-large, facebook/blenderbot-400M-distill, facebook/blenderbot-1B-distill, facebook/blenderbot-3B
  - **AI21 (4 models):** j2-light, j2-mid, j2-ultra, j2-jumbo-instruct
  - **Together AI (3 models):** togethercomputer/RedPajama-INCITE-Chat-3B-v1, togethercomputer/RedPajama-INCITE-Chat-7B-v1, NousResearch/Nous-Hermes-Llama2-13b
- **Additional OpenAI Models (6 new):**
  - `gpt-3` - Legacy GPT-3 model
  - `text-embedding-ada-002` - Ada embedding model
  - `text-embedding-3-small` - Small embedding model
  - `text-embedding-3-large` - Large embedding model
  - `gpt-4-base` - Base GPT-4 model
  - `gpt-3.5-turbo-instruct-0914` - Specific instruct variant
- **Additional Anthropic Models (3 new):**
  - `claude-1` - Legacy Claude model
  - `claude-1.3` - Claude 1.3 model
  - `claude-1.3-100k` - Claude 1.3 with 100k context
- **Additional Cohere Models (2 new):**
  - `command-r-08-2024` - August 2024 Command-R variant
  - `command-r-plus-08-2024` - August 2024 Command-R Plus variant

### Enhanced
- Case-insensitive model name matching for better compatibility
- Provider-specific tokenization approximations for new providers
- Improved model detection and validation across all providers
- Enhanced error handling for complex model names with special characters

### Changed
- Expanded to 10 total providers
- Total model support increased from 84 to 112 models
- Updated documentation with comprehensive provider listings

## [0.4.0] - 2025-01-04

### Added
- 30 new models across all providers, bringing total to 84 models
- **OpenAI Models (5 new):**
  - `gpt-4o-2024-08-06` - Latest GPT-4o variant
  - `gpt-4o-2024-11-20` - Newest GPT-4o release
  - `gpt-4-1106-vision-preview` - Vision-enabled GPT-4 preview
  - `gpt-3.5-turbo-instruct` - Instruction-tuned GPT-3.5
- **Anthropic Models (4 new):**
  - `claude-3-opus` - Short name for Claude-3 Opus
  - `claude-3-sonnet` - Short name for Claude-3 Sonnet
  - `claude-3-haiku` - Short name for Claude-3 Haiku
  - `claude-instant` - Short name for Claude Instant
- **Google Models (5 new):**
  - `gemini-1.5-pro-latest` - Latest Gemini 1.5 Pro
  - `gemini-1.5-flash-latest` - Latest Gemini 1.5 Flash
  - `gemini-1.0-pro` - Gemini 1.0 Pro
  - `gemini-1.0-pro-vision` - Vision-enabled Gemini 1.0 Pro
  - `gemini-ultra` - Gemini Ultra model
- **Meta Models (7 new):**
  - `llama-3-8b` - LLaMA 3 8B parameter model
  - `llama-3-70b` - LLaMA 3 70B parameter model
  - `llama-3.1-8b` - LLaMA 3.1 8B parameter model
  - `llama-3.1-70b` - LLaMA 3.1 70B parameter model
  - `llama-3.1-405b` - LLaMA 3.1 405B parameter model
  - `llama-3.2-1b` - LLaMA 3.2 1B parameter model
  - `llama-3.2-3b` - LLaMA 3.2 3B parameter model
- **Mistral Models (6 new):**
  - `mistral-large` - Mistral Large model
  - `mistral-medium` - Mistral Medium model
  - `mistral-small` - Mistral Small model
  - `mistral-tiny` - Mistral Tiny model
  - `mixtral-8x7b` - Mixtral 8x7B mixture of experts
  - `mixtral-8x22b` - Mixtral 8x22B mixture of experts
- **Cohere Models (4 new):**
  - `command-light` - Lightweight Command model
  - `command-nightly` - Nightly Command model
  - `command-r` - Command-R model
  - `command-r-plus` - Enhanced Command-R model

### Enhanced
- Provider-specific tokenization approximations for better accuracy
- Improved model detection and validation
- Enhanced error handling for unsupported models

### Changed
- Updated model counts in documentation
- Improved README with comprehensive model listings
- Enhanced provider categorization

## [0.3.0] - 2024-12-01

### Added
- 10 new models from 4 new providers
- **Google Models (4 new):**
  - `gemini-pro` - Gemini Pro model
  - `gemini-pro-vision` - Vision-enabled Gemini Pro
  - `gemini-1.5-pro` - Gemini 1.5 Pro model
  - `gemini-1.5-flash` - Gemini 1.5 Flash model
- **Meta Models (3 new):**
  - `llama-2-7b` - LLaMA-2 7B parameter model
  - `llama-2-13b` - LLaMA-2 13B parameter model
  - `llama-2-70b` - LLaMA-2 70B parameter model
- **Mistral Models (2 new):**
  - `mistral-7b` - Mistral 7B model
  - `mistral-8x7b` - Mistral 8x7B mixture of experts
- **Cohere Models (1 new):**
  - `command` - Cohere Command model

### Enhanced
- Expanded to 6 total providers (OpenAI, Anthropic, Google, Meta, Mistral, Cohere)
- Provider-specific approximation algorithms
- Enhanced tokenization accuracy for non-OpenAI models

### Changed
- Total model support increased from 43 to 54 models

## [0.2.0] - 2024-10-01

### Added
- 10 new OpenAI and Anthropic models
- **OpenAI Models (6 new):**
  - `gpt-4-turbo` - GPT-4 Turbo model
  - `gpt-4-turbo-2024-04-09` - Specific GPT-4 Turbo release
  - `gpt-4o` - GPT-4o model
  - `gpt-4o-2024-05-13` - Specific GPT-4o release
  - `gpt-4o-mini` - Compact GPT-4o model
  - `gpt-4o-mini-2024-07-18` - Specific GPT-4o mini release
- **Anthropic Models (4 new):**
  - `claude-3.5-sonnet-20240620` - Claude 3.5 Sonnet
  - `claude-3.5-sonnet-20241022` - Updated Claude 3.5 Sonnet
  - `claude-3.5-haiku-20241022` - Claude 3.5 Haiku
  - `claude-3-5-sonnet-20240620` - Alternative naming for Claude 3.5

### Enhanced
- Updated cost estimation for new models
- Improved model validation and error handling

### Changed
- Total model support increased from 33 to 43 models

## [0.1.0] - 2024-08-01

### Added
- Initial release with 33 models
- Support for OpenAI GPT models (31 models)
- Support for Anthropic Claude models (12 models)
- Token counting for text and chat messages
- Cost estimation functionality
- Comprehensive test suite
- CLI interface
- Type hints and full documentation

### Features
- Accurate tokenization using official tiktoken for OpenAI models
- Smart approximation algorithm for Anthropic models
- Chat message formatting overhead calculation
- Cost estimation based on current pricing
- Easy-to-use functional and object-oriented APIs

[0.7.0]: https://github.com/kactlabs/toksum/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/kactlabs/toksum/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/kactlabs/toksum/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/kactlabs/toksum/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kactlabs/toksum/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kactlabs/toksum/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kactlabs/toksum/releases/tag/v0.1.0
