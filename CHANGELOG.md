# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.4.0]: https://github.com/kactlabs/toksum/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kactlabs/toksum/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kactlabs/toksum/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kactlabs/toksum/releases/tag/v0.1.0
