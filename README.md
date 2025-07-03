# toksum

A Python library for counting tokens in text for major Large Language Models (LLMs).

[![PyPI version](https://badge.fury.io/py/toksum.svg)](https://badge.fury.io/py/toksum)
[![Python Support](https://img.shields.io/pypi/pyversions/toksum.svg)](https://pypi.org/project/toksum/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Multi-LLM Support**: Count tokens for OpenAI GPT models and Anthropic Claude models
- **Accurate Tokenization**: Uses official tokenizers (tiktoken for OpenAI) and smart approximation for Claude
- **Chat Message Support**: Count tokens in chat/conversation format with proper message overhead calculation
- **Cost Estimation**: Estimate API costs based on token counts and current pricing
- **Easy to Use**: Simple API with both functional and object-oriented interfaces
- **Well Tested**: Comprehensive test suite with high coverage
- **Type Hints**: Full type annotation support for better IDE experience

## Supported Models

### OpenAI Models
- GPT-4 (all variants including gpt-4, gpt-4-32k, gpt-4-turbo-preview, etc.)
- GPT-3.5 Turbo (all variants)
- Legacy models (text-davinci-003, text-davinci-002, etc.)

### Anthropic Models
- Claude-3 (Opus, Sonnet, Haiku)
- Claude-2 (2.1, 2.0)
- Claude Instant (all variants)

## Installation

```bash
pip install toksum
```

### Optional Dependencies

For OpenAI models, you'll need `tiktoken`:
```bash
pip install tiktoken
```

For Anthropic models, the library uses built-in approximation (no additional dependencies required).

## Quick Start

```python
from toksum import count_tokens, TokenCounter

# Quick token counting
tokens = count_tokens("Hello, world!", "gpt-4")
print(f"Token count: {tokens}")

# Using TokenCounter class
counter = TokenCounter("gpt-4")
tokens = counter.count("Hello, world!")
print(f"Token count: {tokens}")
```

## Usage Examples

### Basic Token Counting

```python
from toksum import count_tokens

# Count tokens for different models
text = "The quick brown fox jumps over the lazy dog."

gpt4_tokens = count_tokens(text, "gpt-4")
gpt35_tokens = count_tokens(text, "gpt-3.5-turbo")
claude_tokens = count_tokens(text, "claude-3-opus-20240229")

print(f"GPT-4: {gpt4_tokens} tokens")
print(f"GPT-3.5: {gpt35_tokens} tokens") 
print(f"Claude-3 Opus: {claude_tokens} tokens")
```

### Using TokenCounter Class

```python
from toksum import TokenCounter

# Create a counter for a specific model
counter = TokenCounter("gpt-4")

# Count tokens for multiple texts
texts = [
    "Short text",
    "This is a longer text with more words and complexity.",
    "Very long text..." * 100
]

for text in texts:
    tokens = counter.count(text)
    print(f"'{text[:30]}...': {tokens} tokens")
```

### Chat Message Token Counting

```python
from toksum import TokenCounter

counter = TokenCounter("gpt-4")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]

total_tokens = counter.count_messages(messages)
print(f"Total conversation tokens: {total_tokens}")
```

### Cost Estimation

```python
from toksum import count_tokens, estimate_cost

text = "Your text here..." * 1000  # Large text
model = "gpt-4"

tokens = count_tokens(text, model)
input_cost = estimate_cost(tokens, model, input_tokens=True)
output_cost = estimate_cost(tokens, model, input_tokens=False)

print(f"Tokens: {tokens}")
print(f"Estimated input cost: ${input_cost:.4f}")
print(f"Estimated output cost: ${output_cost:.4f}")
```

### List Supported Models

```python
from toksum import get_supported_models

models = get_supported_models()
print("Supported models:")
for provider, model_list in models.items():
    print(f"\n{provider.upper()}:")
    for model in model_list:
        print(f"  - {model}")
```

## API Reference

### Functions

#### `count_tokens(text: str, model: str) -> int`
Count tokens in text for a specific model.

**Parameters:**
- `text`: The text to count tokens for
- `model`: The model name (e.g., "gpt-4", "claude-3-opus-20240229")

**Returns:** Number of tokens as integer

#### `get_supported_models() -> Dict[str, List[str]]`
Get dictionary of supported models by provider.

**Returns:** Dictionary with provider names as keys and model lists as values

#### `estimate_cost(token_count: int, model: str, input_tokens: bool = True) -> float`
Estimate cost for given token count and model.

**Parameters:**
- `token_count`: Number of tokens
- `model`: Model name
- `input_tokens`: Whether tokens are input (True) or output (False)

**Returns:** Estimated cost in USD

### Classes

#### `TokenCounter(model: str)`
Token counter for a specific model.

**Methods:**
- `count(text: str) -> int`: Count tokens in text
- `count_messages(messages: List[Dict[str, str]]) -> int`: Count tokens in chat messages

### Exceptions

#### `UnsupportedModelError`
Raised when an unsupported model is specified.

#### `TokenizationError`
Raised when tokenization fails.

## How It Works

### OpenAI Models
Uses the official `tiktoken` library to get exact token counts using the same tokenizer as OpenAI's API.

### Anthropic Models
Uses a smart approximation algorithm based on:
- Character count analysis
- Whitespace and punctuation detection
- Anthropic's guidance of ~4 characters per token
- Adjustments for different text patterns

The approximation is typically within 10-20% of actual token counts for English text.

## Development

### Setup Development Environment

```bash
git clone https://github.com/kactlabs/toksum.git
cd toksum
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=toksum --cov-report=html
```

### Code Formatting

```bash
black toksum tests examples
```

### Type Checking

```bash
mypy toksum
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v0.2.0
- Added 10 new models:
  - OpenAI: gpt-4-turbo, gpt-4-turbo-2024-04-09, gpt-4o, gpt-4o-2024-05-13, gpt-4o-mini, gpt-4o-mini-2024-07-18
  - Anthropic: claude-3.5-sonnet-20240620, claude-3.5-sonnet-20241022, claude-3.5-haiku-20241022, claude-3-5-sonnet-20240620
- Updated cost estimation for new models
- Enhanced model support (now 43 total models)

### v0.1.0
- Initial release
- Support for OpenAI GPT models and Anthropic Claude models
- Token counting for text and chat messages
- Cost estimation functionality
- Comprehensive test suite

## Acknowledgments

- [tiktoken](https://github.com/openai/tiktoken) for OpenAI tokenization
- [Anthropic](https://www.anthropic.com/) for Claude model guidance
- The open-source community for inspiration and best practices
