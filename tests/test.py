from toksum import count_tokens, TokenCounter

# Quick token counting
tokens = count_tokens("Hello, world!", "gpt-4")
print(f"Token count: {tokens}")

# Using TokenCounter class
counter = TokenCounter("gpt-4")
tokens = counter.count("Hello, world!")
print(f"Token count: {tokens}")