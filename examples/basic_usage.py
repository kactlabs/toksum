"""
Basic usage examples for the toksum library.
"""

from toksum import TokenCounter, count_tokens, get_supported_models, estimate_cost, SmartChunker

def main():
    print("=== toksum Library Examples ===\n")
    
    # Example 1: Quick token counting
    print("1. Quick token counting:")
    text = "Hello, world! This is a sample text for token counting."
    
    gpt4_tokens = count_tokens(text, "gpt-4")
    claude_tokens = count_tokens(text, "claude-3-opus-20240229")
    
    print(f"Text: '{text}'")
    print(f"GPT-4 tokens: {gpt4_tokens}")
    print(f"Claude-3 Opus tokens: {claude_tokens}")
    print()
    
    # Example 2: Using TokenCounter class
    print("2. Using TokenCounter class:")
    counter = TokenCounter("gpt-3.5-turbo")
    
    texts = [
        "Short text",
        "This is a medium-length text with some more words.",
        "This is a much longer text that contains multiple sentences. It should demonstrate how token counts scale with text length. The tokenizer will break this down into individual tokens based on the model's vocabulary."
    ]
    
    for i, text in enumerate(texts, 1):
        tokens = counter.count(text)
        print(f"Text {i} ({len(text)} chars): {tokens} tokens")
    print()
    
    # Example 3: Counting tokens in chat messages
    print("3. Chat message token counting:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about it."}
    ]
    
    gpt_counter = TokenCounter("gpt-4")
    claude_counter = TokenCounter("claude-3-sonnet-20240229")
    
    gpt_total = gpt_counter.count_messages(messages)
    claude_total = claude_counter.count_messages(messages)
    
    print("Chat conversation:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    print(f"\nTotal tokens (GPT-4): {gpt_total}")
    print(f"Total tokens (Claude-3 Sonnet): {claude_total}")
    print()
    
    # Example 4: Cost estimation
    print("4. Cost estimation:")
    sample_text = "This is a sample text for cost estimation. " * 100  # Repeat to get more tokens
    
    models_to_test = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
    
    print(f"Sample text length: {len(sample_text)} characters")
    print("\nToken counts and estimated costs:")
    
    for model in models_to_test:
        try:
            tokens = count_tokens(sample_text, model)
            input_cost = estimate_cost(tokens, model, input_tokens=True)
            output_cost = estimate_cost(tokens, model, input_tokens=False)
            
            print(f"{model}:")
            print(f"  Tokens: {tokens}")
            print(f"  Input cost: ${input_cost:.4f}")
            print(f"  Output cost: ${output_cost:.4f}")
        except Exception as e:
            print(f"{model}: Error - {e}")
    print()
    
    # Example 5: List supported models
    print("5. Supported models:")
    models = get_supported_models()
    
    for provider, model_list in models.items():
        print(f"{provider.upper()} models:")
        for model in model_list[:5]:  # Show first 5 models
            print(f"  - {model}")
        if len(model_list) > 5:
            print(f"  ... and {len(model_list) - 5} more")
        print()
    
    # Example 6: Comparing different text types
    print("6. Token counting for different text types:")
    
    text_samples = {
        "Simple English": "The quick brown fox jumps over the lazy dog.",
        "Technical": "import numpy as np\narray = np.zeros((10, 10))\nprint(array.shape)",
        "With Numbers": "The year 2024 has 365 days, and the temperature is 23.5°C.",
        "Punctuation Heavy": "Hello!!! How are you??? I'm fine... Really, really fine!!!",
        "Mixed Case": "CamelCaseVariable = SomeFunction(parameterOne, parameterTwo)"
    }
    
    counter = TokenCounter("gpt-4")
    
    for text_type, text in text_samples.items():
        tokens = counter.count(text)
        chars = len(text)
        ratio = chars / tokens if tokens > 0 else 0
        print(f"{text_type}:")
        print(f"  Text: '{text}'")
        print(f"  Tokens: {tokens}, Characters: {chars}, Chars/Token: {ratio:.2f}")
        print()
    
    # Example 7: Smart text chunking
    print("7. Smart text chunking:")
    
    # Sample long text for demonstration
    long_text = """
Natural language processing (NLP) is a subfield of artificial intelligence, computer science, and linguistics concerned with the interaction between computers and human language. In practice, this means enabling computers to understand, interpret, and respond to human language in a meaningful way.

NLP draws from various fields including computational linguistics, machine learning, and natural language understanding. It has applications in machine translation, sentiment analysis, question answering, and chatbots.

The field has seen significant advancements with deep learning models like transformers and large language models such as GPT and BERT.
"""
    
    # Sample code for chunking
    sample_code = """
class SmartChunker:
    def __init__(self, model, max_tokens):
        self.model = model
        self.max_tokens = max_tokens
    
    def chunk_by_sentences(self, text):
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            test_chunk = current_chunk + ("" if not current_chunk else " ") + sentence
            if count_tokens(test_chunk, self.model) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def calculate_factorial(n):
    if n == 0:
        return 1
    return n * calculate_factorial(n-1)
"""
    
    # Create chunker instance
    chunker = SmartChunker("gpt-4", max_tokens=50)
    
    # Chunk by sentences
    print("Chunking by sentences:")
    sentence_chunks = chunker.chunk_by_sentences(long_text)
    print(f"Number of sentence chunks: {len(sentence_chunks)}")
    for i, chunk in enumerate(sentence_chunks, 1):
        tokens = count_tokens(chunk, "gpt-4")
        print(f"  Chunk {i} ({tokens} tokens): {chunk[:50]}...")
    print()
    
    # Chunk by paragraphs
    print("Chunking by paragraphs:")
    paragraph_chunks = chunker.chunk_by_paragraphs(long_text)
    print(f"Number of paragraph chunks: {len(paragraph_chunks)}")
    for i, chunk in enumerate(paragraph_chunks, 1):
        tokens = count_tokens(chunk, "gpt-4")
        print(f"  Chunk {i} ({tokens} tokens): {chunk[:50]}...")
    print()
    
    # Chunk code
    print("Chunking code:")
    code_chunks = chunker.chunk_code(sample_code, "python")
    print(f"Number of code chunks: {len(code_chunks)}")
    for i, chunk in enumerate(code_chunks, 1):
        tokens = count_tokens(chunk, "gpt-4")
        print(f"  Code Chunk {i} ({tokens} tokens):")
        print(f"    {chunk[:100]}...")
    print()


if __name__ == "__main__":
    main()
