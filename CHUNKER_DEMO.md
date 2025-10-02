# SmartChunker Demo and Testing

This directory contains demonstration and test files for the SmartChunker functionality in toksum.

## Files

### `test_chunker_demo.py`
A comprehensive demonstration script that shows how SmartChunker works with different types of content:

- **Sentence Chunking**: Splits text at sentence boundaries while respecting token limits
- **Paragraph Chunking**: Splits text at paragraph boundaries 
- **Python Code Chunking**: Intelligently splits Python code at function/class boundaries
- **Mixed Content**: Shows how to handle documentation with embedded code
- **Token Limit Comparison**: Demonstrates how different token limits affect chunking

**Run with:**
```bash
python test_chunker_demo.py
```

### `simple_chunker_test.py`
A focused test suite that verifies core functionality:

- Basic chunking operations
- Edge cases (empty input, single sentences, long text)
- Different model compatibility

**Run with:**
```bash
python simple_chunker_test.py
```

## SmartChunker Features

### 1. Sentence-Based Chunking
```python
from toksum import SmartChunker

chunker = SmartChunker("gpt-4", max_tokens=50)
text = "First sentence. Second sentence. Third sentence."
chunks = chunker.chunk_by_sentences(text)
```

### 2. Paragraph-Based Chunking
```python
chunker = SmartChunker("gpt-4", max_tokens=100)
text = "Para 1.\n\nPara 2.\n\nPara 3."
chunks = chunker.chunk_by_paragraphs(text)
```

### 3. Code Chunking
```python
chunker = SmartChunker("gpt-4", max_tokens=150)
code = '''
def function1():
    pass

class MyClass:
    def method(self):
        pass
'''
chunks = chunker.chunk_code(code, "python")
```

## How It Works

1. **Semantic Boundaries**: The chunker respects natural text boundaries (sentences, paragraphs, code blocks)
2. **Token Counting**: Uses the specified model's tokenizer to count tokens accurately
3. **Intelligent Splitting**: Tries to combine smaller units until token limit is reached
4. **Fallback Handling**: For non-Python code, falls back to paragraph chunking

## Supported Models

The chunker works with any model supported by toksum, including:
- OpenAI models (gpt-4, gpt-3.5-turbo, etc.)
- Anthropic models (claude-3-opus, claude-3-sonnet, etc.)
- Google models (gemini-pro, etc.)
- And many more...

## Example Output

When you run the demo, you'll see output like:

```
=== Sentence Chunking Demo ===
Original text: This is the first sentence. This is the second sentence...
Number of chunks: 3

Chunk 1:
  Content: This is the first sentence. This is the second sentence.
  Token count: gpt-4 - estimated

Chunk 2:
  Content: This is the third sentence.
  Token count: gpt-4 - estimated
...
```

## Use Cases

- **Document Processing**: Split large documents for LLM processing
- **Code Analysis**: Break down code files for review or analysis
- **Content Preparation**: Prepare text for embedding or summarization
- **API Optimization**: Ensure requests stay within token limits