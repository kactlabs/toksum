# Batch Token Counting
# Count tokens for multiple texts at once — useful for documents, datasets, etc.
from toksum import count_tokens
from typing import List

def main():
    """
    Demonstrates batch token counting using the toksum library.
    """
    texts: List[str] = ["Hello", "This is a test", "count the words"]
    model_name: str = "gemini-1.5-flash"

    print(f"Counting tokens for texts using model: {model_name}")
    text_counts: List[int] = [count_tokens(text, model=model_name) for text in texts]
    
    for i, text in enumerate(texts):
        print(f"  Text: '{text}' -> Tokens: {text_counts[i]}")
    print(f"\nBatch Token Counts: {text_counts}")

if __name__ == "__main__":
    main()
