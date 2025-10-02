# Batch Token Counting
# Count tokens for multiple texts at once — useful for documents, datasets, etc.
import toksum

texts = ["def python():", "This is a test","count the words"]

gemini_models = ["gemini-1.5-pro", "gpt-3.5-turbo", "gemini-2.0-flash-exp"]

for model in gemini_models:
    text_counts = [toksum.count_tokens(text, model=model) for text in texts]
    print(f"Batch Token Counting for {model}: {text_counts}")
