# Batch Token Counting
# Count tokens for multiple texts at once — useful for documents, datasets, etc.
import toksum

texts = ["Hello", "This is a test","count the words"]

text_counts = [toksum.count_tokens(text, model="gpt-3.5-turbo") for text in texts]
print("Batch Token Counting",text_counts)  