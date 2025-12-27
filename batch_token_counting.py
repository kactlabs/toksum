"""
Batch Token Counting Example

This script demonstrates efficient batch processing of multiple texts for token counting.
This approach is useful for processing documents, datasets, or collections of text
where you need token counts for multiple items.

Key Features Demonstrated:
    - List comprehension for efficient batch processing
    - Consistent model usage across multiple texts
    - Simple output formatting for batch results

Use Cases:
    - Document analysis and preprocessing
    - Dataset token count analysis
    - Batch cost estimation for multiple texts
    - Content length assessment for collections

Performance Notes:
    - Creates a new TokenCounter for each text (less efficient)
    - For better performance with many texts, create one TokenCounter instance
    - Consider using TokenCounter class directly for large batches

Example Usage:
    python batch_token_counting.py

Improved Batch Processing:
    For better performance with large batches:
    
    counter = toksum.TokenCounter("gpt-3.5-turbo")
    text_counts = [counter.count(text) for text in texts]
"""

# Batch Token Counting
# Count tokens for multiple texts at once — useful for documents, datasets, etc.
import toksum

texts = ["Hello", "This is a test","count the words"]

text_counts = [toksum.count_tokens(text, model="gpt-3.5-turbo") for text in texts]
print("Batch Token Counting",text_counts)  