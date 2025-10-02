"""
Basic usage examples for the toksum library.
"""

from toksum import TokenCounter, count_tokens, get_supported_models, estimate_cost, UnsupportedModelError
from typing import List, Dict, Any

def main() -> None:
    """
    Main function to demonstrate various functionalities of the toksum library.
    Includes examples for quick token counting, TokenCounter class usage,
    chat message token counting, cost estimation, listing supported models,
    and comparing token counts for different text types.
    """
    print("=== toksum Library Examples ===\n")
    
    # Example 1: Quick token counting
    print("1. Quick token counting:")
    text: str = "Hello, world! This is a sample text for token counting."
    
    try:
        gpt4_tokens: int = count_tokens(text, "gpt-4")
        claude_tokens: int = count_tokens(text, "claude-3-opus-20240229")
        
        print(f"Text: '{text}'")
        print(f"GPT-4 tokens: {gpt4_tokens}")
        print(f"Claude-3 Opus tokens: {claude_tokens}")
    except UnsupportedModelError as e:
        print(f"Error in quick token counting: {e}")
    print()
    
    # Example 2: Using TokenCounter class
    print("2. Using TokenCounter class:")
    try:
        counter: TokenCounter = TokenCounter("gpt-3.5-turbo")
        
        texts: List[str] = [
            "Short text",
            "This is a medium-length text with some more words.",
            "This is a much longer text that contains multiple sentences. It should demonstrate how token counts scale with text length. The tokenizer will break this down into individual tokens based on the model's vocabulary."
        ]
        
        for i, t in enumerate(texts, 1):
            tokens: int = counter.count(t)
            print(f"Text {i} ({len(t)} chars): {tokens} tokens")
    except UnsupportedModelError as e:
        print(f"Error using TokenCounter class: {e}")
    print()
    
    # Example 3: Counting tokens in chat messages
    print("3. Chat message token counting:")
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about it."}
    ]
    
    try:
        gpt_counter: TokenCounter = TokenCounter("gpt-4")
        claude_counter: TokenCounter = TokenCounter("claude-3-opus-20240229") # Changed to opus for consistency
        
        gpt_total: int = gpt_counter.count_messages(messages)
        claude_total: int = claude_counter.count_messages(messages)
        
        print("Chat conversation:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content']}")
        
        print(f"\nTotal tokens (GPT-4): {gpt_total}")
        print(f"Total tokens (Claude-3 Opus): {claude_total}")
    except UnsupportedModelError as e:
        print(f"Error in chat message token counting: {e}")
    print()
    
    # Example 4: Cost estimation
    print("4. Cost estimation:")
    sample_text: str = "This is a sample text for cost estimation. " * 100  # Repeat to get more tokens
    
    models_to_test: List[str] = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus-20240229", "gemini-1.5-flash"] # Changed claude-3-haiku-20240307 to gemini-1.5-flash
    
    print(f"Sample text length: {len(sample_text)} characters")
    print("\nToken counts and estimated costs:")
    
    for model in models_to_test:
        try:
            tokens: int = count_tokens(sample_text, model)
            input_cost: float = estimate_cost(tokens, model, input_tokens=True)
            output_cost: float = estimate_cost(tokens, model, input_tokens=False)
            
            print(f"{model}:")
            print(f"  Tokens: {tokens}")
            print(f"  Input cost: ${input_cost:.4f}")
            print(f"  Output cost: ${output_cost:.4f}")
        except UnsupportedModelError as e:
            print(f"{model}: Error - {e}")
        except Exception as e:
            print(f"{model}: General Error - {e}")
    print()
    
    # Example 5: List supported models
    print("5. Supported models:")
    models_dict: Dict[str, List[str]] = get_supported_models()
    
    for provider, model_list in models_dict.items():
        print(f"{provider.upper()} models:")
        for model_name in model_list[:5]:  # Show first 5 models
            print(f"  - {model_name}")
        if len(model_list) > 5:
            print(f"  ... and {len(model_list) - 5} more")
        print()
    
    # Example 6: Comparing different text types
    print("6. Token counting for different text types:")
    
    text_samples: Dict[str, str] = {
        "Simple English": "The quick brown fox jumps over the lazy dog.",
        "Technical": "import numpy as np\narray = np.zeros((10, 10))\nprint(array.shape)",
        "With Numbers": "The year 2024 has 365 days, and the temperature is 23.5°C.",
        "Punctuation Heavy": "Hello!!! How are you??? I'm fine... Really, really fine!!!",
        "Mixed Case": "CamelCaseVariable = SomeFunction(parameterOne, parameterTwo)"
    }
    
    try:
        counter = TokenCounter("gpt-4")
        
        for text_type, t in text_samples.items():
            tokens = counter.count(t)
            chars = len(t)
            ratio = chars / tokens if tokens > 0 else 0
            print(f"{text_type}:")
            print(f"  Text: '{t}'")
            print(f"  Tokens: {tokens}, Characters: {chars}, Chars/Token: {ratio:.2f}")
            print()
    except UnsupportedModelError as e:
        print(f"Error in comparing different text types: {e}")


if __name__ == "__main__":
    main()
