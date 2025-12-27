Examples
========

This section provides comprehensive examples of using toksum for various scenarios.

Basic Usage Examples
--------------------

The ``examples/basic_usage.py`` file demonstrates core toksum functionality:

.. literalinclude:: ../examples/basic_usage.py
   :language: python
   :caption: Basic Usage Examples

Batch Processing
----------------

For processing multiple texts efficiently:

.. literalinclude:: ../batch_token_counting.py
   :language: python
   :caption: Batch Token Counting

Advanced Usage Patterns
------------------------

Model Comparison
~~~~~~~~~~~~~~~~

Compare token counts across different providers:

.. code-block:: python

   from toksum import TokenCounter
   
   text = "Compare tokenization across different models."
   models = ["gpt-4", "claude-3-opus", "gemini-pro", "llama-3-70b"]
   
   print(f"Text: '{text}'")
   print("Token counts by model:")
   
   for model in models:
       counter = TokenCounter(model)
       tokens = counter.count(text)
       print(f"  {model}: {tokens} tokens")

Cost Analysis
~~~~~~~~~~~~~

Analyze costs across different models and scenarios:

.. code-block:: python

   from toksum import count_tokens, estimate_cost
   
   # Sample conversation
   conversation = """
   User: What is machine learning?
   Assistant: Machine learning is a subset of artificial intelligence...
   User: Can you give me some examples?
   Assistant: Sure! Here are some common examples of machine learning...
   """
   
   models = ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-haiku"]
   
   print("Cost comparison for conversation:")
   print(f"Text length: {len(conversation)} characters")
   print()
   
   for model in models:
       try:
           tokens = count_tokens(conversation, model)
           input_cost = estimate_cost(tokens, model, input_tokens=True)
           output_cost = estimate_cost(tokens, model, input_tokens=False)
           
           print(f"{model}:")
           print(f"  Tokens: {tokens}")
           print(f"  Input cost: ${input_cost:.4f}")
           print(f"  Output cost: ${output_cost:.4f}")
           print()
       except Exception as e:
           print(f"{model}: Error - {e}")

Chat Message Processing
~~~~~~~~~~~~~~~~~~~~~~~

Process chat conversations with proper message formatting:

.. code-block:: python

   from toksum import TokenCounter
   
   def analyze_conversation(messages, model="gpt-4"):
       counter = TokenCounter(model)
       
       # Count individual messages
       individual_tokens = []
       for msg in messages:
           tokens = counter.count(msg["content"])
           individual_tokens.append(tokens)
           print(f"{msg['role']}: {tokens} tokens - '{msg['content'][:50]}...'")
       
       # Count as conversation format
       total_tokens = counter.count_messages(messages)
       individual_sum = sum(individual_tokens)
       
       print(f"\\nSummary:")
       print(f"Individual message sum: {individual_sum} tokens")
       print(f"Conversation format: {total_tokens} tokens")
       print(f"Formatting overhead: {total_tokens - individual_sum} tokens")
       
       return total_tokens
   
   # Example conversation
   messages = [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "What is the capital of France?"},
       {"role": "assistant", "content": "The capital of France is Paris."},
       {"role": "user", "content": "Tell me more about it."},
       {"role": "assistant", "content": "Paris is known for its art, fashion, gastronomy, and culture..."}
   ]
   
   analyze_conversation(messages)

Error Handling Patterns
~~~~~~~~~~~~~~~~~~~~~~~~

Robust error handling for production applications:

.. code-block:: python

   from toksum import TokenCounter, get_supported_models
   from toksum.exceptions import UnsupportedModelError, TokenizationError
   
   def safe_token_count(text, model, fallback_models=None):
       """
       Safely count tokens with fallback options.
       """
       fallback_models = fallback_models or ["gpt-3.5-turbo", "claude-3-haiku"]
       
       # Try primary model
       try:
           counter = TokenCounter(model)
           return counter.count(text), model
       except UnsupportedModelError:
           print(f"Model '{model}' not supported, trying fallbacks...")
       except TokenizationError as e:
           print(f"Tokenization failed for '{model}': {e}")
       
       # Try fallback models
       for fallback in fallback_models:
           try:
               counter = TokenCounter(fallback)
               tokens = counter.count(text)
               print(f"Using fallback model: {fallback}")
               return tokens, fallback
           except Exception as e:
               print(f"Fallback '{fallback}' also failed: {e}")
               continue
       
       # All models failed
       raise RuntimeError("All models failed for token counting")
   
   # Usage
   text = "This is a test string for token counting."
   try:
       tokens, used_model = safe_token_count(text, "unknown-model")
       print(f"Successfully counted {tokens} tokens using {used_model}")
   except RuntimeError as e:
       print(f"Failed to count tokens: {e}")

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Optimize performance for large-scale processing:

.. code-block:: python

   from toksum import TokenCounter
   import time
   
   def benchmark_approaches(texts, model="gpt-4"):
       """
       Compare different approaches for batch processing.
       """
       print(f"Benchmarking with {len(texts)} texts using {model}")
       
       # Approach 1: Create new counter each time (inefficient)
       start_time = time.time()
       results1 = []
       for text in texts:
           counter = TokenCounter(model)
           results1.append(counter.count(text))
       time1 = time.time() - start_time
       
       # Approach 2: Reuse counter (efficient)
       start_time = time.time()
       counter = TokenCounter(model)
       results2 = [counter.count(text) for text in texts]
       time2 = time.time() - start_time
       
       print(f"Approach 1 (new counter each time): {time1:.4f}s")
       print(f"Approach 2 (reuse counter): {time2:.4f}s")
       print(f"Speedup: {time1/time2:.2f}x")
       
       # Verify results are identical
       assert results1 == results2, "Results should be identical"
       return results2
   
   # Test with sample texts
   sample_texts = [
       "Short text",
       "Medium length text with more content to tokenize",
       "Much longer text that contains multiple sentences and should demonstrate the performance difference between approaches when processing many texts in batch operations."
   ] * 100  # Repeat to make timing differences visible
   
   benchmark_approaches(sample_texts)

Integration Examples
--------------------

Web Application Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example Flask application with toksum integration:

.. code-block:: python

   from flask import Flask, request, jsonify
   from toksum import TokenCounter, get_supported_models
   from toksum.exceptions import ToksumError
   
   app = Flask(__name__)
   
   @app.route('/count', methods=['POST'])
   def count_tokens():
       try:
           data = request.json
           text = data.get('text', '')
           model = data.get('model', 'gpt-3.5-turbo')
           
           counter = TokenCounter(model)
           tokens = counter.count(text)
           
           return jsonify({
               'tokens': tokens,
               'model': model,
               'text_length': len(text)
           })
       except ToksumError as e:
           return jsonify({'error': str(e)}), 400
       except Exception as e:
           return jsonify({'error': f'Unexpected error: {e}'}), 500
   
   @app.route('/models', methods=['GET'])
   def list_models():
       try:
           models = get_supported_models()
           return jsonify(models)
       except Exception as e:
           return jsonify({'error': str(e)}), 500
   
   if __name__ == '__main__':
       app.run(debug=True)

Data Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

Example data processing pipeline with toksum:

.. code-block:: python

   import pandas as pd
   from toksum import TokenCounter
   from toksum.exceptions import TokenizationError
   
   def process_dataset(df, text_column, model="gpt-3.5-turbo"):
       """
       Add token counts to a pandas DataFrame.
       """
       counter = TokenCounter(model)
       
       def safe_count(text):
           try:
               if pd.isna(text) or text == '':
                   return 0
               return counter.count(str(text))
           except TokenizationError:
               return -1  # Mark as error
       
       # Add token count column
       df[f'{text_column}_tokens'] = df[text_column].apply(safe_count)
       
       # Add statistics
       valid_counts = df[df[f'{text_column}_tokens'] >= 0]
       stats = {
           'total_rows': len(df),
           'valid_counts': len(valid_counts),
           'errors': len(df) - len(valid_counts),
           'avg_tokens': valid_counts[f'{text_column}_tokens'].mean(),
           'max_tokens': valid_counts[f'{text_column}_tokens'].max(),
           'total_tokens': valid_counts[f'{text_column}_tokens'].sum()
       }
       
       return df, stats
   
   # Example usage
   # df = pd.read_csv('your_dataset.csv')
   # processed_df, statistics = process_dataset(df, 'content_column')
   # print(f"Processing statistics: {statistics}")

These examples demonstrate various ways to integrate toksum into different types of applications and workflows.