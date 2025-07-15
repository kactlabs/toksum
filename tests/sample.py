import sys
sys.path.insert(0, "/home/meeran/pypi/toksum") 

from toksum.core import count_tokens, estimate_cost

# Example text and model
text = "This is a sample text to count tokens for."
model_name = "gpt-4o-mini" # Using a model with defined pricing

# 1. Count tokens
tokens = count_tokens(text, model_name)
print(f"Text: '{text}'")
print(f"Model: {model_name}")
print(f"Token count: {tokens}")

# 2. Estimate cost in USD
cost_usd = estimate_cost(tokens, model_name, currency="USD")
print(f"Estimated cost in USD: ${cost_usd:.6f}")

# 3. Estimate cost in INR
cost_inr = estimate_cost(tokens, model_name, currency="INR")
print(f"Estimated cost in INR: ₹{cost_inr:.6f}")

# 4. Estimate cost in EUR (using the new implementation)
cost_eur = estimate_cost(tokens, model_name, currency="EUR")
print(f"Estimated cost in EUR: €{cost_eur:.6f}")
