"""
Tests for the toksum library.
"""

import pytest
from unittest.mock import Mock, patch

from toksum import TokenCounter, count_tokens, get_supported_models
from toksum.exceptions import UnsupportedModelError, TokenizationError


class TestTokenCounter:
    """Test cases for the TokenCounter class."""
    
    def test_unsupported_model(self):
        """Test that unsupported models raise an exception."""
        with pytest.raises(UnsupportedModelError):
            TokenCounter("unsupported-model")
    
    def test_supported_models_detection(self):
        """Test that supported models are detected correctly."""
        # Test OpenAI model detection
        counter = TokenCounter("gpt-4")
        assert counter.provider == "openai"
        
        # Test Anthropic model detection
        counter = TokenCounter("claude-3-opus-20240229")
        assert counter.provider == "anthropic"
    
    def test_case_insensitive_model_names(self):
        """Test that model names are case insensitive."""
        counter1 = TokenCounter("GPT-4")
        counter2 = TokenCounter("gpt-4")
        assert counter1.provider == counter2.provider
    
    @patch('toksum.core.tiktoken')
    def test_openai_token_counting(self, mock_tiktoken):
        """Test token counting for OpenAI models."""
        # Mock tiktoken
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.get_encoding.return_value = mock_encoder
        
        counter = TokenCounter("gpt-4")
        result = counter.count("Hello, world!")
        
        assert result == 5
        mock_encoder.encode.assert_called_once_with("Hello, world!")
    
    def test_anthropic_token_counting(self):
        """Test token counting for Anthropic models."""
        counter = TokenCounter("claude-3-opus-20240229")
        
        # Test basic text
        result = counter.count("Hello, world!")
        assert isinstance(result, int)
        assert result > 0
        
        # Test empty string
        result = counter.count("")
        assert result == 0
        
        # Test longer text should have more tokens
        short_text = "Hi"
        long_text = "This is a much longer text that should have more tokens than the short one."
        
        short_count = counter.count(short_text)
        long_count = counter.count(long_text)
        assert long_count > short_count
    
    def test_invalid_input_type(self):
        """Test that non-string inputs raise an exception."""
        counter = TokenCounter("gpt-4")
        
        with pytest.raises(TokenizationError):
            counter.count(123)
        
        with pytest.raises(TokenizationError):
            counter.count(None)
    
    @patch('toksum.core.tiktoken')
    def test_count_messages_openai(self, mock_tiktoken):
        """Test message counting for OpenAI models."""
        mock_encoder = Mock()
        mock_encoder.encode.side_effect = lambda x: [1] * len(x.split())  # 1 token per word
        mock_tiktoken.get_encoding.return_value = mock_encoder
        
        counter = TokenCounter("gpt-4")
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi how are you"}
        ]
        
        result = counter.count_messages(messages)
        assert isinstance(result, int)
        assert result > 0
    
    def test_count_messages_anthropic(self):
        """Test message counting for Anthropic models."""
        counter = TokenCounter("claude-3-opus-20240229")
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi how are you"}
        ]
        
        result = counter.count_messages(messages)
        assert isinstance(result, int)
        assert result > 0
    
    def test_count_messages_invalid_format(self):
        """Test that invalid message formats raise exceptions."""
        counter = TokenCounter("gpt-4")
        
        # Test non-list input
        with pytest.raises(TokenizationError):
            counter.count_messages("not a list")
        
        # Test message without content
        with pytest.raises(TokenizationError):
            counter.count_messages([{"role": "user"}])
        
        # Test non-dict message
        with pytest.raises(TokenizationError):
            counter.count_messages(["not a dict"])


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @patch('toksum.core.tiktoken')
    def test_count_tokens_function(self, mock_tiktoken):
        """Test the count_tokens convenience function."""
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_encoder
        
        result = count_tokens("Hello", "gpt-4")
        assert result == 3
    
    def test_get_supported_models(self):
        """Test getting supported models."""
        models = get_supported_models()
        
        assert isinstance(models, dict)
        assert "openai" in models
        assert "anthropic" in models
        assert isinstance(models["openai"], list)
        assert isinstance(models["anthropic"], list)
        assert len(models["openai"]) > 0
        assert len(models["anthropic"]) > 0
        assert "gpt-4" in models["openai"]
        assert "claude-3-opus-20240229" in models["anthropic"]


class TestExceptions:
    """Test cases for custom exceptions."""
    
    def test_unsupported_model_error(self):
        """Test UnsupportedModelError exception."""
        supported = ["gpt-4", "claude-3-opus-20240229"]
        error = UnsupportedModelError("invalid-model", supported)
        
        assert error.model == "invalid-model"
        assert error.supported_models == supported
        assert "invalid-model" in str(error)
        assert "gpt-4" in str(error)
    
    def test_tokenization_error(self):
        """Test TokenizationError exception."""
        error = TokenizationError("Test error", model="gpt-4", text_preview="Hello world")
        
        assert error.model == "gpt-4"
        assert error.text_preview == "Hello world"
        assert "Test error" in str(error)
        assert "gpt-4" in str(error)
        assert "Hello world" in str(error)
    
    def test_tokenization_error_long_text(self):
        """Test TokenizationError with long text preview."""
        long_text = "This is a very long text that should be truncated in the error message" * 10
        error = TokenizationError("Test error", text_preview=long_text)
        
        error_str = str(error)
        assert "..." in error_str  # Should be truncated
        assert len(error_str) < len(long_text) + 100  # Should be much shorter


class TestCostEstimation:
    """Test cases for cost estimation functionality."""
    
    def test_estimate_cost_known_models(self):
        """Test cost estimation for known models."""
        from toksum.core import estimate_cost
        
        # Test GPT-4
        cost = estimate_cost(1000, "gpt-4", input_tokens=True)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Test output tokens cost more than input
        input_cost = estimate_cost(1000, "gpt-4", input_tokens=True)
        output_cost = estimate_cost(1000, "gpt-4", input_tokens=False)
        assert output_cost > input_cost
        
        # Test Claude
        cost = estimate_cost(1000, "claude-3-opus-20240229", input_tokens=True)
        assert cost > 0
        assert isinstance(cost, float)
    
    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown models."""
        from toksum.core import estimate_cost
        
        cost = estimate_cost(1000, "unknown-model")
        assert cost == 0.0


class TestNewProviders:
    """Test cases for all new providers and models."""
    
    def test_google_models(self):
        """Test Google/Gemini models."""
        google_models = [
            "gemini-pro", "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash",
            "gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro",
            "gemini-1.0-pro-vision", "gemini-ultra"
        ]
        
        for model in google_models:
            counter = TokenCounter(model)
            assert counter.provider == "google"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
    
    def test_meta_models(self):
        """Test Meta/LLaMA models."""
        meta_models = [
            "llama-2-7b", "llama-2-13b", "llama-2-70b",
            "llama-3-8b", "llama-3-70b",
            "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
            "llama-3.2-1b", "llama-3.2-3b"
        ]
        
        for model in meta_models:
            counter = TokenCounter(model)
            assert counter.provider == "meta"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_mistral_models(self):
        """Test Mistral models."""
        mistral_models = [
            "mistral-7b", "mistral-8x7b", "mistral-large", "mistral-medium",
            "mistral-small", "mistral-tiny", "mixtral-8x7b", "mixtral-8x22b"
        ]
        
        for model in mistral_models:
            counter = TokenCounter(model)
            assert counter.provider == "mistral"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_cohere_models(self):
        """Test Cohere models."""
        cohere_models = [
            "command", "command-light", "command-nightly",
            "command-r", "command-r-plus", "command-r-08-2024", "command-r-plus-08-2024"
        ]
        
        for model in cohere_models:
            counter = TokenCounter(model)
            assert counter.provider == "cohere"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_perplexity_models(self):
        """Test Perplexity models."""
        perplexity_models = [
            "pplx-7b-online", "pplx-70b-online", "pplx-7b-chat",
            "pplx-70b-chat", "codellama-34b-instruct"
        ]
        
        for model in perplexity_models:
            counter = TokenCounter(model)
            assert counter.provider == "perplexity"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_huggingface_models(self):
        """Test Hugging Face models."""
        huggingface_models = [
            "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill", "facebook/blenderbot-1B-distill",
            "facebook/blenderbot-3B"
        ]
        
        for model in huggingface_models:
            counter = TokenCounter(model)
            assert counter.provider == "huggingface"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_ai21_models(self):
        """Test AI21 models."""
        ai21_models = [
            "j2-light", "j2-mid", "j2-ultra", "j2-jumbo-instruct"
        ]
        
        for model in ai21_models:
            counter = TokenCounter(model)
            assert counter.provider == "ai21"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_together_models(self):
        """Test Together AI models."""
        together_models = [
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            "togethercomputer/RedPajama-INCITE-Chat-7B-v1",
            "NousResearch/Nous-Hermes-Llama2-13b"
        ]
        
        for model in together_models:
            counter = TokenCounter(model)
            assert counter.provider == "together"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0


class TestNewOpenAIModels:
    """Test cases for new OpenAI models."""
    
    @patch('toksum.core.tiktoken')
    def test_new_gpt4_variants(self, mock_tiktoken):
        """Test new GPT-4 variants."""
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.get_encoding.return_value = mock_encoder
        
        new_models = [
            "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
            "gpt-4-1106-vision-preview", "gpt-3.5-turbo-instruct"
        ]
        
        for model in new_models:
            counter = TokenCounter(model)
            assert counter.provider == "openai"
            tokens = counter.count("Hello, world!")
            assert tokens == 5
    
    @patch('toksum.core.tiktoken')
    def test_legacy_openai_models(self, mock_tiktoken):
        """Test legacy OpenAI models."""
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_encoder
        
        legacy_models = [
            "gpt-3", "text-embedding-ada-002", "text-embedding-3-small",
            "text-embedding-3-large", "gpt-4-base", "gpt-3.5-turbo-instruct-0914"
        ]
        
        for model in legacy_models:
            counter = TokenCounter(model)
            assert counter.provider == "openai"
            tokens = counter.count("Hello, world!")
            assert tokens == 3


class TestNewAnthropicModels:
    """Test cases for new Anthropic models."""
    
    def test_short_name_models(self):
        """Test Anthropic short name models."""
        short_name_models = [
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-instant"
        ]
        
        for model in short_name_models:
            counter = TokenCounter(model)
            assert counter.provider == "anthropic"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_legacy_claude_models(self):
        """Test legacy Claude models."""
        legacy_models = [
            "claude-1", "claude-1.3", "claude-1.3-100k"
        ]
        
        for model in legacy_models:
            counter = TokenCounter(model)
            assert counter.provider == "anthropic"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0


class TestProviderSpecificApproximations:
    """Test provider-specific tokenization approximations."""
    
    def test_approximation_differences(self):
        """Test that different providers give different approximations."""
        test_text = "This is a test message for tokenization approximation."
        
        # Test different providers
        providers_models = {
            "anthropic": "claude-3-opus",
            "google": "gemini-pro",
            "meta": "llama-3-8b",
            "mistral": "mistral-large",
            "cohere": "command",
            "perplexity": "pplx-7b-online",
            "huggingface": "microsoft/DialoGPT-medium",
            "ai21": "j2-ultra",
            "together": "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
        }
        
        token_counts = {}
        for provider, model in providers_models.items():
            counter = TokenCounter(model)
            tokens = counter.count(test_text)
            token_counts[provider] = tokens
            
            # All should return reasonable token counts
            assert 5 <= tokens <= 25, f"{provider} returned {tokens} tokens, expected 5-25"
        
        # Different providers should give different results (within reason)
        unique_counts = set(token_counts.values())
        assert len(unique_counts) >= 3, "Expected more variation in token counts across providers"
    
    def test_empty_string_handling(self):
        """Test that all providers handle empty strings correctly."""
        providers_models = {
            "anthropic": "claude-3-opus",
            "google": "gemini-pro",
            "meta": "llama-3-8b",
            "mistral": "mistral-large",
            "cohere": "command",
            "perplexity": "pplx-7b-online",
            "huggingface": "microsoft/DialoGPT-medium",
            "ai21": "j2-ultra",
            "together": "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
        }
        
        for provider, model in providers_models.items():
            counter = TokenCounter(model)
            tokens = counter.count("")
            assert tokens == 0, f"{provider} should return 0 tokens for empty string"
    
    def test_whitespace_and_punctuation_handling(self):
        """Test handling of whitespace and punctuation across providers."""
        test_cases = [
            "Hello world",  # Simple text
            "Hello, world!",  # With punctuation
            "Hello    world",  # Multiple spaces
            "Hello\nworld",  # With newline
            "Hello... world???",  # Multiple punctuation
        ]
        
        counter = TokenCounter("claude-3-opus")  # Use one model for consistency
        
        for text in test_cases:
            tokens = counter.count(text)
            assert tokens > 0, f"Should return positive tokens for '{text}'"
            assert tokens < 20, f"Should return reasonable token count for '{text}'"


class TestCaseInsensitiveMatching:
    """Test case-insensitive model name matching."""
    
    def test_case_variations(self):
        """Test various case combinations."""
        test_cases = [
            ("gpt-4", "GPT-4", "Gpt-4", "gPt-4"),
            ("claude-3-opus", "CLAUDE-3-OPUS", "Claude-3-Opus"),
            ("gemini-pro", "GEMINI-PRO", "Gemini-Pro"),
            ("llama-3-8b", "LLAMA-3-8B", "Llama-3-8B"),
            ("mistral-large", "MISTRAL-LARGE", "Mistral-Large"),
        ]
        
        for variations in test_cases:
            providers = []
            for model_name in variations:
                counter = TokenCounter(model_name)
                providers.append(counter.provider)
            
            # All variations should detect the same provider
            assert len(set(providers)) == 1, f"Case variations should detect same provider: {variations}"
    
    def test_complex_model_names(self):
        """Test case insensitivity with complex model names."""
        complex_models = [
            ("microsoft/DialoGPT-medium", "MICROSOFT/DIALOGPT-MEDIUM"),
            ("togethercomputer/RedPajama-INCITE-Chat-3B-v1", "TOGETHERCOMPUTER/REDPAJAMA-INCITE-CHAT-3B-V1"),
            ("facebook/blenderbot-400M-distill", "FACEBOOK/BLENDERBOT-400M-DISTILL"),
        ]
        
        for original, uppercase in complex_models:
            counter1 = TokenCounter(original)
            counter2 = TokenCounter(uppercase)
            assert counter1.provider == counter2.provider


class TestModelCounts:
    """Test that we have the expected number of models."""
    
    def test_total_model_count(self):
        """Test that we have 112 total models."""
        models = get_supported_models()
        total_count = sum(len(model_list) for model_list in models.values())
        assert total_count == 112, f"Expected 112 models, got {total_count}"
    
    def test_provider_counts(self):
        """Test expected model counts per provider."""
        models = get_supported_models()
        expected_counts = {
            "openai": 42,
            "anthropic": 19,
            "google": 9,
            "meta": 10,
            "mistral": 8,
            "cohere": 7,
            "perplexity": 5,
            "huggingface": 5,
            "ai21": 4,
            "together": 3,
        }
        
        for provider, expected_count in expected_counts.items():
            actual_count = len(models[provider])
            assert actual_count == expected_count, f"Expected {expected_count} {provider} models, got {actual_count}"
    
    def test_provider_list(self):
        """Test that we have all expected providers."""
        models = get_supported_models()
        expected_providers = {
            "openai", "anthropic", "google", "meta", "mistral",
            "cohere", "perplexity", "huggingface", "ai21", "together"
        }
        actual_providers = set(models.keys())
        assert actual_providers == expected_providers


class TestIntegration:
    """Integration tests."""
    
    def test_anthropic_approximation_accuracy(self):
        """Test that Anthropic token approximation is reasonable."""
        counter = TokenCounter("claude-3-opus-20240229")
        
        # Test various text patterns
        test_cases = [
            ("Hello", 1, 3),  # Simple word should be 1-3 tokens
            ("Hello, world!", 2, 5),  # With punctuation
            ("The quick brown fox jumps over the lazy dog.", 8, 15),  # Sentence
            ("Python is a programming language.", 5, 10),  # Technical text
        ]
        
        for text, min_tokens, max_tokens in test_cases:
            count = counter.count(text)
            assert min_tokens <= count <= max_tokens, f"Token count {count} for '{text}' not in range [{min_tokens}, {max_tokens}]"
    
    def test_all_providers_basic_functionality(self):
        """Test basic functionality across all providers."""
        test_models = [
            "gpt-4",  # OpenAI
            "claude-3-opus",  # Anthropic
            "gemini-pro",  # Google
            "llama-3-8b",  # Meta
            "mistral-large",  # Mistral
            "command",  # Cohere
            "pplx-7b-online",  # Perplexity
            "microsoft/DialoGPT-medium",  # Hugging Face
            "j2-ultra",  # AI21
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1",  # Together
        ]
        
        test_text = "This is a test message for all providers."
        
        for model in test_models:
            # Test TokenCounter initialization
            counter = TokenCounter(model)
            assert counter.model == model.lower()
            
            # Test token counting
            tokens = counter.count(test_text)
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test convenience function
            tokens2 = count_tokens(test_text, model)
            assert tokens == tokens2
    
    @patch('toksum.core.tiktoken', None)
    def test_missing_tiktoken_dependency(self):
        """Test behavior when tiktoken is not available."""
        with pytest.raises(TokenizationError) as exc_info:
            TokenCounter("gpt-4")
        
        assert "tiktoken is required" in str(exc_info.value)
    
    def test_message_counting_all_providers(self):
        """Test message counting across different providers."""
        test_models = [
            "claude-3-opus",  # Anthropic
            "gemini-pro",  # Google
            "llama-3-8b",  # Meta
            "mistral-large",  # Mistral
            "command",  # Cohere
        ]
        
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi, how can I help you?"}
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            tokens = counter.count_messages(messages)
            assert isinstance(tokens, int)
            assert tokens > 0


if __name__ == "__main__":
    pytest.main([__file__])
