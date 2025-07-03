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


class TestNewProvidersV070:
    """Test cases for all new providers added in v0.7.0."""
    
    def test_xai_models(self):
        """Test xAI/Grok models."""
        xai_models = ["grok-1", "grok-1.5", "grok-2", "grok-beta"]
        
        for model in xai_models:
            counter = TokenCounter(model)
            assert counter.provider == "xai"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
    
    def test_alibaba_models(self):
        """Test Alibaba/Qwen models."""
        alibaba_models = [
            "qwen-1.5-0.5b", "qwen-1.5-1.8b", "qwen-1.5-4b", "qwen-1.5-7b",
            "qwen-1.5-14b", "qwen-1.5-32b", "qwen-1.5-72b", "qwen-1.5-110b",
            "qwen-2-0.5b", "qwen-2-1.5b", "qwen-2-7b", "qwen-2-57b", "qwen-2-72b",
            "qwen-vl", "qwen-vl-chat", "qwen-vl-plus"
        ]
        
        for model in alibaba_models:
            counter = TokenCounter(model)
            assert counter.provider == "alibaba"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test Chinese text (should be optimized)
            chinese_tokens = counter.count("你好，世界！")
            assert isinstance(chinese_tokens, int)
            assert chinese_tokens > 0
    
    def test_baidu_models(self):
        """Test Baidu/ERNIE models."""
        baidu_models = [
            "ernie-4.0", "ernie-3.5", "ernie-3.0", "ernie-speed",
            "ernie-lite", "ernie-tiny", "ernie-bot", "ernie-bot-4"
        ]
        
        for model in baidu_models:
            counter = TokenCounter(model)
            assert counter.provider == "baidu"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test Chinese text (should be optimized)
            chinese_tokens = counter.count("你好，世界！")
            assert isinstance(chinese_tokens, int)
            assert chinese_tokens > 0
    
    def test_huawei_models(self):
        """Test Huawei/PanGu models."""
        huawei_models = [
            "pangu-alpha-2.6b", "pangu-alpha-13b", "pangu-alpha-200b",
            "pangu-coder", "pangu-coder-15b"
        ]
        
        for model in huawei_models:
            counter = TokenCounter(model)
            assert counter.provider == "huawei"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test code (for coder models)
            if "coder" in model:
                code_tokens = counter.count("def hello_world():\n    print('Hello, world!')")
                assert isinstance(code_tokens, int)
                assert code_tokens > 0
    
    def test_yandex_models(self):
        """Test Yandex/YaLM models."""
        yandex_models = ["yalm-100b", "yalm-200b", "yagpt", "yagpt-2"]
        
        for model in yandex_models:
            counter = TokenCounter(model)
            assert counter.provider == "yandex"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test Russian text (should be optimized)
            russian_tokens = counter.count("Привет, мир!")
            assert isinstance(russian_tokens, int)
            assert russian_tokens > 0
    
    def test_stability_models(self):
        """Test Stability AI/StableLM models."""
        stability_models = [
            "stablelm-alpha-3b", "stablelm-alpha-7b", "stablelm-base-alpha-3b",
            "stablelm-base-alpha-7b", "stablelm-tuned-alpha-3b", 
            "stablelm-tuned-alpha-7b", "stablelm-zephyr-3b"
        ]
        
        for model in stability_models:
            counter = TokenCounter(model)
            assert counter.provider == "stability"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_tii_models(self):
        """Test TII/Falcon models."""
        tii_models = [
            "falcon-7b", "falcon-7b-instruct", "falcon-40b",
            "falcon-40b-instruct", "falcon-180b", "falcon-180b-chat"
        ]
        
        for model in tii_models:
            counter = TokenCounter(model)
            assert counter.provider == "tii"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_eleutherai_models(self):
        """Test EleutherAI models."""
        eleutherai_models = [
            "gpt-neo-125m", "gpt-neo-1.3b", "gpt-neo-2.7b", "gpt-neox-20b",
            "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b",
            "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b"
        ]
        
        for model in eleutherai_models:
            counter = TokenCounter(model)
            assert counter.provider == "eleutherai"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_mosaicml_models(self):
        """Test MosaicML/Databricks models."""
        mosaicml_models = [
            "mpt-7b", "mpt-7b-chat", "mpt-7b-instruct",
            "mpt-30b", "mpt-30b-chat", "mpt-30b-instruct",
            "dbrx", "dbrx-instruct"
        ]
        
        for model in mosaicml_models:
            counter = TokenCounter(model)
            assert counter.provider == "mosaicml"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_replit_models(self):
        """Test Replit code models."""
        replit_models = ["replit-code-v1-3b", "replit-code-v1.5-3b", "replit-code-v2-3b"]
        
        for model in replit_models:
            counter = TokenCounter(model)
            assert counter.provider == "replit"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test code (specialized for code models)
            code_tokens = counter.count("def hello_world():\n    print('Hello, world!')")
            assert isinstance(code_tokens, int)
            assert code_tokens > 0
    
    def test_minimax_models(self):
        """Test MiniMax models."""
        minimax_models = [
            "abab5.5-chat", "abab5.5s-chat", "abab6-chat",
            "abab6.5-chat", "abab6.5s-chat"
        ]
        
        for model in minimax_models:
            counter = TokenCounter(model)
            assert counter.provider == "minimax"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test Chinese text (should be optimized)
            chinese_tokens = counter.count("你好，世界！")
            assert isinstance(chinese_tokens, int)
            assert chinese_tokens > 0
    
    def test_aleph_alpha_models(self):
        """Test Aleph Alpha/Luminous models."""
        aleph_alpha_models = [
            "luminous-base", "luminous-extended", 
            "luminous-supreme", "luminous-supreme-control"
        ]
        
        for model in aleph_alpha_models:
            counter = TokenCounter(model)
            assert counter.provider == "aleph_alpha"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_deepseek_models(self):
        """Test DeepSeek models."""
        deepseek_models = [
            "deepseek-coder-1.3b", "deepseek-coder-6.7b", "deepseek-coder-33b",
            "deepseek-coder-instruct", "deepseek-vl-1.3b", "deepseek-vl-7b",
            "deepseek-llm-7b", "deepseek-llm-67b"
        ]
        
        for model in deepseek_models:
            counter = TokenCounter(model)
            assert counter.provider == "deepseek"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test code (for coder models)
            if "coder" in model:
                code_tokens = counter.count("def hello_world():\n    print('Hello, world!')")
                assert isinstance(code_tokens, int)
                assert code_tokens > 0
    
    def test_tsinghua_models(self):
        """Test Tsinghua KEG Lab/ChatGLM models."""
        tsinghua_models = ["chatglm-6b", "chatglm2-6b", "chatglm3-6b", "glm-4", "glm-4v"]
        
        for model in tsinghua_models:
            counter = TokenCounter(model)
            assert counter.provider == "tsinghua"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test Chinese text (should be optimized)
            chinese_tokens = counter.count("你好，世界！")
            assert isinstance(chinese_tokens, int)
            assert chinese_tokens > 0
    
    def test_rwkv_models(self):
        """Test RWKV models."""
        rwkv_models = [
            "rwkv-4-169m", "rwkv-4-430m", "rwkv-4-1b5", "rwkv-4-3b",
            "rwkv-4-7b", "rwkv-4-14b", "rwkv-5-world"
        ]
        
        for model in rwkv_models:
            counter = TokenCounter(model)
            assert counter.provider == "rwkv"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_community_models(self):
        """Test community fine-tuned models."""
        community_models = [
            "vicuna-7b", "vicuna-13b", "vicuna-33b",
            "alpaca-7b", "alpaca-13b",
            "wizardlm-7b", "wizardlm-13b", "wizardlm-30b",
            "orca-mini-3b", "orca-mini-7b", "orca-mini-13b",
            "zephyr-7b-alpha", "zephyr-7b-beta"
        ]
        
        for model in community_models:
            counter = TokenCounter(model)
            assert counter.provider == "community"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0


class TestLanguageSpecificOptimizations:
    """Test language-specific tokenization optimizations."""
    
    def test_chinese_optimized_models(self):
        """Test Chinese-optimized models."""
        chinese_models = [
            ("qwen-2-7b", "alibaba"),
            ("ernie-4.0", "baidu"),
            ("pangu-alpha-13b", "huawei"),
            ("abab6-chat", "minimax"),
            ("chatglm-6b", "tsinghua")
        ]
        
        chinese_text = "你好，世界！这是一个测试消息。"
        english_text = "Hello, world! This is a test message."
        
        for model, expected_provider in chinese_models:
            counter = TokenCounter(model)
            assert counter.provider == expected_provider
            
            chinese_tokens = counter.count(chinese_text)
            english_tokens = counter.count(english_text)
            
            # Both should return reasonable token counts
            assert 5 <= chinese_tokens <= 20
            assert 5 <= english_tokens <= 15
    
    def test_russian_optimized_models(self):
        """Test Russian-optimized models."""
        yandex_models = ["yalm-100b", "yalm-200b", "yagpt", "yagpt-2"]
        
        russian_text = "Привет, мир! Это тестовое сообщение."
        english_text = "Hello, world! This is a test message."
        
        for model in yandex_models:
            counter = TokenCounter(model)
            assert counter.provider == "yandex"
            
            russian_tokens = counter.count(russian_text)
            english_tokens = counter.count(english_text)
            
            # Both should return reasonable token counts
            assert 5 <= russian_tokens <= 20
            assert 5 <= english_tokens <= 15
    
    def test_code_optimized_models(self):
        """Test code-optimized models."""
        code_models = [
            ("replit-code-v2-3b", "replit"),
            ("deepseek-coder-6.7b", "deepseek"),
            ("pangu-coder-15b", "huawei")
        ]
        
        code_text = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        for model, expected_provider in code_models:
            counter = TokenCounter(model)
            assert counter.provider == expected_provider
            
            code_tokens = counter.count(code_text)
            
            # Should return reasonable token count for code
            assert 10 <= code_tokens <= 50


class TestModelCounts:
    """Test that we have the expected number of models."""
    
    def test_total_model_count(self):
        """Test that we have 212+ total models."""
        models = get_supported_models()
        total_count = sum(len(model_list) for model_list in models.values())
        assert total_count >= 212, f"Expected at least 212 models, got {total_count}"
    
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
            "xai": 4,
            "alibaba": 16,
            "baidu": 8,
            "huawei": 5,
            "yandex": 4,
            "stability": 7,
            "tii": 6,
            "eleutherai": 12,
            "mosaicml": 8,
            "replit": 3,
            "minimax": 5,
            "aleph_alpha": 4,
            "deepseek": 8,
            "tsinghua": 5,
            "rwkv": 7,
            "community": 13,
        }
        
        for provider, expected_count in expected_counts.items():
            actual_count = len(models[provider])
            assert actual_count == expected_count, f"Expected {expected_count} {provider} models, got {actual_count}"
    
    def test_provider_list(self):
        """Test that we have all expected providers."""
        models = get_supported_models()
        expected_providers = {
            "openai", "anthropic", "google", "meta", "mistral",
            "cohere", "perplexity", "huggingface", "ai21", "together",
            "xai", "alibaba", "baidu", "huawei", "yandex", "stability",
            "tii", "eleutherai", "mosaicml", "replit", "minimax",
            "aleph_alpha", "deepseek", "tsinghua", "rwkv", "community"
        }
        actual_providers = set(models.keys())
        assert actual_providers == expected_providers


class TestProviderSpecificApproximationsV070:
    """Test provider-specific tokenization approximations for v0.7.0."""
    
    def test_all_new_providers_approximation(self):
        """Test that all new providers give reasonable approximations."""
        test_text = "This is a comprehensive test message for tokenization approximation across all providers."
        
        # Test all new providers
        new_providers_models = {
            "xai": "grok-1",
            "alibaba": "qwen-2-7b",
            "baidu": "ernie-4.0",
            "huawei": "pangu-alpha-13b",
            "yandex": "yalm-100b",
            "stability": "stablelm-alpha-7b",
            "tii": "falcon-7b",
            "eleutherai": "gpt-neo-1.3b",
            "mosaicml": "mpt-7b",
            "replit": "replit-code-v2-3b",
            "minimax": "abab6-chat",
            "aleph_alpha": "luminous-base",
            "deepseek": "deepseek-coder-6.7b",
            "tsinghua": "chatglm-6b",
            "rwkv": "rwkv-4-7b",
            "community": "vicuna-7b"
        }
        
        token_counts = {}
        for provider, model in new_providers_models.items():
            counter = TokenCounter(model)
            tokens = counter.count(test_text)
            token_counts[provider] = tokens
            
            # All should return reasonable token counts
            assert 10 <= tokens <= 35, f"{provider} returned {tokens} tokens, expected 10-35"
        
        # Different providers should give different results (within reason)
        unique_counts = set(token_counts.values())
        assert len(unique_counts) >= 5, "Expected some variation in token counts across new providers"
    
    def test_chinese_vs_english_optimization(self):
        """Test Chinese vs English optimization differences."""
        chinese_models = ["qwen-2-7b", "ernie-4.0", "chatglm-6b"]
        english_models = ["grok-1", "falcon-7b", "vicuna-7b"]
        
        chinese_text = "这是一个中文测试消息，用于测试中文优化的分词器。"
        english_text = "This is an English test message for testing English-optimized tokenizers."
        
        # Chinese models should handle Chinese text more efficiently
        for model in chinese_models:
            counter = TokenCounter(model)
            chinese_tokens = counter.count(chinese_text)
            english_tokens = counter.count(english_text)
            
            # Both should be reasonable, but Chinese should be more efficient for Chinese text
            assert 5 <= chinese_tokens <= 30
            assert 5 <= english_tokens <= 25
        
        # English models should handle English text normally
        for model in english_models:
            counter = TokenCounter(model)
            english_tokens = counter.count(english_text)
            assert 8 <= english_tokens <= 25


class TestCaseInsensitiveMatchingV070:
    """Test case-insensitive model name matching for v0.7.0 models."""
    
    def test_new_providers_case_variations(self):
        """Test case variations for new providers."""
        test_cases = [
            ("grok-1", "GROK-1", "Grok-1"),
            ("qwen-2-7b", "QWEN-2-7B", "Qwen-2-7B"),
            ("ernie-4.0", "ERNIE-4.0", "Ernie-4.0"),
            ("falcon-7b", "FALCON-7B", "Falcon-7B"),
            ("stablelm-alpha-7b", "STABLELM-ALPHA-7B", "StableLM-Alpha-7B"),
            ("deepseek-coder-6.7b", "DEEPSEEK-CODER-6.7B", "DeepSeek-Coder-6.7B"),
            ("chatglm-6b", "CHATGLM-6B", "ChatGLM-6B"),
            ("rwkv-4-7b", "RWKV-4-7B", "RWKV-4-7B"),
        ]
        
        for variations in test_cases:
            providers = []
            for model_name in variations:
                counter = TokenCounter(model_name)
                providers.append(counter.provider)
            
            # All variations should detect the same provider
            assert len(set(providers)) == 1, f"Case variations should detect same provider: {variations}"


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
