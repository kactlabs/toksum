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
            "openai": 49,  # Updated: Added O1 models and vision models
            "anthropic": 27,  # Updated: Added Haiku, Computer Use, Claude 2.1, and Instant 2 models
            "google": 16,  # Updated: Added Gemini 2.0 and PaLM models
            "meta": 12,  # Updated: Added Llama 3.3 models
            "mistral": 10,  # Updated: Added Large 2 models
            "cohere": 9,   # Updated: Added Command R+ variants
            "perplexity": 5,
            "huggingface": 5,
            "ai21": 4,
            "together": 3,
            "xai": 4,
            "alibaba": 20,  # Updated: Added Qwen 2.5 models
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
            "deepseek": 10,  # Updated: Added V3 models
            "tsinghua": 5,
            "rwkv": 7,
            "community": 13,
            "microsoft": 4,  # New provider: Added Phi models
            "amazon": 3,     # New provider: Added Titan models
            "nvidia": 2,     # New provider: Added Nemotron models
            "ibm": 3,        # New provider: Added Granite models
            "salesforce": 3, # New provider: Added CodeGen models
            "bigcode": 3,    # New provider: Added StarCoder models
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
            "aleph_alpha", "deepseek", "tsinghua", "rwkv", "community",
            "microsoft", "amazon", "nvidia", "ibm", "salesforce", "bigcode"
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


class TestNewModelsV080:
    """Test cases for the 20 new models added in v0.8.0."""
    
    @patch('toksum.core.tiktoken')
    def test_openai_o1_models(self, mock_tiktoken):
        """Test OpenAI O1 models."""
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.get_encoding.return_value = mock_encoder
        
        o1_models = [
            "o1-preview", "o1-mini", 
            "o1-preview-2024-09-12", "o1-mini-2024-09-12"
        ]
        
        for model in o1_models:
            counter = TokenCounter(model)
            assert counter.provider == "openai"
            
            tokens = counter.count("Hello, world!")
            assert tokens == 5
            mock_encoder.encode.assert_called_with("Hello, world!")
    
    def test_anthropic_haiku_models(self):
        """Test Anthropic Claude 3.5 Haiku models."""
        haiku_models = [
            "claude-3.5-haiku-20241022",
            "claude-3-5-haiku-20241022"  # Alternative naming
        ]
        
        for model in haiku_models:
            counter = TokenCounter(model)
            assert counter.provider == "anthropic"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test longer text
            long_text = "This is a longer text to test the Haiku model's token counting capabilities."
            long_tokens = counter.count(long_text)
            assert long_tokens > tokens
    
    def test_anthropic_computer_use_models(self):
        """Test Anthropic Computer Use models."""
        computer_use_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3.5-sonnet-computer-use"
        ]
        
        for model in computer_use_models:
            counter = TokenCounter(model)
            assert counter.provider == "anthropic"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test computer-related text
            computer_text = "Click on the button at coordinates (100, 200) and then type 'hello' in the text field."
            computer_tokens = counter.count(computer_text)
            assert isinstance(computer_tokens, int)
            assert computer_tokens > tokens
    
    def test_google_gemini_2_models(self):
        """Test Google Gemini 2.0 models."""
        gemini_2_models = [
            "gemini-2.0-flash-exp", "gemini-2.0-flash",
            "gemini-exp-1206", "gemini-exp-1121"
        ]
        
        for model in gemini_2_models:
            counter = TokenCounter(model)
            assert counter.provider == "google"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test multimodal-like text
            multimodal_text = "Analyze this image and describe what you see in detail."
            multimodal_tokens = counter.count(multimodal_text)
            assert isinstance(multimodal_tokens, int)
            assert multimodal_tokens > tokens
    
    def test_meta_llama_33_models(self):
        """Test Meta Llama 3.3 models."""
        llama_33_models = [
            "llama-3.3-70b",
            "llama-3.3-70b-instruct"
        ]
        
        for model in llama_33_models:
            counter = TokenCounter(model)
            assert counter.provider == "meta"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test instruction-following text (for instruct model)
            if "instruct" in model:
                instruction_text = "Please explain the concept of machine learning in simple terms."
                instruction_tokens = counter.count(instruction_text)
                assert isinstance(instruction_tokens, int)
                assert instruction_tokens > tokens
    
    def test_mistral_large_2_models(self):
        """Test Mistral Large 2 models."""
        mistral_large_2_models = [
            "mistral-large-2",
            "mistral-large-2407"
        ]
        
        for model in mistral_large_2_models:
            counter = TokenCounter(model)
            assert counter.provider == "mistral"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test multilingual text (Mistral supports multiple languages)
            multilingual_text = "Bonjour le monde! Hola mundo! Hallo Welt!"
            multilingual_tokens = counter.count(multilingual_text)
            assert isinstance(multilingual_tokens, int)
            assert multilingual_tokens > tokens
    
    def test_deepseek_v3_models(self):
        """Test DeepSeek V3 models."""
        deepseek_v3_models = [
            "deepseek-v3",
            "deepseek-v3-base"
        ]
        
        for model in deepseek_v3_models:
            counter = TokenCounter(model)
            assert counter.provider == "deepseek"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test code (DeepSeek is good at coding)
            code_text = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
print(fibonacci(10))
"""
            code_tokens = counter.count(code_text)
            assert isinstance(code_tokens, int)
            assert code_tokens > tokens
            
            # Test Chinese text (DeepSeek supports Chinese)
            chinese_text = "你好，世界！这是一个测试消息。"
            chinese_tokens = counter.count(chinese_text)
            assert isinstance(chinese_tokens, int)
            assert chinese_tokens > 0
    
    def test_qwen_25_models(self):
        """Test Qwen 2.5 models."""
        qwen_25_models = [
            "qwen-2.5-72b", "qwen-2.5-32b",
            "qwen-2.5-14b", "qwen-2.5-7b"
        ]
        
        for model in qwen_25_models:
            counter = TokenCounter(model)
            assert counter.provider == "alibaba"
            
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test Chinese text (Qwen is optimized for Chinese)
            chinese_text = "你好，世界！这是阿里巴巴的通义千问模型测试。"
            chinese_tokens = counter.count(chinese_text)
            assert isinstance(chinese_tokens, int)
            assert chinese_tokens > 0
            
            # Test mixed Chinese-English text
            mixed_text = "Hello 你好, this is a mixed language test 这是混合语言测试."
            mixed_tokens = counter.count(mixed_text)
            assert isinstance(mixed_tokens, int)
            assert mixed_tokens > tokens
    
    def test_new_models_case_insensitive(self):
        """Test case insensitive matching for new models."""
        test_cases = [
            ("o1-preview", "O1-PREVIEW", "O1-Preview"),
            ("claude-3.5-haiku-20241022", "CLAUDE-3.5-HAIKU-20241022", "Claude-3.5-Haiku-20241022"),
            ("gemini-2.0-flash", "GEMINI-2.0-FLASH", "Gemini-2.0-Flash"),
            ("llama-3.3-70b", "LLAMA-3.3-70B", "Llama-3.3-70B"),
            ("mistral-large-2", "MISTRAL-LARGE-2", "Mistral-Large-2"),
            ("deepseek-v3", "DEEPSEEK-V3", "DeepSeek-V3"),
            ("qwen-2.5-72b", "QWEN-2.5-72B", "Qwen-2.5-72B"),
        ]
        
        for variations in test_cases:
            providers = []
            for model_name in variations:
                counter = TokenCounter(model_name)
                providers.append(counter.provider)
            
            # All variations should detect the same provider
            assert len(set(providers)) == 1, f"Case variations should detect same provider: {variations}"
    
    def test_new_models_message_counting(self):
        """Test message counting for new models."""
        test_models = [
            "claude-3.5-haiku-20241022",
            "gemini-2.0-flash",
            "llama-3.3-70b-instruct",
            "mistral-large-2",
            "deepseek-v3",
            "qwen-2.5-32b"
        ]
        
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
            {"role": "user", "content": "Can you explain quantum computing?"}
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            tokens = counter.count_messages(messages)
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Should be more than individual message tokens
            individual_tokens = sum(counter.count(msg["content"]) for msg in messages)
            assert tokens >= individual_tokens  # Should include formatting overhead
    
    def test_new_models_approximation_consistency(self):
        """Test that new models provide consistent approximations."""
        test_texts = [
            "Hello",
            "Hello, world!",
            "This is a test message.",
            "This is a longer test message with more words and punctuation!",
            "A very long message that contains multiple sentences. Each sentence should contribute to the token count. The approximation should be reasonable and consistent across different model providers."
        ]
        
        test_models = [
            "claude-3.5-haiku-20241022",
            "gemini-2.0-flash",
            "llama-3.3-70b",
            "mistral-large-2",
            "deepseek-v3",
            "qwen-2.5-32b"
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            previous_tokens = 0
            
            for text in test_texts:
                tokens = counter.count(text)
                assert isinstance(tokens, int)
                assert tokens > 0
                
                # Longer texts should generally have more tokens
                assert tokens >= previous_tokens
                previous_tokens = tokens
    
    def test_new_models_special_characters(self):
        """Test new models with special characters and edge cases."""
        special_texts = [
            "",  # Empty string
            " ",  # Single space
            "\n",  # Single newline
            "🚀🌟💫",  # Emojis
            "Hello\n\nWorld",  # Multiple newlines
            "Hello    World",  # Multiple spaces
            "Hello... World???",  # Multiple punctuation
            "café naïve résumé",  # Accented characters
            "αβγδε",  # Greek letters
            "12345",  # Numbers only
            "!@#$%^&*()",  # Special symbols only
        ]
        
        test_models = [
            "claude-3.5-haiku-20241022",
            "gemini-2.0-flash",
            "llama-3.3-70b",
            "mistral-large-2",
            "deepseek-v3",
            "qwen-2.5-32b"
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            
            for text in special_texts:
                tokens = counter.count(text)
                assert isinstance(tokens, int)
                assert tokens >= 0  # Should never be negative
                
                if text == "":
                    assert tokens == 0  # Empty string should always be 0 tokens
                elif text.strip():  # Non-empty, non-whitespace text
                    assert tokens > 0  # Should have at least 1 token
    
    def test_new_models_error_handling(self):
        """Test error handling for new models."""
        test_models = [
            "claude-3.5-haiku-20241022",
            "gemini-2.0-flash",
            "llama-3.3-70b",
            "mistral-large-2",
            "deepseek-v3",
            "qwen-2.5-32b"
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            
            # Test invalid input types
            with pytest.raises(TokenizationError):
                counter.count(123)
            
            with pytest.raises(TokenizationError):
                counter.count(None)
            
            with pytest.raises(TokenizationError):
                counter.count(["list", "of", "strings"])
            
            # Test invalid message formats
            with pytest.raises(TokenizationError):
                counter.count_messages("not a list")
            
            with pytest.raises(TokenizationError):
                counter.count_messages([{"role": "user"}])  # Missing content
            
            with pytest.raises(TokenizationError):
                counter.count_messages(["not a dict"])


class TestModelCountsV080:
    """Test model counts after adding v0.8.0 models."""
    
    def test_updated_total_model_count(self):
        """Test that we now have 249+ total models."""
        models = get_supported_models()
        total_count = sum(len(model_list) for model_list in models.values())
        assert total_count >= 249, f"Expected at least 249 models, got {total_count}"
    
    def test_updated_provider_counts(self):
        """Test updated model counts per provider."""
        models = get_supported_models()
        
        # Updated expected counts after adding new models
        expected_minimums = {
            "openai": 46,  # Added O1 models
            "anthropic": 23,  # Added Haiku and Computer Use models
            "google": 13,  # Added Gemini 2.0 models
            "meta": 12,  # Added Llama 3.3 models
            "mistral": 10,  # Added Large 2 models
            "alibaba": 20,  # Added Qwen 2.5 models
            "deepseek": 10,  # Added V3 models
        }
        
        for provider, expected_min in expected_minimums.items():
            actual_count = len(models[provider])
            assert actual_count >= expected_min, f"Expected at least {expected_min} {provider} models, got {actual_count}"
    
    def test_new_models_in_supported_list(self):
        """Test that all new models appear in the supported models list."""
        models = get_supported_models()
        
        # Check that specific new models are present
        new_models_to_check = [
            ("openai", "o1-preview"),
            ("openai", "o1-mini"),
            ("anthropic", "claude-3.5-haiku-20241022"),
            ("anthropic", "claude-3.5-sonnet-computer-use"),
            ("google", "gemini-2.0-flash"),
            ("google", "gemini-exp-1206"),
            ("meta", "llama-3.3-70b"),
            ("meta", "llama-3.3-70b-instruct"),
            ("mistral", "mistral-large-2"),
            ("mistral", "mistral-large-2407"),
            ("deepseek", "deepseek-v3"),
            ("deepseek", "deepseek-v3-base"),
            ("alibaba", "qwen-2.5-72b"),
            ("alibaba", "qwen-2.5-7b"),
        ]
        
        for provider, model in new_models_to_check:
            assert model in models[provider], f"Model {model} not found in {provider} models list"


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


class TestNewModelsV090:
    """Test cases for the 30 new models added in v0.9.0."""
    
    def test_microsoft_models(self):
        """Test Microsoft Phi models."""
        microsoft_models = [
            "phi-3-mini", "phi-3-small", "phi-3-medium", "phi-3.5-mini"
        ]
        
        for model in microsoft_models:
            counter = TokenCounter(model)
            assert counter.provider == "microsoft"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test longer text
            long_text = "This is a longer text to test the Microsoft Phi model's token counting capabilities."
            long_tokens = counter.count(long_text)
            assert long_tokens > tokens
            
            # Test code (Phi models are good at coding)
            code_text = "def hello_world():\n    print('Hello, world!')"
            code_tokens = counter.count(code_text)
            assert isinstance(code_tokens, int)
            assert code_tokens > 0
    
    def test_amazon_models(self):
        """Test Amazon Titan models."""
        amazon_models = [
            "titan-text-express", "titan-text-lite", "titan-embed-text"
        ]
        
        for model in amazon_models:
            counter = TokenCounter(model)
            assert counter.provider == "amazon"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test business text (Titan is enterprise-focused)
            business_text = "Our quarterly revenue increased by 15% compared to last year."
            business_tokens = counter.count(business_text)
            assert isinstance(business_tokens, int)
            assert business_tokens > tokens
            
            # Test embedding-like text (for embed model)
            if "embed" in model:
                embed_text = "Document similarity and semantic search capabilities."
                embed_tokens = counter.count(embed_text)
                assert isinstance(embed_tokens, int)
                assert embed_tokens > 0
    
    def test_nvidia_models(self):
        """Test Nvidia Nemotron models."""
        nvidia_models = [
            "nemotron-4-340b", "nemotron-3-8b"
        ]
        
        for model in nvidia_models:
            counter = TokenCounter(model)
            assert counter.provider == "nvidia"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test technical text (Nemotron is good at technical content)
            technical_text = "GPU acceleration enables parallel processing for machine learning workloads."
            technical_tokens = counter.count(technical_text)
            assert isinstance(technical_tokens, int)
            assert technical_tokens > tokens
            
            # Test scientific text
            scientific_text = "The neural network architecture consists of transformer layers with attention mechanisms."
            scientific_tokens = counter.count(scientific_text)
            assert isinstance(scientific_tokens, int)
            assert scientific_tokens > 0
    
    def test_ibm_models(self):
        """Test IBM Granite models."""
        ibm_models = [
            "granite-13b-chat", "granite-13b-instruct", "granite-20b-code"
        ]
        
        for model in ibm_models:
            counter = TokenCounter(model)
            assert counter.provider == "ibm"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test enterprise text (Granite is enterprise-focused)
            enterprise_text = "Enterprise AI solutions require robust security and compliance frameworks."
            enterprise_tokens = counter.count(enterprise_text)
            assert isinstance(enterprise_tokens, int)
            assert enterprise_tokens > tokens
            
            # Test code (for code model)
            if "code" in model:
                code_text = """
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return [x * 2 for x in self.data]
"""
                code_tokens = counter.count(code_text)
                assert isinstance(code_tokens, int)
                assert code_tokens > tokens
            
            # Test instruction-following text (for instruct/chat models)
            if "instruct" in model or "chat" in model:
                instruction_text = "Please analyze the following data and provide insights."
                instruction_tokens = counter.count(instruction_text)
                assert isinstance(instruction_tokens, int)
                assert instruction_tokens > 0
    
    def test_salesforce_models(self):
        """Test Salesforce CodeGen models."""
        salesforce_models = [
            "codegen-16b", "codegen-6b", "codegen-2b"
        ]
        
        for model in salesforce_models:
            counter = TokenCounter(model)
            assert counter.provider == "salesforce"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test code generation (CodeGen's specialty)
            code_text = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
            code_tokens = counter.count(code_text)
            assert isinstance(code_tokens, int)
            assert code_tokens > tokens
            
            # Test different programming languages
            python_code = "import numpy as np\narray = np.zeros((10, 10))"
            python_tokens = counter.count(python_code)
            assert isinstance(python_tokens, int)
            assert python_tokens > 0
            
            javascript_code = "const array = new Array(10).fill(0);"
            javascript_tokens = counter.count(javascript_code)
            assert isinstance(javascript_tokens, int)
            assert javascript_tokens > 0
    
    def test_bigcode_models(self):
        """Test BigCode StarCoder models."""
        bigcode_models = [
            "starcoder", "starcoder2-15b", "starcoderbase"
        ]
        
        for model in bigcode_models:
            counter = TokenCounter(model)
            assert counter.provider == "bigcode"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test code (StarCoder's specialty)
            code_text = """
class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)
"""
            code_tokens = counter.count(code_text)
            assert isinstance(code_tokens, int)
            assert code_tokens > tokens
            
            # Test various programming languages
            languages_code = {
                "python": "def hello(): print('Hello, world!')",
                "javascript": "function hello() { console.log('Hello, world!'); }",
                "java": "public class Hello { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } }",
                "cpp": "#include <iostream>\nint main() { std::cout << \"Hello, world!\" << std::endl; return 0; }",
                "rust": "fn main() { println!(\"Hello, world!\"); }",
            }
            
            for language, code in languages_code.items():
                lang_tokens = counter.count(code)
                assert isinstance(lang_tokens, int)
                assert lang_tokens > 0
    
    def test_extended_anthropic_models(self):
        """Test extended Anthropic models."""
        extended_anthropic_models = [
            "claude-2.1-200k", "claude-2.1-100k",
            "claude-instant-2", "claude-instant-2.0"
        ]
        
        for model in extended_anthropic_models:
            counter = TokenCounter(model)
            assert counter.provider == "anthropic"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test long context (for 200k/100k models)
            if "200k" in model or "100k" in model:
                long_text = "This is a test for long context models. " * 100
                long_tokens = counter.count(long_text)
                assert isinstance(long_tokens, int)
                assert long_tokens > tokens * 50  # Should be much larger
            
            # Test instant response scenarios (for instant models)
            if "instant" in model:
                quick_text = "Quick response needed."
                quick_tokens = counter.count(quick_text)
                assert isinstance(quick_tokens, int)
                assert quick_tokens > 0
    
    def test_extended_openai_models(self):
        """Test extended OpenAI vision models."""
        extended_openai_models = [
            "gpt-4-vision", "gpt-4-vision-preview-0409", "gpt-4-vision-preview-1106"
        ]
        
        # Mock tiktoken for these tests
        with patch('toksum.core.tiktoken') as mock_tiktoken:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
            mock_tiktoken.get_encoding.return_value = mock_encoder
            
            for model in extended_openai_models:
                counter = TokenCounter(model)
                assert counter.provider == "openai"
                
                # Test basic token counting
                tokens = counter.count("Hello, world!")
                assert tokens == 5
                
                # Test vision-related text
                vision_text = "Describe what you see in this image: a cat sitting on a windowsill."
                vision_tokens = counter.count(vision_text)
                assert vision_tokens == 5  # Mocked to return 5 tokens
                
                # Test multimodal instructions
                multimodal_text = "Analyze the image and extract all text visible in the picture."
                multimodal_tokens = counter.count(multimodal_text)
                assert multimodal_tokens == 5  # Mocked to return 5 tokens
    
    def test_extended_cohere_models(self):
        """Test extended Cohere Command R+ models."""
        extended_cohere_models = [
            "command-r-plus-04-2024", "command-r-plus-08-2024"
        ]
        
        for model in extended_cohere_models:
            counter = TokenCounter(model)
            assert counter.provider == "cohere"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test retrieval-augmented generation text (Command R+ specialty)
            rag_text = "Based on the provided context, please answer the following question with citations."
            rag_tokens = counter.count(rag_text)
            assert isinstance(rag_tokens, int)
            assert rag_tokens > tokens
            
            # Test multilingual text (Command R+ supports multiple languages)
            multilingual_text = "Hello, Bonjour, Hola, Guten Tag, Ciao, こんにちは"
            multilingual_tokens = counter.count(multilingual_text)
            assert isinstance(multilingual_tokens, int)
            assert multilingual_tokens > 0
    
    def test_extended_google_models(self):
        """Test extended Google PaLM models."""
        extended_google_models = [
            "palm-2", "palm-2-chat", "palm-2-codechat"
        ]
        
        for model in extended_google_models:
            counter = TokenCounter(model)
            assert counter.provider == "google"
            
            # Test basic token counting
            tokens = counter.count("Hello, world!")
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Test empty string
            assert counter.count("") == 0
            
            # Test chat scenarios (for chat models)
            if "chat" in model:
                chat_text = "Let's have a conversation about artificial intelligence and its applications."
                chat_tokens = counter.count(chat_text)
                assert isinstance(chat_tokens, int)
                assert chat_tokens > tokens
            
            # Test code scenarios (for codechat model)
            if "codechat" in model:
                code_chat_text = "Can you help me debug this Python function and explain what's wrong?"
                code_chat_tokens = counter.count(code_chat_text)
                assert isinstance(code_chat_tokens, int)
                assert code_chat_tokens > tokens
    
    def test_new_models_case_insensitive(self):
        """Test case insensitive matching for all new models."""
        test_cases = [
            ("phi-3-mini", "PHI-3-MINI", "Phi-3-Mini"),
            ("titan-text-express", "TITAN-TEXT-EXPRESS", "Titan-Text-Express"),
            ("nemotron-4-340b", "NEMOTRON-4-340B", "Nemotron-4-340B"),
            ("granite-13b-chat", "GRANITE-13B-CHAT", "Granite-13B-Chat"),
            ("codegen-16b", "CODEGEN-16B", "CodeGen-16B"),
            ("starcoder", "STARCODER", "StarCoder"),
            ("claude-2.1-200k", "CLAUDE-2.1-200K", "Claude-2.1-200K"),
            ("gpt-4-vision", "GPT-4-VISION", "GPT-4-Vision"),
            ("command-r-plus-04-2024", "COMMAND-R-PLUS-04-2024", "Command-R-Plus-04-2024"),
            ("palm-2-chat", "PALM-2-CHAT", "PaLM-2-Chat"),
        ]
        
        for variations in test_cases:
            providers = []
            for model_name in variations:
                counter = TokenCounter(model_name)
                providers.append(counter.provider)
            
            # All variations should detect the same provider
            assert len(set(providers)) == 1, f"Case variations should detect same provider: {variations}"
    
    def test_new_models_message_counting(self):
        """Test message counting for new models."""
        test_models = [
            "phi-3-mini",
            "titan-text-express",
            "nemotron-4-340b",
            "granite-13b-chat",
            "codegen-16b",
            "starcoder",
            "claude-2.1-200k",
            "command-r-plus-04-2024",
            "palm-2-chat"
        ]
        
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
            {"role": "user", "content": "Can you help me with a coding problem?"}
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            tokens = counter.count_messages(messages)
            assert isinstance(tokens, int)
            assert tokens > 0
            
            # Should be more than individual message tokens
            individual_tokens = sum(counter.count(msg["content"]) for msg in messages)
            assert tokens >= individual_tokens  # Should include formatting overhead
    
    def test_new_models_approximation_consistency(self):
        """Test that new models provide consistent approximations."""
        test_texts = [
            "Hello",
            "Hello, world!",
            "This is a test message.",
            "This is a longer test message with more words and punctuation!",
            "A very long message that contains multiple sentences. Each sentence should contribute to the token count. The approximation should be reasonable and consistent across different model providers."
        ]
        
        test_models = [
            "phi-3-mini",
            "titan-text-express",
            "nemotron-4-340b",
            "granite-13b-chat",
            "codegen-16b",
            "starcoder",
            "claude-2.1-200k",
            "command-r-plus-04-2024",
            "palm-2-chat"
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            previous_tokens = 0
            
            for text in test_texts:
                tokens = counter.count(text)
                assert isinstance(tokens, int)
                assert tokens > 0
                
                # Longer texts should generally have more tokens
                assert tokens >= previous_tokens
                previous_tokens = tokens
    
    def test_new_models_special_characters(self):
        """Test new models with special characters and edge cases."""
        special_texts = [
            "",  # Empty string
            " ",  # Single space
            "\n",  # Single newline
            "🚀🌟💫",  # Emojis
            "Hello\n\nWorld",  # Multiple newlines
            "Hello    World",  # Multiple spaces
            "Hello... World???",  # Multiple punctuation
            "café naïve résumé",  # Accented characters
            "αβγδε",  # Greek letters
            "12345",  # Numbers only
            "!@#$%^&*()",  # Special symbols only
            "def func():\n    pass",  # Code with indentation
            "SELECT * FROM users WHERE id = 1;",  # SQL
            "console.log('Hello, world!');",  # JavaScript
        ]
        
        test_models = [
            "phi-3-mini",
            "titan-text-express",
            "nemotron-4-340b",
            "granite-13b-chat",
            "codegen-16b",
            "starcoder",
            "claude-2.1-200k",
            "command-r-plus-04-2024",
            "palm-2-chat"
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            
            for text in special_texts:
                tokens = counter.count(text)
                assert isinstance(tokens, int)
                assert tokens >= 0  # Should never be negative
                
                if text == "":
                    assert tokens == 0  # Empty string should always be 0 tokens
                elif text.strip():  # Non-empty, non-whitespace text
                    assert tokens > 0  # Should have at least 1 token
    
    def test_new_models_error_handling(self):
        """Test error handling for new models."""
        test_models = [
            "phi-3-mini",
            "titan-text-express",
            "nemotron-4-340b",
            "granite-13b-chat",
            "codegen-16b",
            "starcoder",
            "claude-2.1-200k",
            "command-r-plus-04-2024",
            "palm-2-chat"
        ]
        
        for model in test_models:
            counter = TokenCounter(model)
            
            # Test invalid input types
            with pytest.raises(TokenizationError):
                counter.count(123)
            
            with pytest.raises(TokenizationError):
                counter.count(None)
            
            with pytest.raises(TokenizationError):
                counter.count(["list", "of", "strings"])
            
            # Test invalid message formats
            with pytest.raises(TokenizationError):
                counter.count_messages("not a list")
            
            with pytest.raises(TokenizationError):
                counter.count_messages([{"role": "user"}])  # Missing content
            
            with pytest.raises(TokenizationError):
                counter.count_messages(["not a dict"])
    
    def test_new_provider_approximations(self):
        """Test provider-specific approximations for new providers."""
        test_text = "This is a comprehensive test message for tokenization approximation across all new providers."
        
        # Test all new providers
        new_providers_models = {
            "microsoft": "phi-3-mini",
            "amazon": "titan-text-express",
            "nvidia": "nemotron-4-340b",
            "ibm": "granite-13b-chat",
            "salesforce": "codegen-16b",
            "bigcode": "starcoder",
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
        assert len(unique_counts) >= 3, "Expected some variation in token counts across new providers"
    
    def test_code_specialized_models(self):
        """Test code-specialized models with various programming languages."""
        code_models = [
            "codegen-16b",  # Salesforce
            "starcoder",    # BigCode
            "granite-20b-code",  # IBM
            "phi-3-mini",   # Microsoft (good at code)
        ]
        
        code_samples = {
            "python": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""",
            "javascript": """
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

const memoizedFib = (function() {
    const cache = {};
    return function(n) {
        if (n in cache) return cache[n];
        return cache[n] = n <= 1 ? n : memoizedFib(n - 1) + memoizedFib(n - 2);
    };
})();
""",
            "java": """
public class BinarySearchTree {
    private Node root;
    
    private class Node {
        int data;
        Node left, right;
        
        Node(int data) {
            this.data = data;
            left = right = null;
        }
    }
    
    public void insert(int data) {
        root = insertRec(root, data);
    }
    
    private Node insertRec(Node root, int data) {
        if (root == null) {
            root = new Node(data);
            return root;
        }
        
        if (data < root.data)
            root.left = insertRec(root.left, data);
        else if (data > root.data)
            root.right = insertRec(root.right, data);
        
        return root;
    }
}
""",
        }
        
        for model in code_models:
            counter = TokenCounter(model)
            
            for language, code in code_samples.items():
                tokens = counter.count(code)
                assert isinstance(tokens, int)
                assert tokens > 20, f"{model} should return substantial tokens for {language} code"
                # Adjust upper limit based on code complexity - Java code is longer
                max_tokens = 300 if language == "java" else 200
                assert tokens < max_tokens, f"{model} token count for {language} code should be reasonable (got {tokens}, expected < {max_tokens})"
    
    def test_enterprise_models_business_text(self):
        """Test enterprise-focused models with business text."""
        enterprise_models = [
            "titan-text-express",  # Amazon
            "granite-13b-chat",   # IBM
            "phi-3-mini",         # Microsoft
        ]
        
        business_texts = [
            "Our quarterly earnings report shows a 15% increase in revenue year-over-year.",
            "The strategic partnership will enable us to expand into new market segments.",
            "Risk assessment indicates potential compliance issues with the new regulations.",
            "Customer satisfaction metrics have improved by 23% following the implementation.",
            "The digital transformation initiative requires significant investment in cloud infrastructure.",
        ]
        
        for model in enterprise_models:
            counter = TokenCounter(model)
            
            for text in business_texts:
                tokens = counter.count(text)
                assert isinstance(tokens, int)
                assert 10 <= tokens <= 30, f"{model} should handle business text appropriately"
    
    def test_vision_models_multimodal_text(self):
        """Test vision models with multimodal-related text."""
        # Mock tiktoken for OpenAI vision models
        with patch('toksum.core.tiktoken') as mock_tiktoken:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]
            mock_tiktoken.get_encoding.return_value = mock_encoder
            
            vision_models = [
                "gpt-4-vision",
                "gpt-4-vision-preview-0409",
                "gpt-4-vision-preview-1106"
            ]
            
            vision_texts = [
                "Describe what you see in this image.",
                "Extract all text visible in the picture.",
                "Analyze the chart and provide insights about the data trends.",
                "Identify all objects in the image and their locations.",
                "Compare these two images and highlight the differences.",
            ]
            
            for model in vision_models:
                counter = TokenCounter(model)
                
                for text in vision_texts:
                    tokens = counter.count(text)
                    assert tokens == 8  # Mocked to return 8 tokens
                    assert isinstance(tokens, int)
                    assert tokens > 0


class TestModelCountsV090:
    """Test model counts after adding v0.9.0 models."""
    
    def test_updated_total_model_count_v090(self):
        """Test that we now have 279+ total models."""
        models = get_supported_models()
        total_count = sum(len(model_list) for model_list in models.values())
        assert total_count >= 279, f"Expected at least 279 models, got {total_count}"
    
    def test_new_providers_in_supported_list(self):
        """Test that all new providers appear in the supported models list."""
        models = get_supported_models()
        
        # Check that new providers are present
        new_providers = ["microsoft", "amazon", "nvidia", "ibm", "salesforce", "bigcode"]
        for provider in new_providers:
            assert provider in models, f"New provider {provider} not found in supported models"
            assert len(models[provider]) > 0, f"Provider {provider} has no models"
    
    def test_new_provider_model_counts(self):
        """Test expected model counts for new providers."""
        models = get_supported_models()
        
        expected_counts = {
            "microsoft": 4,   # phi-3-mini, phi-3-small, phi-3-medium, phi-3.5-mini
            "amazon": 3,      # titan-text-express, titan-text-lite, titan-embed-text
            "nvidia": 2,      # nemotron-4-340b, nemotron-3-8b
            "ibm": 3,         # granite-13b-chat, granite-13b-instruct, granite-20b-code
            "salesforce": 3,  # codegen-16b, codegen-6b, codegen-2b
            "bigcode": 3,     # starcoder, starcoder2-15b, starcoderbase
        }
        
        for provider, expected_count in expected_counts.items():
            actual_count = len(models[provider])
            assert actual_count == expected_count, f"Expected {expected_count} {provider} models, got {actual_count}"
    
    def test_extended_provider_model_counts(self):
        """Test updated model counts for extended providers."""
        models = get_supported_models()
        
        # These providers should have additional models
        extended_providers_minimums = {
            "openai": 49,     # Added vision models
            "anthropic": 27,  # Added claude-2.1 and instant-2 models
            "cohere": 9,      # Added command-r-plus variants
            "google": 16,     # Added PaLM models
        }
        
        for provider, expected_min in extended_providers_minimums.items():
            actual_count = len(models[provider])
            assert actual_count >= expected_min, f"Expected at least {expected_min} {provider} models, got {actual_count}"
    
    def test_specific_new_models_present(self):
        """Test that specific new models are present in the supported models list."""
        models = get_supported_models()
        
        # Check that specific new models are present
        new_models_to_check = [
            ("microsoft", "phi-3-mini"),
            ("microsoft", "phi-3.5-mini"),
            ("amazon", "titan-text-express"),
            ("amazon", "titan-embed-text"),
            ("nvidia", "nemotron-4-340b"),
            ("nvidia", "nemotron-3-8b"),
            ("ibm", "granite-13b-chat"),
            ("ibm", "granite-20b-code"),
            ("salesforce", "codegen-16b"),
            ("salesforce", "codegen-2b"),
            ("bigcode", "starcoder"),
            ("bigcode", "starcoder2-15b"),
            ("anthropic", "claude-2.1-200k"),
            ("anthropic", "claude-instant-2"),
            ("openai", "gpt-4-vision"),
            ("openai", "gpt-4-vision-preview-0409"),
            ("cohere", "command-r-plus-04-2024"),
            ("cohere", "command-r-plus-08-2024"),
            ("google", "palm-2"),
            ("google", "palm-2-codechat"),
        ]
        
        for provider, model in new_models_to_check:
            assert model in models[provider], f"Model {model} not found in {provider} models list"
    
    def test_total_provider_count(self):
        """Test that we now have 32 total providers."""
        models = get_supported_models()
        total_providers = len(models)
        assert total_providers >= 32, f"Expected at least 32 providers, got {total_providers}"


if __name__ == "__main__":
    pytest.main([__file__])
