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
    
    @patch('toksum.core.tiktoken', None)
    def test_missing_tiktoken_dependency(self):
        """Test behavior when tiktoken is not available."""
        with pytest.raises(TokenizationError) as exc_info:
            TokenCounter("gpt-4")
        
        assert "tiktoken is required" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
