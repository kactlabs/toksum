"""
Tests for the new exception features in toksum.
"""

import pytest
from toksum.exceptions import (
    ModelDeprecationError,
    RateLimitError,
    ConfigurationError,
    BatchProcessingError,
    ToksumError
)


class TestModelDeprecationError:
    """Test cases for ModelDeprecationError."""
    
    def test_basic_deprecation_error(self):
        """Test basic deprecation error creation."""
        error = ModelDeprecationError("old-model")
        
        assert error.model == "old-model"
        assert error.replacement_model is None
        assert error.deprecation_date is None
        assert error.removal_date is None
        assert "old-model" in str(error)
        assert "deprecated" in str(error)
    
    def test_full_deprecation_error(self):
        """Test deprecation error with all parameters."""
        error = ModelDeprecationError(
            model="text-davinci-002",
            replacement_model="gpt-3.5-turbo",
            deprecation_date="2023-01-04",
            removal_date="2024-01-04"
        )
        
        assert error.model == "text-davinci-002"
        assert error.replacement_model == "gpt-3.5-turbo"
        assert error.deprecation_date == "2023-01-04"
        assert error.removal_date == "2024-01-04"
        
        error_str = str(error)
        assert "text-davinci-002" in error_str
        assert "gpt-3.5-turbo" in error_str
        assert "2023-01-04" in error_str
        assert "2024-01-04" in error_str
    
    def test_deprecation_error_inheritance(self):
        """Test that ModelDeprecationError inherits from ToksumError."""
        error = ModelDeprecationError("test-model")
        assert isinstance(error, ToksumError)
        assert isinstance(error, Exception)


class TestRateLimitError:
    """Test cases for RateLimitError."""
    
    def test_basic_rate_limit_error(self):
        """Test basic rate limit error creation."""
        error = RateLimitError("Too many requests")
        
        assert error.retry_after is None
        assert error.provider is None
        assert "Too many requests" in str(error)
        assert "Rate limit exceeded" in str(error)
    
    def test_full_rate_limit_error(self):
        """Test rate limit error with all parameters."""
        error = RateLimitError(
            message="API quota exceeded",
            retry_after=300,
            provider="openai"
        )
        
        assert error.retry_after == 300
        assert error.provider == "openai"
        
        error_str = str(error)
        assert "API quota exceeded" in error_str
        assert "openai" in error_str
        assert "300 seconds" in error_str
    
    def test_rate_limit_error_inheritance(self):
        """Test that RateLimitError inherits from ToksumError."""
        error = RateLimitError("test message")
        assert isinstance(error, ToksumError)
        assert isinstance(error, Exception)


class TestConfigurationError:
    """Test cases for ConfigurationError."""
    
    def test_basic_configuration_error(self):
        """Test basic configuration error creation."""
        error = ConfigurationError("Invalid setting")
        
        assert error.config_key is None
        assert error.suggested_fix is None
        assert "Invalid setting" in str(error)
        assert "Configuration error" in str(error)
    
    def test_full_configuration_error(self):
        """Test configuration error with all parameters."""
        error = ConfigurationError(
            message="Missing API key",
            config_key="OPENAI_API_KEY",
            suggested_fix="Set environment variable OPENAI_API_KEY"
        )
        
        assert error.config_key == "OPENAI_API_KEY"
        assert error.suggested_fix == "Set environment variable OPENAI_API_KEY"
        
        error_str = str(error)
        assert "Missing API key" in error_str
        assert "OPENAI_API_KEY" in error_str
        assert "Set environment variable" in error_str
    
    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from ToksumError."""
        error = ConfigurationError("test message")
        assert isinstance(error, ToksumError)
        assert isinstance(error, Exception)


class TestBatchProcessingError:
    """Test cases for BatchProcessingError."""
    
    def test_basic_batch_processing_error(self):
        """Test basic batch processing error creation."""
        error = BatchProcessingError("Batch failed")
        
        assert error.failed_items == []
        assert error.total_items is None
        assert "Batch failed" in str(error)
        assert "Batch processing failed" in str(error)
    
    def test_batch_processing_error_with_items(self):
        """Test batch processing error with failed items."""
        error = BatchProcessingError(
            message="Multiple items failed",
            failed_items=[1, 3, 5],
            total_items=10
        )
        
        assert error.failed_items == [1, 3, 5]
        assert error.total_items == 10
        
        error_str = str(error)
        assert "Multiple items failed" in error_str
        assert "7/10 items processed successfully" in error_str
    
    def test_batch_processing_error_failed_only(self):
        """Test batch processing error with only failed items count."""
        error = BatchProcessingError(
            message="Some items failed",
            failed_items=[0, 2, 4]
        )
        
        assert error.failed_items == [0, 2, 4]
        assert error.total_items is None
        
        error_str = str(error)
        assert "Some items failed" in error_str
        assert "3 items failed" in error_str
    
    def test_batch_processing_error_inheritance(self):
        """Test that BatchProcessingError inherits from ToksumError."""
        error = BatchProcessingError("test message")
        assert isinstance(error, ToksumError)
        assert isinstance(error, Exception)


class TestExceptionIntegration:
    """Test integration of new exceptions with existing ones."""
    
    def test_all_exceptions_inherit_from_toksum_error(self):
        """Test that all custom exceptions inherit from ToksumError."""
        exceptions = [
            ModelDeprecationError("test"),
            RateLimitError("test"),
            ConfigurationError("test"),
            BatchProcessingError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, ToksumError)
            assert isinstance(exc, Exception)
    
    def test_exception_string_representations(self):
        """Test that all exceptions have meaningful string representations."""
        exceptions = [
            ModelDeprecationError("test-model", "new-model"),
            RateLimitError("rate limit", retry_after=60),
            ConfigurationError("config issue", config_key="TEST_KEY"),
            BatchProcessingError("batch failed", failed_items=[1, 2])
        ]
        
        for exc in exceptions:
            error_str = str(exc)
            assert len(error_str) > 0
            assert error_str != "ToksumError"  # Should have specific message
    
    def test_exception_attributes_accessible(self):
        """Test that exception-specific attributes are accessible."""
        # ModelDeprecationError
        dep_error = ModelDeprecationError("old", "new", "2023-01-01", "2024-01-01")
        assert hasattr(dep_error, 'model')
        assert hasattr(dep_error, 'replacement_model')
        assert hasattr(dep_error, 'deprecation_date')
        assert hasattr(dep_error, 'removal_date')
        
        # RateLimitError
        rate_error = RateLimitError("test", retry_after=30, provider="test")
        assert hasattr(rate_error, 'retry_after')
        assert hasattr(rate_error, 'provider')
        
        # ConfigurationError
        config_error = ConfigurationError("test", config_key="KEY", suggested_fix="fix")
        assert hasattr(config_error, 'config_key')
        assert hasattr(config_error, 'suggested_fix')
        
        # BatchProcessingError
        batch_error = BatchProcessingError("test", failed_items=[1], total_items=5)
        assert hasattr(batch_error, 'failed_items')
        assert hasattr(batch_error, 'total_items')


if __name__ == "__main__":
    pytest.main([__file__])