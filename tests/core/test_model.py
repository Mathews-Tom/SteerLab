"""
Tests for SteerableModel class.

This module tests the core steerable model functionality including
hook management, vector loading, and generation with steering.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from steerlab.core.model import SteerableModel


@pytest.fixture
def mock_model():
    """Create a mock HuggingFace model for testing."""
    model = Mock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token_id = 0

    # Mock the call to return proper tensor dict
    def tokenize_side_effect(*args, **kwargs):
        result = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        # Add .to() method to the result dict
        result_mock = Mock()
        result_mock.__getitem__ = lambda _, key: result[key]
        result_mock.to = Mock(return_value=result)
        return result_mock

    tokenizer.side_effect = tokenize_side_effect
    tokenizer.decode.return_value = (
        "Test promptGenerated text"  # Include prompt for slicing
    )
    return tokenizer


@pytest.fixture
def steerable_model(mock_model, mock_tokenizer):
    """Create a SteerableModel instance with mocked dependencies."""
    with (
        patch(
            "steerlab.core.model.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
        patch(
            "steerlab.core.model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
    ):
        model = SteerableModel("test-model")
        model.model = mock_model
        model.tokenizer = mock_tokenizer
        return model


class TestSteerableModel:
    """Test cases for SteerableModel class."""

    def test_initialization(self, steerable_model):
        """Test model initialization."""
        assert steerable_model.model_name == "test-model"
        assert steerable_model.model is not None
        assert steerable_model.tokenizer is not None
        assert steerable_model.active_hooks == []
        assert steerable_model.steering_vectors == {}
        assert steerable_model.steering_config == {}

    def test_load_steering_vectors(self, steerable_model):
        """Test loading steering vectors."""
        vectors = {"layer.0": torch.randn(768), "layer.1": torch.randn(768)}

        steerable_model.load_steering_vectors(vectors)

        assert steerable_model.steering_vectors == vectors
        assert len(steerable_model.steering_vectors) == 2

    def test_set_steering_config(self, steerable_model):
        """Test setting steering configuration."""
        config = {"formality": 0.5, "creativity": -0.3}

        # Mock _register_hooks to avoid actual hook registration
        steerable_model._register_hooks = Mock()

        steerable_model.set_steering(config)

        assert steerable_model.steering_config == config
        steerable_model._register_hooks.assert_called_once()

    def test_clear_steering(self, steerable_model):
        """Test clearing steering configuration."""
        steerable_model.steering_config = {"test": 0.5}
        steerable_model._remove_hooks = Mock()

        steerable_model.clear_steering()

        assert steerable_model.steering_config == {}
        steerable_model._remove_hooks.assert_called_once()

    def test_hook_registration_and_removal(self, steerable_model):
        """Test hook registration and removal."""
        # Mock model layers
        mock_layer = Mock()
        mock_hook = Mock()
        mock_layer.register_forward_hook.return_value = mock_hook

        steerable_model._get_layer_by_name = Mock(return_value=mock_layer)
        steerable_model.steering_vectors = {"test.layer": torch.randn(768)}

        # Test hook registration
        steerable_model._register_hooks()

        assert len(steerable_model.active_hooks) == 1
        assert steerable_model.active_hooks[0] == mock_hook
        mock_layer.register_forward_hook.assert_called_once()

        # Test hook removal
        steerable_model._remove_hooks()

        assert len(steerable_model.active_hooks) == 0
        mock_hook.remove.assert_called_once()

    def test_generate_basic(self, steerable_model):
        """Test basic text generation."""
        prompt = "Test prompt"

        result = steerable_model.generate(prompt)

        # Should call tokenizer and model.generate
        steerable_model.tokenizer.assert_called()
        steerable_model.model.generate.assert_called_once()
        assert isinstance(result, str)

    def test_generate_with_steering(self, steerable_model):
        """Test text generation with steering applied."""
        # Setup steering
        vectors = {"test.layer": torch.randn(768)}
        steerable_model.load_steering_vectors(vectors)

        # Mock hook registration/removal
        steerable_model._register_hooks = Mock()
        steerable_model._remove_hooks = Mock()

        config = {"test": 0.5}
        steerable_model.set_steering(config)

        prompt = "Test prompt"
        result = steerable_model.generate(prompt)

        # Should clear steering after generation
        steerable_model._remove_hooks.assert_called()
        assert isinstance(result, str)

    def test_context_manager(self, steerable_model):
        """Test context manager behavior."""
        steerable_model.clear_steering = Mock()

        with steerable_model:
            pass

        # Should clear steering on exit
        steerable_model.clear_steering.assert_called_once()

    def test_get_layer_by_name_success(self, steerable_model):
        """Test successful layer retrieval."""
        # Mock nested model structure
        layer = Mock()
        steerable_model.model.test = Mock()
        steerable_model.model.test.layer = layer

        result = steerable_model._get_layer_by_name("test.layer")

        assert result == layer

    def test_get_layer_by_name_failure(self, steerable_model):
        """Test failed layer retrieval."""
        # Create a fresh mock model that doesn't have the nonexistent attribute
        steerable_model.model = Mock(spec=[])  # Empty spec means no attributes

        result = steerable_model._get_layer_by_name("nonexistent.layer")

        assert result is None

    def test_steering_hook_function(self, steerable_model):
        """Test the steering hook function behavior."""
        vector = torch.randn(768)
        steerable_model.steering_config = {"test": 0.5}

        hook_fn = steerable_model._create_steering_hook(vector)

        # Test with tuple output (typical transformer output)
        hidden_states = torch.randn(1, 10, 768)
        mock_module = Mock()
        output = (hidden_states, Mock())

        result = hook_fn(mock_module, None, output)

        assert isinstance(result, tuple)
        assert result[0].shape == hidden_states.shape
        # The result should be different from input due to steering
        assert not torch.equal(result[0], hidden_states)

    def test_steering_hook_function_no_steering(self, steerable_model):
        """Test hook function with no active steering."""
        vector = torch.randn(768)
        steerable_model.steering_config = {}  # No active steering

        hook_fn = steerable_model._create_steering_hook(vector)

        hidden_states = torch.randn(1, 10, 768)
        mock_module = Mock()
        output = (hidden_states, Mock())

        result = hook_fn(mock_module, None, output)

        # Should return original output unchanged
        assert result == output
