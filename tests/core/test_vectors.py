"""
Tests for SteeringVectorManager and CAA algorithm.

This module tests the vector computation, loading/saving functionality,
and the Contrastive Activation Addition algorithm.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from steerlab.core.vectors import SteeringVectorManager


@pytest.fixture
def mock_model():
    """Create a mock HuggingFace model for testing."""
    model = Mock()
    model.device = torch.device("cpu")
    model.named_modules.return_value = [
        ("model.layers.0", Mock()),
        ("model.layers.1", Mock()),
        ("model.embed_tokens", Mock()),
        ("model.norm", Mock()),
    ]
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"

    # Mock the call to return proper tensor dict
    def tokenize_side_effect(*args, **kwargs):
        result = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        # Add .to() method to the result dict
        result_mock = Mock()
        result_mock.__getitem__ = lambda _, key: result[key]
        result_mock.to = Mock(return_value=result)
        return result_mock

    tokenizer.side_effect = tokenize_side_effect
    return tokenizer


@pytest.fixture
def vector_manager(mock_model, mock_tokenizer):
    """Create a SteeringVectorManager instance with mocked dependencies."""
    with (
        patch(
            "steerlab.core.vectors.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
        patch(
            "steerlab.core.vectors.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
    ):
        manager = SteeringVectorManager("test-model")
        manager.model = mock_model
        manager.tokenizer = mock_tokenizer
        manager.target_layers = ["model.layers.0", "model.layers.1"]
        return manager


class TestSteeringVectorManager:
    """Test cases for SteeringVectorManager class."""

    def test_initialization(self):
        """Test vector manager initialization."""
        manager = SteeringVectorManager("test-model")

        assert manager.model_name == "test-model"
        assert manager.model is None
        assert manager.tokenizer is None
        assert manager.target_layers == []

    def test_identify_target_layers(self, vector_manager):
        """Test identification of target layers."""
        vector_manager._identify_target_layers()

        # Should identify the layer blocks but not embeddings/norms
        assert "model.layers.0" in vector_manager.target_layers
        assert "model.layers.1" in vector_manager.target_layers
        assert "model.embed_tokens" not in vector_manager.target_layers
        assert "model.norm" not in vector_manager.target_layers

    def test_collect_activations(self, vector_manager):
        """Test activation collection from examples."""
        examples = ["Example 1", "Example 2"]

        # Mock layer and hooks - only return layer for layers that exist in target_layers
        mock_layer = Mock()
        mock_hook = Mock()
        mock_layer.register_forward_hook.return_value = mock_hook

        def get_layer_by_name_side_effect(layer_name):
            if layer_name in vector_manager.target_layers:
                return mock_layer
            return None

        vector_manager._get_layer_by_name = Mock(
            side_effect=get_layer_by_name_side_effect
        )

        # Mock model forward pass to trigger hooks
        def mock_forward(**kwargs):
            # Simulate calling the hook
            for layer_name in vector_manager.target_layers:
                hook_fn = mock_layer.register_forward_hook.call_args[0][0]
                hidden_states = torch.randn(
                    2, 5, 768
                )  # batch_size=2, seq_len=5, hidden_dim=768
                hook_fn(mock_layer, None, hidden_states)
            return Mock()

        vector_manager.model.side_effect = mock_forward

        activations = vector_manager._collect_activations(examples, max_length=256)

        # Should return activations for each target layer
        assert len(activations) == len(vector_manager.target_layers)
        for layer_name in vector_manager.target_layers:
            assert layer_name in activations
            assert isinstance(activations[layer_name], torch.Tensor)

    @patch("steerlab.core.vectors.SteeringVectorManager._collect_activations")
    def test_compute_steering_vectors(self, mock_collect, vector_manager):
        """Test steering vector computation using CAA."""
        positive_examples = ["Positive example 1", "Positive example 2"]
        negative_examples = ["Negative example 1", "Negative example 2"]

        # Mock activations
        pos_activations = {
            "layer.0": torch.tensor([[1.0, 2.0], [1.5, 2.5]]),  # 2 examples, 2 dims
            "layer.1": torch.tensor([[0.5, 1.0], [0.8, 1.2]]),
        }
        neg_activations = {
            "layer.0": torch.tensor([[-1.0, -2.0], [-1.5, -2.5]]),
            "layer.1": torch.tensor([[-0.5, -1.0], [-0.8, -1.2]]),
        }

        mock_collect.side_effect = [pos_activations, neg_activations]
        vector_manager.target_layers = ["layer.0", "layer.1"]

        vectors = vector_manager.compute_steering_vectors(
            positive_examples, negative_examples, "test_preference"
        )

        # Should compute vectors as difference between positive and negative centroids
        assert len(vectors) == 2
        assert "layer.0" in vectors
        assert "layer.1" in vectors

        # Check that vectors are computed correctly (pos_mean - neg_mean)
        expected_layer0 = pos_activations["layer.0"].mean(dim=0) - neg_activations[
            "layer.0"
        ].mean(dim=0)
        torch.testing.assert_close(vectors["layer.0"], expected_layer0)

    def test_save_and_load_vectors(self):
        """Test saving and loading vectors in safetensors format."""
        vectors = {"layer.0": torch.randn(768), "layer.1": torch.randn(768)}
        metadata = {"test_key": "test_value"}

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_vectors.safetensors"

            # Test saving
            manager = SteeringVectorManager("test-model")
            manager.save_vectors(vectors, output_path, metadata)

            assert output_path.exists()

            # Test loading
            loaded_vectors, loaded_metadata = manager.load_vectors(output_path)

            assert len(loaded_vectors) == len(vectors)
            for key in vectors:
                assert key in loaded_vectors
                torch.testing.assert_close(loaded_vectors[key], vectors[key])

    def test_load_examples_from_json_list(self):
        """Test loading examples from JSON list format."""
        examples = ["Example 1", "Example 2", "Example 3"]

        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "examples.json"
            with open(json_path, "w") as f:
                json.dump(examples, f)

            loaded = SteeringVectorManager.load_examples_from_json(json_path)

            assert loaded == examples

    def test_load_examples_from_json_dict(self):
        """Test loading examples from JSON dict format."""
        data = {"examples": ["Example 1", "Example 2"]}

        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "examples.json"
            with open(json_path, "w") as f:
                json.dump(data, f)

            loaded = SteeringVectorManager.load_examples_from_json(json_path)

            assert loaded == data["examples"]

    def test_load_examples_from_json_texts(self):
        """Test loading examples from JSON with 'texts' key."""
        data = {"texts": ["Text 1", "Text 2"]}

        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "examples.json"
            with open(json_path, "w") as f:
                json.dump(data, f)

            loaded = SteeringVectorManager.load_examples_from_json(json_path)

            assert loaded == data["texts"]

    def test_load_examples_invalid_format(self):
        """Test loading examples with invalid JSON format."""
        data = {"invalid_key": ["Example 1", "Example 2"]}

        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "examples.json"
            with open(json_path, "w") as f:
                json.dump(data, f)

            with pytest.raises(ValueError):
                SteeringVectorManager.load_examples_from_json(json_path)

    def test_get_layer_by_name_success(self, vector_manager):
        """Test successful layer retrieval."""
        # Mock nested model structure
        layer = Mock()
        vector_manager.model.test = Mock()
        vector_manager.model.test.layer = layer

        result = vector_manager._get_layer_by_name("test.layer")

        assert result == layer

    def test_get_layer_by_name_failure(self, vector_manager):
        """Test failed layer retrieval."""
        # Create a fresh mock model that doesn't have the nonexistent attribute
        vector_manager.model = Mock(spec=[])  # Empty spec means no attributes

        result = vector_manager._get_layer_by_name("nonexistent.layer")

        assert result is None

    def test_cleanup(self, vector_manager):
        """Test resource cleanup."""
        # Set up some resources
        vector_manager.model = Mock()
        vector_manager.tokenizer = Mock()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
        ):
            vector_manager.cleanup()

            assert vector_manager.model is None
            assert vector_manager.tokenizer is None
            mock_empty_cache.assert_called_once()

    def test_load_model_for_analysis(self, mock_model, mock_tokenizer):
        """Test model loading for vector analysis."""
        with (
            patch(
                "steerlab.core.vectors.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steerlab.core.vectors.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            manager = SteeringVectorManager("test-model")
            manager._load_model_for_analysis()

            assert manager.model == mock_model
            assert manager.tokenizer == mock_tokenizer
            assert len(manager.target_layers) > 0  # Should identify some layers
