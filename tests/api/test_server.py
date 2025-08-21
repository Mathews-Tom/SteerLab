"""
Tests for FastAPI server.

This module tests the API endpoints, request/response handling,
and server lifecycle management.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from steerlab.api.schemas import ChatRequest, InteractionMode, PreferenceState
from steerlab.api.server import app, get_model, get_session


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_steerable_model():
    """Create a mock SteerableModel for testing."""
    model = Mock()
    model.model_name = "test-model"
    model.model = Mock()
    model.model.device = "cpu"
    model.generate.return_value = "Generated test response"
    model.load_steering_vectors = Mock()
    model.set_steering = Mock()
    model.clear_steering = Mock()
    return model


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_health_check_basic(self, client):
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "available_vectors" in data
        assert "uptime" in data

    def test_health_check_with_model(self, client, mock_steerable_model):
        """Test health check with loaded model."""
        with patch(
            "steerlab.api.server.app_state",
            {
                "model": mock_steerable_model,
                "available_vectors": {},
                "sessions": {},
                "start_time": 0,
            },
        ):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is True


class TestModelInfoEndpoint:
    """Test cases for model info endpoint."""

    def test_model_info(self, client, mock_steerable_model):
        """Test model info endpoint."""
        with patch("steerlab.api.server.get_model", return_value=mock_steerable_model):
            response = client.get("/info")

            assert response.status_code == 200
            data = response.json()
            assert (
                data["model_name"] == mock_steerable_model.model_name
            )  # Use dynamic value
            assert "available_preferences" in data
            assert "device_info" in data


class TestChatEndpoint:
    """Test cases for chat/generation endpoint."""

    def test_chat_basic(self, client, mock_steerable_model):
        """Test basic chat generation."""
        chat_request = {
            "prompt": "Tell me a joke",
            "max_length": 100,
            "temperature": 0.7,
        }

        mock_app_state = {
            "model": mock_steerable_model,
            "available_vectors": {},
            "sessions": {},
            "start_time": 0,
        }

        with (
            patch("steerlab.api.server.get_model", return_value=mock_steerable_model),
            patch("steerlab.api.server.app_state", mock_app_state),
        ):
            response = client.post("/chat", json=chat_request)

            assert response.status_code == 200
            data = response.json()
            assert data["generated_text"] == "Generated test response"
            assert "preferences_applied" in data
            assert "model_info" in data

    def test_chat_with_preferences(self, client, mock_steerable_model):
        """Test chat with preference overrides."""
        chat_request = {
            "prompt": "Be formal and creative",
            "preferences": {"formality": 0.8, "creativity": 0.6},
            "session_id": "test_session",
        }

        mock_app_state = {
            "model": mock_steerable_model,
            "available_vectors": {},
            "sessions": {},
            "start_time": 0,
        }

        with (
            patch("steerlab.api.server.get_model", return_value=mock_steerable_model),
            patch("steerlab.api.server.app_state", mock_app_state),
        ):
            response = client.post("/chat", json=chat_request)

            assert response.status_code == 200
            data = response.json()
            assert data["preferences_applied"]["formality"] == 0.8
            assert data["preferences_applied"]["creativity"] == 0.6
            assert data["session_id"] == "test_session"

    def test_chat_with_steering_vectors(self, client, mock_steerable_model):
        """Test chat with available steering vectors."""
        chat_request = {"prompt": "Test prompt", "preferences": {"formality": 0.5}}

        # Mock available vectors with tensor-like behavior
        mock_tensor_0 = torch.randn(768)
        mock_tensor_1 = torch.randn(768)
        mock_vectors = {"layer.0": mock_tensor_0, "layer.1": mock_tensor_1}
        available_vectors = {"formality": mock_vectors}

        mock_app_state = {
            "model": mock_steerable_model,
            "available_vectors": available_vectors,
            "sessions": {},
            "start_time": 0,
        }

        with (
            patch("steerlab.api.server.get_model", return_value=mock_steerable_model),
            patch("steerlab.api.server.app_state", mock_app_state),
        ):
            response = client.post("/chat", json=chat_request)

            assert response.status_code == 200
            # Should have called load_steering_vectors and set_steering
            mock_steerable_model.load_steering_vectors.assert_called_once()
            mock_steerable_model.set_steering.assert_called_once()

    def test_chat_invalid_request(self, client):
        """Test chat with invalid request data."""
        invalid_request = {
            "prompt": "",  # Empty prompt should be invalid
            "temperature": 3.0,  # Invalid temperature
        }

        response = client.post("/chat", json=invalid_request)

        assert response.status_code == 422  # Validation error


class TestPreferenceEndpoints:
    """Test cases for preference management endpoints."""

    def test_update_preferences(self, client):
        """Test updating user preferences."""
        preference_request = {
            "preferences": {"formality": 0.8, "creativity": -0.3},
            "mode": "select",
            "session_id": "test_session",
        }

        response = client.post("/preferences", json=preference_request)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["updated_preferences"]["preferences"]["formality"] == 0.8
        assert data["updated_preferences"]["preferences"]["creativity"] == -0.3

    def test_get_preferences_default(self, client):
        """Test getting default preferences."""
        response = client.get("/preferences")

        assert response.status_code == 200
        data = response.json()
        assert "preferences" in data
        assert "mode" in data
        assert data["mode"] == "select"

    def test_get_preferences_with_session(self, client):
        """Test getting preferences for specific session."""
        # First set some preferences
        preference_request = {
            "preferences": {"test": 0.5},
            "session_id": "specific_session",
        }
        client.post("/preferences", json=preference_request)

        # Then retrieve them
        response = client.get("/preferences?session_id=specific_session")

        assert response.status_code == 200
        data = response.json()
        assert data["preferences"]["test"] == 0.5

    def test_update_preferences_invalid_values(self, client):
        """Test updating preferences with invalid values."""
        invalid_request = {
            "preferences": {"test": 2.0}  # Out of range [-1, 1]
        }

        response = client.post("/preferences", json=invalid_request)

        assert response.status_code == 422  # Validation error


class TestVectorEndpoints:
    """Test cases for vector management endpoints."""

    @patch("steerlab.api.server.SteeringVectorManager")
    def test_compute_vectors(self, mock_vector_manager_class, client):
        """Test vector computation endpoint."""
        mock_manager = Mock()
        mock_manager.compute_steering_vectors.return_value = {
            "layer.0": Mock(),
            "layer.1": Mock(),
        }
        mock_manager.save_vectors = Mock()
        mock_manager.cleanup = Mock()
        mock_vector_manager_class.return_value = mock_manager

        compute_request = {
            "model_name": "test-model",
            "preference_name": "formality",
            "positive_examples": ["Formal example 1", "Formal example 2"],
            "negative_examples": ["Informal example 1", "Informal example 2"],
            "max_length": 256,
        }

        response = client.post("/compute-vectors", json=compute_request)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["preference_name"] == "formality"
        assert data["num_vectors"] == 2

        # Verify manager was called correctly
        mock_manager.compute_steering_vectors.assert_called_once()
        mock_manager.save_vectors.assert_called_once()
        mock_manager.cleanup.assert_called_once()

    def test_list_vectors_empty(self, client):
        """Test listing vectors when none are available."""
        with patch("steerlab.api.server.app_state", {"available_vectors": {}}):
            response = client.get("/vectors")

            assert response.status_code == 200
            data = response.json()
            assert data["available_vectors"] == []
            assert data["count"] == 0

    def test_list_vectors_with_data(self, client):
        """Test listing vectors with available data."""
        available_vectors = {"formality": {}, "creativity": {}}

        with patch(
            "steerlab.api.server.app_state", {"available_vectors": available_vectors}
        ):
            response = client.get("/vectors")

            assert response.status_code == 200
            data = response.json()
            assert set(data["available_vectors"]) == {"formality", "creativity"}
            assert data["count"] == 2


class TestSessionManagement:
    """Test cases for session management."""

    def test_delete_session_existing(self, client):
        """Test deleting an existing session."""
        # First create a session by updating preferences
        preference_request = {"preferences": {"test": 0.5}, "session_id": "delete_me"}
        client.post("/preferences", json=preference_request)

        # Then delete it
        response = client.delete("/sessions/delete_me")

        assert response.status_code == 200
        data = response.json()
        assert "deleted" in data["message"].lower()

    def test_delete_session_nonexistent(self, client):
        """Test deleting a non-existent session."""
        response = client.delete("/sessions/nonexistent")

        assert response.status_code == 404


class TestDependencies:
    """Test cases for dependency injection functions."""

    @pytest.mark.asyncio
    async def test_get_model_new(self):
        """Test getting a new model instance."""
        with (
            patch("steerlab.api.server.app_state", {"model": None}),
            patch("steerlab.api.server.SteerableModel") as mock_model_class,
        ):
            mock_instance = Mock()
            mock_model_class.return_value = mock_instance

            result = await get_model("test-model")

            assert result == mock_instance
            mock_model_class.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_get_model_existing(self):
        """Test getting existing model instance."""
        existing_model = Mock()
        existing_model.model_name = "test-model"

        with patch("steerlab.api.server.app_state", {"model": existing_model}):
            result = await get_model("test-model")

            assert result == existing_model

    def test_get_session_new(self):
        """Test getting a new session."""
        with patch("steerlab.api.server.app_state", {"sessions": {}}):
            session = get_session("new_session")

            assert isinstance(session, PreferenceState)
            assert session.mode == InteractionMode.SELECT

    def test_get_session_existing(self):
        """Test getting an existing session."""
        existing_session = PreferenceState(preferences={"test": 0.5})

        with patch(
            "steerlab.api.server.app_state",
            {"sessions": {"existing": existing_session}},
        ):
            session = get_session("existing")

            assert session == existing_session
            assert session.preferences["test"] == 0.5
