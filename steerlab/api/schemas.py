"""
Pydantic schemas for SteerLab API.

This module defines the data models used for API requests and responses,
providing type safety and automatic validation for all API interactions.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class InteractionMode(str, Enum):
    """Enum for different interaction modes."""

    SELECT = "select"
    CALIBRATE = "calibrate"
    LEARN = "learn"


class PreferenceState(BaseModel):
    """
    Represents the current state of user preferences.

    This model captures both explicit preference settings and learned preferences
    from user interactions.
    """

    preferences: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of preference names to their strengths (-1.0 to 1.0)",
    )
    mode: InteractionMode = Field(
        default=InteractionMode.SELECT, description="Current interaction mode"
    )

    @field_validator("preferences")
    @classmethod
    def validate_preference_values(cls, v):
        """Ensure preference values are within valid range."""
        for pref_name, value in v.items():
            if not isinstance(value, int | float):
                raise ValueError(f"Preference value must be numeric: {pref_name}")
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"Preference value must be between -1.0 and 1.0: {pref_name} = {value}"
                )
        return v


class ChatRequest(BaseModel):
    """
    Request model for chat/generation endpoints.
    """

    prompt: str = Field(
        ...,
        description="The input prompt for text generation",
        min_length=1,
        max_length=8192,
    )
    preferences: dict[str, float] | None = Field(
        default=None, description="Optional preference overrides for this request"
    )
    max_length: int = Field(
        default=512, description="Maximum length of generated text", gt=0, le=2048
    )
    temperature: float = Field(
        default=0.7, description="Sampling temperature for generation", ge=0.0, le=2.0
    )
    do_sample: bool = Field(
        default=True, description="Whether to use sampling during generation"
    )
    session_id: str | None = Field(
        default=None, description="Optional session identifier for tracking"
    )

    @field_validator("preferences")
    @classmethod
    def validate_preference_values(cls, v):
        """Ensure preference values are within valid range."""
        if v is not None:
            for pref_name, value in v.items():
                if not isinstance(value, int | float):
                    raise ValueError(f"Preference value must be numeric: {pref_name}")
                if not -1.0 <= value <= 1.0:
                    raise ValueError(
                        f"Preference value must be between -1.0 and 1.0: {pref_name} = {value}"
                    )
        return v


class ChatResponse(BaseModel):
    """
    Response model for chat/generation endpoints.
    """

    generated_text: str = Field(..., description="The generated text response")
    preferences_applied: dict[str, float] = Field(
        ..., description="The preference settings that were applied during generation"
    )
    session_id: str | None = Field(
        default=None, description="Session identifier if provided in request"
    )
    model_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the model and generation parameters used",
    )
    generation_stats: dict[str, Any] | None = Field(
        default=None, description="Optional statistics about the generation process"
    )


class VectorComputeRequest(BaseModel):
    """
    Request model for computing steering vectors.
    """

    model_name: str = Field(..., description="HuggingFace model identifier")
    preference_name: str = Field(
        ...,
        description="Name of the preference being learned",
        min_length=1,
        max_length=100,
    )
    positive_examples: list[str] = Field(
        ...,
        description="list of positive examples for the preference",
        min_length=1,
        max_length=1000,
    )
    negative_examples: list[str] = Field(
        ...,
        description="List of negative examples for the preference",
        min_length=1,
        max_length=1000,
    )
    max_length: int = Field(
        default=256,
        description="Maximum sequence length for processing examples",
        gt=0,
        le=1024,
    )
    output_path: str | None = Field(
        default=None, description="Optional custom output path for vectors"
    )


class VectorComputeResponse(BaseModel):
    """
    Response model for vector computation requests.
    """

    success: bool = Field(..., description="Whether vector computation was successful")
    preference_name: str = Field(..., description="Name of the computed preference")
    vector_path: str | None = Field(
        default=None, description="Path where vectors were saved"
    )
    num_vectors: int = Field(
        default=0, description="Number of steering vectors computed"
    )
    message: str = Field(default="", description="Status message or error details")
    computation_stats: dict[str, Any] | None = Field(
        default=None, description="Optional statistics about the computation process"
    )


class PreferenceUpdateRequest(BaseModel):
    """
    Request model for updating user preferences.
    """

    preferences: dict[str, float] = Field(..., description="Updated preference values")
    mode: InteractionMode | None = Field(
        default=None, description="Optional mode change"
    )
    session_id: str | None = Field(default=None, description="Session identifier")

    @field_validator("preferences")
    @classmethod
    def validate_preference_values(cls, v):
        """Ensure preference values are within valid range."""
        for pref_name, value in v.items():
            if not isinstance(value, int | float):
                raise ValueError(f"Preference value must be numeric: {pref_name}")
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"Preference value must be between -1.0 and 1.0: {pref_name} = {value}"
                )
        return v


class PreferenceUpdateResponse(BaseModel):
    """
    Response model for preference updates.
    """

    success: bool = Field(..., description="Whether the update was successful")
    updated_preferences: PreferenceState = Field(
        ..., description="The updated preference state"
    )
    message: str = Field(
        default="Preferences updated successfully", description="Status message"
    )


class ModelInfoResponse(BaseModel):
    """
    Response model for model information endpoint.
    """

    model_name: str = Field(..., description="Name of the loaded model")
    available_preferences: list[str] = Field(
        default_factory=list, description="List of available preference types"
    )
    model_parameters: dict[str, Any] | None = Field(
        default=None, description="Model configuration parameters"
    )
    device_info: str | None = Field(
        default=None, description="Information about the device the model is running on"
    )


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.
    """

    status: str = Field(default="healthy", description="Service health status")
    model_loaded: bool = Field(
        ..., description="Whether the model is successfully loaded"
    )
    available_vectors: list[str] = Field(
        default_factory=list, description="List of available steering vector sets"
    )
    uptime: float | None = Field(default=None, description="Service uptime in seconds")
