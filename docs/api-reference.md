# API Reference

This document provides a reference for the core classes, Pydantic schemas, and API endpoints in the SteerLab framework.

---

## Core Python Modules

### `steering_vectors.py`

#### `class SteeringVectorManager`

A utility class to handle the loading, caching, and combination of steering vectors.

- `load_vectors(path: str) -> dict`: Loads a steering vector dictionary from a `.safetensors` file.
- `get_vector(axis: str, layer: int) -> torch.Tensor`: Retrieves a specific vector for a given preference axis and layer.
- `combine_vectors(prefs: dict[str, float], layer: int) -> torch.Tensor`: Computes the final, combined steering vector to be injected at a specific layer, weighted by the preference strengths.

### `model.py`

#### `class SteerableModel`

A wrapper class to manage a Hugging Face model and the steering lifecycle.

- `__init__(self, model_name_or_path: str)`: Loads a base model from the Hugging Face Hub or a local path.
- `generate_steered(self, prompt: str, prefs: dict[str, float], vector_manager: SteeringVectorManager, target_layers: list[int], **kwargs) -> str`: The primary method for generating steered text. It orchestrates the entire process: registering hooks, running generation, and ensuring hooks are cleared in a `finally` block.

---

### Pydantic Schemas (`schemas.py`)

These models define the data structures for API requests and responses.

#### `class PreferenceState(BaseModel)`

- `user_id: str`
- `preferences: dict[str, float]`

#### `class ChatRequest(BaseModel)`

- `prompt: str`
- `user_id: str`
- `mode: Literal['select', 'calibrate', 'learn']`
- `settings: dict | None = None`

#### `class ChatResponse(BaseModel)`

- `text: str`
- `updated_prefs: dict[str, float]`

---

## Chat Application Server (FastAPI)

The backend server exposes RESTful endpoints for interaction.

| Endpoint | Method | Request Body | Response Body | Description |
|---|---|---|---|---|
| `/chat` | `POST` | `ChatRequest` | `ChatResponse` | Main endpoint for a conversational turn. Dispatches to SELECT, CALIBRATE, or LEARN logic based on the `mode`. |
| `/preferences/{user_id}` | `GET` | - | `PreferenceState` | Retrieves the current steering preferences for a user. |
| `/preferences/{user_id}` | `PUT` | `dict[str, float]` | `PreferenceState` | Directly sets or updates the steering preferences for a user (used by SELECT mode). |
| `/calibrate/start/{user_id}` | `POST` | `{axis: str}` | `CalibrationState` | Initiates a calibration session for a preference axis. |
| `/calibrate/respond/{user_id}`| `POST` | `CalibrationResponse` | `CalibrationState` | Submits a user's choice for a calibration pair and gets the next pair or the final result. |
