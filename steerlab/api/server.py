"""
FastAPI server for SteerLab steerable inference engine.

This module implements the main API server that orchestrates steerable text generation,
manages user sessions, and provides endpoints for all SteerLab functionality.
"""

import asyncio
import logging
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# Filter out HuggingFace Hub deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.model import SteerableModel
from ..core.vectors import SteeringVectorManager
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthCheckResponse,
    InteractionMode,
    ModelInfoResponse,
    PreferenceState,
    PreferenceUpdateRequest,
    PreferenceUpdateResponse,
    VectorComputeRequest,
    VectorComputeResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global application state
app_state = {"model": None, "available_vectors": {}, "sessions": {}, "start_time": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    app_state["start_time"] = time.time()
    logger.info("SteerLab API server starting up...")

    # Load default model if specified in environment
    # This would typically come from configuration
    try:
        # For now, we'll initialize lazily when first requested
        logger.info("Server ready - models will be loaded on first request")
    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    # Shutdown
    logger.info("SteerLab API server shutting down...")
    if app_state["model"]:
        # Cleanup model resources
        del app_state["model"]


# Create FastAPI application
app = FastAPI(
    title="SteerLab API",
    description="Steerable LLM inference engine with preference-based personalization",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_model(model_name: str = "microsoft/DialoGPT-medium") -> SteerableModel:
    """
    Dependency to get or create a steerable model instance.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        SteerableModel instance
    """
    if app_state["model"] is None or app_state["model"].model_name != model_name:
        logger.info(f"Loading model: {model_name}")
        try:
            app_state["model"] = SteerableModel(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {e}"
            ) from e

    return app_state["model"]


def get_session(session_id: str | None = None) -> PreferenceState:
    """
    Get or create a user session.

    Args:
        session_id: Optional session identifier

    Returns:
        PreferenceState for the session
    """
    if session_id is None:
        session_id = "default"

    if session_id not in app_state["sessions"]:
        app_state["sessions"][session_id] = PreferenceState()

    return app_state["sessions"][session_id]


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "name": "SteerLab API",
        "version": "0.1.0",
        "description": "Steerable LLM inference engine with preference-based personalization",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "chat": "/chat",
            "preferences": "/preferences",
            "vectors": "/vectors",
            "compute_vectors": "/compute-vectors"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = app_state["model"] is not None
    uptime = time.time() - app_state["start_time"] if app_state["start_time"] else 0

    return HealthCheckResponse(
        model_loaded=model_loaded,
        available_vectors=list(app_state["available_vectors"].keys()),
        uptime=uptime,
    )


@app.get("/info", response_model=ModelInfoResponse)
async def get_model_info(model: SteerableModel = Depends(get_model)):
    """Get information about the loaded model."""
    return ModelInfoResponse(
        model_name=model.model_name,
        available_preferences=list(app_state["available_vectors"].keys()),
        device_info=str(model.model.device) if model.model else None,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, model: SteerableModel = Depends(get_model)):
    """
    Generate steered text response.

    This is the main inference endpoint that applies steering based on
    user preferences and generates text responses.
    """
    try:
        # Get session preferences
        session = get_session(request.session_id)

        # Merge session preferences with request overrides
        active_preferences = session.preferences.copy()
        if request.preferences:
            active_preferences.update(request.preferences)

        # Load appropriate steering vectors
        vector_keys = set(active_preferences.keys()) & set(
            app_state["available_vectors"].keys()
        )
        if vector_keys:
            combined_vectors = {}
            for key in vector_keys:
                vectors = app_state["available_vectors"][key]
                strength = active_preferences[key]

                # Scale vectors by preference strength
                for layer_name, vector in vectors.items():
                    if layer_name not in combined_vectors:
                        combined_vectors[layer_name] = strength * vector
                    else:
                        combined_vectors[layer_name] += strength * vector

            model.load_steering_vectors(combined_vectors)

        # Set steering configuration
        if active_preferences:
            model.set_steering(active_preferences)

        # Generate response
        generated_text = model.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            do_sample=request.do_sample,
        )

        # Prepare response
        return ChatResponse(
            generated_text=generated_text,
            preferences_applied=active_preferences,
            session_id=request.session_id,
            model_info={
                "model_name": model.model_name,
                "max_length": request.max_length,
                "temperature": request.temperature,
            },
        )

    except Exception as e:
        logger.error(f"Chat generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e


@app.post("/preferences", response_model=PreferenceUpdateResponse)
async def update_preferences(request: PreferenceUpdateRequest):
    """Update user preferences for a session."""
    try:
        session = get_session(request.session_id)

        # Update preferences
        session.preferences.update(request.preferences)

        # Update mode if specified
        if request.mode:
            session.mode = request.mode

        # Store updated session
        session_id = request.session_id or "default"
        app_state["sessions"][session_id] = session

        return PreferenceUpdateResponse(success=True, updated_preferences=session)

    except Exception as e:
        logger.error(f"Preference update error: {e}")
        return PreferenceUpdateResponse(
            success=False, updated_preferences=session, message=f"Update failed: {e}"
        )


@app.get("/preferences")
async def get_preferences(session_id: str | None = None) -> PreferenceState:
    """Get current preferences for a session."""
    return get_session(session_id)


@app.post("/compute-vectors", response_model=VectorComputeResponse)
async def compute_vectors(
    request: VectorComputeRequest, background_tasks: BackgroundTasks
):
    """
    Compute steering vectors from contrastive examples.

    This endpoint handles the offline computation of steering vectors.
    For large models, this is run as a background task.
    """
    try:
        # Create vector manager
        vector_manager = SteeringVectorManager(request.model_name)

        # Compute vectors
        vectors = vector_manager.compute_steering_vectors(
            positive_examples=request.positive_examples,
            negative_examples=request.negative_examples,
            preference_name=request.preference_name,
            max_length=request.max_length,
        )

        # Determine output path
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            output_path = Path(f"vectors/{request.preference_name}_vectors.safetensors")

        # Save vectors
        metadata = {
            "preference_name": request.preference_name,
            "model_name": request.model_name,
            "num_positive": len(request.positive_examples),
            "num_negative": len(request.negative_examples),
        }

        vector_manager.save_vectors(vectors, output_path, metadata)

        # Store in available vectors
        app_state["available_vectors"][request.preference_name] = vectors

        # Cleanup
        vector_manager.cleanup()

        return VectorComputeResponse(
            success=True,
            preference_name=request.preference_name,
            vector_path=str(output_path),
            num_vectors=len(vectors),
            message=f"Successfully computed {len(vectors)} steering vectors",
        )

    except Exception as e:
        logger.error(f"Vector computation error: {e}")
        return VectorComputeResponse(
            success=False,
            preference_name=request.preference_name,
            message=f"Computation failed: {e}",
        )


@app.post("/load-vectors")
async def load_vectors(vector_path: str, preference_name: str | None = None):
    """Load pre-computed steering vectors from file."""
    try:
        vector_manager = SteeringVectorManager("dummy")  # Model not needed for loading
        vectors, metadata = vector_manager.load_vectors(vector_path)

        # Use provided name or extract from metadata/filename
        if preference_name is None:
            preference_name = Path(vector_path).stem.replace("_vectors", "")

        # Store in available vectors
        app_state["available_vectors"][preference_name] = vectors

        return {
            "success": True,
            "preference_name": preference_name,
            "num_vectors": len(vectors),
            "message": f"Loaded {len(vectors)} vectors for {preference_name}",
        }

    except Exception as e:
        logger.error(f"Vector loading error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load vectors: {e}"
        ) from e


@app.get("/vectors")
async def list_available_vectors():
    """List all available steering vector sets."""
    return {
        "available_vectors": list(app_state["available_vectors"].keys()),
        "count": len(app_state["available_vectors"]),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a user session."""
    if session_id in app_state["sessions"]:
        del app_state["sessions"][session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
