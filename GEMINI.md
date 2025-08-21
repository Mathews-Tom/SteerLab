# Project Overview

This project, **SteerLab**, is a Python framework for implementing preference-based activation steering in Large Language Models (LLMs). It allows researchers and developers to build personalized and steerable chatbots by modifying a model's internal activations at inference time.

The framework's architecture is designed for robustness and scalability, featuring a critical separation between an **offline Steering Vector Generation Service** and an **online Steerable Inference Engine**. This ensures that the computationally heavy task of vector creation does not interfere with the low-latency requirements of real-time inference. The core algorithm for vector computation is **Contrastive Activation Addition (CAA)**, a direct and powerful method for finding preference directions in a model's activation space.

**Main Technologies:**
- **Python 3.10+**
- **PyTorch** & **HuggingFace Transformers** for core ML operations.
- **FastAPI** & **Pydantic** for building a robust, type-safe API server.
- **Gradio** for creating interactive web-based user interfaces.
- **uv**, **ruff**, **pytest**, and **mypy** for a modern development toolchain.
- **Docker** for ensuring reproducible environments.

**Architecture:**
- **Offline Service:** Generates steering vectors from contrastive datasets using the CAA algorithm and saves them as `.safetensors` files.
- **Online Engine:** A `SteerableModel` wrapper class loads a base LLM and uses PyTorch hooks to inject the pre-computed steering vectors into the model's residual stream during inference. Hook management is handled carefully on a per-request basis to prevent state leakage.
- **Chat Application Backend:** A FastAPI server that manages user sessions, orchestrates the SELECT, CALIBRATE, and LEARN interaction modes, and handles all API requests.

# Building and Running

The project uses `uv` for dependency management.

**1. Setup Environment:**
```bash
# Create a virtual environment
uv venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

**2. Running the Application:**

*   **Offline - Compute Steering Vectors:**
    The primary offline task is generating vectors via the CLI.
    ```bash
    steerlab compute-vectors \
      --model "gemma-2-9b-it" \
      --preference "formality" \
      --positive-data "data/formal_examples.json" \
      --negative-data "data/informal_examples.json" \
      --output "vectors/formality_vectors.safetensors"
    ```

*   **Online - Start the API Server:**
    The main application is the FastAPI server.
    ```bash
    uvicorn steerlab.api.server:app --reload
    ```

*   **Launch a Demo UI:**
    A Gradio or Streamlit app will be provided to interact with the API.
    ```bash
    # (Example command)
    python examples/run_select_ui.py
    ```

**3. Running Tests:**
```bash
# Run all tests
pytest
```

# Development Conventions

-   **Code Style:** The project uses `ruff` for linting and formatting, enforced by `pre-commit` hooks.
-   **API First:** The core logic is exposed via a well-defined FastAPI server with Pydantic schemas.
-   **Testing:** All new features must be accompanied by `pytest` tests.
-   **Contributions:** The `contributing.md` file details the process for setting up the development environment and submitting pull requests.
-   **Vector Storage:** Steering vectors are stored in the secure and efficient `.safetensors` format.