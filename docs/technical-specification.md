# Technical Specifications

This document outlines the technical stack, module breakdown, and development practices for the SteerLab framework, updated to reflect the latest design decisions.

## Technical Stack

The framework is built on Python 3.10+ and leverages modern, high-performance libraries.

### Core Technologies

| Component | Technology | Version | Rationale |
|---|---|---|---|
| **Language** | Python | 3.10+ | De facto standard for AI development. |
| **ML Framework** | PyTorch | 2.1+ | Essential for its `register_forward_hook` functionality. |
| **LLM Library** | Hugging Face Transformers | Latest | Provides access to a vast array of open-source LLMs. |
| **API Server** | FastAPI | Latest | High-performance framework for building the Chat Application's API. |
| **Data Validation**| Pydantic | V2 | Ensures type safety and provides automatic API documentation. |
| **Inference Server**| Uvicorn | Latest | High-speed ASGI server for running the FastAPI application. |
| **Containerization**| Docker | Latest | Ensures reproducible development and deployment environments. |

### Development & Tooling

| Component | Technology | Purpose |
|---|---|---|
| **Package Manager** | `uv` | Fast dependency management and virtual environments. |
| **Code Quality** | `ruff` | Linting, formatting, and code analysis. |
| **Testing** | `pytest` | Comprehensive testing suite. |
| **Type Checking** | `mypy` | Static type analysis. |

### A Note on `vLLM`

For this project, we will **not** use highly optimized inference engines like `vLLM`. The core mechanism of activation steering requires dynamic, per-request modification of the model's forward pass via PyTorch hooks. The aggressive optimizations in engines like `vLLM` (e.g., PagedAttention, static computation graphs) are fundamentally incompatible with this dynamic intervention. Prioritizing correctness and faithfulness to the research methodology, we will use a standard Hugging Face `model.generate()` call served by FastAPI/Uvicorn.

## Project Structure

```
steerlab/
├── steerlab/
│   ├── __init__.py
│   ├── api/                  # FastAPI application
│   │   ├── __init__.py
│   │   ├── server.py         # API endpoint definitions
│   │   └── schemas.py        # Pydantic data models
│   ├── core/                 # Core steering engine
│   │   ├── __init__.py
│   │   ├── model.py          # SteerableModel wrapper class
│   │   └── vectors.py        # SteeringVectorManager and CAA logic
│   ├── interfaces/           # Logic for SELECT, CALIBRATE, LEARN
│   │   ├── __init__.py
│   │   ├── calibrate.py
│   │   ├── learn.py
│   │   └── select.py
│   ├── cli.py                # Command-line interface
│   └── data/                 # Data loading and processing (TBD)
├── tests/
├── examples/
├── vectors/                  # Directory for pre-computed .safetensors files
├── docs/
├── pyproject.toml
└── README.md
```

## Module Breakdown

-   **`steerlab.core.vectors`**: Contains the `SteeringVectorManager` for loading and combining vectors, and the logic for the **Contrastive Activation Addition (CAA)** algorithm.
-   **`steerlab.core.model`**: Implements the `SteerableModel` wrapper class, which manages the Hugging Face model, and crucially, the registration and cleanup of PyTorch hooks on a per-request basis.
-   **`steerlab.api.schemas`**: Defines the Pydantic models (`PreferenceState`, `ChatRequest`, `ChatResponse`) that structure all data flowing into and out of the API.
-   **`steerlab.api.server`**: The FastAPI application that defines the RESTful endpoints (`/chat`, `/preferences`, etc.) and orchestrates the calls to the steering engine.
-   **`steerlab.interfaces`**: Contains the specific business logic for each of the three user interaction modes described in the paper (SELECT, CALIBRATE, and LEARN).
-   **`steerlab.cli`**: The command-line interface for offline tasks, primarily for running the steering vector generation process.