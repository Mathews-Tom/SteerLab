# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SteerLab is a Python framework for implementing preference-based activation steering in Large Language Models (LLMs). It enables building personalized and steerable chatbots by modifying model internal activations at inference time using the Contrastive Activation Addition (CAA) algorithm.

## Key Architecture

The system has a **critical separation** between offline and online components:

- **Offline**: Steering Vector Generation Service computes vectors using CAA algorithm and saves as `.safetensors` files
- **Online**: Steerable Inference Engine loads pre-trained LLM and uses PyTorch hooks to inject steering vectors during inference
- **API Layer**: FastAPI server managing user sessions and SELECT/CALIBRATE/LEARN interaction modes

**Hook Management**: PyTorch hooks are managed on a per-request basis to prevent state leakage between requests. Hooks are registered via `set_steering()` and cleaned up via `clear_steering()` in finally blocks.

## Development Commands

### Package Management

- Package manager: `uv` (not pip/conda)
- Install dependencies: `uv pip install -e ".[dev]"`
- Create environment: `uv venv`
- Run commands: `uv run <command>`

### Core Development Tasks

```bash
# Setup environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Code quality (uses ruff for linting and formatting)
uv run ruff check
uv run ruff format

# Type checking
uv run mypy

# Generate steering vectors (offline)
uv run steerlab compute-vectors \
  --model "gemma-2-9b-it" \
  --preference "formality" \
  --positive-data "data/formal_examples.json" \
  --negative-data "data/informal_examples.json" \
  --output "vectors/formality_vectors.safetensors"

# Start API server (online)
uv run uvicorn steerlab.api.server:app --reload

# Launch demo UI
uv run examples/run_select_ui.py
```

## Module Structure

```
steerlab/
├── steerlab/
│   ├── api/                  # FastAPI application (server.py, schemas.py)
│   ├── core/                 # Core steering engine (model.py, vectors.py)
│   ├── interfaces/           # SELECT/CALIBRATE/LEARN logic
│   ├── cli.py                # Command-line interface
│   └── data/                 # Data loading and processing
├── tests/                    # pytest test suite
├── vectors/                  # Pre-computed .safetensors files
└── examples/                 # Demo applications
```

### Key Modules

- `steerlab.core.model`: `SteerableModel` wrapper managing PyTorch hooks per-request
- `steerlab.core.vectors`: `SteeringVectorManager` and CAA algorithm implementation
- `steerlab.api.schemas`: Pydantic models (`PreferenceState`, `ChatRequest`, `ChatResponse`)
- `steerlab.api.server`: FastAPI endpoints orchestrating steering engine calls
- `steerlab.interfaces`: Business logic for the three interaction modes

## Technical Constraints

- **Python**: 3.10+ required
- **PyTorch**: 2.1+ required for `register_forward_hook` functionality
- **No vLLM**: Intentionally avoid optimized inference engines as they're incompatible with dynamic PyTorch hook injection
- **Hook Lifecycle**: Always ensure hook cleanup in finally blocks to prevent memory leaks and state contamination
- **Vector Storage**: Use `.safetensors` format for steering vectors (secure and efficient)

## Testing Philosophy

- All new features require pytest tests
- Test hook registration/cleanup thoroughly to prevent state leakage
- Test vector loading and application logic
- Focus on per-request isolation in multi-tenant scenarios
