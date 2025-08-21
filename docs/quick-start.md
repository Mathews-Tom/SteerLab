# Quick Start Guide

This guide provides a fast path to setting up your development environment and running the SteerLab framework.

## 1. Prerequisites

- Python 3.12+
- An NVIDIA GPU with CUDA support (recommended for performance)
- `uv` installed (`pip install uv`)

## 2. Installation

We recommend using `uv` for fast and reliable dependency management.

```bash
# 1. Clone the repository
git clone https://github.com/aether-forge/steerlab.git
cd steerlab

# 2. Create a virtual environment
uv venv

# 3. Activate the environment
source .venv/bin/activate

# 4. Install dependencies
uv pip install -e ".[dev]"
```

## 3. 30-Second Example

This example demonstrates how to load a model and apply a single pre-computed steering vector.

```python
import steerlab as sl
from pathlib import Path

# Assume pre-computed vectors are in a 'vectors' directory
vector_path = Path("vectors/cost_vectors.pkl")

# Load a model supported by the framework
model = sl.load_model("gemma-2-9b-it")

# Load the pre-computed steering vectors for the "cost" preference
cost_vectors = sl.load_vectors(vector_path)

# Create a simple steerable chatbot interface
# This will launch a Gradio app
chatbot = sl.create_chatbot(model, {"cost": cost_vectors})
chatbot.launch()
```

## 4. Command-Line Interface (CLI)

The SteerLab framework includes a CLI for common tasks.

### Compute Steering Vectors

You can generate steering vectors from your own contrastive datasets.

```bash
steerlab compute-vectors \
  --model "gemma-2-9b-it" \
  --preference "ambiance" \
  --positive-data "data/hipster_examples.json" \
  --negative-data "data/touristy_examples.json" \
  --output "vectors/ambiance_vectors.pkl"
```

### Launch Interfaces

Launch one of the three interactive interfaces.

```bash
# Launch the SELECT interface with sliders
steerlab interface select --model gemma-2-9b-it --port 7860

# Launch the CALIBRATE interface for preference tuning
steerlab interface calibrate --model gemma-2-9b-it --port 7861

# Launch the LEARN interface for adaptive learning
steerlab interface learn --model gemma-2-9b-it --port 7862
```

### Reproduce Paper Experiments

The CLI also provides tools to reproduce the experiments from the original paper.

```bash
# Run Experiment E1 (Effect of steering strength) for a specific model
steerlab reproduce --experiment "E1" --model "gemma-2-9b-it"
```
