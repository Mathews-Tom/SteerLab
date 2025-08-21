# 🧪 SteerLab

Modern Python laboratory for activation steering research and development. Build, test, and deploy steerable LLMs with preference-based personalization, supporting research reproducibility and production applications.

**Implementation of "Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering"** ([arXiv:2505.04260v2](https://arxiv.org/abs/2505.04260))

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![Paper](https://img.shields.io/badge/arXiv-2505.04260v2-b31b1b.svg)](https://arxiv.org/abs/2505.04260)

## 📄 About the Research

This framework implements the methodology from **"Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering"** by Bo et al. (2025). The research introduces **activation steering** - a technique to personalize LLMs at inference time without retraining, using:

- **🔬 Contrastive Activation Addition (CAA)**: Compute steering vectors from contrastive examples
- **🎯 Multi-dimensional Control**: Steer along multiple preference dimensions simultaneously  
- **⚡ Real-time Personalization**: Apply steering during inference via PyTorch hooks
- **📊 Validated Results**: Tested on 5 models (StableLM, Gemma, Mistral, Qwen) across 5 preference dimensions

## ✨ Features

- 🎯 **Preference-Based Steering**: Control LLM behavior through learned preference vectors
- ⚡ **Real-Time Inference**: Fast API server with per-request hook management
- 🧠 **Multiple Learning Modes**: SELECT, CALIBRATE, and LEARN interaction paradigms
- 🔒 **Production Ready**: Thread-safe, memory-efficient, with proper cleanup
- 📊 **Research Reproducible**: Implements Contrastive Activation Addition (CAA) algorithm
- 🚀 **Easy to Use**: Simple CLI and comprehensive API
- 📈 **Quantitative Evaluation**: Built-in metrics and comparison tools

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mathews-Tom/SteerLab.git
cd SteerLab

# Install with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Basic Usage

1. **Create training data:**

```bash
steerlab create-template formality
# Edit data/formality_positive.json and data/formality_negative.json
```

2. **Compute steering vectors:**

```bash
steerlab compute-vectors \
  --model microsoft/DialoGPT-medium \
  --preference formality \
  --positive-data data/formality_positive.json \
  --negative-data data/formality_negative.json
```

3. **Start the API server:**

```bash
steerlab serve
```

4. **Generate steered text:**

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I need help with my project",
    "preferences": {"formality": 0.8},
    "max_length": 100
  }'
```

## 🏗️ Architecture

SteerLab implements a **clean separation** between offline vector computation and online inference:

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Training Data     │───▶│  Vector Computation  │───▶│   Steering Vectors  │
│  (Contrastive)      │    │      (CAA)           │    │   (.safetensors)    │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                                                   │
                                                                   ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│      User           │◀──▶│    FastAPI Server    │───▶│  Steerable Model    │
│   (Web/CLI)         │    │   (Session Mgmt)     │    │  (PyTorch Hooks)    │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Key Components

- **SteerableModel**: Wraps HuggingFace models with per-request PyTorch hook management
- **SteeringVectorManager**: Implements CAA algorithm and vector I/O
- **FastAPI Server**: Provides REST endpoints with session management
- **Interface Modes**: SELECT (explicit), CALIBRATE (guided), LEARN (adaptive)

## 📚 User Interaction Modes

### SELECT Mode 🎚️

Direct preference control through sliders/dropdowns:

```python
from steerlab.interfaces.select import create_default_select_interface

interface = create_default_select_interface()
interface.set_preference("formality", 0.8)
interface.set_preference("creativity", -0.2)
preferences = interface.get_current_preferences()
```

### CALIBRATE Mode 🎯

Interactive preference discovery through pairwise comparisons:

```python
from steerlab.interfaces.calibrate import CalibrateInterface

interface = CalibrateInterface(["formality", "creativity"])
status = interface.start_calibration()
# Present options, collect feedback, iterate until convergence
final_preferences = interface.get_calibration_status()["current_preferences"]
```

### LEARN Mode 🧠

Adaptive learning from implicit user feedback:

```python
from steerlab.interfaces.learn import LearnInterface

interface = LearnInterface(["formality", "creativity"])
event_id = interface.record_interaction(prompt, response, preferences_used)
interface.record_feedback(event_id, FeedbackSignal.THUMBS_UP)
learned_preferences = interface.get_current_preferences()
```

## 🛠️ API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check and service status |
| `/chat` | POST | Generate steered text responses |
| `/preferences` | GET/POST | Manage user preference sessions |
| `/compute-vectors` | POST | Compute new steering vectors |
| `/load-vectors` | POST | Load pre-computed vectors |
| `/vectors` | GET | List available vector sets |

### Example API Usage

```python
import requests

# Generate steered response
response = requests.post("http://localhost:8000/chat", json={
    "prompt": "Explain machine learning",
    "preferences": {
        "formality": 0.7,
        "detail_level": 0.5
    },
    "max_length": 200,
    "session_id": "user123"
})

result = response.json()
print(result["generated_text"])
```

## 🧪 Examples

### Research Paper Reproduction

```bash
# Reproduce paper methodology with validated models
uv run examples/paper_reproduction_example.py

# Run quantitative evaluation with metrics
uv run examples/evaluate_steering_demo.py

# Quick evaluation (faster testing)
uv run python quick_eval.py
```

### Complete Workflow Example

```bash
# Run the comprehensive example
uv run examples/basic_steering_example.py
```

This example demonstrates:

1. ✅ Creating training data for formality preference
2. ✅ Computing steering vectors using CAA algorithm
3. ✅ Testing steered vs unsteered generation
4. ✅ Using vectors via the API
5. ✅ Quantitative evaluation with metrics

### Custom Preference Training

```python
from steerlab.core.vectors import SteeringVectorManager

# Load your training data
positive_examples = ["Very formal text examples..."]
negative_examples = ["Casual text examples..."]

# Compute vectors
manager = SteeringVectorManager("microsoft/DialoGPT-medium")
vectors = manager.compute_steering_vectors(
    positive_examples, negative_examples, "custom_preference"
)

# Save for later use
manager.save_vectors(vectors, "vectors/custom_preference.safetensors")
```

## 🧬 Research & Technical Details

### Contrastive Activation Addition (CAA)

SteerLab implements the CAA algorithm for computing steering vectors:

1. **Collect Activations**: Forward pass positive and negative examples through the model
2. **Compute Centroids**: Calculate mean activations for each set
3. **Vector Difference**: Steering vector = positive_centroid - negative_centroid
4. **Apply During Inference**: Add scaled vectors to residual stream via PyTorch hooks

### Hook Management

Critical for production safety:

- **Per-request isolation**: Hooks registered and cleaned up for each request
- **Thread safety**: No shared state between concurrent requests
- **Memory efficiency**: Automatic cleanup prevents memory leaks
- **Error handling**: Hooks cleaned up even if generation fails

### Supported Models

SteerLab works with any HuggingFace causal language model:

- ✅ GPT-2, DialoGPT family
- ✅ Llama, Llama 2, Code Llama
- ✅ Mistral, Mixtral
- ✅ Gemma models
- ✅ Any transformer with standard residual connections

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/core/
uv run pytest tests/api/

# Run with coverage
uv run pytest --cov=steerlab --cov-report=html
```

## 📖 Development

### Project Structure

```text
steerlab/
├── steerlab/               # Main package
│   ├── api/               # FastAPI server and schemas
│   ├── core/              # SteerableModel and vector management
│   ├── interfaces/        # SELECT/CALIBRATE/LEARN modes
│   └── cli.py             # Command-line interface
├── tests/                 # Comprehensive test suite
├── examples/              # Working examples and tutorials
├── vectors/               # Pre-computed steering vectors
└── docs/                  # Additional documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes with tests: `uv run pytest`
4. Check code quality: `uv run ruff check && uv run mypy`
5. Submit a pull request

### Code Quality

- **Formatting**: `ruff` for fast, consistent code formatting
- **Type Checking**: `mypy` for static type analysis
- **Testing**: `pytest` with comprehensive coverage
- **Documentation**: Clear docstrings and examples

## 🔬 Advanced Usage

### Custom Vector Computation

```python
from steerlab.core.vectors import SteeringVectorManager

class CustomVectorManager(SteeringVectorManager):
    def compute_custom_vectors(self, data, method="caa"):
        # Implement custom vector computation logic
        pass

# Use your custom implementation
manager = CustomVectorManager("your-model")
vectors = manager.compute_custom_vectors(your_data)
```

### Batch Processing

```python
import asyncio
from steerlab.core.model import SteerableModel

async def batch_generate(prompts, preferences):
    model = SteerableModel("microsoft/DialoGPT-medium")
    model.load_steering_vectors(your_vectors)

    results = []
    for prompt in prompts:
        model.set_steering(preferences)
        result = model.generate(prompt)
        results.append(result)

    return results
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv pip install -e "."

EXPOSE 8000
CMD ["uvicorn", "steerlab.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Configuration

- **Memory Management**: Configure model caching and cleanup
- **Rate Limiting**: Add request throttling for vector computation endpoints
- **Authentication**: Implement API key or OAuth-based access control
- **Monitoring**: Add logging, metrics, and health checks
- **Scaling**: Use horizontal scaling for concurrent inference requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the "Steerable Chatbots" research paper
- Built with HuggingFace Transformers and FastAPI
- Inspired by the broader activation steering research community

## 🔗 Links

- [Documentation](docs/)
- [API Reference](docs/api-reference.md)
- [Technical Specification](docs/technical-specification.md)
- [Contributing Guide](docs/contributing.md)
- [Roadmap](docs/roadmap.md)

---

**Built with ❤️ for the AI research and development community**
