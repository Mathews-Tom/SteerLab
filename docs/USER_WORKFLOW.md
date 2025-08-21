# SteerLab User Workflow Guide

This document describes the complete workflow for using SteerLab to create personalized and steerable chatbots through preference-based activation steering.

## Overview

SteerLab enables you to modify how language models behave by "steering" their internal activations based on your preferences. The system uses a two-phase approach:

1. **Offline Phase**: Generate steering vectors from training data
2. **Online Phase**: Apply steering vectors during text generation

**Research Paper Implementation**: This implementation is based on "Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering" and includes validated models, preference dimensions, and experimental configurations from the research.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- At least 8GB RAM (16GB+ recommended for larger models)
- CUDA-compatible GPU (optional but recommended)

### Installation

```bash
# Clone and setup
git clone <your-repository>
cd SteerLab
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Research Paper Models and Preferences

### Supported Models from Research Paper

SteerLab supports all five models tested in the research paper with validated configurations:

```bash
# List paper-tested models
uv run steerlab list-paper-models
```

#### Supported Models

- `stabilityai/stablelm-2-1_6b-chat` (1.6B parameters)
- `google/gemma-2-2b-it` (2B parameters)
- `mistralai/Mistral-7B-Instruct-v0.3` (7B parameters)
- `Qwen/Qwen2.5-7B-Instruct` (7B parameters)
- `google/gemma-2-9b-it` (9B parameters)

### Research Paper Preference Dimensions

The paper validated five preference dimensions for lifestyle planning tasks:

```bash
# List paper preference dimensions
uv run steerlab list-paper-preferences
```

#### Preference Dimensions

- **Cost**: Budget ↔ Luxury
- **Ambiance**: Touristy ↔ Hipster
- **Age**: Kids-friendly ↔ Adults-oriented
- **Time**: Evening ↔ Morning
- **Culture**: Asian ↔ American

## Complete Workflow

### Phase 1: Offline Vector Generation

#### Step 1: Prepare Training Data

You can create training data using either generic templates or research paper templates:

```bash
# Generic template
uv run steerlab create-template formality \
  --positive-file data/formal_examples.json \
  --negative-file data/informal_examples.json

# Research paper template (recommended)
uv run steerlab create-template cost --use-paper-templates
```

Edit the generated files:

**data/formal_examples.json**

```json
[
  "I would be delighted to assist you with this matter.",
  "Please allow me to express my sincere appreciation.",
  "It would be most appropriate to schedule a meeting."
]
```

**data/informal_examples.json**

```json
[
  "Hey! Can you help me out with this?",
  "Thanks a bunch, you're awesome!",
  "Let's grab coffee and chat about it."
]
```

#### Step 2: Compute Steering Vectors

Generate steering vectors using the Contrastive Activation Addition (CAA) algorithm with research paper models:

```bash
# Using a research paper model with validated configuration
uv run steerlab compute-vectors \
  --model "google/gemma-2-2b-it" \
  --preference "cost" \
  --positive-data "data/cost_positive.json" \
  --negative-data "data/cost_negative.json" \
  --output "vectors/cost_vectors.safetensors"

# The system automatically uses paper configurations:
# - Functional steering range: (-30, 30)
# - Top-k layers: 16
# - Probe type: logistic regression
```

This process:

- Loads the specified language model
- Processes your training examples
- Computes contrastive activation differences
- Saves vectors in `.safetensors` format

#### Step 3: Test Your Vectors

Validate that your vectors work as expected:

```bash
uv run steerlab test-vectors \
  --vector-path "vectors/formality_vectors.safetensors" \
  --prompt "I need help with my project" \
  --strength 0.8
```

This shows both steered and unsteered outputs for comparison.

### Phase 2: Online Steerable Generation

#### Method A: Command Line Interface

List available vectors:

```bash
uv run steerlab list-vectors
```

Start the API server:

```bash
uv run steerlab serve --reload
```

#### Method B: Python API

```python
from steerlab.core.model import SteerableModel
from steerlab.core.vectors import SteeringVectorManager

# Load model and vectors
model = SteerableModel("microsoft/DialoGPT-medium")
vector_manager = SteeringVectorManager("dummy")
vectors, metadata = vector_manager.load_vectors("vectors/formality_vectors.safetensors")
model.load_steering_vectors(vectors)

# Generate steered text
model.set_steering({"formality": 0.8})
response = model.generate("I need help with my project", max_length=100)
print(response)
```

#### Method C: REST API

Start the server:

```bash
uv run uvicorn steerlab.api.server:app --reload
```

Make requests:

```bash
# Generate steered text
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I need help with my project",
    "preferences": {"formality": 0.8},
    "max_length": 100
  }'

# Update preferences
curl -X POST "http://localhost:8000/preferences" \
  -H "Content-Type: application/json" \
  -d '{
    "preferences": {"formality": 0.5},
    "session_id": "user123"
  }'
```

## Advanced Workflows

### Multi-Preference Steering

You can combine multiple preferences:

```python
# Compute vectors for different preferences
# formality_vectors.safetensors
# politeness_vectors.safetensors
# creativity_vectors.safetensors

# Apply multiple preferences simultaneously
model.set_steering({
    "formality": 0.7,
    "politeness": 0.9,
    "creativity": -0.3
})
```

### Session Management

Track user preferences across conversations:

```python
# API request with session tracking
{
  "prompt": "Hello, how are you?",
  "preferences": {"formality": 0.8},
  "session_id": "user123",
  "max_length": 100
}
```

### Interactive Modes

SteerLab supports three interaction modes:

#### SELECT Mode

Users directly choose preference settings:

```python
{
  "mode": "select",
  "preferences": {"formality": 0.8, "creativity": 0.2}
}
```

#### CALIBRATE Mode

System learns preferences from user feedback:

```python
# User rates responses, system adjusts preferences
{
  "mode": "calibrate",
  "feedback": {"response_id": "abc123", "rating": 4.5}
}
```

#### LEARN Mode

System adapts based on user interactions:

```python
{
  "mode": "learn",
  "implicit_feedback": true
}
```

## Best Practices

### Data Quality

- Use 8-20 examples per preference side (positive/negative)
- Make examples diverse but clearly contrasting
- Keep examples focused on the specific preference
- Avoid mixing multiple preferences in examples

### Vector Generation

- Start with smaller models for testing (DialoGPT-medium)
- Use appropriate max_length for your use case (128-512)
- Save vectors with descriptive metadata
- Test vectors thoroughly before production use

### Steering Strength

- Use values between -1.0 and 1.0
- Start with moderate values (±0.3 to ±0.7)
- Extreme values (±0.9 to ±1.0) may degrade text quality
- Test different strengths for your specific preference

### Production Deployment

- Precompute all vectors offline
- Load vectors once at startup
- Use session management for user state
- Monitor hook cleanup to prevent memory leaks
- Set appropriate timeout and resource limits

## Troubleshooting

### Common Issues

**Memory errors during vector computation:**

- Use smaller models or reduce max_length
- Process examples in smaller batches
- Ensure adequate GPU/CPU memory

**Poor steering effectiveness:**

- Increase contrast between positive/negative examples
- Add more diverse training examples
- Adjust steering strength
- Verify examples represent the desired preference

**API timeout errors:**

- Reduce max_length for generation
- Use faster models for real-time applications
- Implement request queuing for high load

### Debug Commands

```bash
# Check model and vectors
uv run steerlab list-vectors -d vectors/

# Test vector loading
uv run python -c "
from steerlab.core.vectors import SteeringVectorManager
vm = SteeringVectorManager('dummy')
vectors, meta = vm.load_vectors('vectors/formality_vectors.safetensors')
print(f'Loaded {len(vectors)} vectors, meta: {meta}')
"

# Verify API health
curl http://localhost:8000/health
```

## Example: Research Paper Reproduction

Here's a complete example reproducing the research paper methodology:

```bash
# 1. List available paper models and preferences
uv run steerlab list-paper-models
uv run steerlab list-paper-preferences

# 2. Create research paper training data
uv run steerlab create-template cost --use-paper-templates

# 3. Compute vectors with paper-validated model
uv run steerlab compute-vectors \
  -m "google/gemma-2-2b-it" \
  -p "cost" \
  --positive-data "data/cost_positive.json" \
  --negative-data "data/cost_negative.json"

# 4. Test vectors within functional range (-30 to 30)
uv run steerlab test-vectors \
  -v "vectors/cost_vectors.safetensors" \
  -p "Help me choose a present for a friend" \
  -s 15

# 5. Test different steering strengths
uv run steerlab test-vectors -v vectors/cost_vectors.safetensors -s -30  # Budget
uv run steerlab test-vectors -v vectors/cost_vectors.safetensors -s 0    # Neutral
uv run steerlab test-vectors -v vectors/cost_vectors.safetensors -s 30   # Luxury

# 6. Start API server for interactive use
uv run steerlab serve --reload

# 7. Use via API with multiple preferences
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Plan a weekend trip for me",
    "preferences": {"cost": 0.8, "age": -0.5}
  }'
```

### Paper Reproduction Example

Run the complete paper reproduction example:

```bash
uv run examples/paper_reproduction_example.py
```

This demonstrates:
- All five paper models with validated configurations
- All five preference dimensions with template data
- Functional steering range validation
- Multi-preference steering capabilities
- Research paper experimental methodology

## Quantitative Evaluation Framework

SteerLab includes comprehensive evaluation tools to measure steering effectiveness:

### Built-in Evaluation Commands

```bash
# Comprehensive evaluation with all metrics
uv run steerlab evaluate-steering \
  -m "google/gemma-2-2b-it" \
  -v "vectors/cost_vectors.safetensors" \
  --strengths "-1.0,-0.5,0.0,0.5,1.0"

# Full evaluation demo with Rich progress bars
uv run examples/evaluate_steering_demo.py

# Quick evaluation for testing (faster)
uv run python quick_eval.py
```

### Evaluation Metrics

The framework computes four key metrics:

1. **Preference Alignment Score (0-1)**: How well outputs align with desired vs undesired examples
2. **Semantic Coherence Score (0-1)**: Consistency between different steered outputs
3. **Fluency Score (0-1)**: Quality and naturalness of generated text
4. **Lexical Diversity Score (0-1)**: Vocabulary richness and variety

### Publication-Ready Results

Evaluation generates:

- **Quantitative metrics** showing X% improvement over baseline
- **Visualization plots** with steering effectiveness curves  
- **Comparison tables** across different steering strengths
- **Markdown reports** with key findings and statistical summaries
- **Before/after examples** demonstrating concrete changes

### Example Results

```
🎯 Steering Effectiveness Analysis
📊 Model: google/gemma-2-2b-it
🔄 Preference: cost
📝 Test Prompts: 8

┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Strength    ┃ Alignment ┃ Fluency ┃ Coherence ┃ Diversity ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Baseline    │ 0.520     │ -       │ -         │ -         │
│ +0.5        │ 0.742     │ 0.856   │ 0.791     │ 0.634     │
└─────────────┴───────────┴─────────┴───────────┴───────────┘

🏆 Key Improvements
🚀 Best Alignment Improvement: +0.222
🎯 Best Steering Strength: +0.5  
📈 Relative Improvement: +42.7%
```

This quantitative framework enables researchers to:
- **Validate steering effectiveness** with concrete metrics
- **Compare different approaches** systematically
- **Generate publication-ready results** for papers and presentations
- **Debug and optimize** steering vector quality

This workflow provides a complete path from data preparation to production deployment and rigorous evaluation of steerable language models.
