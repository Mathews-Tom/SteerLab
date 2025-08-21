# SteerLab Framework Overview

**A Python Framework for Preference-Based Activation Steering in Large Language Models**

*Implementing "Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering" ([arXiv:2505.04260v2](https://arxiv.org/abs/2505.04260))*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-HuggingFace%20Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/arXiv-2505.04260v2-b31b1b.svg)](https://arxiv.org/abs/2505.04260)

## 📄 Research Paper Background

This framework implements the methodology from **"Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering"** by Bo et al. (2025), published on arXiv:2505.04260v2.

![SteerLab](steerlab.png)

### Paper Summary

The research introduces a novel approach to personalizing Large Language Models (LLMs) through **activation steering** - a technique that modifies the model's internal representations during inference without requiring model retraining. The key contributions include:

#### 🔬 Core Innovation

- **Contrastive Activation Addition (CAA)**: A method to compute steering vectors by analyzing activation differences between contrasting examples (e.g., formal vs. informal language)
- **Multi-dimensional preference control**: Ability to steer models along multiple preference dimensions simultaneously
- **Real-time personalization**: Inference-time steering without model parameter updates

#### 🎯 Three Interface Paradigms

1. **SELECT**: Users directly specify preference values via sliders/controls
2. **CALIBRATE**: Interactive preference discovery through pairwise comparisons
3. **LEARN**: Adaptive learning from implicit user feedback and behavior

#### 📊 Experimental Validation

- Tested on 5 model families: StableLM (1.6B), Gemma (2B, 9B), Mistral (7B), Qwen (7B)
- 5 preference dimensions: Cost (Budget↔Luxury), Ambiance (Touristy↔Hipster), Age (Kids↔Adults), Time (Evening↔Morning), Culture (Asian↔American)
- Demonstrated effective preference alignment with minimal computational overhead
- Functional steering range validated at [-30, +30] with top-k layer selection (k=16)

#### 🏗️ Technical Architecture

- Offline vector computation using contrastive datasets
- Online steering via PyTorch forward hooks on residual streams
- Session-based preference management for multi-turn conversations

## 🎯 Core Mission

SteerLab provides a **production-ready implementation** of this research, enabling:

- **🔬 Research Reproducibility**: Faithful reproduction of all paper experiments and results
- **🚀 Production Deployment**: Robust API server with proper session management and error handling  
- **🧩 Extensibility**: Modular design supporting new models, preferences, and steering methods
- **📚 Developer Experience**: Comprehensive CLI, documentation, and evaluation tools

The framework is architected with a clear separation between an **offline Steering Vector Generation Service** and an **online Steerable Inference Engine**. This design ensures that the computationally intensive process of vector creation does not impact the performance of the real-time inference service.

## Key Objectives

1. **🔬 Research Reproducibility**: Faithfully reproduce all experimental results from the original paper.
2. **🧩 Modular Design**: Create reusable components for different aspects of activation steering.
3. **⚡ Production Ready**: Support real-world deployment scenarios with robust interfaces.
4. **🔄 Extensible Architecture**: Allow for easy integration of new models, datasets, and preference dimensions.
5. **📚 Developer Friendly**: Provide comprehensive documentation and examples for quick adoption.

## Success Metrics

- ✅ Reproduce computational experiments E1-E4 from the paper.
- ✅ Implement all three interface designs (SELECT, CALIBRATE, LEARN).
- ✅ Support all recommended models (StableLM, Gemma, Mistral, Qwen).
- ✅ Achieve performance parity with paper results.
- ✅ Enable custom preference dimensions and datasets.

## Supported Models

The framework is designed to be extensible to any HuggingFace compatible model. The models validated in the original research paper are:

| Model Family | Specific Models | Parameters |
|--------------|-----------------|------------|
| **StableLM** | stablelm-21-6b-chat | 1.6B |
| **Gemma** | gemma-2-2b-it, gemma-2-9b-it | 2B, 9B |
| **Mistral** | Mistral-7B-Instruct-v0.3 | 7B |
| **Qwen** | Qwen2.5-7B-Instruct | 7B |

## 📄 Research Citation

```bibtex
@article{bo2025steerable,
  title={Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering},
  author={Bo, et al.},
  journal={arXiv preprint arXiv:2505.04260v2},
  year={2025}
}
```
