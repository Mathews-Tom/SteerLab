# SteerLab Framework Overview

**A Python Framework for Preference-Based Activation Steering in Large Language Models**

*Implementing "Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering" (Bo et al., 2024)*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-HuggingFace%20Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🎯 Core Mission

To develop a **modular, extensible Python framework** that faithfully implements the activation steering methodology from Bo et al. (2024), enabling researchers and developers to build personalized chatbots with preference-based control.

The framework is architected with a clear separation between an **offline Steering Vector Generation Service** and an **online Steerable Inference Engine**. This design ensures that the computationally intensive process of vector creation does not impact the performance of the real-time inference service. The core vector computation is performed using **Contrastive Activation Addition (CAA)**, a robust and direct method for identifying preference directions in activation space.

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
