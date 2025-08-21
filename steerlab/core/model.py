"""
SteerableModel: Wrapper class for LLMs with activation steering capabilities.

This module implements the core SteerableModel class that wraps HuggingFace models
and provides per-request PyTorch hook management for activation steering.
"""

import logging
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Filter out HuggingFace Hub deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

logger = logging.getLogger(__name__)


class SteerableModel:
    """
    A wrapper around HuggingFace models that enables activation steering.

    This class manages PyTorch hooks on a per-request basis to ensure thread safety
    and prevent state leakage between requests in multi-tenant scenarios.

    Supported models from research paper:
    - stablelm-2-1.6b-chat
    - gemma-2-2b-it
    - Mistral-7B-Instruct-v0.3
    - Qwen2.5-7B-Instruct
    - gemma-2-9b-it
    """

    # Model configurations from paper Table A.1
    PAPER_MODEL_CONFIGS = {
        "stabilityai/stablelm-2-1_6b-chat": {
            "top_k_layers": 16,
            "functional_range": (-10, 10),
            "probe_type": "logistic",
        },
        "google/gemma-2-2b-it": {
            "top_k_layers": 16,
            "functional_range": (-30, 30),
            "probe_type": "logistic",
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "top_k_layers": 24,
            "functional_range": (-30, 30),
            "probe_type": "logistic",
        },
        "Qwen/Qwen2.5-7B-Instruct": {
            "top_k_layers": 24,
            "functional_range": (-10, 10),
            "probe_type": "logistic",
        },
        "google/gemma-2-9b-it": {
            "top_k_layers": 32,
            "functional_range": (-30, 30),
            "probe_type": "logistic",
        },
    }

    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the steerable model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.active_hooks = []
        self.steering_vectors = {}
        self.steering_config = {}

        # Load model configuration from paper if available
        self.model_config = self.PAPER_MODEL_CONFIGS.get(
            model_name,
            {
                "top_k_layers": 16,
                "functional_range": (-10, 10),
                "probe_type": "logistic",
            },
        )

        logger.info(f"Using config for {model_name}: {self.model_config}")

        self._load_model()

    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if self.device != "auto" else None,
            trust_remote_code=True,
        )

        logger.info(f"Model loaded on device: {self.model.device}")

    def load_steering_vectors(self, vectors: dict[str, torch.Tensor]):
        """
        Load steering vectors for use in inference.

        Args:
            vectors: Dictionary mapping layer names to steering vectors
        """
        self.steering_vectors = vectors
        logger.info(f"Loaded steering vectors for {len(vectors)} layers")

    def validate_steering_strength(self, strength: float) -> float:
        """
        Validate and clamp steering strength to model's functional range.

        Args:
            strength: Raw steering strength

        Returns:
            Clamped steering strength within functional range
        """
        min_strength, max_strength = self.model_config["functional_range"]

        if strength < min_strength or strength > max_strength:
            clamped = max(min_strength, min(max_strength, strength))
            logger.warning(
                f"Steering strength {strength} clamped to {clamped} for model {self.model_name}"
            )
            return clamped

        return strength

    def set_steering(self, config: dict[str, float]):
        """
        Set steering configuration and register hooks.

        Args:
            config: Dictionary mapping preference names to steering strengths
        """
        # Validate and clamp steering strengths
        validated_config = {}
        for pref_name, strength in config.items():
            validated_config[pref_name] = self.validate_steering_strength(strength)

        self.steering_config = validated_config
        self._register_hooks()
        logger.debug(f"Steering enabled with validated config: {validated_config}")

    def clear_steering(self):
        """Remove all steering hooks and clear configuration."""
        self._remove_hooks()
        self.steering_config = {}
        logger.debug("Steering cleared")

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        if not self.steering_vectors:
            logger.warning("No steering vectors loaded")
            return

        for layer_name, vector in self.steering_vectors.items():
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                hook = layer.register_forward_hook(self._create_steering_hook(vector))
                self.active_hooks.append(hook)
                logger.debug(f"Registered hook on layer: {layer_name}")

    def _remove_hooks(self):
        """Remove all active hooks."""
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks = []
        logger.debug("All hooks removed")

    def _get_layer_by_name(self, layer_name: str):
        """Get a layer by its name from the model."""
        try:
            parts = layer_name.split(".")
            layer = self.model
            for part in parts:
                layer = getattr(layer, part)
            return layer
        except AttributeError:
            logger.error(f"Layer not found: {layer_name}")
            return None

    def _create_steering_hook(self, steering_vector: torch.Tensor):
        """Create a forward hook function for steering."""

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Apply steering based on current configuration
            steering_strength = sum(self.steering_config.values())
            if steering_strength != 0:
                steered_output = hidden_states + steering_strength * steering_vector.to(
                    hidden_states.device
                )

                if isinstance(output, tuple):
                    return (steered_output,) + output[1:]
                else:
                    return steered_output

            return output

        return hook_fn

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate text with optional steering applied.

        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            # Generate with current steering configuration
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Return only the newly generated portion
            return generated_text[len(prompt) :].strip()

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

        finally:
            # Always clear hooks after generation to prevent state leakage
            self.clear_steering()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure hooks are cleaned up."""
        self.clear_steering()
