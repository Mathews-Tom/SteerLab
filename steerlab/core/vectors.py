"""
Steering Vector Management and Contrastive Activation Addition (CAA) Algorithm.

This module implements the core CAA algorithm for computing steering vectors
and manages loading/saving vectors in safetensors format.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Filter out HuggingFace Hub deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

logger = logging.getLogger(__name__)


class SteeringVectorManager:
    """
    Manages steering vectors and implements the Contrastive Activation Addition algorithm.

    This class handles the offline computation of steering vectors from contrastive
    datasets and provides utilities for loading/saving vectors.

    Supports preference dimensions from research paper:
    - Cost: Budget ↔ Luxury
    - Ambiance: Touristy ↔ Hipster
    - Age: Kids ↔ Adults
    - Time: Evening ↔ Morning
    - Culture: Asian ↔ American
    """

    # Preference dimensions from research paper Table 1
    PAPER_PREFERENCES = {
        "cost": {
            "negative": "budget",
            "positive": "luxury",
            "description": "Price preference from budget-friendly to luxury",
        },
        "ambiance": {
            "negative": "touristy",
            "positive": "hipster",
            "description": "Ambiance preference from touristy to hipster",
        },
        "age": {
            "negative": "kids",
            "positive": "adults",
            "description": "Age-group preference from kids-friendly to adults-oriented",
        },
        "time": {
            "negative": "evening",
            "positive": "morning",
            "description": "Time preference from evening to morning",
        },
        "culture": {
            "negative": "asian",
            "positive": "american",
            "description": "Cultural preference from Asian to American",
        },
    }

    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the steering vector manager.

        Args:
            model_name: HuggingFace model identifier
            device: Device to perform computations on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.target_layers = []

    def _load_model_for_analysis(self):
        """Load model and tokenizer for vector computation."""
        if self.model is None:
            logger.info(f"Loading model for analysis: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Try to load model with fallback options
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                    if torch.cuda.is_available()
                    else torch.float32,
                    device_map=self.device if self.device != "auto" else None,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"Failed to load model with standard config: {e}")
                logger.info("Trying with fallback configuration...")
                # Fallback: try without device_map and with float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            # Common patterns for different model architectures
            layer_patterns = [
                "layers",  # Llama, Mistral, Gemma
                "h",  # GPT-2 style
                "model.layers",  # Gemma2
                "layer",  # BERT style
                "blocks",  # Some transformer variants
            ]

            for name, _ in self.model.named_modules():
                for pattern in layer_patterns:
                    if pattern in name and ("mlp" in name or "attention" in name):
                        continue  # Skip sub-components
                    if pattern in name and any(
                        x in name for x in ["norm", "embed", "head"]
                    ):
                        continue  # Skip normalization and embedding layers
                    if pattern in name and name.count(".") <= 2:  # Top-level layer blocks
                        self.target_layers.append(name)
                        break

        self.target_layers = sorted(list(set(self.target_layers)))
        logger.info(f"Identified {len(self.target_layers)} target layers")

    def compute_steering_vectors(
        self,
        positive_examples: list[str],
        negative_examples: list[str],
        preference_name: str,
        max_length: int = 256,
    ) -> dict[str, torch.Tensor]:
        """
        Compute steering vectors using Contrastive Activation Addition (CAA).

        Args:
            positive_examples: List of texts representing positive preference
            negative_examples: List of texts representing negative preference
            preference_name: Name of the preference being learned
            max_length: Maximum sequence length for processing

        Returns:
            Dictionary mapping layer names to steering vectors
        """
        self._load_model_for_analysis()

        logger.info(f"Computing steering vectors for preference: {preference_name}")
        logger.info(
            f"Positive examples: {len(positive_examples)}, Negative: {len(negative_examples)}"
        )

        # Collect activations for both positive and negative examples
        positive_activations = self._collect_activations(positive_examples, max_length)
        negative_activations = self._collect_activations(negative_examples, max_length)

        # Compute steering vectors via contrastive difference
        steering_vectors = {}
        for layer_name in self.target_layers:
            if (
                layer_name in positive_activations
                and layer_name in negative_activations
            ):
                pos_mean = positive_activations[layer_name].mean(dim=0)
                neg_mean = negative_activations[layer_name].mean(dim=0)

                # The steering vector is the difference between positive and negative centroids
                steering_vector = pos_mean - neg_mean
                steering_vectors[layer_name] = steering_vector

                logger.debug(
                    f"Computed steering vector for {layer_name}: {steering_vector.shape}"
                )

        logger.info(f"Generated {len(steering_vectors)} steering vectors")
        return steering_vectors

    def _collect_activations(
        self, examples: list[str], max_length: int
    ) -> dict[str, torch.Tensor]:
        """
        Collect activations from target layers for given examples.

        Args:
            examples: List of text examples
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping layer names to collected activations
        """
        activations = {layer: [] for layer in self.target_layers}
        hooks = []

        def create_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Store mean activation across sequence dimension
                mean_activation = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
                activations[layer_name].append(mean_activation.detach().cpu())

            return hook_fn

        # Register hooks
        for layer_name in self.target_layers:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                hook = layer.register_forward_hook(create_hook(layer_name))
                hooks.append(hook)

        try:
            # Process examples in batches
            batch_size = 4
            with torch.no_grad():
                for i in range(0, len(examples), batch_size):
                    batch = examples[i : i + batch_size]

                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(self.model.device)

                    # Forward pass to collect activations
                    _ = self.model(**inputs, output_hidden_states=True)

        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

        # Convert lists to tensors
        final_activations = {}
        for layer_name, layer_activations in activations.items():
            if layer_activations:
                final_activations[layer_name] = torch.cat(layer_activations, dim=0)

        return final_activations

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

    def save_vectors(
        self,
        vectors: dict[str, torch.Tensor],
        output_path: str | Path,
        metadata: dict | None = None,
    ):
        """
        Save steering vectors to safetensors format.

        Args:
            vectors: Dictionary of steering vectors
            output_path: Path to save vectors
            metadata: Optional metadata to include
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "model_name": self.model_name,
                "num_vectors": str(len(vectors)),
                "vector_names": json.dumps(list(vectors.keys())),
            }
        )

        # Save vectors
        save_file(vectors, output_path, metadata=metadata)
        logger.info(f"Saved {len(vectors)} steering vectors to {output_path}")

    def load_vectors(
        self, vector_path: str | Path
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """
        Load steering vectors from safetensors format.

        Args:
            vector_path: Path to vector file

        Returns:
            Tuple of (vectors dictionary, metadata dictionary)
        """
        vector_path = Path(vector_path)

        if not vector_path.exists():
            raise FileNotFoundError(f"Vector file not found: {vector_path}")

        vectors = load_file(vector_path)

        # Load metadata if available
        try:
            with open(vector_path, "rb") as f:
                # This is a simplified way to get metadata - in practice you'd use
                # safetensors metadata reading functionality
                metadata = {}
        except Exception as _:
            metadata = {}

        logger.info(f"Loaded {len(vectors)} steering vectors from {vector_path}")
        return vectors, metadata

    @classmethod
    def get_paper_preference_info(cls, preference_name: str) -> dict:
        """
        Get preference information from the research paper.

        Args:
            preference_name: Name of the preference dimension

        Returns:
            Dictionary with negative, positive traits and description
        """
        return cls.PAPER_PREFERENCES.get(preference_name.lower(), {})

    @classmethod
    def list_paper_preferences(cls) -> list[str]:
        """Get list of preference dimensions from the research paper."""
        return list(cls.PAPER_PREFERENCES.keys())

    @staticmethod
    def load_examples_from_json(file_path: str | Path) -> list[str]:
        """
        Load text examples from a JSON file.

        Args:
            file_path: Path to JSON file containing examples

        Returns:
            List of text examples
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "examples" in data:
            return data["examples"]
        elif isinstance(data, dict) and "texts" in data:
            return data["texts"]
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")

    @classmethod
    def generate_paper_template_data(
        cls, preference_name: str
    ) -> tuple[list[str], list[str]]:
        """
        Generate template training data based on research paper preferences.

        Args:
            preference_name: Name of the preference dimension

        Returns:
            Tuple of (positive_examples, negative_examples)
        """
        pref_info = cls.get_paper_preference_info(preference_name)
        if not pref_info:
            raise ValueError(f"Unknown preference: {preference_name}")

        # Template examples based on the paper's methodology
        templates = {
            "cost": {
                "positive": [  # Luxury examples
                    "Visit the exclusive rooftop restaurant with stunning city views and premium cuisine.",
                    "Book a private suite at the five-star hotel with personalized concierge service.",
                    "Experience the Michelin-starred tasting menu with wine pairings.",
                    "Enjoy a premium spa day with luxury treatments and champagne service.",
                    "Take a private yacht charter with gourmet catering and scenic coastline.",
                    "Stay at the boutique resort with infinity pools and world-class amenities.",
                ],
                "negative": [  # Budget examples
                    "Try the local food truck with affordable and delicious street tacos.",
                    "Stay at the budget hostel with shared rooms and basic amenities.",
                    "Visit the free public park with walking trails and picnic areas.",
                    "Check out the discount grocery store for cost-effective meal options.",
                    "Use public transportation and walk to save on travel costs.",
                    "Find the happy hour deals at local bars and restaurants.",
                ],
            },
            "ambiance": {
                "positive": [  # Hipster examples
                    "Discover the hidden speakeasy behind the vintage bookstore.",
                    "Visit the underground coffee roastery in the converted warehouse.",
                    "Check out the indie record store with vinyl listening stations.",
                    "Try the farm-to-table restaurant in the trendy arts district.",
                    "Explore the artisanal craft brewery with experimental flavors.",
                    "Find the vintage clothing shop with curated retro pieces.",
                ],
                "negative": [  # Touristy examples
                    "Visit the famous landmark that's featured on all the postcards.",
                    "Take the hop-on-hop-off bus tour to see the main attractions.",
                    "Check out the souvenir shops along the main tourist strip.",
                    "Dine at the popular chain restaurant near the visitor center.",
                    "Take photos at the iconic viewpoint where everyone goes.",
                    "Visit the large museum with the famous exhibitions and crowds.",
                ],
            },
            "age": {
                "positive": [  # Adults examples
                    "Enjoy the wine tasting at the sophisticated vineyard.",
                    "Visit the upscale cocktail lounge with live jazz music.",
                    "Experience the fine dining restaurant with an extensive wine list.",
                    "Take the cultural walking tour focusing on architecture and history.",
                    "Attend the art gallery opening with contemporary exhibitions.",
                    "Relax at the adults-only spa resort with wellness treatments.",
                ],
                "negative": [  # Kids examples
                    "Visit the interactive children's museum with hands-on exhibits.",
                    "Spend time at the playground with slides, swings and climbing structures.",
                    "Go to the petting zoo where kids can feed and interact with animals.",
                    "Check out the family restaurant with kids' menus and play areas.",
                    "Take the scenic train ride with child-friendly commentary.",
                    "Visit the ice cream parlor with colorful flavors and fun toppings.",
                ],
            },
            "time": {
                "positive": [  # Morning examples
                    "Start early with the sunrise yoga session in the park.",
                    "Visit the farmers market that opens at dawn with fresh produce.",
                    "Try the breakfast cafe known for artisanal coffee and pastries.",
                    "Take the morning hiking trail while it's cool and peaceful.",
                    "Enjoy the early bird special at the local diner.",
                    "Visit the morning meditation center for a peaceful start.",
                ],
                "negative": [  # Evening examples
                    "Experience the vibrant nightlife scene with live music venues.",
                    "Dine at the restaurant that comes alive after sunset.",
                    "Visit the night market with street food and evening entertainment.",
                    "Enjoy cocktails at the rooftop bar with city lights views.",
                    "Take the evening ghost tour through historic districts.",
                    "Attend the late-night comedy show at the local theater.",
                ],
            },
            "culture": {
                "positive": [  # American examples
                    "Try the classic American BBQ joint with pulled pork and ribs.",
                    "Visit the baseball stadium for the all-American game experience.",
                    "Check out the diner serving burgers, fries, and milkshakes.",
                    "Experience the country music venue with line dancing.",
                    "Visit the steakhouse known for prime cuts and bourbon.",
                    "Try the Southern comfort food restaurant with fried chicken.",
                ],
                "negative": [  # Asian examples
                    "Visit the authentic dim sum restaurant in Chinatown.",
                    "Try the ramen shop with traditional Japanese broth and noodles.",
                    "Experience the Korean BBQ with table-side grilling.",
                    "Check out the Thai restaurant with spicy curries and pad thai.",
                    "Visit the sushi bar with fresh fish and traditional preparation.",
                    "Try the Vietnamese pho restaurant with aromatic herbs and broth.",
                ],
            },
        }

        template_data = templates.get(preference_name.lower())
        if not template_data:
            # Generate generic examples if specific templates not available
            pos_trait = pref_info["positive"]
            neg_trait = pref_info["negative"]

            positive_examples = [
                f"This recommendation focuses on {pos_trait} options that align with your preferences.",
                f"Here are some {pos_trait}-oriented suggestions that match what you're looking for.",
                f"I'd recommend these {pos_trait} choices based on your requirements.",
            ]
            negative_examples = [
                f"These {neg_trait} options provide a different perspective on your request.",
                f"Here are some {neg_trait}-focused alternatives to consider.",
                f"These {neg_trait} suggestions offer a contrasting approach.",
            ]
        else:
            positive_examples = template_data["positive"]
            negative_examples = template_data["negative"]

        return positive_examples, negative_examples

    def cleanup(self):
        """Clean up model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleaned up model resources")
