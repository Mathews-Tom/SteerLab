"""Quantitative metrics for evaluating steering effectiveness."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from steering evaluation."""

    preference: str
    steering_strength: float
    preference_alignment_score: float
    semantic_coherence_score: float
    fluency_score: float
    diversity_score: float
    example_outputs: Dict[str, str]
    baseline_outputs: Dict[str, str]


class SteeringEvaluator:
    """Evaluates effectiveness of steering vectors."""

    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2"):
        """Initialize evaluator with sentence transformer model."""
        self.sentence_model = SentenceTransformer(sentence_model)
        logger.info(f"Initialized evaluator with model: {sentence_model}")

    def compute_preference_alignment(
        self,
        outputs: List[str],
        positive_examples: List[str],
        negative_examples: List[str],
    ) -> float:
        """
        Compute how well outputs align with positive vs negative examples.

        Returns score between 0-1 where:
        - 1.0: Perfect alignment with positive examples
        - 0.0: Perfect alignment with negative examples
        - 0.5: Neutral/mixed alignment
        """
        if not outputs or not positive_examples or not negative_examples:
            return 0.5

        # Encode all texts
        output_embeddings = self.sentence_model.encode(outputs)
        positive_embeddings = self.sentence_model.encode(positive_examples)
        negative_embeddings = self.sentence_model.encode(negative_examples)

        # Compute average similarities
        pos_similarities = []
        neg_similarities = []

        for output_emb in output_embeddings:
            # Average similarity to positive examples
            pos_sim = np.mean(
                cosine_similarity(output_emb.reshape(1, -1), positive_embeddings)
            )
            pos_similarities.append(pos_sim)

            # Average similarity to negative examples
            neg_sim = np.mean(
                cosine_similarity(output_emb.reshape(1, -1), negative_embeddings)
            )
            neg_similarities.append(neg_sim)

        # Overall alignment: how much more similar to positive vs negative
        avg_pos_sim = np.mean(pos_similarities)
        avg_neg_sim = np.mean(neg_similarities)

        # Normalize to 0-1 scale
        if avg_pos_sim + avg_neg_sim == 0:
            return 0.5

        alignment_score = avg_pos_sim / (avg_pos_sim + avg_neg_sim)
        return float(alignment_score)

    def compute_semantic_coherence(self, outputs: List[str]) -> float:
        """
        Compute semantic coherence of outputs.
        High coherence = outputs are semantically similar to each other.
        """
        if len(outputs) < 2:
            return 1.0

        embeddings = self.sentence_model.encode(outputs)

        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)

        return float(np.mean(similarities))

    def compute_fluency_score(self, outputs: List[str]) -> float:
        """
        Compute fluency score based on linguistic patterns.
        Simple heuristics for demonstration - could be enhanced with language models.
        """
        if not outputs:
            return 0.0

        scores = []
        for output in outputs:
            # Basic fluency indicators
            words = output.strip().split()

            # Length check (not too short, not too long)
            length_score = (
                min(1.0, len(words) / 10)
                if len(words) <= 50
                else max(0.5, 1 - (len(words) - 50) / 100)
            )

            # Sentence completeness (ends with punctuation)
            completion_score = 1.0 if output.strip()[-1] in ".!?" else 0.7

            # Basic grammar patterns (contains some structure words)
            structure_words = [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "with",
                "for",
                "to",
                "of",
                "in",
                "on",
            ]
            structure_score = min(
                1.0, sum(1 for word in words if word.lower() in structure_words) / 3
            )

            scores.append((length_score + completion_score + structure_score) / 3)

        return float(np.mean(scores))

    def compute_diversity_score(self, outputs: List[str]) -> float:
        """
        Compute lexical diversity of outputs.
        Higher diversity = more varied vocabulary and structures.
        """
        if not outputs:
            return 0.0

        all_words = []
        for output in outputs:
            words = re.findall(r"\b\w+\b", output.lower())
            all_words.extend(words)

        if not all_words:
            return 0.0

        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words)
        return float(diversity)

    def evaluate_steering_effectiveness(
        self,
        steered_outputs: List[str],
        baseline_outputs: List[str],
        positive_examples: List[str],
        negative_examples: List[str],
        preference: str,
        steering_strength: float,
        test_prompts: List[str],
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of steering effectiveness.

        Args:
            steered_outputs: Model outputs with steering applied
            baseline_outputs: Model outputs without steering
            positive_examples: Training examples for positive preference
            negative_examples: Training examples for negative preference
            preference: Name of the preference dimension
            steering_strength: Steering strength used (-1 to 1)
            test_prompts: Prompts used for generation

        Returns:
            EvaluationResult with comprehensive metrics
        """
        logger.info(
            f"Evaluating steering for preference '{preference}' with strength {steering_strength}"
        )

        # Compute preference alignment for steered outputs
        steered_alignment = self.compute_preference_alignment(
            steered_outputs, positive_examples, negative_examples
        )

        # Compute other metrics
        coherence = self.compute_semantic_coherence(steered_outputs)
        fluency = self.compute_fluency_score(steered_outputs)
        diversity = self.compute_diversity_score(steered_outputs)

        # Create example mappings
        example_outputs = {}
        baseline_example = {}
        for i, prompt in enumerate(test_prompts[: min(3, len(test_prompts))]):
            if i < len(steered_outputs):
                example_outputs[f"prompt_{i + 1}"] = {
                    "prompt": prompt,
                    "steered_output": steered_outputs[i],
                }
            if i < len(baseline_outputs):
                baseline_example[f"prompt_{i + 1}"] = {
                    "prompt": prompt,
                    "baseline_output": baseline_outputs[i],
                }

        result = EvaluationResult(
            preference=preference,
            steering_strength=steering_strength,
            preference_alignment_score=steered_alignment,
            semantic_coherence_score=coherence,
            fluency_score=fluency,
            diversity_score=diversity,
            example_outputs=example_outputs,
            baseline_outputs=baseline_example,
        )

        logger.info(
            f"Evaluation complete - Alignment: {steered_alignment:.3f}, "
            f"Coherence: {coherence:.3f}, Fluency: {fluency:.3f}, "
            f"Diversity: {diversity:.3f}"
        )

        return result


def load_evaluation_data(data_dir: Path) -> Tuple[List[str], List[str]]:
    """Load positive and negative examples from data directory."""
    positive_file = data_dir / "cost_positive.json"
    negative_file = data_dir / "cost_negative.json"

    with open(positive_file) as f:
        positive_examples = json.load(f)

    with open(negative_file) as f:
        negative_examples = json.load(f)

    return positive_examples, negative_examples
