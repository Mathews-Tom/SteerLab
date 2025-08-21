"""Tool for comparing steered vs unsteered model outputs."""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler

from steerlab.core.model import SteerableModel
from steerlab.core.vectors import SteeringVectorManager
from steerlab.evaluation.metrics import (
    EvaluationResult,
    SteeringEvaluator,
    load_evaluation_data,
)

# Setup rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results from comparing steered vs baseline outputs."""

    preference: str
    model_name: str
    test_prompts: list[str]
    steering_strengths: list[float]
    results: list[EvaluationResult]
    baseline_alignment: float
    improvement_summary: dict[str, float]


class SteeringComparator:
    """Compares steered vs unsteered model performance."""

    def __init__(self, model_name: str):
        """Initialize comparator with model."""
        self.model_name = model_name
        self.model = None
        self.evaluator = SteeringEvaluator()
        self.vector_manager = SteeringVectorManager(model_name)

    async def load_model(self):
        """Load the steerable model."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SteerableModel(self.model_name)
            logger.info("Model loaded successfully")

    async def generate_outputs(
        self,
        prompts: list[str],
        preference_settings: dict[str, float] | None = None,
        max_length: int = 100,
        progress: Progress | None = None,
        task_id: int | None = None,
    ) -> list[str]:
        """Generate outputs with optional steering."""
        if self.model is None:
            await self.load_model()

        outputs = []
        try:
            if preference_settings:
                self.model.set_steering(preference_settings)

            for i, prompt in enumerate(prompts):
                if progress and task_id is not None:
                    progress.update(task_id, description=f"Generating: {prompt[:40]}...")
                
                output = self.model.generate(prompt, max_length=max_length)
                outputs.append(output)
                
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

        finally:
            if preference_settings:
                self.model.clear_steering()

        return outputs

    async def compare_steering_effectiveness(
        self,
        vector_path: Path,
        test_prompts: list[str],
        steering_strengths: list[float],
        data_dir: Path,
        max_length: int = 100,
    ) -> ComparisonResult:
        """
        Compare model performance across different steering strengths.

        Args:
            vector_path: Path to steering vectors
            test_prompts: List of prompts for evaluation
            steering_strengths: List of strengths to test
            data_dir: Directory containing training data
            max_length: Maximum generation length

        Returns:
            ComparisonResult with comprehensive comparison
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            expand=True,
        ) as progress:
            
            # Setup task
            total_steps = len(steering_strengths) + 1  # +1 for baseline
            main_task = progress.add_task("🧪 Evaluating steering effectiveness", total=total_steps)
            
            # Load model and vectors
            progress.update(main_task, description="🔧 Loading model and vectors...")
            await self.load_model()

            vectors, metadata = self.vector_manager.load_vectors(vector_path)
            preference = metadata.get("preference_name", metadata.get("preference", "unknown"))
            self.model.load_steering_vectors(vectors)

            # Load evaluation data
            positive_examples, negative_examples = load_evaluation_data(data_dir)

            # Generate baseline outputs (no steering)
            baseline_task = progress.add_task("📊 Generating baseline", total=len(test_prompts))
            baseline_outputs = await self.generate_outputs(
                test_prompts, None, max_length, progress, baseline_task
            )
            progress.update(main_task, advance=1)

            # Compute baseline preference alignment  
            baseline_alignment = self.evaluator.compute_preference_alignment(
                baseline_outputs, positive_examples, negative_examples
            )

            # Test different steering strengths
            results = []
            for strength in steering_strengths:
                # Generate steered outputs
                steering_task = progress.add_task(f"⚖️  Testing strength {strength:+.1f}", total=len(test_prompts))
                preference_settings = {preference: strength}
                steered_outputs = await self.generate_outputs(
                    test_prompts, preference_settings, max_length, progress, steering_task
                )

                # Evaluate this configuration
                progress.update(steering_task, description=f"📈 Evaluating strength {strength:+.1f}")
                result = self.evaluator.evaluate_steering_effectiveness(
                    steered_outputs=steered_outputs,
                    baseline_outputs=baseline_outputs,
                    positive_examples=positive_examples,
                    negative_examples=negative_examples,
                    preference=preference,
                    steering_strength=strength,
                    test_prompts=test_prompts,
                )

                results.append(result)
                progress.update(main_task, advance=1)

        # Calculate improvement summary
        improvement_summary = self._calculate_improvements(results, baseline_alignment)

        comparison_result = ComparisonResult(
            preference=preference,
            model_name=self.model_name,
            test_prompts=test_prompts,
            steering_strengths=steering_strengths,
            results=results,
            baseline_alignment=baseline_alignment,
            improvement_summary=improvement_summary,
        )

        logger.info(
            f"Comparison complete - tested {len(steering_strengths)} configurations"
        )
        return comparison_result

    def _calculate_improvements(
        self, results: list[EvaluationResult], baseline_alignment: float
    ) -> dict[str, float]:
        """Calculate improvement metrics."""
        if not results:
            return {}

        # Find best performing configuration
        best_result = max(results, key=lambda r: r.preference_alignment_score)

        alignment_improvement = (
            best_result.preference_alignment_score - baseline_alignment
        )

        # Calculate average improvements
        avg_alignment = sum(r.preference_alignment_score for r in results) / len(
            results
        )
        avg_fluency = sum(r.fluency_score for r in results) / len(results)
        avg_coherence = sum(r.semantic_coherence_score for r in results) / len(results)

        return {
            "max_alignment_improvement": alignment_improvement,
            "best_steering_strength": best_result.steering_strength,
            "best_alignment_score": best_result.preference_alignment_score,
            "baseline_alignment": baseline_alignment,
            "avg_alignment_score": avg_alignment,
            "avg_fluency_score": avg_fluency,
            "avg_coherence_score": avg_coherence,
            "relative_improvement": (alignment_improvement / baseline_alignment * 100)
            if baseline_alignment > 0
            else 0,
        }

    def display_results(self, comparison: ComparisonResult):
        """Display comparison results in a formatted table."""
        console.print(
            Panel(
                f"🎯 Steering Effectiveness Analysis\n"
                f"📊 Model: {comparison.model_name}\n"
                f"🔄 Preference: {comparison.preference}\n"
                f"📝 Test Prompts: {len(comparison.test_prompts)}",
                title="Evaluation Summary",
            )
        )

        # Results table
        table = Table(title="Steering Performance by Strength")
        table.add_column("Strength", style="cyan", no_wrap=True)
        table.add_column("Alignment", style="green")
        table.add_column("Fluency", style="blue")
        table.add_column("Coherence", style="magenta")
        table.add_column("Diversity", style="yellow")

        # Add baseline row
        table.add_row(
            "Baseline (0.0)", f"{comparison.baseline_alignment:.3f}", "-", "-", "-"
        )

        # Add steering results
        for result in comparison.results:
            table.add_row(
                f"{result.steering_strength:+.1f}",
                f"{result.preference_alignment_score:.3f}",
                f"{result.fluency_score:.3f}",
                f"{result.semantic_coherence_score:.3f}",
                f"{result.diversity_score:.3f}",
            )

        console.print(table)

        # Improvement summary
        summary = comparison.improvement_summary
        console.print(
            Panel(
                f"🚀 Best Alignment Improvement: {summary.get('max_alignment_improvement', 0):.3f}\n"
                f"🎯 Best Steering Strength: {summary.get('best_steering_strength', 0):+.1f}\n"
                f"📈 Relative Improvement: {summary.get('relative_improvement', 0):+.1f}%\n"
                f"📊 Best Alignment Score: {summary.get('best_alignment_score', 0):.3f}",
                title="🏆 Key Improvements",
            )
        )

    def save_results(self, comparison: ComparisonResult, output_path: Path):
        """Save comparison results to JSON file."""
        output_data = {
            "comparison_summary": asdict(comparison),
            "evaluation_details": [asdict(result) for result in comparison.results],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def plot_results(self, comparison: ComparisonResult, save_path: Path | None = None):
        """Create visualization of steering effectiveness."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        strengths = comparison.steering_strengths
        alignments = [r.preference_alignment_score for r in comparison.results]
        fluencies = [r.fluency_score for r in comparison.results]
        coherences = [r.semantic_coherence_score for r in comparison.results]
        diversities = [r.diversity_score for r in comparison.results]

        # Preference Alignment
        ax1.plot(strengths, alignments, "o-", color="green", linewidth=2, markersize=6)
        ax1.axhline(
            y=comparison.baseline_alignment,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Baseline",
        )
        ax1.set_xlabel("Steering Strength")
        ax1.set_ylabel("Preference Alignment")
        ax1.set_title("Preference Alignment vs Steering Strength")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Fluency
        ax2.plot(strengths, fluencies, "o-", color="blue", linewidth=2, markersize=6)
        ax2.set_xlabel("Steering Strength")
        ax2.set_ylabel("Fluency Score")
        ax2.set_title("Fluency vs Steering Strength")
        ax2.grid(True, alpha=0.3)

        # Coherence
        ax3.plot(strengths, coherences, "o-", color="purple", linewidth=2, markersize=6)
        ax3.set_xlabel("Steering Strength")
        ax3.set_ylabel("Semantic Coherence")
        ax3.set_title("Coherence vs Steering Strength")
        ax3.grid(True, alpha=0.3)

        # Diversity
        ax4.plot(
            strengths, diversities, "o-", color="orange", linewidth=2, markersize=6
        )
        ax4.set_xlabel("Steering Strength")
        ax4.set_ylabel("Lexical Diversity")
        ax4.set_title("Diversity vs Steering Strength")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(
            f"Steering Analysis: {comparison.preference} ({comparison.model_name})",
            y=0.98,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()
