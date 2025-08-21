"""
Command-line interface for SteerLab.

This module provides CLI commands for offline tasks like computing steering vectors,
managing vector files, and running the API server.
"""

import asyncio
import json
import logging
import warnings
from pathlib import Path

import click
import uvicorn

# Filter out HuggingFace Hub deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

from .core.model import SteerableModel
from .core.vectors import SteeringVectorManager
from .evaluation.compare import SteeringComparator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """SteerLab: Implementation of 'Steerable Chatbots' research (arXiv:2505.04260v2)."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--model", "-m", required=True, help="HuggingFace model identifier")
@click.option(
    "--preference", "-p", required=True, help="Name of the preference to learn"
)
@click.option(
    "--positive-data", required=True, help="Path to JSON file with positive examples"
)
@click.option(
    "--negative-data", required=True, help="Path to JSON file with negative examples"
)
@click.option(
    "--output",
    "-o",
    help="Output path for steering vectors (default: vectors/{preference}_vectors.safetensors)",
)
@click.option(
    "--max-length", default=256, help="Maximum sequence length for processing examples"
)
@click.option(
    "--device", default="auto", help="Device to run computation on (auto, cuda, cpu)"
)
def compute_vectors(
    model: str,
    preference: str,
    positive_data: str,
    negative_data: str,
    output: str | None,
    max_length: int,
    device: str,
):
    """
    Compute steering vectors using Contrastive Activation Addition (CAA).

    This command implements the offline vector generation process, taking
    contrastive datasets and producing steering vectors for a specific preference.
    """
    try:
        click.echo(f"🚀 Computing steering vectors for preference: {preference}")
        click.echo(f"📦 Model: {model}")
        click.echo(f"📊 Positive examples: {positive_data}")
        click.echo(f"📊 Negative examples: {negative_data}")

        # Load example data
        click.echo("\n📁 Loading example data...")
        positive_examples = SteeringVectorManager.load_examples_from_json(positive_data)
        negative_examples = SteeringVectorManager.load_examples_from_json(negative_data)

        click.echo(f"✅ Loaded {len(positive_examples)} positive examples")
        click.echo(f"✅ Loaded {len(negative_examples)} negative examples")

        # Initialize vector manager
        click.echo("\n🔧 Initializing vector manager...")
        vector_manager = SteeringVectorManager(model, device=device)

        # Compute vectors
        click.echo("\n⚡ Computing steering vectors (this may take a while)...")
        with click.progressbar(length=1, label="Computing vectors") as bar:
            vectors = vector_manager.compute_steering_vectors(
                positive_examples=positive_examples,
                negative_examples=negative_examples,
                preference_name=preference,
                max_length=max_length,
            )
            bar.update(1)

        click.echo(f"✅ Computed {len(vectors)} steering vectors")

        # Determine output path
        if output is None:
            output = f"vectors/{preference}_vectors.safetensors"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save vectors
        click.echo(f"\n💾 Saving vectors to: {output_path}")
        metadata = {
            "preference_name": preference,
            "model_name": model,
            "num_positive": str(len(positive_examples)),
            "num_negative": str(len(negative_examples)),
            "max_length": str(max_length),
        }

        vector_manager.save_vectors(vectors, output_path, metadata)

        # Cleanup
        vector_manager.cleanup()

        click.echo(f"🎉 Successfully computed and saved steering vectors!")
        click.echo(f"📍 Vectors saved to: {output_path.absolute()}")

    except Exception as e:
        logger.error(f"Vector computation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--vector-path", "-v", required=True, help="Path to steering vector file")
@click.option(
    "--model",
    "-m",
    help="Model to test with (optional, uses model from vectors if available)",
)
@click.option(
    "--prompt", "-p", default="Tell me about the weather.", help="Test prompt"
)
@click.option("--strength", "-s", default=0.5, help="Steering strength (-1.0 to 1.0)")
def test_vectors(vector_path: str, model: str | None, prompt: str, strength: float):
    """
    Test steering vectors by generating text with and without steering.

    This command helps validate that computed vectors work as expected
    by comparing steered vs unsteered generation.
    """
    try:
        click.echo(f"🧪 Testing steering vectors: {vector_path}")
        click.echo(f"📝 Test prompt: {prompt}")
        click.echo(f"⚡ Steering strength: {strength}")

        # Load vectors
        click.echo("\n📁 Loading steering vectors...")
        vector_manager = SteeringVectorManager("dummy")  # Dummy model for loading
        vectors, metadata = vector_manager.load_vectors(vector_path)

        # Determine model to use
        if model is None:
            if "model_name" in metadata:
                model = metadata["model_name"]
            else:
                model = "microsoft/DialoGPT-medium"  # Default fallback

        click.echo(f"📦 Using model: {model}")

        # Initialize steerable model
        click.echo("\n🔧 Loading model...")
        steerable_model = SteerableModel(model)
        steerable_model.load_steering_vectors(vectors)

        # Generate without steering
        click.echo("\n📝 Generating unsteered text...")
        unsteered_text = steerable_model.generate(prompt, max_length=100)

        # Generate with steering
        click.echo(f"📝 Generating steered text (strength={strength})...")
        preference_name = metadata.get("preference_name", "test_preference")
        steerable_model.set_steering({preference_name: strength})
        steered_text = steerable_model.generate(prompt, max_length=100)

        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("RESULTS")
        click.echo("=" * 60)
        click.echo("\n🔹 UNSTEERED:")
        click.echo(f"   {unsteered_text}")
        click.echo(f"\n🔸 STEERED ({preference_name}={strength}):")
        click.echo(f"   {steered_text}")
        click.echo("\n" + "=" * 60)

        click.echo("\n✅ Vector testing complete!")

    except Exception as e:
        logger.error(f"Vector testing failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort() from e


@cli.command()
@click.option(
    "--vector-dir", "-d", default="vectors", help="Directory containing vector files"
)
def list_vectors(vector_dir: str):
    """List available steering vector files."""
    try:
        vector_path = Path(vector_dir)
        if not vector_path.exists():
            click.echo(f"❌ Vector directory not found: {vector_path}")
            return

        vector_files = list(vector_path.glob("*.safetensors"))

        if not vector_files:
            click.echo(f"📁 No vector files found in: {vector_path}")
            return

        click.echo(f"📁 Found {len(vector_files)} vector file(s) in {vector_path}:")
        click.echo()

        for vector_file in sorted(vector_files):
            click.echo(f"  📄 {vector_file.name}")

            # Try to load metadata
            try:
                vector_manager = SteeringVectorManager("dummy")
                vectors, metadata = vector_manager.load_vectors(vector_file)

                if metadata:
                    click.echo(
                        f"     └─ Model: {metadata.get('model_name', 'unknown')}"
                    )
                    click.echo(
                        f"     └─ Preference: {metadata.get('preference_name', 'unknown')}"
                    )
                    click.echo(f"     └─ Vectors: {len(vectors)}")
                else:
                    click.echo(f"     └─ Vectors: {len(vectors)} (no metadata)")

            except Exception as e:
                click.echo(f"     └─ Error loading: {e}")

            click.echo()

    except Exception as e:
        logger.error(f"Failed to list vectors: {e}")
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the SteerLab API server."""
    try:
        click.echo("🚀 Starting SteerLab API server...")
        click.echo(f"🌐 Host: {host}")
        click.echo(f"🔌 Port: {port}")
        click.echo(f"🔄 Reload: {reload}")
        click.echo(f"📖 API docs will be available at: http://{host}:{port}/docs")

        uvicorn.run("steerlab.api.server:app", host=host, port=port, reload=reload)

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option("--model", "-m", required=True, help="HuggingFace model identifier")
@click.option(
    "--vector-path", "-v", required=True, help="Path to steering vectors file"
)
@click.option(
    "--data-dir", "-d", default="data", help="Directory containing training data"
)
@click.option(
    "--strengths",
    default="-1.0,-0.5,0.0,0.5,1.0",
    help="Comma-separated steering strengths to test",
)
@click.option("--output", "-o", help="Output directory for results (default: results/)")
@click.option("--max-length", default=100, help="Maximum generation length")
@click.option("--show-plot", is_flag=True, help="Display interactive plot")
def evaluate_steering(
    model: str,
    vector_path: str,
    data_dir: str,
    strengths: str,
    output: str | None,
    max_length: int,
    show_plot: bool,
):
    """Evaluate steering effectiveness and generate comparison report."""
    try:
        click.echo("🧪 Evaluating steering effectiveness")
        click.echo(f"📦 Model: {model}")
        click.echo(f"📊 Vectors: {vector_path}")
        click.echo()

        # Parse steering strengths
        strength_values = [float(s.strip()) for s in strengths.split(",")]

        # Default test prompts for demonstration
        test_prompts = [
            "Help me plan a weekend getaway",
            "Recommend a restaurant for dinner tonight",
            "Suggest a gift for my friend's birthday",
            "Plan a fun activity for this afternoon",
            "Choose a movie for us to watch",
        ]

        # Setup paths
        vector_path = Path(vector_path)
        data_path = Path(data_dir)
        output_path = Path(output) if output else Path("results")
        output_path.mkdir(exist_ok=True)

        # Initialize comparator
        comparator = SteeringComparator(model)

        # Run evaluation
        async def run_evaluation():
            return await comparator.compare_steering_effectiveness(
                vector_path=vector_path,
                test_prompts=test_prompts,
                steering_strengths=strength_values,
                data_dir=data_path,
                max_length=max_length,
            )

        comparison_result = asyncio.run(run_evaluation())

        # Display results
        comparator.display_results(comparison_result)

        # Save results
        results_file = (
            output_path
            / f"evaluation_{comparison_result.preference}_{model.replace('/', '_')}.json"
        )
        comparator.save_results(comparison_result, results_file)

        # Generate plot
        plot_file = (
            output_path
            / f"evaluation_{comparison_result.preference}_{model.replace('/', '_')}.png"
        )
        comparator.plot_results(comparison_result, plot_file if not show_plot else None)

        click.echo()
        click.echo(f"✅ Evaluation complete!")
        click.echo(f"📄 Results saved to: {results_file}")
        click.echo(f"📊 Plot saved to: {plot_file}")

        # Print summary
        summary = comparison_result.improvement_summary
        click.echo(
            f"🚀 Best improvement: {summary.get('max_alignment_improvement', 0):+.3f} at strength {summary.get('best_steering_strength', 0):+.1f}"
        )
        click.echo(
            f"📈 Relative improvement: {summary.get('relative_improvement', 0):+.1f}%"
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)

    except ImportError as e:
        click.echo("❌ Error: uvicorn is required to run the server.", err=True)
        click.echo("Install it with: uv add uvicorn", err=True)
        raise click.Abort() from e
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort() from e


@cli.command()
@click.argument("preference_name")
@click.option("--positive-file", help="Create template for positive examples")
@click.option("--negative-file", help="Create template for negative examples")
@click.option(
    "--use-paper-templates", is_flag=True, help="Use research paper template data"
)
def create_template(
    preference_name: str,
    positive_file: str | None,
    negative_file: str | None,
    use_paper_templates: bool,
):
    """Create template JSON files for training data."""
    if positive_file is None:
        positive_file = f"data/{preference_name}_positive.json"
    if negative_file is None:
        negative_file = f"data/{preference_name}_negative.json"

    # Create data directory
    for file_path in [positive_file, negative_file]:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    if use_paper_templates:
        try:
            # Use research paper template data
            positive_template, negative_template = (
                SteeringVectorManager.generate_paper_template_data(preference_name)
            )

            pref_info = SteeringVectorManager.get_paper_preference_info(preference_name)
            click.echo(
                f"🧪 Using research paper templates for preference: {preference_name}"
            )
            click.echo(f"📝 {pref_info.get('description', 'No description available')}")
            click.echo(f"📊 Negative trait: {pref_info.get('negative', 'unknown')}")
            click.echo(f"📊 Positive trait: {pref_info.get('positive', 'unknown')}")

        except ValueError as e:
            click.echo(f"❌ Error: {e}", err=True)
            available_prefs = SteeringVectorManager.list_paper_preferences()
            click.echo(f"💡 Available paper preferences: {', '.join(available_prefs)}")
            raise click.Abort() from e
    else:
        # Use generic template data
        positive_template = [
            f"This is an example of text that demonstrates {preference_name} in a positive way.",
            f"Another example showing positive {preference_name}.",
            "Add more examples here that represent what you want the model to learn.",
        ]

        negative_template = [
            f"This is an example that lacks {preference_name} or demonstrates it negatively.",
            f"Another example showing what you don't want regarding {preference_name}.",
            "Add more negative examples here for contrast.",
        ]

    # Save templates
    with open(positive_file, "w") as f:
        json.dump(positive_template, f, indent=2)

    with open(negative_file, "w") as f:
        json.dump(negative_template, f, indent=2)

    click.echo("✅ Created template files:")
    click.echo(f"   📄 Positive examples: {positive_file}")
    click.echo(f"   📄 Negative examples: {negative_file}")
    click.echo("\n💡 Edit these files with your training data, then run:")
    click.echo(f"   steerlab compute-vectors -m <model> -p {preference_name} \\")
    click.echo(f"     --positive-data {positive_file} --negative-data {negative_file}")


@cli.command()
def list_paper_models():
    """List supported models from the research paper."""
    from .core.model import SteerableModel

    click.echo("🧪 Supported models from research paper:")
    click.echo("=" * 50)

    for model_name, config in SteerableModel.PAPER_MODEL_CONFIGS.items():
        click.echo(f"\n📦 {model_name}")
        click.echo(f"   └─ Top-k layers: {config['top_k_layers']}")
        click.echo(f"   └─ Functional range: {config['functional_range']}")
        click.echo(f"   └─ Probe type: {config['probe_type']}")


@cli.command()
def list_paper_preferences():
    """List preference dimensions from the research paper."""
    click.echo("🧪 Preference dimensions from research paper:")
    click.echo("=" * 50)

    for pref_name, pref_info in SteeringVectorManager.PAPER_PREFERENCES.items():
        click.echo(f"\n📊 {pref_name}")
        click.echo(f"   └─ Range: {pref_info['negative']} ↔ {pref_info['positive']}")
        click.echo(f"   └─ Description: {pref_info['description']}")


if __name__ == "__main__":
    cli()
