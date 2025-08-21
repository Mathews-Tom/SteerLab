#!/usr/bin/env python3
"""
Basic Steering Example for SteerLab

This example demonstrates the complete workflow:
1. Create training data for a 'formality' preference
2. Compute steering vectors using CAA
3. Test steered vs unsteered generation
4. Use vectors via API

Run with: uv run examples/basic_steering_example.py
"""

import json
import logging
from pathlib import Path

import torch

from steerlab.core.model import SteerableModel
from steerlab.core.vectors import SteeringVectorManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_training_data():
    """Create sample training data for formality preference."""
    logger.info("🔧 Creating training data...")

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Formal examples (positive)
    formal_examples = [
        "I would be delighted to assist you with this matter.",
        "Please allow me to express my sincere appreciation for your consideration.",
        "I respectfully request that you review the attached documentation.",
        "It would be most appropriate to schedule a meeting at your earliest convenience.",
        "I am writing to formally inquire about the status of my application.",
        "May I suggest that we proceed with the proposal as outlined?",
        "I trust this communication finds you in good health and spirits.",
        "Please accept my humble apologies for any inconvenience caused.",
    ]

    # Informal examples (negative)
    informal_examples = [
        "Hey! Can you help me out with this?",
        "Thanks a bunch, you're awesome!",
        "Let's grab coffee and chat about it.",
        "No worries, we'll figure it out as we go.",
        "What's up? How's everything going?",
        "Cool, sounds good to me!",
        "Oops, my bad! I totally forgot about that.",
        "Yeah, that's definitely the way to go.",
    ]

    # Save to JSON files
    positive_file = data_dir / "formality_positive.json"
    negative_file = data_dir / "formality_negative.json"

    with open(positive_file, "w") as f:
        json.dump(formal_examples, f, indent=2)

    with open(negative_file, "w") as f:
        json.dump(informal_examples, f, indent=2)

    logger.info(
        f"✅ Created {positive_file} with {len(formal_examples)} formal examples"
    )
    logger.info(
        f"✅ Created {negative_file} with {len(informal_examples)} informal examples"
    )

    return str(positive_file), str(negative_file)


def compute_steering_vectors(model_name="microsoft/DialoGPT-medium"):
    """Compute steering vectors for formality preference."""
    logger.info(f"⚡ Computing steering vectors for model: {model_name}")

    # Create training data
    positive_file, negative_file = create_training_data()

    # Load training data
    positive_examples = SteeringVectorManager.load_examples_from_json(positive_file)
    negative_examples = SteeringVectorManager.load_examples_from_json(negative_file)

    # Initialize vector manager
    vector_manager = SteeringVectorManager(model_name)

    # Compute vectors
    vectors = vector_manager.compute_steering_vectors(
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        preference_name="formality",
        max_length=128,  # Shorter for faster processing
    )

    # Save vectors
    vectors_dir = Path("vectors")
    vectors_dir.mkdir(exist_ok=True)
    output_path = vectors_dir / "formality_vectors.safetensors"

    metadata = {
        "preference_name": "formality",
        "model_name": model_name,
        "description": "Formal vs informal communication style",
    }

    vector_manager.save_vectors(vectors, output_path, metadata)
    vector_manager.cleanup()

    logger.info(f"💾 Saved {len(vectors)} vectors to {output_path}")
    return str(output_path), vectors


def test_steering(
    model_name="microsoft/DialoGPT-medium", vector_path=None, vectors=None
):
    """Test steering with different preference strengths."""
    logger.info("🧪 Testing steering effects...")

    # Initialize steerable model
    steerable_model = SteerableModel(model_name)

    # Load vectors
    if vectors is None:
        if vector_path is None:
            raise ValueError("Either vector_path or vectors must be provided")
        vector_manager = SteeringVectorManager("dummy")
        vectors, _ = vector_manager.load_vectors(vector_path)

    steerable_model.load_steering_vectors(vectors)

    # Test prompts
    test_prompts = [
        "I need help with my project",
        "Thanks for the information",
        "Can we schedule a meeting",
    ]

    steering_strengths = [0.0, 0.5, 1.0, -0.5, -1.0]

    print("\n" + "=" * 80)
    print("STEERING TEST RESULTS")
    print("=" * 80)

    for prompt in test_prompts:
        print(f"\n🔹 Prompt: '{prompt}'")
        print("-" * 60)

        for strength in steering_strengths:
            # Set steering
            if strength != 0:
                steerable_model.set_steering({"formality": strength})

            # Generate
            try:
                response = steerable_model.generate(
                    prompt=prompt, max_length=50, temperature=0.7, do_sample=True
                )

                strength_label = f"Strength {strength:+.1f}"
                if strength == 0:
                    strength_label = "No steering"
                elif strength > 0:
                    strength_label = f"Formal {strength:+.1f}"
                else:
                    strength_label = f"Informal {strength:+.1f}"

                print(f"  {strength_label:>12}: {response}")

            except Exception as e:
                print(f"  {strength:+.1f}: Error - {e}")

        print()

    print("=" * 80)


def demonstrate_api_usage():
    """Show how to use steering vectors via the API."""
    logger.info("📡 API Usage Example")

    print("\n" + "=" * 60)
    print("API USAGE EXAMPLE")
    print("=" * 60)

    api_example = """
# 1. Start the API server
uv run steerlab serve

# 2. Load steering vectors
curl -X POST "http://localhost:8000/load-vectors" \\
  -H "Content-Type: application/json" \\
  -d '{"vector_path": "vectors/formality_vectors.safetensors"}'

# 3. Generate steered text
curl -X POST "http://localhost:8000/chat" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "I need help with my project",
    "preferences": {"formality": 0.8},
    "max_length": 100
  }'

# 4. Update session preferences
curl -X POST "http://localhost:8000/preferences" \\
  -H "Content-Type: application/json" \\
  -d '{
    "preferences": {"formality": 0.5},
    "session_id": "user123"
  }'
"""

    print(api_example)
    print("=" * 60)


def main():
    """Run the complete steering example."""
    print("🚀 SteerLab Basic Steering Example")
    print("==================================")

    model_name = "microsoft/DialoGPT-medium"  # Smaller model for demo

    try:
        # Step 1: Compute steering vectors
        vector_path, vectors = compute_steering_vectors(model_name)

        # Step 2: Test steering effects
        test_steering(model_name, vectors=vectors)

        # Step 3: Show API usage
        demonstrate_api_usage()

        print("\n✅ Example completed successfully!")
        print(f"📁 Vectors saved to: {vector_path}")
        print("🔧 You can now use these vectors in the SteerLab API")

    except KeyboardInterrupt:
        print("\n❌ Example interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    main()
