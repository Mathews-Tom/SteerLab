#!/usr/bin/env python3
"""
Research Paper Reproduction Example for SteerLab

This example demonstrates how to reproduce key experiments from the research paper:
"Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering"

The script shows:
1. Using models tested in the paper
2. Working with the five preference dimensions studied
3. Computing steering vectors with paper methodology
4. Testing across different steering strengths
5. Validating the functional steering ranges

Run with: uv run examples/paper_reproduction_example.py
"""

import json
import logging
from pathlib import Path

from steerlab.core.model import SteerableModel
from steerlab.core.vectors import SteeringVectorManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_paper_models():
    """Show the models tested in the research paper."""
    print("🧪 RESEARCH PAPER MODELS")
    print("=" * 50)

    for model_name, config in SteerableModel.PAPER_MODEL_CONFIGS.items():
        print(f"\n📦 {model_name}")
        print(f"   ├─ Parameters: {model_name.split('/')[-1]}")
        print(f"   ├─ Top-k layers: {config['top_k_layers']}")
        print(f"   ├─ Functional range: {config['functional_range']}")
        print(f"   └─ Probe type: {config['probe_type']}")


def demonstrate_paper_preferences():
    """Show the preference dimensions from the research paper."""
    print("\n🧪 RESEARCH PAPER PREFERENCE DIMENSIONS")
    print("=" * 50)

    for pref_name, pref_info in SteeringVectorManager.PAPER_PREFERENCES.items():
        print(f"\n📊 {pref_name.upper()}")
        print(f"   ├─ Negative trait: {pref_info['negative']}")
        print(f"   ├─ Positive trait: {pref_info['positive']}")
        print(f"   └─ Description: {pref_info['description']}")


def create_paper_datasets():
    """Create datasets based on the research paper methodology."""
    print("\n🧪 CREATING PAPER-BASED DATASETS")
    print("=" * 50)

    data_dir = Path("data/paper_reproduction")
    data_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    for preference in SteeringVectorManager.list_paper_preferences():
        try:
            positive_examples, negative_examples = (
                SteeringVectorManager.generate_paper_template_data(preference)
            )

            # Save datasets
            pos_file = data_dir / f"{preference}_positive.json"
            neg_file = data_dir / f"{preference}_negative.json"

            with open(pos_file, "w") as f:
                json.dump(positive_examples, f, indent=2)

            with open(neg_file, "w") as f:
                json.dump(negative_examples, f, indent=2)

            pref_info = SteeringVectorManager.get_paper_preference_info(preference)
            print(f"\n📊 {preference.upper()}")
            print(
                f"   ├─ {pref_info['negative']} examples: {len(negative_examples)} → {neg_file.name}"
            )
            print(
                f"   └─ {pref_info['positive']} examples: {len(positive_examples)} → {pos_file.name}"
            )

            created_files.extend([str(pos_file), str(neg_file)])

        except Exception as e:
            print(f"❌ Failed to create dataset for {preference}: {e}")

    return created_files


def test_paper_model_with_preference(
    model_name="google/gemma-2-2b-it", preference="cost"
):
    """
    Reproduce paper experiments with a specific model and preference.

    This demonstrates:
    - Loading a paper-tested model
    - Computing steering vectors using paper methodology
    - Testing with different steering strengths within functional range
    - Showing validation of steering strength ranges
    """
    print(f"\n🧪 TESTING PAPER MODEL: {model_name}")
    print(f"📊 PREFERENCE: {preference}")
    print("=" * 70)

    try:
        # Get model configuration from paper
        model_config = SteerableModel.PAPER_MODEL_CONFIGS.get(model_name, {})
        if model_config:
            print(f"📝 Using paper configuration:")
            print(f"   ├─ Top-k layers: {model_config['top_k_layers']}")
            print(f"   ├─ Functional range: {model_config['functional_range']}")
            print(f"   └─ Probe type: {model_config['probe_type']}")
        else:
            print(f"⚠️  Model not in paper configurations, using defaults")

        # Generate paper-based training data
        print(f"\n📊 Generating training data for {preference}...")
        positive_examples, negative_examples = (
            SteeringVectorManager.generate_paper_template_data(preference)
        )

        pref_info = SteeringVectorManager.get_paper_preference_info(preference)
        print(
            f"   ├─ Negative ({pref_info['negative']}): {len(negative_examples)} examples"
        )
        print(
            f"   └─ Positive ({pref_info['positive']}): {len(positive_examples)} examples"
        )

        # Compute steering vectors
        print(f"\n⚡ Computing steering vectors...")
        vector_manager = SteeringVectorManager(model_name)

        vectors = vector_manager.compute_steering_vectors(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            preference_name=preference,
            max_length=128,  # Shorter for demo
        )

        print(f"✅ Computed {len(vectors)} steering vectors")

        # Save vectors with paper metadata
        vectors_dir = Path("vectors/paper_reproduction")
        vectors_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            vectors_dir
            / f"{preference}_{model_name.split('/')[-1]}_vectors.safetensors"
        )

        metadata = {
            "preference_name": preference,
            "model_name": model_name,
            "paper_reproduction": "true",
            "negative_trait": pref_info["negative"],
            "positive_trait": pref_info["positive"],
            "description": pref_info["description"],
            **{f"paper_{k}": str(v) for k, v in model_config.items()},
        }

        vector_manager.save_vectors(vectors, output_path, metadata)
        print(f"💾 Vectors saved to: {output_path}")

        # Test steerable model with paper methodology
        print(f"\n🧪 Testing steerable model...")
        steerable_model = SteerableModel(model_name)
        steerable_model.load_steering_vectors(vectors)

        # Test prompts from paper domain (lifestyle planning)
        test_prompts = [
            "Help me choose a present for a friend who likes jewelry.",
            "Plan things to do on vacation to Paris.",
            "Give me some date night restaurants in San Francisco.",
        ]

        # Test with strengths within paper's functional range
        min_strength, max_strength = model_config.get("functional_range", (-10, 10))
        test_strengths = [
            0,
            min_strength // 2,
            max_strength // 2,
            min_strength,
            max_strength,
        ]

        print(
            f"\n📊 Testing steering strengths within functional range {model_config.get('functional_range', (-10, 10))}"
        )
        print("-" * 70)

        for prompt in test_prompts[:1]:  # Test with one prompt for demo
            print(f"\n🔹 Prompt: '{prompt}'")
            print("-" * 50)

            for strength in test_strengths:
                if strength == 0:
                    steerable_model.clear_steering()
                    label = "No steering"
                else:
                    # This will validate and clamp the strength automatically
                    steerable_model.set_steering({preference: strength})
                    trait = (
                        pref_info["positive"] if strength > 0 else pref_info["negative"]
                    )
                    label = f"{trait.title()} {abs(strength)}"

                try:
                    response = steerable_model.generate(
                        prompt=prompt, max_length=50, temperature=0.7, do_sample=True
                    )
                    print(f"  {label:>15}: {response}")

                except Exception as e:
                    print(f"  {label:>15}: Error - {e}")

        vector_manager.cleanup()
        print(f"\n✅ Paper reproduction test completed successfully!")

        return output_path

    except Exception as e:
        logger.error(f"Paper reproduction failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return None


def demonstrate_multi_preference_steering():
    """Demonstrate multi-preference steering as shown in paper E3 experiments."""
    print(f"\n🧪 MULTI-PREFERENCE STEERING (Paper E3)")
    print("=" * 50)

    # The paper tested age + culture (least correlated dimensions)
    preferences = ["age", "culture"]

    print(f"📊 Testing preferences: {', '.join(preferences)}")
    print("   (Selected as least correlated dimensions in paper)")

    # Show what this would look like conceptually
    print(f"\n💡 Example multi-preference configuration:")
    print(f"   steerable_model.set_steering({{")
    print(f"       'age': 0.7,      # toward adults")
    print(f"       'culture': -0.5   # toward asian")
    print(f"   }})")

    for pref in preferences:
        pref_info = SteeringVectorManager.get_paper_preference_info(pref)
        print(f"\n📊 {pref.upper()}: {pref_info['negative']} ↔ {pref_info['positive']}")


def main():
    """Run the complete paper reproduction example."""
    print("🚀 STEERLAB PAPER REPRODUCTION EXAMPLE")
    print(
        "Based on: Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering"
    )
    print("=" * 80)

    # Show paper models and preferences
    demonstrate_paper_models()
    demonstrate_paper_preferences()

    # Create paper-based datasets
    created_files = create_paper_datasets()
    print(f"\n✅ Created {len(created_files)} dataset files")

    # Test with a smaller model for demonstration
    try:
        model_name = "google/gemma-2-2b-it"  # Smallest paper model
        preference = "cost"  # Well-understood preference

        print(f"\n🎯 FOCUSED TEST")
        print(f"Testing {model_name} with {preference} preference...")

        vector_path = test_paper_model_with_preference(model_name, preference)

        if vector_path:
            print(f"\n💡 You can now use these vectors:")
            print(f"   uv run steerlab test-vectors -v {vector_path}")
            print(f"   uv run steerlab serve  # then load vectors via API")

        # Show multi-preference capabilities
        demonstrate_multi_preference_steering()

        print(f"\n📚 NEXT STEPS:")
        print(f"   1. Try other paper models: uv run steerlab list-paper-models")
        print(f"   2. Explore preferences: uv run steerlab list-paper-preferences")
        print(
            f"   3. Create paper templates: uv run steerlab create-template <pref> --use-paper-templates"
        )
        print(f"   4. Test different strengths within functional ranges")
        print(f"   5. Implement the three interface modes (SELECT/CALIBRATE/LEARN)")

        print(f"\n🎉 Paper reproduction example completed!")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print(f"💡 This might be due to model loading issues or insufficient resources")
        print(f"💡 Try with CPU-only or ensure you have adequate GPU memory")


if __name__ == "__main__":
    main()
