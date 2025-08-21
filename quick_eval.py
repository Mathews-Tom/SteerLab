#!/usr/bin/env python3
"""Quick evaluation demo with fewer prompts for faster testing."""

import asyncio
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

from steerlab.evaluation.compare import SteeringComparator

# Setup rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)


async def quick_evaluation():
    """Quick evaluation with minimal prompts."""
    console.print(Panel(
        "⚡ Quick Steering Evaluation\n\n"
        "Testing with 2 prompts and 3 steering strengths for faster results.",
        title="🚀 Quick Demo",
        border_style="blue"
    ))
    
    # Configuration - minimal for speed
    model_name = "google/gemma-2-2b-it"
    vector_path = Path("vectors/cost_vectors.safetensors")
    data_dir = Path("data")
    
    # Just 2 test prompts
    test_prompts = [
        "Help me choose a restaurant for dinner",
        "Suggest a vacation destination"
    ]
    
    # Just 3 steering strengths  
    steering_strengths = [-0.5, 0.0, 0.5]
    
    console.print(f"📦 Model: {model_name}")
    console.print(f"📝 Test prompts: {len(test_prompts)}")
    console.print(f"⚖️  Steering strengths: {steering_strengths}")
    console.print()
    
    # Initialize and run
    comparator = SteeringComparator(model_name)
    
    comparison_result = await comparator.compare_steering_effectiveness(
        vector_path=vector_path,
        test_prompts=test_prompts,
        steering_strengths=steering_strengths,
        data_dir=data_dir,
        max_length=80  # Shorter for speed
    )
    
    # Display results
    console.print("\\n" + "="*60)
    console.print("📊 QUICK EVALUATION RESULTS")
    console.print("="*60)
    
    comparator.display_results(comparison_result)
    
    # Show key finding
    summary = comparison_result.improvement_summary
    console.print("\\n" + Panel(
        f"🎯 **QUICK RESULTS**\\n\\n"
        f"• **Best Improvement:** {summary.get('max_alignment_improvement', 0):+.3f}\\n"
        f"• **Optimal Strength:** {summary.get('best_steering_strength', 0):+.1f}\\n"
        f"• **Relative Improvement:** {summary.get('relative_improvement', 0):+.1f}%",
        title="⚡ Quick Summary"
    ))


if __name__ == "__main__":
    asyncio.run(quick_evaluation())