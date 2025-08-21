#!/usr/bin/env python3
"""Debug script to isolate evaluation issues."""

import asyncio
import logging
from pathlib import Path
from steerlab.core.model import SteerableModel
from steerlab.core.vectors import SteeringVectorManager

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_basic_generation():
    """Test basic model generation."""
    logger.info("Starting debug session...")
    
    model_name = "google/gemma-2-2b-it"
    vector_path = Path("vectors/cost_vectors.safetensors")
    
    # Test 1: Basic model loading
    logger.info("Test 1: Loading model...")
    try:
        model = SteerableModel(model_name)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return
    
    # Test 2: Basic generation
    logger.info("Test 2: Basic generation...")
    try:
        test_prompt = "Help me choose a restaurant"
        logger.info(f"Generating for prompt: {test_prompt}")
        
        # Set a reasonable max_length and add timeout handling
        output = model.generate(test_prompt, max_length=50)
        logger.info(f"✅ Generated output: {output}")
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return
    
    # Test 3: Vector loading
    logger.info("Test 3: Loading vectors...")
    try:
        vector_manager = SteeringVectorManager(model_name)
        vectors, metadata = vector_manager.load_vectors(vector_path)
        logger.info(f"✅ Loaded {len(vectors)} vectors, metadata: {metadata}")
        
        model.load_steering_vectors(vectors)
        logger.info("✅ Vectors loaded into model")
    except Exception as e:
        logger.error(f"❌ Vector loading failed: {e}")
        return
    
    # Test 4: Steered generation
    logger.info("Test 4: Steered generation...")
    try:
        preference = metadata.get('preference_name', 'cost')
        model.set_steering({preference: 0.5})
        
        steered_output = model.generate(test_prompt, max_length=50)
        logger.info(f"✅ Steered output: {steered_output}")
        
        model.clear_steering()
        logger.info("✅ Steering cleared")
    except Exception as e:
        logger.error(f"❌ Steered generation failed: {e}")
        return
    
    logger.info("🎉 All tests passed!")

if __name__ == "__main__":
    asyncio.run(debug_basic_generation())