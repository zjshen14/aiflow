#!/usr/bin/env python3
"""
Generate training samples for ML pipeline
"""

import random
import json
import os
from typing import List, Dict, Any
import time
from . import component


@component(
    name="data_generator",
    description="""Generate synthetic training data for machine learning models. This component creates pseudo-random feature vectors with corresponding binary labels. Each sample contains 10-dimensional feature vectors with values uniformly distributed between -1 and 1, and binary classification labels (0 or 1). The generated data is saved as JSON format with metadata including sample ID, features array, label, and timestamp. This is ideal for prototyping ML pipelines, testing data processing workflows, or when real training data is not available. The output includes comprehensive metadata about the generated dataset including sample count, feature dimensions, and file location.""",
    inputs={
        "num_samples": "Number of samples to generate",
        "output_path": "Path to save the generated samples"
    },
    outputs={
        "num_samples": "Number of generated samples",
        "output_path": "Path where samples were saved",
        "feature_dim": "Dimension of features",
        "num_classes": "Number of target classes"
    }
)
def generate_training_samples(num_samples: int = 1000, output_path: str = "training_data.json") -> Dict[str, Any]:
    """
    Generate pseudo training samples for ML model
    
    Args:
        num_samples: Number of samples to generate
        output_path: Path to save the generated samples
        
    Returns:
        Dict containing metadata about generated samples
    """
    print(f"Generating {num_samples} training samples...")
    
    # Simulate data generation process
    samples = []
    for i in range(num_samples):
        sample = {
            "id": i,
            "features": [random.uniform(-1, 1) for _ in range(10)],
            "label": random.choice([0, 1]),
            "timestamp": time.time()
        }
        samples.append(sample)
        
        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} samples")
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    metadata = {
        "num_samples": len(samples),
        "output_path": output_path,
        "feature_dim": 10,
        "num_classes": 2,
        "status": "completed"
    }
    
    print(f"Training samples saved to {output_path}")
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training samples")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="training_data.json", help="Output file path")
    
    args = parser.parse_args()
    
    result = generate_training_samples(args.num_samples, args.output)
    print(f"Generation completed: {result}")