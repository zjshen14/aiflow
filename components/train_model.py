#!/usr/bin/env python3
"""
Train ML model from training samples
"""

import json
import pickle
import os
import time
from typing import Dict, Any, List
import random
from components.components import component


class PseudoModel:
    """Simple pseudo ML model for demonstration"""
    
    def __init__(self):
        self.weights = None
        self.is_trained = False
        self.accuracy = 0.0
        
    def fit(self, features: List[List[float]], labels: List[int]) -> None:
        """Train the model"""
        print("Training model...")
        
        # Simulate training process
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(len(features[0]))]
        
        # Simulate training iterations
        for epoch in range(5):
            time.sleep(0.1)  # Simulate training time
            print(f"Epoch {epoch + 1}/5")
            
        # Simulate final accuracy
        self.accuracy = random.uniform(0.7, 0.95)
        self.is_trained = True
        print(f"Training completed. Accuracy: {self.accuracy:.3f}")
        
    def save(self, path: str) -> None:
        """Save the trained model"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)


@component(
    name="model_trainer",
    description="""Train a machine learning model using provided training data. This component loads training samples from a JSON file, extracts features and labels, and trains a pseudo ML model using a simulated training process. The training includes multiple epochs with progress tracking and generates a final accuracy score. The trained model is serialized using pickle and saved to disk for later use. The component handles feature extraction from JSON format, validates data existence, performs training with realistic timing simulation, and outputs comprehensive training metadata including sample count, accuracy metrics, and model persistence information. Suitable for binary classification tasks with numerical features.""",
    inputs={
        "data_path": "Path to training data JSON file",
        "model_path": "Path to save the trained model"
    },
    outputs={
        "num_samples": "Number of training samples used",
        "model_path": "Path where model was saved",
        "accuracy": "Training accuracy achieved",
        "feature_dim": "Dimension of input features"
    }
)
def train_model(data_path: str = "training_data.json", model_path: str = "trained_model.pkl") -> Dict[str, Any]:
    """
    Train ML model from training samples
    
    Args:
        data_path: Path to training data JSON file
        model_path: Path to save the trained model
        
    Returns:
        Dict containing training metadata
    """
    print(f"Loading training data from {data_path}...")
    
    # Load training data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")
        
    with open(data_path, 'r') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} training samples")
    
    # Extract features and labels
    features = [sample["features"] for sample in samples]
    labels = [sample["label"] for sample in samples]
    
    # Initialize and train model
    model = PseudoModel()
    model.fit(features, labels)
    
    # Save trained model
    model.save(model_path)
    
    metadata = {
        "num_samples": len(samples),
        "model_path": model_path,
        "accuracy": model.accuracy,
        "feature_dim": len(features[0]),
        "training_completed": True,
        "status": "completed"
    }
    
    print(f"Model saved to {model_path}")
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--data", type=str, default="training_data.json", help="Path to training data")
    parser.add_argument("--output", type=str, default="trained_model.pkl", help="Output model path")
    
    args = parser.parse_args()
    
    result = train_model(args.data, args.output)
    print(f"Training completed: {result}")