#!/usr/bin/env python3
"""
Serve ML model for inference
"""

import json
import pickle
import os
import time
from typing import Dict, Any, List
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import random
from . import component
from .train_model import PseudoModel


class ModelServer:
    """Simple model server for demonstration"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.request_count = 0
        
    def load_model(self) -> None:
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        self.is_loaded = True
        print("Model loaded successfully")
        
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make prediction"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
            
        self.request_count += 1
        
        # Simulate prediction
        time.sleep(0.01)  # Simulate inference time
        
        # Simple prediction based on feature sum
        feature_sum = sum(features)
        prediction = 1 if feature_sum > 0 else 0
        confidence = random.uniform(0.6, 0.9)
        
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "request_id": self.request_count,
            "timestamp": time.time()
        }
        
        return result


@component(
    name="model_server",
    description="""Deploy and serve a trained machine learning model for real-time inference requests. This component loads a pickled model from disk, starts a simulation of a model serving environment, and processes incoming prediction requests over a specified duration. It simulates realistic serving conditions with random request arrivals (30% chance per second), generates synthetic input features, and returns predictions with confidence scores. The server tracks request counts, response times, and maintains session statistics. Ideal for testing model serving infrastructure, evaluating inference performance, or simulating production-like serving scenarios before actual deployment.""",
    inputs={
        "model_path": "Path to the trained model",
        "port": "Port to serve on",
        "duration": "How long to serve (seconds)"
    },
    outputs={
        "model_path": "Path to served model",
        "port": "Port used for serving",
        "total_requests": "Total requests served",
        "serving_duration": "Duration of serving session"
    }
)
def serve_model(model_path: str = "trained_model.pkl", port: int = 8080, duration: int = 30) -> Dict[str, Any]:
    """
    Serve ML model for inference
    
    Args:
        model_path: Path to the trained model
        port: Port to serve on
        duration: How long to serve (seconds)
        
    Returns:
        Dict containing serving metadata
    """
    print(f"Starting model server on port {port}...")
    
    # Initialize server
    server = ModelServer(model_path)
    server.load_model()
    
    # Simulate serving requests
    print(f"Serving model for {duration} seconds...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Simulate incoming requests
        if random.random() < 0.3:  # 30% chance of request per second
            fake_features = [random.uniform(-1, 1) for _ in range(10)]
            result = server.predict(fake_features)
            print(f"Request {result['request_id']}: prediction={result['prediction']}, confidence={result['confidence']:.3f}")
        
        time.sleep(1)
    
    metadata = {
        "model_path": model_path,
        "port": port,
        "total_requests": server.request_count,
        "serving_duration": duration,
        "status": "completed"
    }
    
    print(f"Server stopped. Served {server.request_count} requests")
    return metadata


@component(
    name="model_inference",
    description="""Perform batch inference using a trained model for testing and evaluation purposes. This component loads a serialized model and processes multiple test samples in sequence, generating predictions with confidence scores for each input. It creates synthetic test data with the same feature structure as training data, runs inference for each sample, collects comprehensive results including prediction outcomes and confidence metrics, and calculates aggregate statistics. The component provides detailed logging for each prediction and returns both individual results and summary statistics. Perfect for model validation, performance testing, A/B testing scenarios, or generating inference benchmarks before production deployment.""",
    inputs={
        "model_path": "Path to the trained model",
        "test_samples": "Number of test samples to process"
    },
    outputs={
        "model_path": "Path to used model",
        "test_samples": "Number of test samples processed",
        "results": "List of prediction results",
        "avg_confidence": "Average confidence across all predictions"
    }
)
def serve_model_batch(model_path: str = "trained_model.pkl", test_samples: int = 10) -> Dict[str, Any]:
    """
    Serve model in batch mode for testing
    
    Args:
        model_path: Path to the trained model
        test_samples: Number of test samples to process
        
    Returns:
        Dict containing batch serving metadata
    """
    print(f"Starting batch inference with {test_samples} samples...")
    
    # Initialize server
    server = ModelServer(model_path)
    server.load_model()
    
    results = []
    for i in range(test_samples):
        # Generate random test sample
        features = [random.uniform(-1, 1) for _ in range(10)]
        result = server.predict(features)
        results.append(result)
        print(f"Sample {i+1}: prediction={result['prediction']}, confidence={result['confidence']:.3f}")
    
    metadata = {
        "model_path": model_path,
        "test_samples": test_samples,
        "results": results,
        "avg_confidence": sum(r['confidence'] for r in results) / len(results),
        "status": "completed"
    }
    
    print(f"Batch inference completed. Average confidence: {metadata['avg_confidence']:.3f}")
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Serve ML model")
    parser.add_argument("--model", type=str, default="trained_model.pkl", help="Path to trained model")
    parser.add_argument("--mode", type=str, choices=["server", "batch"], default="batch", help="Serving mode")
    parser.add_argument("--port", type=int, default=8080, help="Server port (server mode)")
    parser.add_argument("--duration", type=int, default=30, help="Serving duration in seconds (server mode)")
    parser.add_argument("--samples", type=int, default=10, help="Number of test samples (batch mode)")
    
    args = parser.parse_args()
    
    if args.mode == "server":
        result = serve_model(args.model, args.port, args.duration)
    else:
        result = serve_model_batch(args.model, args.samples)
    
    print(f"Serving completed: {result}")