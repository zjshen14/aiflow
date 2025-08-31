# Model Training Pipeline

This pipeline demonstrates a basic ML training workflow with data generation, model training, and inference.

## Pipeline Steps

1. **Data Generation** (`data_generator`)
   - Generate 100 synthetic training samples
   - Output: `training_data.json`
   - Dependencies: None

2. **Model Training** (`model_trainer`) 
   - Train ML model using generated data
   - Output: `trained_model.pkl`
   - Dependencies: Step 1 (Data Generation)

3. **Model Inference** (`model_inference`)
   - Run batch inference for testing
   - Process 5 test samples
   - Dependencies: Step 2 (Model Training)

## Environment

Each pipeline run should be isolated in a separate directory:
- Base directory: `testdata/model_training_pipeline/`
- Run directories: `run_1`, `run_2`, `run_3`, etc.
- Each new execution bumps up the index to ensure isolation