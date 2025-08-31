# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIFlow is a proof-of-concept agentic ML pipeline orchestration framework. AI agents understand and execute ML pipelines through semantic component descriptions rather than traditional orchestration tools.

## Architecture

### Components Directory (`components/`)
Contains pipeline components decorated with `@component` that provide rich semantic descriptions:
- `generate_training_samples.py` - `data_generator` component
- `train_model.py` - `model_trainer` component  
- `serve_model.py` - `model_server` and `model_inference` components
- `components.py` - Component decorator and registry system

### Pipelines Directory (`pipelines/`)
Pipeline definitions in markdown format describing:
- Pipeline steps and dependencies
- Component configurations
- Environment isolation requirements

### Key Files
- `components.py` - Main decorator framework for component annotation
- `README.md` - Project documentation
- `testdata/` - Isolated pipeline execution artifacts

## Common Development Tasks

### Running Components
Components are executed as Python modules:
```bash
python3 -m components.generate_training_samples --num-samples 100 --output path/to/output.json
python3 -m components.train_model --data path/to/data.json --output path/to/model.pkl
python3 -m components.serve_model --model path/to/model.pkl --mode batch --samples 5
```

### Adding New Components
1. Create new Python file in `components/` directory
2. Import: `from components.components import component`
3. Decorate function with `@component()` including detailed description
4. Ensure component can be executed as standalone module

### Pipeline Execution
- Each pipeline run creates isolated directory: `testdata/{pipeline_name}/run_{N}/`
- Run indices auto-increment to prevent conflicts
- All artifacts (data, models, logs) stored in run-specific directories

### Testing
Components include pseudo-implementations suitable for testing pipeline orchestration without requiring real ML infrastructure.

## Component System

Components use semantic decorators with:
- `name`: Unique component identifier
- `description`: Rich semantic description for AI agent understanding
- `inputs`: Parameter descriptions
- `outputs`: Return value descriptions

The decorator provides automatic execution tracking, timing, and metadata collection.

## Development Notes

- Components are chainable and composable based on use case
- Rich descriptions enable AI agents to understand component capabilities
- Each component is independently executable for testing
- Import paths use `components.components` for the decorator framework