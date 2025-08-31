# Pipeline Orchestration is Dead. Claude is All You Need.

**AIFlow** - A proof-of-concept framework where AI agents orchestrate machine learning pipelines through natural language understanding, eliminating the need for traditional pipeline orchestration tools.

## Overview

AIFlow demonstrates how AI agents can understand, compose, and execute ML pipelines through semantic component descriptions. The framework uses decorators to annotate pipeline components with rich metadata, enabling intelligent orchestration and execution.

## Project Structure

```
aiflow/
├── components/           # Pipeline components with @component decorators
│   ├── generate_training_samples.py  # data_generator component
│   ├── train_model.py                # model_trainer component  
│   ├── serve_model.py               # model_server & model_inference components
│   └── __init__.py                  # Package initialization
├── pipelines/           # Pipeline definitions in markdown
│   └── model_training_pipeline.md  # Example training pipeline
├── testdata/           # Isolated pipeline run outputs (excluded from git)
└── decorator.py        # Component decorator framework
```

## Key Features

### Component System
- **Semantic Decorators**: Components use `@component` decorators with detailed descriptions
- **Automatic Discovery**: Components are registered and trackable via ComponentRegistry  
- **Execution Tracking**: Built-in timing, logging, and metadata collection
- **Modular Design**: Each component is independently executable

### Available Components
- `data_generator` - Generate synthetic training data
- `model_trainer` - Train ML models from data
- `model_server` - Serve models for real-time inference  
- `model_inference` - Batch inference for testing

### Pipeline Orchestration
- **Markdown Definitions**: Pipelines defined in human-readable markdown format
- **Dependency Management**: Clear step dependencies (1 → 2 → 3)
- **Isolated Execution**: Each run uses separate directories with auto-incrementing indices
- **Component Chaining**: Flexible composition for different use cases

## Example Usage

### Running a Pipeline
Simply tell Claude to run a pipeline:

```
run a pipeline defined in pipelines/model_training_pipeline.md
```

That's it! Claude understands the pipeline definition and executes all steps automatically with proper dependency management and isolated run directories.

### Component Output
Each component provides execution metadata:
```json
{
  "status": "completed",
  "_component_metadata": {
    "name": "data_generator", 
    "execution_time": 0.0015,
    "status": "success"
  }
}
```

## Architecture Benefits

1. **Agent-Friendly**: Rich semantic descriptions enable AI agents to understand component capabilities
2. **Composable**: Components can be chained in different configurations per use case
3. **Traceable**: Full execution tracking and metadata for debugging and monitoring  
4. **Isolated**: Each pipeline run maintains separate artifacts preventing conflicts
5. **Extensible**: Easy to add new components following the decorator pattern

## Future Directions

- Dynamic pipeline generation from natural language descriptions
- Intelligent component selection and parameter optimization
- Parallel execution and dependency resolution
- Integration with existing ML orchestration platforms
- Advanced error handling and recovery mechanisms