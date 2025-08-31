#!/usr/bin/env python3
"""
Component decorator for agentic pipeline orchestration
"""

from functools import wraps
from typing import Dict, Any, Callable, Optional
import time
import json


class ComponentRegistry:
    """Registry to keep track of all pipeline components"""
    
    _components: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, func: Callable, metadata: Dict[str, Any]) -> None:
        cls._components[name] = {
            'function': func,
            'metadata': metadata,
            'registered_at': time.time()
        }
    
    @classmethod
    def get_component(cls, name: str) -> Optional[Dict[str, Any]]:
        return cls._components.get(name)
    
    @classmethod
    def list_components(cls) -> Dict[str, Dict[str, Any]]:
        return cls._components.copy()


def component(
    name: str,
    description: str = "",
    inputs: Optional[Dict[str, str]] = None,
    outputs: Optional[Dict[str, str]] = None,
    dependencies: Optional[list] = None
):
    """
    Decorator to mark functions as pipeline components
    
    Args:
        name: Component name for identification
        description: Description of what the component does
        inputs: Dictionary describing input parameters
        outputs: Dictionary describing output format
        dependencies: List of component dependencies
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[COMPONENT] Starting {name}...")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Add execution metadata to result if it's a dict
                if isinstance(result, dict):
                    result['_component_metadata'] = {
                        'name': name,
                        'execution_time': execution_time,
                        'status': 'success'
                    }
                
                print(f"[COMPONENT] Completed {name} in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"[COMPONENT] Failed {name} after {execution_time:.2f}s: {str(e)}")
                raise
        
        # Register component metadata
        metadata = {
            'name': name,
            'description': description,
            'inputs': inputs or {},
            'outputs': outputs or {},
            'dependencies': dependencies or [],
            'original_function': func.__name__
        }
        
        ComponentRegistry.register(name, wrapper, metadata)
        
        return wrapper
    return decorator