#!/usr/bin/env python3
"""
Stub decorators for backward compatibility.
These are minimal stubs to allow existing code to work while the deprecated
decorator system is being phased out.
"""

def sem_simple_command(backend=None, description=None, params=None, examples=None):
    """Stub decorator for simple commands - just returns the function unchanged."""
    def decorator(func):
        # Just return the function unchanged - no registration needed
        return func
    return decorator

def sem_command(cmd=None, params=None, examples=None, help_text=None):
    """Stub decorator for commands - just returns the function unchanged."""
    def decorator(func):
        return func
    return decorator

def sem_param(*args, **kwargs):
    """Stub decorator for parameters - just returns the function unchanged."""
    def decorator(func):
        return func
    return decorator
