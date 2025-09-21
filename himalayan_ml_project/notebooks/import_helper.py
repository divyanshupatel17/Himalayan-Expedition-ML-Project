"""
Import helper for notebooks to ensure consistent imports across different execution contexts
"""
import sys
import os

def setup_imports():
    """Setup import paths for notebooks"""
    # Get current working directory
    cwd = os.getcwd()
    
    # Possible paths to src directory
    possible_paths = [
        os.path.join('..', '..', 'src'),  # From notebooks/models/
        os.path.join('..', 'src'),       # From notebooks/
        os.path.join('.', 'src'),        # From project root
        os.path.join(cwd, 'myProject', 'himalayan_ml_project', 'src'),  # Absolute from workspace
        os.path.join(cwd, 'himalayan_ml_project', 'src'),  # Relative from workspace
    ]
    
    # Add all possible paths
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        if normalized_path not in sys.path:
            sys.path.append(normalized_path)
    
    print(f"Added import paths. Current working directory: {cwd}")
    print("Available paths:")
    for path in sys.path[-len(possible_paths):]:
        print(f"  - {path}")

if __name__ == "__main__":
    setup_imports()