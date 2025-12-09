import importlib

m = importlib.import_module("sklearn.preprocessing._data")
print("module loaded", m)
"""Removed: debugging helper script.

This file was used to check sklearn internals during development and
is no longer needed for the public repository.
"""

raise SystemExit("Removed during cleanup")
