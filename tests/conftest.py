# tests/conftest.py
import sys
from pathlib import Path

# Insert project root (one level above tests/) to sys.path so "import src" works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
