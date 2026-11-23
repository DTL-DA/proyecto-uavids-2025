"""Hace importables `src` y `app` desde la raíz del Entregable 3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
