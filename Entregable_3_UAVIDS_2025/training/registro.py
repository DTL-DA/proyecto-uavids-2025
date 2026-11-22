"""Registra el modelo original en el catálogo del motor del E2 sin modificar el código congelado del E2.

Así el modelo original pasa exactamente por el mismo pipeline que los rivales (sección 6.5 del enunciado):
mismos datos, transformación, pérdida ponderada, optimizador, planificador, parada temprana, recorte del
gradiente, conteo de MAC y registro de corridas.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

RAIZ_E3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_E3))
sys.path.insert(0, str(RAIZ_E3.parent / "Entregable_2_UAVIDS_2025"))

from codigo import modelos  # noqa: E402
from src.original_model import ModeloOriginal  # noqa: E402

NOMBRE = "Original"


def registrar(**configuracion):
    """Agrega (o reemplaza) la entrada 'Original' del catálogo con la configuración dada."""
    modelos.CATALOGO[NOMBRE] = {"clase": functools.partial(ModeloOriginal, **configuracion), "familia": "Mezcla con compuerta",
                                "grupo": "Original", "busqueda": False}
    return modelos.CATALOGO[NOMBRE]
