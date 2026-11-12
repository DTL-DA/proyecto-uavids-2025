"""Genera el cuaderno del Entregable 2 y conserva las salidas de las celdas cuyo código no cambió.

    python Entregable_2_UAVIDS_2025/herramientas/crear_cuaderno.py "Entregable 2 - UAVIDS-2025.ipynb" [secciones]

Las secciones están en herramientas/cuaderno/ y se agregan en orden; con una lista de secciones
se genera solo esa parte, para probarla.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuaderno import base  # noqa: E402

SECCIONES = ["s0_portada", "s1_problema", "s2_eda", "s3_literatura", "s3b_fichas", "s4_implementacion", "s5_paper_guia", "s6_benchmark", "s7_ruta",
             "s8_reproducibilidad", "s9_conclusiones"]

if __name__ == "__main__":
    elegidas = sys.argv[2].split(",") if len(sys.argv) > 2 else SECCIONES
    for nombre in elegidas:
        if (Path(__file__).resolve().parent / "cuaderno" / f"{nombre}.py").exists():
            importlib.import_module(f"cuaderno.{nombre}")
    base.escribir(sys.argv[1])
