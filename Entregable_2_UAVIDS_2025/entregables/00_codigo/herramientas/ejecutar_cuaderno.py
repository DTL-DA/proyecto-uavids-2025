"""Ejecuta el cuaderno de principio a fin con nbclient, desde la raíz del repositorio, y lo guarda tras cada celda.

    python Entregable_2_UAVIDS_2025/herramientas/ejecutar_cuaderno.py "Entregable 2 - UAVIDS-2025.ipynb"
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ruta = Path(sys.argv[1]).resolve()
nb = nbformat.read(ruta, as_version=4)
cliente = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(ruta.parent)}}, allow_errors=False)
inicio = time.time()
with cliente.setup_kernel():
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        t = time.time()
        cliente.execute_cell(celda, i)
        nbformat.write(nb, ruta)
        print(f"{i:>3} {celda.get('id')} {time.time() - t:.1f} s", flush=True)
print(f"Listo en {(time.time() - inicio) / 60:.1f} min")
