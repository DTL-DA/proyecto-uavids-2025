"""Descarga UAVIDS-2025 desde Kaggle y verifica su suma SHA-256, con los mismos pasos que el cuaderno.

El conjunto de datos tiene licencia CC BY 4.0 y también está en Zenodo: https://doi.org/10.5281/zenodo.15336998

Uso:
    python obtener_datos.py
"""

import hashlib
import subprocess
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/qinglizeng1997/uavids-2025"
SHA_ESPERADO = "d50d339f68be7b23f0bf089dd438b20a1835c13182d8641220538121440164d0"

PROYECTO = next(p for p in Path(__file__).resolve().parents if p.name == "Entregable_1_UAVIDS_2025")
CARPETA = PROYECTO / "datos"
CSV = CARPETA / "UAVIDS-2025.csv"

CARPETA.mkdir(parents=True, exist_ok=True)
if not CSV.exists():
    comprimido = CARPETA / "archive.zip"
    subprocess.run(["curl", "-L", "-sS", "-o", str(comprimido), URL], check=True)
    with zipfile.ZipFile(comprimido) as z:
        z.extract(CSV.name, CARPETA)
    comprimido.unlink()

sha = hashlib.sha256(CSV.read_bytes()).hexdigest()
assert sha == SHA_ESPERADO, "El SHA-256 no coincide: la copia está dañada o cambió de versión"
print(f"Datos verificados en {CSV.relative_to(PROYECTO.parent)}\nSHA-256: {sha}")
