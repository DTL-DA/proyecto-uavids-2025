"""Arma el paquete de reproducción que se publica con la release entregable-3.

    python verificacion_externa/empaquetar.py      ->   ../paquete_reproduccion_e3.zip  (+ .sha256)

Incluye los archivos versionados del repositorio (los que no ignora Git) y, aparte, los archivos ignorados que hacen
falta para regenerar la tabla principal y compilar el libro sin volver a entrenar todo:
  - el CSV de UAVIDS-2025 (CC BY 4.0), verificado con su SHA-256;
  - las probabilidades de prueba (*_eva.npy) de las corridas definitivas del E2 y del E3;
  - los pesos del modelo original de las semillas 6001 a 6005 (capítulo 5, verificación numérica de la cota).
No incluye los conjuntos adicionales en bruto (data/raw), el libro compilado ni la salida de una reproducción anterior.
"""
from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
E3 = AQUI.parent
RAIZ = E3.parent
DESTINO = RAIZ / "paquete_reproduccion_e3.zip"
EXCLUIR = ("Entregable_3_UAVIDS_2025/verificacion_externa/salida/", "Entregable_3_UAVIDS_2025/book/_build/", "Entregable_3_UAVIDS_2025/data/raw/")
SHA_CSV = "d50d339f68be7b23f0bf089dd438b20a1835c13182d8641220538121440164d0"


def versionados():
    salida = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=RAIZ, capture_output=True, check=True).stdout
    return [p for p in salida.decode("utf-8").split("\0") if p and not p.startswith(EXCLUIR) and (RAIZ / p).is_file()]


def extras():
    csv = RAIZ / "Entregable_1_UAVIDS_2025" / "datos" / "UAVIDS-2025.csv"
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == SHA_CSV, "El CSV de UAVIDS-2025 no coincide con el del Entregable 1"
    rutas = [csv]
    rutas += sorted((RAIZ / "Entregable_2_UAVIDS_2025" / "resultados" / "corridas" / "definitiva").glob("definitiva_*_eva.npy"))
    rutas += sorted((E3 / "experiments" / "resultados" / "corridas" / "definitiva").glob("definitiva_*_eva.npy"))
    rutas += [E3 / "experiments" / "resultados" / "corridas" / "definitiva" / f"definitiva_Original_{s}.pt" for s in range(6001, 6006)]
    return [p.relative_to(RAIZ).as_posix() for p in rutas if p.is_file()]


def main():
    archivos = sorted(set(versionados()) | set(extras()))
    with zipfile.ZipFile(DESTINO, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in archivos:
            z.write(RAIZ / p, p)
    sha = hashlib.sha256(DESTINO.read_bytes()).hexdigest()
    DESTINO.with_suffix(".zip.sha256").write_bytes(f"{sha}  {DESTINO.name}\n".encode("utf-8"))  # LF: sha256sum -c lo lee en cualquier sistema
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    print(f"{len(archivos)} archivos | {DESTINO.stat().st_size / 1e6:.1f} MB | commit {commit} | SHA-256 {sha}")


if __name__ == "__main__":
    main()
