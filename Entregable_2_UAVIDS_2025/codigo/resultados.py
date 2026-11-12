"""Lectura de las corridas guardadas por el ejecutor."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_E2 = Path(__file__).resolve().parents[1]
CORRIDAS = RAIZ_E2 / "resultados" / "corridas"


@lru_cache(maxsize=None)
def registros(fase):
    carpeta = CORRIDAS / fase
    if not carpeta.exists():
        return ()
    return tuple(json.loads(p.read_text(encoding="utf-8")) for p in sorted(carpeta.glob("*.json")))


def tabla_fase(fase):
    filas = []
    for r in registros(fase):
        f = {k: v for k, v in r.items() if k not in ("curvas", "hp", "arquitectura")}
        f.update({f"hp_{k}": v for k, v in r.get("hp", {}).items()})
        if r.get("arquitectura"):
            f["arquitectura"] = tuple(r["arquitectura"])
        filas.append(f)
    return pd.DataFrame(filas)


def curvas(fase, modelo=None):
    filas = []
    for r in registros(fase):
        if modelo is None or r["modelo"] == modelo:
            for c in r.get("curvas", []):
                filas.append({"modelo": r["modelo"], "semilla": r["semilla"], "id": r["id"], **c})
    return pd.DataFrame(filas)


def probabilidades(fase, id_corrida, conjunto="eva"):
    ruta = CORRIDAS / fase / f"{id_corrida}_{conjunto}.npy"
    return np.load(ruta) if ruta.exists() else None


def matriz(df, valor, fila="semilla", columna="modelo"):
    return df.pivot_table(index=fila, columns=columna, values=valor, aggfunc="first")
