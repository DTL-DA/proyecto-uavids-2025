"""Carga y caché de los datos precalculados que muestra el dashboard (se leen una sola vez al iniciar el servidor)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
TABLERO = RAIZ / "artifacts" / "dashboard"
VERSION = "1.0.0"
JBOOK = "https://dtl-da.github.io/proyecto-uavids-2025/entregable-3/"


@lru_cache(maxsize=1)
def uavids() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data" / "uavids.parquet")


@lru_cache(maxsize=1)
def lideres() -> pd.DataFrame:
    return pd.read_csv(TABLERO / "lideres.csv")


@lru_cache(maxsize=1)
def diagnosticos() -> dict:
    return json.loads((TABLERO / "diagnosticos.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def contrastes() -> pd.DataFrame | None:
    ruta = TABLERO / "contrastes.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@lru_cache(maxsize=1)
def resumen() -> dict:
    ruta = TABLERO / "resumen.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


@lru_cache(maxsize=1)
def robustez() -> pd.DataFrame | None:
    ruta = TABLERO / "robustez.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@lru_cache(maxsize=1)
def ficha_original() -> dict:
    ruta = TABLERO / "ficha_original.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


@lru_cache(maxsize=1)
def adicionales() -> pd.DataFrame | None:
    ruta = TABLERO / "adicionales.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@lru_cache(maxsize=1)
def adicionales_contrastes() -> pd.DataFrame | None:
    ruta = TABLERO / "adicionales_contrastes.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@lru_cache(maxsize=1)
def balance_r3() -> list:
    ruta = TABLERO / "balance_r3.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []
