"""Conjuntos adicionales (sección 8.3 y protocolo, sección 5.3): carga, preprocesamiento y partición agrupada.

    python training/conjuntos_adicionales.py preparar      # genera data/raw/<conjunto>/procesado.npz y data/<conjunto>_resumen.json

Reglas fijadas en el protocolo antes de ver resultados:
  1. Se excluyen identificadores, marcas de tiempo, columnas categóricas, etiquetas binarias y columnas constantes.
  2. Se eliminan las filas duplicadas exactas (atributos y etiqueta).
  3. Se eliminan las clases con menos de 300 filas tras la deduplicación.
  4. Transformación sign(x) * log(1 + |x|) y estandarización con el entrenamiento.
  5. Bloques de 250 filas contiguas dentro de cada clase (orden de la fuente), repartidos 70/15/15 dentro de cada clase (semilla 2027).
  6. Submuestreo dentro de cada partición con tope por clase por llenado: 28 000 / 6 000 / 6 000 filas.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CRUDOS = RAIZ / "data" / "raw"
SEMILLA = 2027
LARGO_BLOQUE = 250
FRACCIONES = (0.70, 0.15, 0.15)
TOPES = {"ent": 28_000, "val": 6_000, "eva": 6_000}
MINIMO_CLASE = 300
CONJUNTOS = ("uav_nidd", "unsw_nb15", "unr_idd")


def _uav_nidd():
    ruta = CRUDOS / "uav_nidd" / "descarga" / "GCS-Case 3.zip"
    with zipfile.ZipFile(ruta).open("GCS-Case 3/GSC Case3 Label .csv") as f:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
    y = df.pop("label").astype(str).str.strip()
    return df, y


def _unsw_nb15():
    d = CRUDOS / "unsw_nb15" / "descarga"
    nombres = pd.read_csv(d / "NUSW-NB15_features.csv", encoding="latin-1")["Name"].str.strip().str.lower().tolist()
    partes = [pd.read_csv(d / f"UNSW-NB15_{i}.csv", header=None, names=nombres, low_memory=False, encoding="latin-1") for i in range(1, 5)]
    df = pd.concat(partes, ignore_index=True)
    df = df.iloc[np.argsort(pd.to_numeric(df["stime"], errors="coerce").to_numpy(), kind="stable")].reset_index(drop=True)
    y = df.pop("attack_cat").fillna("Normal").astype(str).str.strip().replace({"": "Normal", "Backdoors": "Backdoor"})
    df = df.drop(columns=["srcip", "sport", "dstip", "dsport", "proto", "state", "service", "stime", "ltime", "label"])
    return df, y


def _unr_idd():
    df = pd.read_csv(CRUDOS / "unr_idd" / "descarga" / "UNR-IDD.csv")
    y = df.pop("Label").astype(str).str.strip()
    df = df.drop(columns=["Switch ID", "Port Number", "Connection Point", "Binary Label"])
    return df, y


CARGAR = {"uav_nidd": _uav_nidd, "unsw_nb15": _unsw_nb15, "unr_idd": _unr_idd}


def _llenado(conteos: np.ndarray, tope: int) -> np.ndarray:
    """Cuántas filas tomar de cada clase para no pasar el tope, repartiéndolo por llenado (water-filling)."""
    if conteos.sum() <= tope:
        return conteos.copy()
    bajo, alto = 0, int(conteos.max())
    while bajo < alto:
        medio = (bajo + alto + 1) // 2
        if np.minimum(conteos, medio).sum() <= tope:
            bajo = medio
        else:
            alto = medio - 1
    toma = np.minimum(conteos, bajo)
    resto = tope - toma.sum()
    for c in np.argsort(-conteos):  # el sobrante (menos que el número de clases) va a las clases más grandes
        if resto == 0:
            break
        if conteos[c] > toma[c]:
            toma[c] += 1
            resto -= 1
    return toma


def preparar(nombre: str) -> dict:
    df, y = CARGAR[nombre]()
    X = df.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any()]
    X = X.fillna(0.0)
    X = X.loc[:, X.nunique() > 1]                      # 1. columnas constantes
    filas_originales = len(X)
    duplicada = pd.concat([X, y.rename("__y__")], axis=1).duplicated().to_numpy()
    X, y = X[~duplicada].reset_index(drop=True), y[~duplicada].reset_index(drop=True)   # 2. duplicados exactos
    conteo = y.value_counts()
    clases = sorted(conteo[conteo >= MINIMO_CLASE].index)                                 # 3. clases raras
    quedan = y.isin(clases).to_numpy()
    X, y = X[quedan].reset_index(drop=True), y[quedan].reset_index(drop=True)
    yi = y.map({c: i for i, c in enumerate(clases)}).to_numpy().astype(np.int64)
    # 5. Bloques de filas contiguas DENTRO de cada clase (las fuentes vienen ordenadas por ejecución de ataque), repartidos
    #    70/15/15 dentro de cada clase: partición agrupada y estratificada, con todas las clases en las tres particiones.
    rng = np.random.default_rng(SEMILLA)
    bloques = np.empty(len(X), dtype=np.int64)
    asignacion = {"ent": [], "val": [], "eva": []}
    siguiente = 0
    for c in range(len(clases)):
        filas_c = np.flatnonzero(yi == c)
        n_bloques = max(3, int(np.ceil(len(filas_c) / LARGO_BLOQUE)))
        trozos = np.array_split(filas_c, n_bloques)
        ids_bloque = np.arange(siguiente, siguiente + n_bloques)
        for b, trozo in zip(ids_bloque, trozos):
            bloques[trozo] = b
        siguiente += n_bloques
        orden = rng.permutation(ids_bloque)
        n_val = max(1, int(round(FRACCIONES[1] * n_bloques)))
        n_eva = max(1, int(round(FRACCIONES[2] * n_bloques)))
        asignacion["val"] += list(orden[:n_val])
        asignacion["eva"] += list(orden[n_val:n_val + n_eva])
        asignacion["ent"] += list(orden[n_val + n_eva:])
    ids = {}
    for parte, blq in asignacion.items():                                                  # 6. submuestreo por llenado
        candidatas = np.flatnonzero(np.isin(bloques, blq))
        conteos = np.bincount(yi[candidatas], minlength=len(clases))
        toma = _llenado(conteos, TOPES[parte])
        elegidas = [rng.choice(candidatas[yi[candidatas] == c], toma[c], replace=False) for c in range(len(clases)) if toma[c] > 0]
        ids[parte] = np.sort(np.concatenate(elegidas))
    M = X.to_numpy(dtype=np.float64)
    M = np.sign(M) * np.log1p(np.abs(M))                                                   # 4. transformación
    media, desv = M[ids["ent"]].mean(0), M[ids["ent"]].std(0)
    desv[desv == 0] = 1.0
    M = ((M - media) / desv).astype(np.float32)
    frecuencia = np.bincount(yi[ids["ent"]], minlength=len(clases))
    pesos = (frecuencia.sum() / (len(clases) * np.maximum(frecuencia, 1))).astype(np.float32)
    destino = CRUDOS / nombre / "procesado.npz"
    np.savez_compressed(destino, X=M, y=yi, ent=ids["ent"], val=ids["val"], eva=ids["eva"], pesos=pesos, bloques=bloques)
    resumen = {"conjunto": nombre, "filas_fuente": int(filas_originales), "duplicadas_eliminadas": int(duplicada.sum()),
               "clases": clases, "atributos": list(X.columns), "n_atributos": int(M.shape[1]),
               "conteo_por_particion": {p: np.bincount(yi[v], minlength=len(clases)).tolist() for p, v in ids.items()},
               "bloques_compartidos": int(len(set(bloques[ids["ent"]]) & set(bloques[ids["eva"]])) + len(set(bloques[ids["ent"]]) & set(bloques[ids["val"]]))),
               "sha256_procesado": hashlib.sha256(destino.read_bytes()).hexdigest()}
    (RAIZ / "data" / f"{nombre}_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    return resumen


def cargar(nombre: str) -> dict:
    d = np.load(CRUDOS / nombre / "procesado.npz")
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    if sys.argv[1] == "preparar":
        for nombre in (sys.argv[2:] or CONJUNTOS):
            r = preparar(nombre)
            print(json.dumps({k: v for k, v in r.items() if k != "atributos"}, ensure_ascii=False))
