"""Datos heredados del Entregable 1 y preparación por pliegue.

El Entregable 2 no cambia la partición ni la transformación: lee `particion.npz` y
`datos_preparados.npz` del Entregable 1 y verifica la suma SHA-256 del conjunto de prueba.
Para la validación cruzada y las curvas de aprendizaje, la estandarización y los pesos de clase
se vuelven a ajustar solo con la parte de entrenamiento de cada pliegue.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
E1 = RAIZ / "Entregable_1_UAVIDS_2025"
E2 = RAIZ / "Entregable_2_UAVIDS_2025"
RESULTADOS = E2 / "resultados"
CSV = E1 / "datos" / "UAVIDS-2025.csv"
SHA_CSV = "d50d339f68be7b23f0bf089dd438b20a1835c13182d8641220538121440164d0"
SHA_PRUEBA = "2a2dab99c08ed1f19f24add07f5c6b0375e5ce0df7951a99090f54f4f2cd32b4"
LARGO_BLOQUE = 1000
ATRIBUTOS = ["FlowDuration/s", "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes", "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s",
             "RxByteRate/s", "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps", "MeanPacketSize", "PacketDropRate", "AverageHopCount"]
IDENTIFICADORES = ["FlowID", "SrcAddr", "SrcPort", "DstAddr", "DstPort", "Protocol"]
CLASES = ["Blackhole", "Flooding", "Normal", "Sybil", "Wormhole"]
TOPE = np.where(np.array(ATRIBUTOS) == "PacketDropRate", 1.0, np.inf)


def transformar(matriz):
    """La transformación del Entregable 1: recorte de PacketDropRate en 1 y log(1 + x)."""
    return np.log1p(np.minimum(matriz, TOPE))


@lru_cache(maxsize=1)
def heredados():
    """Partición del Entregable 1, con la prueba verificada, y la matriz estandarizada con los estadísticos
    de su entrenamiento. Se regenera desde el CSV y coincide exactamente con `datos_preparados.npz` del Entregable 1."""
    part = np.load(E1 / "resultados" / "particion.npz")
    ids = {"entrenamiento": part["entrenamiento"], "validacion": part["validacion"], "prueba": part["prueba"]}
    sha = hashlib.sha256(np.sort(ids["prueba"]).tobytes()).hexdigest()
    assert sha == SHA_PRUEBA, "La suma SHA-256 del conjunto de prueba no coincide con la del Entregable 1"
    datos, _, x_log = crudo()
    X, pesos = preparar_pliegue(ids["entrenamiento"], x_log)
    return {"X": X, "y": etiquetas(), "pesos": pesos, "media": x_log[ids["entrenamiento"]].mean(0), "desviacion": x_log[ids["entrenamiento"]].std(0),
            "atributos": ATRIBUTOS, "clases": CLASES, "tx_packets": datos["TxPackets"].to_numpy(), "ids": ids}


@lru_cache(maxsize=1)
def etiquetas():
    datos = crudo()[0]
    nombre = datos["label"].str.replace(" Attack", "").str.replace(" Traffic", "")
    return nombre.map({c: i for i, c in enumerate(CLASES)}).to_numpy().astype(np.int64)


@lru_cache(maxsize=1)
def crudo():
    """CSV original, con la suma SHA-256 verificada, los bloques de 1000 flujos y la matriz transformada."""
    sha = hashlib.sha256(CSV.read_bytes()).hexdigest()
    assert sha == SHA_CSV, "El SHA-256 del CSV no coincide con la versión del Entregable 1"
    datos = pd.read_csv(CSV)
    bloques = ((datos["FlowID"].to_numpy() - 1) // LARGO_BLOQUE).astype(np.int64)
    x_log = transformar(datos[ATRIBUTOS].to_numpy(dtype=np.float64))
    return datos, bloques, x_log


def preparar_pliegue(ids_ent, x_log=None):
    """Estandariza con la media y la desviación de `ids_ent` y calcula los pesos de clase de esa parte."""
    if x_log is None:
        x_log = crudo()[2]
    y = etiquetas()
    media, desviacion = x_log[ids_ent].mean(0), x_log[ids_ent].std(0)
    X = ((x_log - media) / desviacion).astype(np.float32)
    frecuencia = np.bincount(y[ids_ent], minlength=5)
    pesos = (frecuencia.sum() / (5 * frecuencia)).astype(np.float32)
    return X, pesos


def cargar_particion(nombre):
    """Partición con nombre: 'principal' es la del Entregable 1; las demás están en resultados/particiones."""
    h = heredados()
    if nombre == "principal":
        return {"ent": h["ids"]["entrenamiento"], "val": h["ids"]["validacion"], "eva": h["ids"]["prueba"]}
    d = np.load(RESULTADOS / "particiones" / f"{nombre}.npz")
    return {k: d[k] for k in d.files}
