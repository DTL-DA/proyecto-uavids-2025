"""Inferencia en CPU con ONNX Runtime: carga bajo demanda, preprocesamiento, calibración, conformal y fuera de dominio.

El dashboard nunca entrena ni importa PyTorch: lee los artefactos de ``artifacts/<modelo>/`` (``model.onnx`` y
``metadata.json``) que genera ``training/exportar_onnx.py`` y ``artifacts/dashboard/calibracion.json`` que genera
``training/preparar_dashboard.py``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
ARTEFACTOS = RAIZ / "artifacts"
CLASES = ["Blackhole", "Flooding", "Normal", "Sybil", "Wormhole"]
ATRIBUTOS = ["FlowDuration/s", "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes", "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s",
             "RxByteRate/s", "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps", "MeanPacketSize", "PacketDropRate", "AverageHopCount"]


def carpeta(modelo: str) -> Path:
    return ARTEFACTOS / modelo.replace(" ", "_").replace("í", "i")


def modelos_disponibles() -> list[str]:
    nombres = []
    for p in sorted(ARTEFACTOS.glob("*/metadata.json")):
        nombres.append(json.loads(p.read_text(encoding="utf-8"))["modelo"])
    orden = {"Original": 0, "LightGBM": 1}
    return sorted(nombres, key=lambda m: (orden.get(m, 2), m))


@lru_cache(maxsize=None)
def metadatos(modelo: str) -> dict:
    return json.loads((carpeta(modelo) / "metadata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=6)  # carga bajo demanda; como mucho 6 sesiones vivas para cuidar la memoria del plan gratuito
def sesion(modelo: str):
    import onnxruntime as ort

    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = 1
    opciones.inter_op_num_threads = 1
    opciones.log_severity_level = 3  # silencia el aviso de forma de la salida 'label' de LightGBM
    return ort.InferenceSession(str(carpeta(modelo) / "model.onnx"), sess_options=opciones, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def calibracion() -> dict:
    ruta = ARTEFACTOS / "dashboard" / "calibracion.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def preprocesar(crudo: np.ndarray, modelo: str) -> np.ndarray:
    """Transformación del E1: log(1 + min(x, tope)) con tope 1 en PacketDropRate, y estandarización del entrenamiento."""
    m = metadatos(modelo)["preprocesamiento"]
    x = np.asarray(crudo, dtype=np.float64).reshape(-1, len(ATRIBUTOS))
    tope = np.where(np.array(ATRIBUTOS) == "PacketDropRate", 1.0, np.inf)
    z = (np.log1p(np.minimum(np.maximum(x, 0.0), tope)) - np.array(m["media"])) / np.array(m["desviacion"])
    return z.astype(np.float32)


def probabilidades(crudo: np.ndarray, modelo: str, calibrar: bool = True) -> np.ndarray:
    """Probabilidades por clase (n x 5). Con ``calibrar``, aplica la temperatura ajustada en validación."""
    x = preprocesar(crudo, modelo)
    salidas = sesion(modelo).run(None, {"x": x})
    if metadatos(modelo)["salida"]["nombre"] == "logits":
        logits = np.asarray(salidas[0], dtype=np.float64)
    else:
        logits = np.log(np.clip(np.asarray(salidas[1] if len(salidas) > 1 else salidas[0], dtype=np.float64), 1e-12, 1))
    T = calibracion().get(modelo, {}).get("temperatura", 1.0) if calibrar else 1.0
    z = logits / T
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def conjunto_conforme(p: np.ndarray, modelo: str) -> list[list[str]]:
    """Conjunto de predicción conforme (LAC, cobertura nominal del 90 %) con el cuantil ajustado en validación."""
    q = calibracion().get(modelo, {}).get("cuantil_lac_90")
    if q is None:
        return [[CLASES[int(i)]] for i in p.argmax(1)]
    return [[CLASES[j] for j in np.flatnonzero(1 - fila <= q)] or [CLASES[int(fila.argmax())]] for fila in p]


def fuera_de_dominio(crudo: np.ndarray, modelo: str = "Original") -> dict:
    """Aviso fuera de dominio: atributos fuera del rango del entrenamiento y distancia de Mahalanobis frente al percentil 99."""
    meta = metadatos(modelo)
    x = np.asarray(crudo, dtype=np.float64).reshape(-1, len(ATRIBUTOS))[0]
    fuera = [e["nombre"] for e, v in zip(meta["entrada"], x) if v < e["minimo_entrenamiento"] or v > e["maximo_entrenamiento"]]
    cal = calibracion().get("_dominio", {})
    distancia, umbral = None, cal.get("umbral_p99")
    if "precision" in cal:
        z = preprocesar(x, modelo)[0].astype(np.float64)
        distancia = float(np.sqrt(z @ np.array(cal["precision"]) @ z))
    return {"atributos_fuera_de_rango": fuera, "distancia": distancia, "umbral": umbral,
            "fuera": bool(fuera) or (distancia is not None and umbral is not None and distancia > umbral)}


def explicacion_local(crudo: np.ndarray, modelo: str, referencia: np.ndarray) -> list[dict]:
    """Explicación local por oclusión: cuánto cae la probabilidad de la clase predicha al reemplazar cada atributo por su
    mediana del entrenamiento (positivo = el atributo apoya la predicción). Es agnóstica al modelo y cuesta 16 inferencias."""
    x = np.asarray(crudo, dtype=np.float64).reshape(1, -1)
    p0 = probabilidades(x, modelo)[0]
    c = int(p0.argmax())
    lote = np.repeat(x, len(ATRIBUTOS), 0)
    lote[np.arange(len(ATRIBUTOS)), np.arange(len(ATRIBUTOS))] = referencia
    p = probabilidades(lote, modelo)[:, c]
    return sorted([{"atributo": a, "aporte": float(p0[c] - p[j])} for j, a in enumerate(ATRIBUTOS)], key=lambda d: -abs(d["aporte"]))
