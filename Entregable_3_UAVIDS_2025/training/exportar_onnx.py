"""Bloque B: selección de la mejor configuración, exportación a ONNX, metadatos y prueba de ida y vuelta.

    python training/exportar_onnx.py [modelos...]

Política idéntica para todos los modelos (sección 5 del enunciado):
- Configuración: la elegida en validación (ajuste del E2 o del E3; arquitectura ganadora de la fase 2 cuando aplica).
- Reentrenamiento final: cada corrida definitiva ya está entrenada sobre el conjunto de entrenamiento completo, con el
  número de épocas fijado en validación por la parada temprana. Se exporta la corrida cuya F1 de validación es la más
  cercana a la mediana de las semillas del modelo: es representativa y se elige sin mirar la prueba.
- Exportación: ONNX (opset 17) para las redes con torch.onnx, LightGBM con onnxmltools y la logística con skl2onnx.
- Ida y vuelta sobre TODA la prueba congelada (22 883 flujos): error absoluto máximo en probabilidades (< 1e-4),
  coincidencia de clases y diferencia de F1 macro entre ONNX y el modelo en memoria.
- metadata.json: versiones, semilla, hiperparámetros, preprocesamiento (transformación y estadísticos), esquema de entrada
  con rangos válidos del entrenamiento, clases y métricas de validación y prueba.
"""
from __future__ import annotations

import json
import pickle
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
RAIZ_E3 = AQUI.parent
sys.path.insert(0, str(AQUI))
import registro  # noqa: E402,F401  (agrega el modelo original al catálogo)
from ejecutar_original import CONFIGURACION  # noqa: E402
from codigo import datos as D  # noqa: E402
from codigo import modelos  # noqa: E402

E2_DEF = RAIZ_E3.parent / "Entregable_2_UAVIDS_2025" / "resultados" / "corridas" / "definitiva"
E3_DEF = RAIZ_E3 / "experiments" / "resultados" / "corridas" / "definitiva"
ARTEFACTOS = RAIZ_E3 / "artifacts"
PROFUNDOS = ["DNN", "CNN 1D", "LSTM", "CNN-LSTM", "Autoencoder", "RNN", "Transformer", "BiLSTM", "ResNet", "GRU", "KAN", "TabNet", "DBN"]
TODOS = ["Original"] + PROFUNDOS + ["LightGBM", "Logística"]


def archivo(m):
    return m.replace(" ", "_").replace("í", "i")


def elegir_corrida(modelo):
    carpeta = E3_DEF if modelo == "Original" else E2_DEF
    regs = [json.loads(p.read_text(encoding="utf-8")) for p in carpeta.glob(f"definitiva_{archivo(modelo)}_*.json")]
    regs = [r for r in regs if r["semilla"] <= 6020]  # mismas 20 semillas para todos
    mediana = np.median([r["f1_val"] for r in regs])
    r = min(regs, key=lambda r: (abs(r["f1_val"] - mediana), r["semilla"]))
    return r, carpeta / r["id"]


def exportar(modelo):
    import onnxruntime as ort
    import torch

    r, base = elegir_corrida(modelo)
    destino = ARTEFACTOS / archivo(modelo)
    destino.mkdir(parents=True, exist_ok=True)
    h = D.heredados()
    X = h["X"]
    ids_eva = h["ids"]["prueba"]
    y_eva = h["y"][ids_eva]
    x_eva = X[ids_eva].astype(np.float32)
    ruta = destino / "model.onnx"
    if modelo in ("LightGBM", "Logística"):
        est = pickle.loads(Path(str(base) + ".pkl").read_bytes())
        p_mem = est.predict_proba(x_eva, **({"num_iteration": r["arboles"]} if modelo == "LightGBM" else {}))
        if modelo == "LightGBM":
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType

            booster = est.booster_
            mo = onnxmltools.convert_lightgbm(est, initial_types=[("x", FloatTensorType([None, X.shape[1]]))], zipmap=False, target_opset=15)
            if r["arboles"] < booster.num_trees() // 5:  # la exportación usa todos los árboles: se recorta al mejor número de iteraciones
                est2 = pickle.loads(Path(str(base) + ".pkl").read_bytes())
                est2._Booster = booster.__class__(model_str=booster.model_to_string(num_iteration=r["arboles"]))
                mo = onnxmltools.convert_lightgbm(est2, initial_types=[("x", FloatTensorType([None, X.shape[1]]))], zipmap=False, target_opset=15)
        else:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            mo = convert_sklearn(est, initial_types=[("x", FloatTensorType([None, X.shape[1]]))], options={id(est): {"zipmap": False}}, target_opset=17)
        ruta.write_bytes(mo.SerializeToString())
        sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        salida = sesion.run(None, {"x": x_eva})
        p_onnx = np.asarray(salida[1] if len(salida) > 1 else salida[0])
        parametros = r.get("parametros")
    else:
        if modelo == "Original":
            registro.registrar(**CONFIGURACION)
        m = modelos.construir(modelo, r.get("arquitectura"))
        m.load_state_dict(torch.load(str(base) + ".pt", map_location="cpu"))
        m.eval()
        with torch.no_grad():
            p_mem = torch.softmax(torch.cat([m(torch.from_numpy(x_eva[i:i + 4096])) for i in range(0, len(x_eva), 4096)]), 1).numpy()
        torch.onnx.export(m, (torch.from_numpy(x_eva[:2]),), str(ruta), input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "lote"}, "logits": {0: "lote"}}, opset_version=17, dynamo=False)
        sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        logits = sesion.run(None, {"x": x_eva})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        p_onnx = e / e.sum(1, keepdims=True)
        parametros = modelos.parametros(m)
    from sklearn.metrics import f1_score

    f1_mem = f1_score(y_eva, p_mem.argmax(1), average="macro")
    f1_onnx = f1_score(y_eva, p_onnx.argmax(1), average="macro")
    crudo = D.crudo()[0][D.ATRIBUTOS].to_numpy()[h["ids"]["entrenamiento"]]
    meta = {"modelo": modelo, "semilla": r["semilla"], "id_corrida": r["id"], "criterio_de_seleccion": "F1 de validación más cercana a la mediana de 20 semillas",
            "hiperparametros": r.get("hp"), "arquitectura": r.get("arquitectura"), "configuracion_original": CONFIGURACION if modelo == "Original" else None,
            "epocas": r.get("epocas"), "parametros": parametros, "mac": r.get("mac"),
            "preprocesamiento": {"transformacion": "log(1 + min(x, tope)), tope = 1 para PacketDropRate e infinito para el resto",
                                 "media": h["media"].tolist(), "desviacion": h["desviacion"].tolist()},
            "entrada": [{"nombre": a, "tipo": "float", "minimo_entrenamiento": float(crudo[:, j].min()), "maximo_entrenamiento": float(crudo[:, j].max())}
                        for j, a in enumerate(D.ATRIBUTOS)],
            "salida": {"nombre": "logits" if modelo not in ("LightGBM", "Logística") else "probabilities", "clases": D.CLASES},
            "metricas": {"f1_val": r.get("f1_val"), "f1_prueba": r.get("f1_eva"), "f1_prueba_onnx": float(f1_onnx)},
            "temperatura": None, "cuantil_conforme_90": None,
            "versiones": {"python": platform.python_version(), "torch": __import__("torch").__version__, "onnxruntime": ort.__version__,
                          "onnx": __import__("onnx").__version__}}
    (destino / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"modelo": modelo, "semilla": r["semilla"], "error_max_prob": float(np.abs(p_onnx - p_mem).max()),
            "misma_clase_%": float(100 * (p_onnx.argmax(1) == p_mem.argmax(1)).mean()), "f1_memoria": float(f1_mem), "f1_onnx": float(f1_onnx),
            "dif_f1": float(f1_onnx - f1_mem), "kb": round(ruta.stat().st_size / 1024, 1), "cumple_1e-4": bool(np.abs(p_onnx - p_mem).max() < 1e-4)}


if __name__ == "__main__":
    import pandas as pd

    filas = []
    for m in (sys.argv[1:] or TODOS):
        try:
            filas.append(exportar(m))
        except Exception as e:  # se reporta y se sigue con los demás
            filas.append({"modelo": m, "error": repr(e)[:300]})
        print(filas[-1], flush=True)
    ruta = ARTEFACTOS / "reporte_ida_y_vuelta.csv"
    previo = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()
    t = pd.concat([previo[~previo["modelo"].isin([f["modelo"] for f in filas])] if len(previo) else previo, pd.DataFrame(filas)])
    t.to_csv(ruta, index=False, encoding="utf-8-sig")
