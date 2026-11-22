"""Diseño d01: ¿cuánto margen tiene una compuerta por flujo entre un experto de umbrales y uno neuronal?

Entrena, en la partición de desarrollo (solo filas de entrenamiento del E2), LightGBM y tres redes del E2 con sus
hiperparámetros definitivos y el mismo protocolo, y mide en el subconjunto 'diseno':
  - el F1 macro de cada modelo,
  - el del promedio simple de probabilidades (regla estándar de fusión),
  - el del ORÁCULO por flujo (acierta si alguno de los dos acierta): techo de cualquier compuerta entre ese par,
  - la confusión Blackhole-Wormhole.
No toca la validación ni la prueba del E2.

Uso: python d01_techo_oraculo.py [procesos]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1] / "training"))
SALIDA = AQUI / "resultados" / "d01"
CORRIDAS = AQUI / "corridas" / "d01"
REDES = {"DNN": ({"lr": 0.001, "weight_decay": 0.0, "batch": 256}, None),
         "Transformer": ({"lr": 0.003, "weight_decay": 1e-05, "batch": 512}, None),
         "CNN-LSTM": ({"lr": 0.003, "weight_decay": 0.0, "batch": 256}, [64, 64, 64, 0, 1, 0])}
HP_LGBM = {"learning_rate": 0.1, "num_leaves": 127, "min_child_samples": 10, "reg_lambda": 1.0, "colsample_bytree": 0.7, "subsample": 1.0}
SEMILLA = 6001
B, W = 0, 4  # índices de Blackhole y Wormhole en D.CLASES


def f1_macro(y, p):
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0))


def confusion_bw(y, p):
    sel = (y == B) | (y == W)
    return float(((y[sel] == B) & (p[sel] == W) | (y[sel] == W) & (p[sel] == B)).mean())


def correr_particion(semilla_part):
    import torch

    torch.set_num_threads(3)
    from particion_desarrollo import particion_desarrollo, matrices
    from codigo import motor

    ids = particion_desarrollo(semilla_part)
    X, pesos, y = matrices(ids["ajuste"])
    probs = {}
    destino = CORRIDAS / f"p{semilla_part}"
    destino.mkdir(parents=True, exist_ok=True)
    import lightgbm as lgb

    inicio = time.time()
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, subsample_freq=1, class_weight="balanced", random_state=SEMILLA,
                           n_jobs=3, verbose=-1, **HP_LGBM)
    m.fit(X[ids["ajuste"]], y[ids["ajuste"]], eval_set=[(X[ids["parada"]], y[ids["parada"]])], callbacks=[lgb.early_stopping(100, verbose=False)])
    probs["LightGBM"] = m.predict_proba(X[ids["diseno"]], num_iteration=m.best_iteration_)
    tiempos = {"LightGBM": round(time.time() - inicio, 1)}
    D = {"X": torch.from_numpy(X), "pesos": torch.from_numpy(pesos), "Y": torch.from_numpy(y)}
    for nombre, (hp, arq) in REDES.items():
        inicio = time.time()
        modelo, curvas, divergente, _ = motor.entrenar(nombre, arq, hp, SEMILLA, D, torch.from_numpy(ids["ajuste"]), torch.from_numpy(ids["parada"]))
        logits = motor.predecir(modelo, D["X"], torch.from_numpy(ids["diseno"]))
        probs[nombre] = torch.softmax(logits, 1).numpy()
        tiempos[nombre] = round(time.time() - inicio, 1)
    for k, v in probs.items():
        np.save(destino / f"{k}.npy", v.astype(np.float32))
    np.save(destino / "y_diseno.npy", y[ids["diseno"]])
    return semilla_part, tiempos


def resumir():
    import pandas as pd

    filas = []
    for carpeta in sorted(CORRIDAS.glob("p*")):
        y = np.load(carpeta / "y_diseno.npy")
        P = {f.stem: np.load(f) for f in carpeta.glob("*.npy") if f.stem != "y_diseno"}
        pred = {k: v.argmax(1) for k, v in P.items()}
        for k in P:
            filas.append({"particion": carpeta.name, "modelo": k, "f1": f1_macro(y, pred[k]), "conf_BW": confusion_bw(y, pred[k])})
        for red in REDES:
            prom = ((P["LightGBM"] + P[red]) / 2).argmax(1)
            oraculo = np.where(pred[red] == y, y, pred["LightGBM"])
            filas.append({"particion": carpeta.name, "modelo": f"Promedio LightGBM+{red}", "f1": f1_macro(y, prom), "conf_BW": confusion_bw(y, prom)})
            filas.append({"particion": carpeta.name, "modelo": f"Oráculo LightGBM|{red}", "f1": f1_macro(y, oraculo), "conf_BW": confusion_bw(y, oraculo),
                          "desacuerdo_%": 100 * float((pred[red] != pred["LightGBM"]).mean())})
        todos = np.where(np.any([pred[k] == y for k in pred], axis=0), y, pred["LightGBM"])
        filas.append({"particion": carpeta.name, "modelo": "Oráculo de los 4", "f1": f1_macro(y, todos), "conf_BW": confusion_bw(y, todos)})
    r = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    r.to_csv(SALIDA / "por_particion.csv", index=False, encoding="utf-8-sig")
    t = r.groupby("modelo", sort=False).agg(f1_medio=("f1", "mean"), f1_de=("f1", "std"), conf_BW=("conf_BW", "mean"), desacuerdo=("desacuerdo_%", "mean"))
    t.to_csv(SALIDA / "resumen.csv", encoding="utf-8-sig")
    print(t.round(4).to_string())


def main():
    from particion_desarrollo import SEMILLAS_PARTICION

    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with ProcessPoolExecutor(max_workers=procesos) as ex:
        for semilla_part, tiempos in ex.map(correr_particion, SEMILLAS_PARTICION):
            print("partición", semilla_part, "segundos:", tiempos, flush=True)
    resumir()


if __name__ == "__main__":
    main()
