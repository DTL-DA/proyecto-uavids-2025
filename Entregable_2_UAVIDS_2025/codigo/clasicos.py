"""Líneas base sin Deep Learning: trivial (clase mayoritaria), regresión logística penalizada y
árboles potenciados (LightGBM), con la misma partición, el mismo preprocesamiento y los mismos
pesos de clase que los modelos profundos."""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .motor import datos_particion, memoria_pico_mb

CLASICOS = ("Trivial", "Logística", "LightGBM")


def _ajustar(trabajo, X, y, ids):
    hp, semilla = trabajo["hp"], trabajo["semilla"]
    ent, val = ids["ent"], ids["val"]
    if trabajo["modelo"] == "Trivial":
        mayoritaria = int(np.bincount(y[ent]).argmax())
        prob = lambda indices: np.eye(5, dtype=np.float32)[np.full(len(indices), mayoritaria)]
        return prob, {"clase_mayoritaria": mayoritaria}
    if trabajo["modelo"] == "Logística":
        m = LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=5000, random_state=semilla).fit(X[ent], y[ent])
        return (lambda indices: m.predict_proba(X[indices]).astype(np.float32)), {"parametros": int(m.coef_.size + m.intercept_.size), "iteraciones": int(np.max(m.n_iter_)), "modelo": m}
    import lightgbm as lgb

    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, learning_rate=hp["learning_rate"], num_leaves=hp["num_leaves"],
                           min_child_samples=hp["min_child_samples"], reg_lambda=hp["reg_lambda"], colsample_bytree=hp["colsample_bytree"],
                           subsample=hp["subsample"], subsample_freq=1, class_weight="balanced", random_state=semilla, n_jobs=trabajo.get("hilos", 2), verbose=-1)
    m.fit(X[ent], y[ent], eval_set=[(X[val], y[val])], callbacks=[lgb.early_stopping(100, verbose=False)])
    arboles = int(m.best_iteration_ or m.n_estimators)
    hojas = int(sum(t["num_leaves"] for t in m.booster_.dump_model()["tree_info"][: arboles * 5]))
    return (lambda indices: m.predict_proba(X[indices], num_iteration=arboles).astype(np.float32)), {"arboles": arboles, "hojas": hojas, "parametros": 2 * hojas, "modelo": m}


def ejecutar_clasico(trabajo, carpeta):
    carpeta = Path(carpeta)
    ruta = carpeta / f"{trabajo['id']}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    D = datos_particion(trabajo.get("particion", "principal"))
    X, y = D["X"].numpy(), D["Y"].numpy()
    ids = {k: v.numpy() for k, v in D["ids"].items()}
    inicio = time.time()
    prob, extra = _ajustar(trabajo, X, y, ids)
    modelo_ajustado = extra.pop("modelo", None)
    registro = {k: trabajo[k] for k in trabajo}
    registro.update(extra)
    registro.update({"segundos": round(time.time() - inicio, 2), "epocas": None, "divergente": False, "mac": None,
                     "equipo": f"{platform.processor()} | {platform.system()} {platform.release()} | Python {platform.python_version()}", "pid": os.getpid()})
    p_val = prob(ids["val"])
    pred_val = p_val.argmax(1)
    registro.update({"f1_val": f1_score(y[ids["val"]], pred_val, average="macro", zero_division=0), "acc_val": accuracy_score(y[ids["val"]], pred_val),
                     "bacc_val": balanced_accuracy_score(y[ids["val"]], pred_val)})
    muestra = np.random.default_rng(trabajo["semilla"] + 1).permutation(ids["ent"])[: len(ids["val"])]
    registro["f1_ent"] = f1_score(y[muestra], prob(muestra).argmax(1), average="macro", zero_division=0)
    if trabajo.get("guardar_prob_val"):
        np.save(ruta.with_name(ruta.stem + "_val.npy"), p_val)
    if trabajo.get("guardar_pesos") and modelo_ajustado is not None:
        import pickle

        ruta.with_suffix(".pkl").write_bytes(pickle.dumps(modelo_ajustado))
    if trabajo.get("evaluar") and "eva" in ids:
        inicio_eva = time.time()
        p_eva = prob(ids["eva"])
        registro["ms_por_muestra_lote"] = round(1000 * (time.time() - inicio_eva) / len(ids["eva"]), 6)
        pred = p_eva.argmax(1)
        registro.update({"f1_eva": f1_score(y[ids["eva"]], pred, average="macro", zero_division=0), "acc_eva": accuracy_score(y[ids["eva"]], pred)})
        if trabajo.get("guardar_prob_eva", True):
            np.save(ruta.with_name(ruta.stem + "_eva.npy"), p_eva)
        muestras = ids["eva"][:200]
        inicio_lat = time.perf_counter()
        for i in muestras:
            prob(np.array([i]))
        registro["ms_por_muestra"] = round(1000 * (time.perf_counter() - inicio_lat) / len(muestras), 5)
    registro["memoria_pico_mb"] = memoria_pico_mb()
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(registro, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta)
    return registro
