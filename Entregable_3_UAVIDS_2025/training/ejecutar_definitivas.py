"""Corridas definitivas del protocolo (solo después de la etiqueta `e3-protocolo`).

    python training/ejecutar_definitivas.py principal [procesos]   # original 6001-6048, CNN-LSTM 6021-6048, TabNet 6021-6038
    python training/ejecutar_definitivas.py ablacion [procesos]    # A1 a A6, semillas 6001-6020
    python training/ejecutar_definitivas.py adicionales [procesos] [modelos...]  # 12 rivales + original, 3 conjuntos, 6001-6010

Todas las corridas del problema principal usan ``codigo.motor.ejecutar`` del E2 (mismo registro JSON, probabilidades de
validación y prueba, pesos, latencia y MAC). Las de los conjuntos adicionales usan el mismo bucle de entrenamiento
(``motor.entrenar`` o ``entrenar_original.entrenar``) sobre las matrices de ``conjuntos_adicionales.py``.
Cada corrida se salta si su JSON ya existe (reanudación).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from ejecutar_original import CONFIGURACION, CORRIDAS, mejores_hp, preparar_proceso  # noqa: E402

E2_DEF = AQUI.parents[1] / "Entregable_2_UAVIDS_2025" / "resultados" / "corridas" / "definitiva"
FAMILIA = ["CNN-LSTM", "Autoencoder", "RNN", "Transformer", "BiLSTM", "ResNet", "GRU", "KAN", "TabNet", "DBN", "LightGBM", "CNN 1D"]
COSTO = {"BiLSTM": 50, "GRU": 24, "TabNet": 14, "CNN-LSTM": 12, "RNN": 8, "ResNet": 4.5, "LSTM": 4.4, "Transformer": 2.8, "CNN 1D": 2.5,
         "Autoencoder": 2.5, "KAN": 2.3, "DBN": 1.5, "Original": 2, "LightGBM": 1}
ABLACION = {"A1_sin_compuerta": {"compuerta": "promedio"},
            "A2_sin_escalones": {"codificador": "ninguno"},
            "A3_conjunto": {"__conjunto__": True},
            "A4_sin_balance": {"gamma": 0.0},
            "A5_solo_suave": {"compuerta": "suave"},
            "A6_solo_umbral": {"compuerta": "umbral"}}


def hp_e2(modelo):
    nombre = modelo.replace(" ", "_").replace("í", "i")
    d = json.loads((E2_DEF / f"definitiva_{nombre}_6001.json").read_text(encoding="utf-8"))
    return d["hp"], d.get("arquitectura")


# ------------------------------------------------------------------ problema principal
def plan_principal():
    from codigo import plan as plan_e2

    hp = mejores_hp()
    t = [plan_e2.trabajo("definitiva", "Original", hp, s, evaluar=True, guardar_prob_val=True, guardar_pesos=True) for s in range(6001, 6049)]
    for modelo, hasta in (("CNN-LSTM", 6048), ("TabNet", 6038)):
        h, arq = hp_e2(modelo)
        extra = {"arquitectura": arq} if arq else {}
        t += [plan_e2.trabajo("definitiva", modelo, h, s, evaluar=True, guardar_prob_val=True, guardar_pesos=True, **extra) for s in range(6021, hasta + 1)]
    return sorted(t, key=lambda x: -x["costo"])


def plan_ablacion():
    from codigo import plan as plan_e2

    hp = mejores_hp()
    return [plan_e2.trabajo("ablacion", "Original", hp, s, f"_{v}", variante_original=v, evaluar=True, guardar_prob_val=True)
            for v in ABLACION for s in range(6001, 6021)]


def correr_principal(trabajo, carpeta):
    import entrenar_original
    import registro
    from codigo import motor

    v = trabajo.get("variante_original")
    if v:
        cambios = dict(ABLACION[v])
        conjunto = cambios.pop("__conjunto__", False)
        registro.registrar(**{**CONFIGURACION, **cambios})
        if conjunto:
            original = entrenar_original.entrenar
            motor_entrenar = motor.entrenar

            def entrenar_conjunto(nombre, arquitectura, hp, semilla, D, ids_ent, ids_val, max_epocas=30, paciencia=6, etiquetas=None, variante=None):
                return original(arquitectura, hp, semilla, D, ids_ent, ids_val, max_epocas, paciencia, etiquetas, variante, cooperativo=False)

            motor.entrenar = entrenar_conjunto
            try:
                return motor.ejecutar(trabajo, carpeta)
            finally:
                motor.entrenar = motor_entrenar
    else:
        registro.registrar(**CONFIGURACION)
    return motor.ejecutar(trabajo, carpeta)


# ------------------------------------------------------------------ conjuntos adicionales
def plan_adicionales(modelos=None):
    from conjuntos_adicionales import CONJUNTOS

    modelos = modelos or FAMILIA + ["Original"]
    t = [{"id": f"{c}_{m.replace(' ', '_')}_{s}", "conjunto": c, "modelo": m, "semilla": s, "costo": COSTO.get(m, 1) * (3 if c != "unr_idd" else 1)}
         for c in CONJUNTOS for m in modelos for s in range(6001, 6011)]
    return sorted(t, key=lambda x: -x["costo"])


def correr_adicional(trabajo, carpeta):
    ruta = Path(carpeta) / f"{trabajo['id']}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    import torch
    from sklearn.metrics import f1_score

    import registro
    from codigo import modelos, motor
    from conjuntos_adicionales import cargar

    d = cargar(trabajo["conjunto"])
    X, y = d["X"], d["y"]
    K = int(y.max()) + 1
    modelos.N_ATRIBUTOS, modelos.N_CLASES = X.shape[1], K
    ids = {k: d[k] for k in ("ent", "val", "eva")}
    f1 = lambda v, p: float(f1_score(v, p, labels=np.unique(v), average="macro", zero_division=0))  # noqa: E731
    inicio = time.time()
    m = trabajo["modelo"]
    registro_corrida = dict(trabajo)
    if m == "LightGBM":
        import lightgbm as lgb

        hp, _ = hp_e2(m)
        clf = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, subsample_freq=1, class_weight="balanced", random_state=trabajo["semilla"],
                                 n_jobs=2, verbose=-1, learning_rate=hp["learning_rate"], num_leaves=hp["num_leaves"], min_child_samples=hp["min_child_samples"],
                                 reg_lambda=hp["reg_lambda"], colsample_bytree=hp["colsample_bytree"], subsample=hp["subsample"])
        clf.fit(X[ids["ent"]], y[ids["ent"]], eval_set=[(X[ids["val"]], y[ids["val"]])], callbacks=[lgb.early_stopping(100, verbose=False)])
        p_val = clf.predict_proba(X[ids["val"]], num_iteration=clf.best_iteration_)
        p_eva = clf.predict_proba(X[ids["eva"]], num_iteration=clf.best_iteration_)
        registro_corrida.update({"epocas": int(clf.best_iteration_ or 0)})
    else:
        if m == "Original":
            hp, arq = mejores_hp(), None
            registro.registrar(**CONFIGURACION, d=X.shape[1], n_clases=K)
        else:
            hp, arq = hp_e2(m)
        D = {"X": torch.from_numpy(X), "pesos": torch.from_numpy(d["pesos"]), "Y": torch.from_numpy(y)}
        modelo, curvas, divergente, _ = motor.entrenar(m, arq, hp, trabajo["semilla"], D, torch.from_numpy(ids["ent"]), torch.from_numpy(ids["val"]))
        p_val = torch.softmax(motor.predecir(modelo, D["X"], torch.from_numpy(ids["val"])), 1).numpy()
        p_eva = torch.softmax(motor.predecir(modelo, D["X"], torch.from_numpy(ids["eva"])), 1).numpy()
        registro_corrida.update({"epocas": len(curvas), "divergente": divergente, "parametros": modelos.parametros(modelo)})
    registro_corrida.update({"f1_val": f1(y[ids["val"]], p_val.argmax(1)), "f1_eva": f1(y[ids["eva"]], p_eva.argmax(1)),
                             "segundos": round(time.time() - inicio, 1), "hilos": torch.get_num_threads(),
                             "equipo": f"{platform.processor()} | {platform.system()} {platform.release()} | torch {torch.__version__}", "pid": os.getpid()})
    np.save(ruta.with_name(ruta.stem + "_eva.npy"), p_eva.astype(np.float32))
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(registro_corrida, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ruta)
    return registro_corrida


def main():
    fase = sys.argv[1]
    procesos = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    preparar_proceso()
    if fase == "principal":
        trabajos, funcion, carpeta = plan_principal(), correr_principal, CORRIDAS / "definitiva"
    elif fase == "ablacion":
        trabajos, funcion, carpeta = plan_ablacion(), correr_principal, CORRIDAS / "ablacion"
    elif fase == "adicionales":
        trabajos, funcion, carpeta = plan_adicionales(sys.argv[3:] or None), correr_adicional, CORRIDAS / "adicionales"
    else:
        raise ValueError(fase)
    carpeta.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=procesos, initializer=preparar_proceso) as ex:
        futuros = {ex.submit(funcion, t, str(carpeta)): t for t in trabajos}
        for i, f in enumerate(as_completed(futuros), 1):
            try:
                r = f.result()
                print(f"[{i}/{len(trabajos)}] {r['id']}: f1_val={r.get('f1_val', float('nan')):.4f} f1_eva={r.get('f1_eva', float('nan')):.4f} "
                      f"({r.get('segundos')} s; {(time.time() - inicio) / 60:.1f} min)", flush=True)
            except Exception as e:  # una corrida fallida no detiene las demás
                print(f"[{i}/{len(trabajos)}] ERROR en {futuros[f]['id']}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
