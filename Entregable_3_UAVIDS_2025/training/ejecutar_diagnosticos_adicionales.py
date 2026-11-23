"""Diagnósticos de ajuste y sobreajuste del original en cada conjunto adicional (cuadro 4, fila "Varios conjuntos de
datos": "mismos diagnósticos aplicados en cada conjunto adicional").

Las corridas de ejecutar_definitivas.py (fase adicionales) no guardaron el F1 de entrenamiento ni las probabilidades de
validación, que hacen falta para la brecha, la temperatura y la consistencia entre validación y prueba. Aquí:
  - se reentrena el original con las mismas semillas 6001 a 6010 y la misma función de entrenamiento (CPU determinista):
    su F1 de prueba debe coincidir con el registrado, lo que además comprueba la reproducibilidad;
  - se guardan F1 de entrenamiento (muestra del tamaño de la validación, como motor.ejecutar), validación y prueba, y
    las probabilidades de validación y de prueba;
  - prueba de permutación: 200 entrenamientos con las etiquetas de entrenamiento permutadas (semillas 9000 a 9199),
    semilla de modelo 6001, como en el problema principal.

Uso: python training/ejecutar_diagnosticos_adicionales.py [procesos]   ->   experiments/resultados/corridas/diagnostico_adicionales/
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from ejecutar_original import CONFIGURACION, CORRIDAS, mejores_hp, preparar_proceso  # noqa: E402

CARPETA = CORRIDAS / "diagnostico_adicionales"
CONJUNTOS = ("uav_nidd", "unsw_nb15", "unr_idd")
B = 200


def plan():
    t = [{"id": f"{c}_Original_{s}", "conjunto": c, "semilla": s, "permutar": None} for c in CONJUNTOS for s in range(6001, 6011)]
    t += [{"id": f"{c}_Original_6001_perm{b:03d}", "conjunto": c, "semilla": 6001, "permutar": 9000 + b} for c in CONJUNTOS for b in range(B)]
    return t


def correr(trabajo, carpeta):
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
    registro.registrar(**CONFIGURACION, d=X.shape[1], n_clases=K)
    ent, val, eva = (torch.from_numpy(d[k]) for k in ("ent", "val", "eva"))
    D = {"X": torch.from_numpy(X), "pesos": torch.from_numpy(d["pesos"]), "Y": torch.from_numpy(y)}
    etiquetas = None
    if trabajo["permutar"] is not None:
        g = np.random.default_rng(trabajo["permutar"])
        y_perm = y.copy()
        y_perm[d["ent"]] = y[d["ent"][g.permutation(len(d["ent"]))]]
        etiquetas = torch.from_numpy(y_perm)
    inicio = time.time()
    modelo, curvas, divergente, _ = motor.entrenar("Original", None, mejores_hp(), trabajo["semilla"], D, ent, val, etiquetas=etiquetas)
    f1 = lambda v, p: float(f1_score(v, p, labels=np.unique(v), average="macro", zero_division=0))  # noqa: E731
    p_val = torch.softmax(motor.predecir(modelo, D["X"], val), 1).numpy()
    p_eva = torch.softmax(motor.predecir(modelo, D["X"], eva), 1).numpy()
    r = dict(trabajo)
    r.update({"epocas": len(curvas), "divergente": divergente, "f1_val": f1(y[d["val"]], p_val.argmax(1)), "f1_eva": f1(y[d["eva"]], p_eva.argmax(1)),
              "segundos": round(time.time() - inicio, 1)})
    if trabajo["permutar"] is None:
        g = torch.Generator().manual_seed(trabajo["semilla"] + 1)
        muestra = ent[torch.randperm(len(ent), generator=g)[:len(val)]]
        r["f1_ent"] = f1(y[muestra.numpy()], motor.predecir(modelo, D["X"], muestra).argmax(1).numpy())
        np.save(ruta.with_name(ruta.stem + "_val.npy"), p_val.astype(np.float32))
        np.save(ruta.with_name(ruta.stem + "_eva.npy"), p_eva.astype(np.float32))
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ruta)
    return r


def main():
    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    CARPETA.mkdir(parents=True, exist_ok=True)
    trabajos = plan()
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=procesos, initializer=preparar_proceso) as ex:
        futuros = {ex.submit(correr, t, str(CARPETA)): t for t in trabajos}
        for i, f in enumerate(as_completed(futuros), 1):
            try:
                r = f.result()
                print(f"[{i}/{len(trabajos)}] {r['id']}: f1_val={r['f1_val']:.4f} f1_eva={r['f1_eva']:.4f} ({r['segundos']} s; {(time.time() - inicio) / 60:.1f} min)", flush=True)
            except Exception as e:  # una corrida fallida no detiene las demás
                print(f"[{i}/{len(trabajos)}] ERROR en {futuros[f]['id']}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
