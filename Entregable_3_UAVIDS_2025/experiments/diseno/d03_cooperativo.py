"""Diseño d03: esquema de entrenamiento cooperativo en la partición de desarrollo por bloques.

Variantes (todas con el motor del E2, la misma semilla 6001 y los hiperparámetros de optimización del DNN del E2):
  V4  compuerta aprendida que ve solo x
  V5  compuerta aprendida que ve x y las distribuciones de ambos expertos
  V6  solo el experto de umbrales            (diagnóstico)
  V7  solo el experto suave                  (diagnóstico)
  V8  promedio fijo de los dos expertos      (diagnóstico)
Referencias de d01 en las mismas reparticiones: LightGBM 0,845 y DNN 0,837 de F1 macro medio.

Uso: python d02_variantes_compuerta.py [procesos]
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
SALIDA = AQUI / "resultados" / "d03"
HP = {"lr": 0.001, "weight_decay": 0.0, "batch": 256}
SEMILLA = 6001
VARIANTES = {"V9_coop_x_expertos": {"entrada_compuerta": "x+expertos"},
             "V10_coop_x": {"entrada_compuerta": "x"}}
B, W = 0, 4


def f1_macro(y, p):
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0))


def correr(trabajo):
    variante, semilla_part = trabajo
    ruta = SALIDA / f"{variante}_p{semilla_part}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    import torch

    torch.set_num_threads(2)
    import registro
    from particion_desarrollo import matrices, particion_desarrollo
    import entrenar_original

    registro.registrar(**VARIANTES[variante])
    ids = particion_desarrollo(semilla_part)
    X, pesos, y = matrices(ids["ajuste"])
    D = {"X": torch.from_numpy(X), "pesos": torch.from_numpy(pesos), "Y": torch.from_numpy(y)}
    inicio = time.time()
    modelo, curvas, divergente, _ = entrenar_original.entrenar(None, HP, SEMILLA, D, torch.from_numpy(ids["ajuste"]), torch.from_numpy(ids["parada"]))
    segundos = time.time() - inicio
    xd = D["X"][torch.from_numpy(ids["diseno"])]
    yd = y[ids["diseno"]]
    modelo.eval()
    with torch.no_grad():
        p = modelo(xd).exp()
        log_u, log_s = modelo.expertos(xd)
        g = modelo.compuerta(xd, log_u, log_s)
    pred = p.argmax(1).numpy()
    sel = (yd == B) | (yd == W)
    registro_corrida = {"variante": variante, "particion": semilla_part, "f1": f1_macro(yd, pred),
                        "f1_experto_umbral": f1_macro(yd, log_u.argmax(1).numpy()), "f1_experto_suave": f1_macro(yd, log_s.argmax(1).numpy()),
                        "conf_BW": float(((yd[sel] == B) & (pred[sel] == W) | (yd[sel] == W) & (pred[sel] == B)).mean()),
                        "g_umbral_por_clase": [round(float(g[yd == c, 0].mean()), 3) if (yd == c).any() else None for c in range(5)],
                        "epocas": len(curvas), "segundos": round(segundos, 1), "parametros": sum(q.numel() for q in modelo.parameters()),
                        "cota_lipschitz": modelo.cota_lipschitz(), "divergente": divergente}
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(registro_corrida, ensure_ascii=False), encoding="utf-8")
    return registro_corrida


def main():
    import pandas as pd
    from particion_desarrollo import SEMILLAS_PARTICION

    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    trabajos = [(v, s) for v in VARIANTES for s in SEMILLAS_PARTICION]
    with ProcessPoolExecutor(max_workers=procesos) as ex:
        filas = list(ex.map(correr, trabajos))
    r = pd.DataFrame(filas)
    r.to_csv(SALIDA / "por_particion.csv", index=False, encoding="utf-8-sig")
    t = r.groupby("variante").agg(f1=("f1", "mean"), f1_de=("f1", "std"), f1_umbral=("f1_experto_umbral", "mean"), f1_suave=("f1_experto_suave", "mean"),
                                  conf_BW=("conf_BW", "mean"), epocas=("epocas", "mean"), segundos=("segundos", "mean"), parametros=("parametros", "first"))
    t.to_csv(SALIDA / "resumen.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(t.round(4).to_string())
    print(r.pivot(index="variante", columns="particion", values="f1").round(4).to_string())
    print(r[["variante", "particion", "g_umbral_por_clase"]].to_string())


if __name__ == "__main__":
    main()
