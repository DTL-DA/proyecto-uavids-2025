"""Complemento de la auditoría: dentro de cada clase, ¿el error crece con la distancia (en filas) al flujo de
entrenamiento más cercano de la misma clase? Un gradiente positivo indica que la cercanía a la serie ayuda.

Solo lectura. Uso: python auditoria_gradiente.py <raiz_repo_E3> <carpeta_corridas_definitiva_E2> <carpeta_salida>
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raiz, corridas, salida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(raiz / "Entregable_2_UAVIDS_2025"))
from codigo import datos as D  # noqa: E402

datos, bloques, _ = D.crudo()
y = D.etiquetas()
ids = D.heredados()["ids"]
K = len(D.CLASES)
CERCA, LEJOS = 100, 500  # filas: cerca ≤ 100; lejos > 500


def distancia_misma_clase(filas):
    d = np.full(len(filas), np.inf)
    for c in range(K):
        pos = np.sort(ids["entrenamiento"][y[ids["entrenamiento"]] == c])
        sel = np.flatnonzero(y[filas] == c)
        j = np.searchsorted(pos, filas[sel])
        izq = pos[np.clip(j - 1, 0, len(pos) - 1)]
        der = pos[np.clip(j, 0, len(pos) - 1)]
        d[sel] = np.minimum(np.abs(filas[sel] - izq), np.abs(der - filas[sel]))
    return d


modelos = sorted(set(re.sub(r"_\d+_eva\.npy$", "", os.path.basename(f)).replace("definitiva_", "")
                     for f in glob.glob(str(corridas / "*_eva.npy"))) - {"Trivial"})
rng = np.random.default_rng(42)
filas_salida = []
for nombre, sufijo in [("validacion", "_val.npy"), ("prueba", "_eva.npy")]:
    filas = ids[nombre]
    d = distancia_misma_clase(filas)
    clases, blq = y[filas], bloques[filas]
    error = {}
    for modelo in modelos:
        aciertos = [np.load(f).argmax(1) == clases for f in sorted(glob.glob(str(corridas / f"definitiva_{modelo}_*{sufijo}")))]
        error[modelo] = 1 - np.mean(aciertos, axis=0)
    grupos = {"LightGBM": error["LightGBM"],
              "Redes (promedio de 13)": np.mean([error[m] for m in modelos if m not in ("LightGBM", "Logistica")], axis=0)}
    unicos = np.unique(blq)
    por_bloque = {b: np.flatnonzero(blq == b) for b in unicos}
    for grupo, e in grupos.items():
        for c in range(K):
            cerca = (clases == c) & (d <= CERCA)
            lejos = (clases == c) & (d > LEJOS)
            if cerca.sum() < 30 or lejos.sum() < 30:
                filas_salida.append({"particion": nombre, "grupo": grupo, "clase": D.CLASES[c], "filas_cerca": int(cerca.sum()),
                                     "filas_lejos": int(lejos.sum()), "nota": "menos de 30 filas en un extremo"})
                continue
            obs = e[lejos].mean() - e[cerca].mean()
            boot = []
            for _ in range(2000):
                idx = np.concatenate([por_bloque[b] for b in rng.choice(unicos, len(unicos), replace=True)])
                ce, le = idx[cerca[idx]], idx[lejos[idx]]
                if len(ce) and len(le):
                    boot.append(e[le].mean() - e[ce].mean())
            boot = np.array(boot)
            filas_salida.append({"particion": nombre, "grupo": grupo, "clase": D.CLASES[c], "filas_cerca": int(cerca.sum()),
                                 "filas_lejos": int(lejos.sum()), "error_cerca": round(float(e[cerca].mean()), 4),
                                 "error_lejos": round(float(e[lejos].mean()), 4), "lejos_menos_cerca": round(float(obs), 4),
                                 "IC95": f"[{np.percentile(boot, 2.5):.4f}; {np.percentile(boot, 97.5):.4f}]",
                                 "p_unilateral": round(float(np.mean(boot <= 0)), 4)})
tabla = pd.DataFrame(filas_salida)
tabla.to_csv(salida / "gradiente_por_clase.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(tabla.to_string(index=False))
