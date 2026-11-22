"""Análisis de potencia a partir del piloto (sección 8.7.2 del enunciado).

Para cada rival k de la familia confirmatoria (m = 12) se calcula, con las semillas 6001 a 6005 y en VALIDACIÓN:
    d_s = F1_val(original, s) - F1_val(k, s)      (las corridas de los rivales son las definitivas del E2)
    sigma_d = desviación estándar muestral de d_s
    R = techo(((z_{1 - alpha'/2} + z_{0,90}) * sigma_d / delta)^2),  alpha' = 0,05 / m,  delta = 0,01
y la diferencia mínima detectable con las 20 semillas disponibles:
    MDE_20 = (z_{1 - alpha'/2} + z_{0,90}) * sigma_d / sqrt(20)
Se reporta también la potencia conjunta aproximada (producto de las potencias por contraste) suponiendo que el
efecto verdadero vale delta.

Uso: python experiments/potencia.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

AQUI = Path(__file__).resolve().parent
E2 = AQUI.parents[1] / "Entregable_2_UAVIDS_2025" / "resultados" / "corridas" / "definitiva"
PILOTO = AQUI / "resultados" / "corridas" / "piloto"
FAMILIA = ["CNN-LSTM", "Autoencoder", "RNN", "Transformer", "BiLSTM", "ResNet", "GRU", "KAN", "TabNet", "DBN", "LightGBM", "CNN 1D"]
EXPLORATORIOS = ["DNN", "LSTM", "Logística"]
SEMILLAS = [6001, 6002, 6003, 6004, 6005]
DELTA, ALFA, POTENCIA, R_DISPONIBLE = 0.01, 0.05, 0.90, 20


def f1_val(carpeta, prefijo, modelo, semilla):
    nombre = modelo.replace(" ", "_").replace("í", "i")
    return json.loads((carpeta / f"{prefijo}_{nombre}_{semilla}.json").read_text(encoding="utf-8"))["f1_val"]


def main():
    m = len(FAMILIA)
    z = norm.ppf(1 - ALFA / m / 2) + norm.ppf(POTENCIA)
    original = np.array([f1_val(PILOTO, "piloto", "Original", s) for s in SEMILLAS])
    filas = []
    for k in FAMILIA + EXPLORATORIOS:
        rival = np.array([f1_val(E2, "definitiva", k, s) for s in SEMILLAS])
        d = original - rival
        sd = float(d.std(ddof=1))
        R = math.ceil((z * sd / DELTA) ** 2)
        mde = z * sd / math.sqrt(R_DISPONIBLE)
        # potencia con 20 semillas si el efecto verdadero fuese delta (t aproximada por normal)
        pot20 = float(norm.cdf(DELTA / (sd / math.sqrt(R_DISPONIBLE)) - norm.ppf(1 - ALFA / m / 2))) if sd > 0 else 1.0
        filas.append({"rival": k, "familia": "confirmatoria" if k in FAMILIA else "exploratoria", "dif_media_val": round(float(d.mean()), 4),
                      "sigma_d": round(sd, 5), "R_necesarias": max(R, 10), "MDE_con_20": round(mde, 4), "potencia_con_20": round(pot20, 3)})
    conjunta = float(np.prod([f["potencia_con_20"] for f in filas if f["familia"] == "confirmatoria"]))
    salida = {"alfa_ajustado": ALFA / m, "z_total": round(z, 4), "f1_val_original_piloto": original.round(4).tolist(), "contrastes": filas,
              "potencia_conjunta_con_20": round(conjunta, 4)}
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "potencia.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    import pandas as pd

    print(pd.DataFrame(filas).to_string(index=False))
    print("F1 de validación del original en el piloto:", original.round(4), "| potencia conjunta con 20 semillas:", round(conjunta, 4))


if __name__ == "__main__":
    main()
