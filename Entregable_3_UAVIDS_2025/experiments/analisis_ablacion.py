"""Ablación (familia secundaria con Holm propio) y predicciones secundarias P1 a P3 del protocolo.

Compara el modelo completo (A0: corridas definitivas 6001-6020) con cada variante A1 a A6 (mismas semillas) en F1 macro
de prueba (t pareado y Wilcoxon, Holm sobre las 6 variantes), confusión Blackhole-Wormhole y F1 por clase. Calcula además
el peso medio de la compuerta hacia el experto de umbrales por clase real (P2) y la diferencia por clase frente a la
mejor red del Top 10 (P3).

Uso: python experiments/analisis_ablacion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "training"))
import registro  # noqa: E402
from ejecutar_original import CONFIGURACION  # noqa: E402
from codigo import datos as D  # noqa: E402
from codigo import estadistica as E  # noqa: E402
from codigo import modelos as M  # noqa: E402

CORR = AQUI / "resultados" / "corridas"
E2_DEF = AQUI.parents[1] / "Entregable_2_UAVIDS_2025" / "resultados" / "corridas" / "definitiva"
SALIDA = AQUI / "resultados" / "ablacion"
VARIANTES = {"A1_sin_compuerta": "Sin compuerta (promedio 1/2)", "A2_sin_escalones": "Sin escalones suaves (MLP sobre x)",
             "A3_conjunto": "Entrenamiento conjunto clásico", "A4_sin_balance": "Sin regularizador de balance (γ = 0)",
             "A5_solo_suave": "Solo el experto suave (modelo base)", "A6_solo_umbral": "Solo el experto de umbrales"}
SEMILLAS = range(6001, 6021)
CLASES = D.CLASES


def metricas_prueba(prob, y):
    pred = prob.argmax(1)
    sel = (y == 0) | (y == 4)
    return {"f1": float(f1_score(y, pred, average="macro")), "conf_BW": float(((y[sel] == 0) & (pred[sel] == 4) | (y[sel] == 4) & (pred[sel] == 0)).mean()),
            **{f"F1 {c}": float(v) for c, v in zip(CLASES, f1_score(y, pred, average=None, labels=range(5)))}}


def cargar(base):
    return np.load(str(base) + "_eva.npy")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    h = D.heredados()
    y = h["y"][h["ids"]["prueba"]]
    filas = []
    for s in SEMILLAS:
        filas.append({"variante": "A0_completo", "semilla": s, **metricas_prueba(cargar(CORR / "definitiva" / f"definitiva_Original_{s}"), y)})
        for v in VARIANTES:
            filas.append({"variante": v, "semilla": s, **metricas_prueba(cargar(CORR / "ablacion" / f"ablacion_Original_{s}_{v}"), y)})
        filas.append({"variante": "CNN-LSTM (mejor red)", "semilla": s, **metricas_prueba(cargar(E2_DEF / f"definitiva_CNN-LSTM_{s}"), y)})
        filas.append({"variante": "LightGBM", "semilla": s, **metricas_prueba(cargar(E2_DEF / f"definitiva_LightGBM_{s}"), y)})
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "por_semilla.csv", index=False, encoding="utf-8-sig")
    medias = t.groupby("variante").mean(numeric_only=True).drop(columns="semilla")
    a0 = t[t.variante == "A0_completo"].set_index("semilla")
    contrastes = []
    for v, nombre in VARIANTES.items():
        b = t[t.variante == v].set_index("semilla")
        r = E.comparar_par(a0["f1"], b["f1"])
        contrastes.append({"variante": v, "descripción": nombre, "F1 variante": b["f1"].mean(), "F1 completo": a0["f1"].mean(),
                           "completo − variante": r["Diferencia media"], "IC95 t": r["IC 95 % t"], "p t": r["p t pareada"], "p Wilcoxon": r["p Wilcoxon"],
                           "δ Cliff": r["δ de Cliff"], "conf B–W variante": b["conf_BW"].mean(), "conf B–W completo": a0["conf_BW"].mean()})
    c = pd.DataFrame(contrastes)
    c["p t Holm"], c["p Wilcoxon Holm"] = E.holm(c["p t"]), E.holm(c["p Wilcoxon"])
    c["el componente aporta"] = (c[["p t Holm", "p Wilcoxon Holm"]].max(axis=1) < 0.05) & (c["completo − variante"] > 0)
    c["el componente resta"] = (c[["p t Holm", "p Wilcoxon Holm"]].max(axis=1) < 0.05) & (c["completo − variante"] < 0)
    c.to_csv(SALIDA / "contrastes.csv", index=False, encoding="utf-8-sig")
    # P2: peso de la compuerta por clase real (modelos definitivos 6001-6020)
    import torch

    registro.registrar(**CONFIGURACION)
    x = torch.from_numpy(h["X"][h["ids"]["prueba"]])
    pesos = []
    for s in SEMILLAS:
        m = M.construir("Original")
        m.load_state_dict(torch.load(CORR / "definitiva" / f"definitiva_Original_{s}.pt", map_location="cpu"))
        m.eval()
        with torch.no_grad():
            g = m.compuerta(x)[:, 0].numpy()
        pesos.append({"semilla": s, **{c: float(g[y == i].mean()) for i, c in enumerate(CLASES)}})
    p2 = pd.DataFrame(pesos)
    p2.to_csv(SALIDA / "peso_compuerta_por_clase.csv", index=False, encoding="utf-8-sig")
    # P3: diferencia por clase frente a la mejor red
    p3 = (medias.loc["A0_completo", [f"F1 {c}" for c in CLASES]] - medias.loc["CNN-LSTM (mejor red)", [f"F1 {c}" for c in CLASES]]).rename("original − CNN-LSTM")
    resumen = {"P1 (sin escalones sube la confusión B–W)": bool(medias.loc["A2_sin_escalones", "conf_BW"] > medias.loc["A0_completo", "conf_BW"]),
               "conf_BW": medias["conf_BW"].round(4).to_dict(),
               "P2 (más peso al experto de umbrales en B y W que en Sybil)": bool(min(p2["Blackhole"].mean(), p2["Wormhole"].mean()) > p2["Sybil"].mean()),
               "peso_umbral_por_clase": p2[CLASES].mean().round(3).to_dict(),
               "P3 (ventaja concentrada en B y W frente a CNN-LSTM)": bool(p3[["F1 Blackhole", "F1 Wormhole"]].mean() > p3[["F1 Flooding", "F1 Normal", "F1 Sybil"]].mean()),
               "diferencia_por_clase_vs_CNN-LSTM": p3.round(4).to_dict()}
    (SALIDA / "predicciones_secundarias.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(medias.round(4).to_string())
    print(c.drop(columns=["IC95 t"]).round(4).to_string(index=False))
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
