"""Sensibilidad de R3 con las pruebas estrictamente admisibles de la fila "F1. Tabular" del cuadro 4 del enunciado.

Desviación declarada (23/11/2026): el protocolo preregistró T1 (t pareada sobre semillas), T2 (Wilcoxon) y T3 (bootstrap
por bloques de prueba). Al revisar el cuadro 4 se constató que T3 figura en la fila de problemas espaciales y que la
fila tabular pide la t pareada con corrección de Nadeau-Bengio. Este análisis NO reemplaza al preregistrado: lo repite
con la batería admitida para F1 y comprueba si la conclusión cambia.

Pruebas por contraste (original frente a cada rival de la familia, m = 12, Holm aparte para cada prueba):
  A1. Wilcoxon de rangos con signo sobre las semillas compartidas.
  A2. t pareada con corrección de Nadeau-Bengio sobre las semillas: varianza × (1/J + n_prueba/n_entrenamiento).
      Con la partición fija la corrección es conservadora (el entrenamiento no cambia entre semillas).
  A3. McNemar exacta sobre los aciertos del ensamble (promedio de probabilidades de las semillas compartidas) en los
      22 883 flujos de prueba. Supone flujos independientes; con la dependencia por bloques es anticonservadora.
  A4. Friedman sobre los 13 modelos (semillas 6001 a 6020) con post hoc de Holm: z de la diferencia de rangos medios.
  A5. Bayesiana de rangos con signo con ROPE ±δ: "rechaza a favor" si P(original > rival) ≥ 0,95.
  A6. DeLong (1988) sobre el AUC uno contra todos de cada clase, con las probabilidades del ensamble; Holm sobre los
      60 contrastes (12 rivales × 5 clases). "Rechaza a favor" si al menos 3 de las 5 clases favorecen al original.
  (La F de 5×2 cv y Nadeau-Bengio sobre 5×2 están en la sensibilidad del capítulo 8, solo para LightGBM y CNN 1D.)
R3 frente a k (estricto): media favorable, al menos tres de A1 a A5 rechazan a favor del original, diferencia > δ,
|δ de Cliff| ≥ 0,147 y límite inferior del IC 95 % > 0.

Uso: python experiments/analisis_cuadro4.py   ->   experiments/resultados/confirmatorio/sensibilidad_cuadro4.{csv,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import analisis_confirmatorio as AC  # noqa: E402
from analisis_confirmatorio import D, E  # noqa: E402

SALIDA = AQUI / "resultados" / "confirmatorio"


def main():
    y = D.etiquetas()
    ids = D.heredados()["ids"]
    y_eva = y[ids["prueba"]]
    n_prueba, n_ent = len(ids["prueba"]), len(ids["entrenamiento"])
    orig = AC.corridas("Original")
    modelos = ["Original"] + AC.FAMILIA
    M = np.array([[AC.corridas(m)[s][0]["f1_eva"] for m in modelos] for s in range(6001, 6021)])
    rangos = E.rangos_medios(M)
    k, N = M.shape[1], M.shape[0]
    escala = np.sqrt(k * (k + 1) / (6 * N))
    filas = []
    for j, rival in enumerate(AC.FAMILIA, start=1):
        riv = AC.corridas(rival)
        semillas = sorted(set(orig) & set(riv))
        a = np.array([orig[s][0]["f1_eva"] for s in semillas])
        b = np.array([riv[s][0]["f1_eva"] for s in semillas])
        r = E.comparar_par(a, b, AC.DELTA)
        nb = E.nadeau_bengio(a - b, n_prueba, n_ent)
        Pa = np.mean([AC.prob_prueba(orig[s][1]) for s in semillas], 0)
        Pb = np.mean([AC.prob_prueba(riv[s][1]) for s in semillas], 0)
        pa, pb = Pa.argmax(1), Pb.argmax(1)
        delong = [E.delong(y_eva == c, Pa[:, c], Pb[:, c]) for c in range(Pa.shape[1])]
        mc = E.mcnemar_exacta(pa == y_eva, pb == y_eva)
        z = (rangos[0] - rangos[j]) / escala
        ic_nb = (r["Diferencia media"] - stats.t.ppf(0.975, nb["gl"]) * nb["error corregido"], r["Diferencia media"] + stats.t.ppf(0.975, nb["gl"]) * nb["error corregido"])
        filas.append({"rival": rival, "n_semillas": len(semillas), "diferencia": r["Diferencia media"], "delta Cliff": r["δ de Cliff"],
                      "IC95 Nadeau-Bengio": ic_nb, "p A1 Wilcoxon": r["p Wilcoxon"], "p A2 t Nadeau-Bengio": nb["p"],
                      "McNemar b (solo acierta el original)": mc["b"], "McNemar c (solo acierta el rival)": mc["c"], "p A3 McNemar": mc["p"],
                      "rango original": rangos[0], "rango rival": rangos[j], "p A4 Friedman post hoc": float(2 * stats.norm.sf(abs(z))),
                      "P(original>rival) bayes": r["P(A > B)"], "P(ROPE)": r["P(ROPE)"], "P(original<rival) bayes": r["P(A < B)"],
                      "DeLong dif AUC por clase": [round(float(x["diferencia"]), 5) for x in delong], "DeLong p por clase": [float(x["p"]) for x in delong]})
    t = pd.DataFrame(filas)
    p_dl = E.holm(np.concatenate(t["DeLong p por clase"].to_list())).reshape(len(t), -1)
    dif_dl = np.array(t["DeLong dif AUC por clase"].to_list())
    t["DeLong clases a favor"] = ((p_dl < AC.ALFA) & (dif_dl > 0)).sum(1)
    t["DeLong clases en contra"] = ((p_dl < AC.ALFA) & (dif_dl < 0)).sum(1)
    for col in ("p A1 Wilcoxon", "p A2 t Nadeau-Bengio", "p A3 McNemar", "p A4 Friedman post hoc"):
        t[col + " Holm"] = E.holm(t[col])
    a_favor = pd.DataFrame({
        "A1": (t["p A1 Wilcoxon Holm"] < AC.ALFA) & (t["diferencia"] > 0),
        "A2": (t["p A2 t Nadeau-Bengio Holm"] < AC.ALFA) & (t["diferencia"] > 0),
        "A3": (t["p A3 McNemar Holm"] < AC.ALFA) & (t["McNemar b (solo acierta el original)"] > t["McNemar c (solo acierta el rival)"]),
        "A4": (t["p A4 Friedman post hoc Holm"] < AC.ALFA) & (t["rango original"] < t["rango rival"]),
        "A5": t["P(original>rival) bayes"] >= 0.95,
        "A6": t["DeLong clases a favor"] >= 3})
    en_contra = pd.DataFrame({
        "A1": (t["p A1 Wilcoxon Holm"] < AC.ALFA) & (t["diferencia"] < 0),
        "A2": (t["p A2 t Nadeau-Bengio Holm"] < AC.ALFA) & (t["diferencia"] < 0),
        "A3": (t["p A3 McNemar Holm"] < AC.ALFA) & (t["McNemar b (solo acierta el original)"] < t["McNemar c (solo acierta el rival)"]),
        "A4": (t["p A4 Friedman post hoc Holm"] < AC.ALFA) & (t["rango original"] > t["rango rival"]),
        "A5": t["P(original<rival) bayes"] >= 0.95,
        "A6": t["DeLong clases en contra"] >= 3})
    t["pruebas a favor del original"] = a_favor.apply(lambda f: ", ".join(c for c in f.index if f[c]), axis=1)
    t["n a favor"] = a_favor.sum(1)
    t["pruebas en contra del original"] = en_contra.apply(lambda f: ", ".join(c for c in f.index if f[c]), axis=1)
    t["n en contra"] = en_contra.sum(1)
    t["condición 1 (media favorable)"] = t["diferencia"] > 0
    t["supera (R3 estricto)"] = (t["condición 1 (media favorable)"] & (t["n a favor"] >= 3) & (t["diferencia"] > AC.DELTA)
                                 & (t["delta Cliff"].abs() >= AC.CLIFF_MIN) & t["IC95 Nadeau-Bengio"].map(lambda ic: ic[0] > 0))
    t.to_csv(SALIDA / "sensibilidad_cuadro4.csv", index=False, encoding="utf-8-sig")
    resumen = {"R3 estricto (supera a los 12)": bool(t["supera (R3 estricto)"].all()),
               "rivales con media mayor que la del original": t.loc[~t["condición 1 (media favorable)"], "rival"].tolist(),
               "contrastes que supera (R3 estricto)": t.loc[t["supera (R3 estricto)"], "rival"].tolist(),
               "rivales mejores con al menos 3 pruebas admisibles": t.loc[t["n en contra"] >= 3, "rival"].tolist(),
               "rivales peores con al menos 3 pruebas admisibles": t.loc[t["n a favor"] >= 3, "rival"].tolist(),
               "n_prueba": n_prueba, "n_entrenamiento": n_ent, "factor Nadeau-Bengio n_prueba/n_entrenamiento": n_prueba / n_ent}
    (SALIDA / "sensibilidad_cuadro4.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(t[["rival", "diferencia", "p A1 Wilcoxon Holm", "p A2 t Nadeau-Bengio Holm", "p A3 McNemar Holm", "p A4 Friedman post hoc Holm",
             "P(original>rival) bayes", "P(original<rival) bayes", "DeLong clases a favor", "DeLong clases en contra", "pruebas a favor del original", "pruebas en contra del original", "supera (R3 estricto)"]].round(4).to_string(index=False))
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
