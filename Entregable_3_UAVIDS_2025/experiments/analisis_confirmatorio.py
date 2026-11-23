"""Análisis confirmatorio del protocolo (secciones 6 y 7 de protocolo.md; secciones 8.4 a 8.7 del enunciado).

Para cada rival k de la familia (m = 12) y cada contraste exploratorio:
  T1  t pareado sobre las semillas compartidas (F1 macro de prueba por semilla), con Shapiro-Wilk de las diferencias
  T2  Wilcoxon de rangos con signo sobre las mismas semillas
  T3  bootstrap pareado por bloques de 1000 flujos de prueba (10 000 réplicas) de la diferencia de F1 macro entre los
      promedios de probabilidades sobre las semillas compartidas; IC 95 % BCa (aceleración por jackknife de bloques)
Holm sobre los 12 contrastes, aparte para cada prueba; la decisión usa el máximo de los p ajustados de T1 a T3. R3 frente a
k exige además: diferencia media > delta, |delta de Cliff| >= 0,147 y límite inferior del IC 95 % > 0.
También: Friedman con Iman-Davenport sobre los 13 modelos (semillas 6001 a 6020), potencia observada y sensibilidad
(bayesiana con ROPE, exactitud balanceada y solo las 20 primeras semillas).

Uso: python experiments/analisis_confirmatorio.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

AQUI = Path(__file__).resolve().parent
RAIZ_E2 = AQUI.parents[1] / "Entregable_2_UAVIDS_2025"
sys.path.insert(0, str(RAIZ_E2))
from codigo import datos as D  # noqa: E402
from codigo import estadistica as E  # noqa: E402

E2_DEF = RAIZ_E2 / "resultados" / "corridas" / "definitiva"
E3_DEF = AQUI / "resultados" / "corridas" / "definitiva"
SALIDA = AQUI / "resultados" / "confirmatorio"
FAMILIA = ["CNN-LSTM", "Autoencoder", "RNN", "Transformer", "BiLSTM", "ResNet", "GRU", "KAN", "TabNet", "DBN", "LightGBM", "CNN 1D"]
EXPLORATORIOS = ["DNN", "LSTM", "Logística"]
DELTA, ALFA, CLIFF_MIN = 0.01, 0.05, 0.147
B_REPLICAS = 10_000


def archivo(modelo):
    return modelo.replace(" ", "_").replace("í", "i")


def corridas(modelo):
    """{semilla: (registro JSON, ruta base)} juntando las corridas del E2 y las del E3."""
    salida = {}
    for carpeta in (E2_DEF, E3_DEF):
        for p in carpeta.glob(f"definitiva_{archivo(modelo)}_*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            if "f1_eva" in r:
                salida[r["semilla"]] = (r, p.with_suffix(""))
    return salida


def prob_prueba(base):
    return np.load(str(base) + "_eva.npy")


def bca(dif_obs, replicas, jackknife):
    """IC 95 % BCa del bootstrap (Efron, 1987) con aceleración por jackknife sobre bloques."""
    z0 = stats.norm.ppf(np.clip((replicas < dif_obs).mean(), 1e-6, 1 - 1e-6))
    media_j = jackknife.mean()
    a = ((media_j - jackknife) ** 3).sum() / (6 * (((media_j - jackknife) ** 2).sum() ** 1.5) + 1e-12)
    limites = []
    for alfa in (0.025, 0.975):
        z = stats.norm.ppf(alfa)
        q = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        limites.append(float(np.quantile(replicas, q)))
    return tuple(limites)


def bootstrap_bloques_f1(y, pa, pb, bloques, semilla=42):
    """Diferencia de F1 macro (a - b) con bootstrap de bloques; devuelve observada, IC BCa y p bilateral."""
    obs = E.f1_macro_rapido(y, pa) - E.f1_macro_rapido(y, pb)
    unicos = np.unique(bloques)
    filas = [np.flatnonzero(bloques == b) for b in unicos]
    rng = np.random.default_rng(semilla)
    rep = np.empty(B_REPLICAS)
    for i in range(B_REPLICAS):
        idx = np.concatenate([filas[j] for j in rng.integers(0, len(unicos), len(unicos))])
        rep[i] = E.f1_macro_rapido(y[idx], pa[idx]) - E.f1_macro_rapido(y[idx], pb[idx])
    jack = np.array([E.f1_macro_rapido(y[np.concatenate(filas[:j] + filas[j + 1:])], pa[np.concatenate(filas[:j] + filas[j + 1:])])
                     - E.f1_macro_rapido(y[np.concatenate(filas[:j] + filas[j + 1:])], pb[np.concatenate(filas[:j] + filas[j + 1:])]) for j in range(len(filas))])
    centradas = rep - rep.mean()
    p = float((1 + (np.abs(centradas) >= abs(obs)).sum()) / (B_REPLICAS + 1))
    return obs, bca(obs, rep, jack), p


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    y = D.etiquetas()
    ids_prueba = D.heredados()["ids"]["prueba"]
    y_eva = y[ids_prueba]
    bloques = D.crudo()[1][ids_prueba]
    orig = corridas("Original")
    filas, res_brutos = [], {}
    for k in FAMILIA + EXPLORATORIOS:
        riv = corridas(k)
        semillas = sorted(set(orig) & set(riv))
        a = np.array([orig[s][0]["f1_eva"] for s in semillas])
        b = np.array([riv[s][0]["f1_eva"] for s in semillas])
        r = E.comparar_par(a, b, DELTA)
        pa = np.mean([prob_prueba(orig[s][1]) for s in semillas], 0).argmax(1)
        pb = np.mean([prob_prueba(riv[s][1]) for s in semillas], 0).argmax(1)
        obs, ic_bca, p_boot = bootstrap_bloques_f1(y_eva, pa, pb, bloques)
        bacc_a = np.array([balanced_accuracy_score(y_eva, prob_prueba(orig[s][1]).argmax(1)) for s in semillas])
        bacc_b = np.array([balanced_accuracy_score(y_eva, prob_prueba(riv[s][1]).argmax(1)) for s in semillas])
        r20 = E.comparar_par(a[:20], b[:20], DELTA) if len(semillas) > 20 else r
        filas.append({"rival": k, "familia": "confirmatoria" if k in FAMILIA else "exploratoria", "n_semillas": len(semillas),
                      "F1 original": a.mean(), "F1 rival": b.mean(), "diferencia": r["Diferencia media"], "DE diferencia": r["DE de la diferencia"],
                      "IC95 t": r["IC 95 % t"], "Shapiro p": r["Shapiro-Wilk p"], "p T1 t": r["p t pareada"], "p T2 Wilcoxon": r["p Wilcoxon"],
                      "dif F1 ensamble": obs, "IC95 BCa bloques": ic_bca, "p T3 bootstrap bloques": p_boot,
                      "delta Cliff": r["δ de Cliff"], "d Cohen": r["d de Cohen pareada"], "P(original>rival) bayes": r["P(A > B)"],
                      "P(ROPE)": r["P(ROPE)"], "P(original<rival) bayes": r["P(A < B)"], "potencia para delta": r["Potencia para δ"],
                      "MDE": r["Mínima detectable"], "dif exactitud balanceada": float((bacc_a - bacc_b).mean()),
                      "p t bacc": float(stats.ttest_rel(bacc_a, bacc_b).pvalue), "p t primeras 20": r20["p t pareada"]})
    t = pd.DataFrame(filas)
    conf = t["familia"] == "confirmatoria"
    for col in ("p T1 t", "p T2 Wilcoxon", "p T3 bootstrap bloques", "p t bacc", "p t primeras 20"):
        t.loc[conf, col + " Holm"] = E.holm(t.loc[conf, col])
        t.loc[~conf, col + " Holm"] = E.holm(t.loc[~conf, col])
    t["p máximo ajustado"] = t[["p T1 t Holm", "p T2 Wilcoxon Holm", "p T3 bootstrap bloques Holm"]].max(axis=1)
    ic_inf = t.apply(lambda r: min(r["IC95 t"][0], r["IC95 BCa bloques"][0]), axis=1)
    t["supera (R3)"] = (t["p máximo ajustado"] < ALFA) & (t["diferencia"] > DELTA) & (t["delta Cliff"].abs() >= CLIFF_MIN) & (ic_inf > 0)
    t["peor con significancia"] = (t["p máximo ajustado"] < ALFA) & (t["diferencia"] < 0)
    t["sensibilidad: conserva decisión con bacc"] = ((t["p t bacc Holm"] < ALFA) & (t["dif exactitud balanceada"] > 0)) == t["supera (R3)"]
    t.to_csv(SALIDA / "tabla_contrastes.csv", index=False, encoding="utf-8-sig")
    # Friedman sobre las semillas 6001-6020 de los 13 modelos
    modelos = ["Original"] + FAMILIA
    M = np.array([[corridas(m)[s][0]["f1_eva"] for m in modelos] for s in range(6001, 6021)])
    fr = E.friedman(M)
    rangos = dict(zip(modelos, E.rangos_medios(M).round(2).tolist()))
    pot = t.loc[conf, "potencia para delta"]
    resumen = {"R3 (supera a los 12)": bool(t.loc[conf, "supera (R3)"].all()), "contrastes que supera": t.loc[conf & t["supera (R3)"], "rival"].tolist(),
               "contrastes donde es peor con significancia": t.loc[t["peor con significancia"], "rival"].tolist(),
               "potencia conjunta observada": float(np.prod(pot)), "friedman": {k: float(v) for k, v in fr.items()}, "rangos medios (1 = mejor)": rangos,
               "F1 original medio (todas las semillas)": float(np.mean([v[0]["f1_eva"] for v in orig.values()])), "semillas original": len(orig)}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(t[["rival", "n_semillas", "F1 original", "F1 rival", "diferencia", "p T1 t Holm", "p T2 Wilcoxon Holm", "p T3 bootstrap bloques Holm",
             "delta Cliff", "supera (R3)", "peor con significancia"]].round(4).to_string(index=False))
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
