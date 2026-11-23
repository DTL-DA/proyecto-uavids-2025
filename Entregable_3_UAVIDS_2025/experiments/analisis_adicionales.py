"""Análisis de los tres conjuntos adicionales (sección 8.3 y condición 4 de la sección 8.4 del enunciado; 5.3 de protocolo.md).

Por conjunto (familia secundaria propia, m = 12, Holm dentro del conjunto y aparte para cada prueba):
  - tabla de los 13 modelos: F1 macro de prueba medio ± DE e IC 95 % sobre las semillas 6001 a 6010, y rango;
  - original frente a cada rival: T1 t pareada, T2 Wilcoxon, T3 bootstrap por bloques de 250 filas de prueba (10 000
    réplicas, IC BCa); δ de Cliff, bayesiana con ROPE ±δ, potencia y diferencia mínima detectable;
  - "peor con significancia" con la misma regla del problema principal (máximo de los p de Holm de T1 a T3 < 0,05) y,
    como lectura adicional, con T1 y T2.
Entre conjuntos (Demšar, 2006): Friedman con Iman-Davenport sobre las medias por conjunto (bloques = conjuntos), con los
3 adicionales y, como sensibilidad, con UAVIDS-2025; Wilcoxon-Holm por pares y bayesiana de rangos con signo con ROPE.
Consistencia (condición 4): el original es mejor en la métrica media en al menos 2 de 3 conjuntos y no es
significativamente peor que ningún rival en ninguno.
Además: corridas que tocan el tope de 30 épocas y dispositivo de cada corrida (CPU o GPU).

Uso: python experiments/analisis_adicionales.py   ->   experiments/resultados/adicionales/
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
E3 = AQUI.parent
sys.path[:0] = [str(AQUI), str(E3 / "training"), str(E3.parent / "Entregable_2_UAVIDS_2025")]
import analisis_confirmatorio as AC  # noqa: E402
from codigo import estadistica as E  # noqa: E402
from conjuntos_adicionales import cargar  # noqa: E402

CARPETA = AQUI / "resultados" / "corridas" / "adicionales"
SALIDA = AQUI / "resultados" / "adicionales"
CONJUNTOS = {"uav_nidd": "UAV-NIDD", "unsw_nb15": "UNSW-NB15", "unr_idd": "UNR-IDD"}
FAMILIA = AC.FAMILIA
MODELOS = ["Original"] + FAMILIA
DELTA, ALFA, B = AC.DELTA, AC.ALFA, AC.B_REPLICAS
SEMILLAS = list(range(6001, 6011))


def corridas(conjunto):
    salida = defaultdict(dict)
    for p in CARPETA.glob(f"{conjunto}_*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("pendiente_gpu") or "f1_eva" not in r:
            continue
        salida[r["modelo"]][r["semilla"]] = (r, p.with_suffix(""))
    return salida


def bootstrap_bloques(y, pa, pb, bloques, k, semilla=42):
    """Igual que analisis_confirmatorio.bootstrap_bloques_f1, con k clases del conjunto."""
    f1 = lambda v, p: E.f1_macro_rapido(v, p, k)  # noqa: E731
    obs = f1(y, pa) - f1(y, pb)
    unicos = np.unique(bloques)
    filas = [np.flatnonzero(bloques == b) for b in unicos]
    rng = np.random.default_rng(semilla)
    rep = np.empty(B)
    for i in range(B):
        idx = np.concatenate([filas[j] for j in rng.integers(0, len(unicos), len(unicos))])
        rep[i] = f1(y[idx], pa[idx]) - f1(y[idx], pb[idx])
    jack = []
    for j in range(len(filas)):
        idx = np.concatenate(filas[:j] + filas[j + 1:])
        jack.append(f1(y[idx], pa[idx]) - f1(y[idx], pb[idx]))
    p = float((1 + (np.abs(rep - rep.mean()) >= abs(obs)).sum()) / (B + 1))
    return obs, AC.bca(obs, rep, np.array(jack)), p, len(unicos)


def por_conjunto(conjunto):
    d = cargar(conjunto)
    k = int(d["y"].max()) + 1
    y_eva, bloques = d["y"][d["eva"]], d["bloques"][d["eva"]]
    c = corridas(conjunto)
    faltan = {m: sorted(set(SEMILLAS) - set(c.get(m, {}))) for m in MODELOS if set(SEMILLAS) - set(c.get(m, {}))}
    if faltan:
        raise RuntimeError(f"{conjunto}: faltan corridas {faltan}")
    tabla = []
    for m in MODELOS:
        f = np.array([c[m][s][0]["f1_eva"] for s in SEMILLAS])
        ep = [c[m][s][0].get("epocas") for s in SEMILLAS]
        disp = [c[m][s][0].get("dispositivo", "cpu") for s in SEMILLAS]
        tabla.append({"conjunto": CONJUNTOS[conjunto], "modelo": m, "semillas": len(f), "F1 macro": f.mean(), "DE": f.std(ddof=1),
                      "IC95 inf": f.mean() - stats.t.ppf(0.975, len(f) - 1) * f.std(ddof=1) / np.sqrt(len(f)),
                      "IC95 sup": f.mean() + stats.t.ppf(0.975, len(f) - 1) * f.std(ddof=1) / np.sqrt(len(f)),
                      "F1 validación": np.mean([c[m][s][0]["f1_val"] for s in SEMILLAS]),
                      "corridas en el tope de 30 épocas": None if m == "LightGBM" else sum(e == 30 for e in ep),
                      "corridas en GPU": sum(x != "cpu" for x in disp)})
    tabla = pd.DataFrame(tabla)
    tabla["rango"] = tabla["F1 macro"].rank(ascending=False).astype(int)
    a = np.array([c["Original"][s][0]["f1_eva"] for s in SEMILLAS])
    pa = np.mean([np.load(str(c["Original"][s][1]) + "_eva.npy") for s in SEMILLAS], 0).argmax(1)
    filas = []
    for rival in FAMILIA:
        b = np.array([c[rival][s][0]["f1_eva"] for s in SEMILLAS])
        r = E.comparar_par(a, b, DELTA)
        pb = np.mean([np.load(str(c[rival][s][1]) + "_eva.npy") for s in SEMILLAS], 0).argmax(1)
        obs, ic_bca, p_boot, n_bloques = bootstrap_bloques(y_eva, pa, pb, bloques, k)
        filas.append({"conjunto": CONJUNTOS[conjunto], "rival": rival, "F1 original": a.mean(), "F1 rival": b.mean(), "diferencia": r["Diferencia media"],
                      "IC95 t": r["IC 95 % t"], "Shapiro p": r["Shapiro-Wilk p"], "p T1 t": r["p t pareada"], "p T2 Wilcoxon": r["p Wilcoxon"],
                      "dif F1 ensamble": obs, "IC95 BCa bloques": ic_bca, "p T3 bootstrap bloques": p_boot, "bloques de prueba": n_bloques,
                      "delta Cliff": r["δ de Cliff"], "P(original>rival) bayes": r["P(A > B)"], "P(ROPE)": r["P(ROPE)"], "P(original<rival) bayes": r["P(A < B)"],
                      "potencia para delta": r["Potencia para δ"], "MDE": r["Mínima detectable"]})
    t = pd.DataFrame(filas)
    for col in ("p T1 t", "p T2 Wilcoxon", "p T3 bootstrap bloques"):
        t[col + " Holm"] = E.holm(t[col])
    t["p máximo ajustado"] = t[["p T1 t Holm", "p T2 Wilcoxon Holm", "p T3 bootstrap bloques Holm"]].max(axis=1)
    t["p máximo T1 y T2"] = t[["p T1 t Holm", "p T2 Wilcoxon Holm"]].max(axis=1)
    ic_inf = t.apply(lambda x: min(x["IC95 t"][0], x["IC95 BCa bloques"][0]), axis=1)
    t["supera"] = (t["p máximo ajustado"] < ALFA) & (t["diferencia"] > DELTA) & (t["delta Cliff"].abs() >= AC.CLIFF_MIN) & (ic_inf > 0)
    t["peor con significancia (T1 a T3)"] = (t["p máximo ajustado"] < ALFA) & (t["diferencia"] < 0)
    t["peor con significancia (T1 y T2)"] = (t["p máximo T1 y T2"] < ALFA) & (t["diferencia"] < 0)
    t["mejor con significancia (T1 y T2)"] = (t["p máximo T1 y T2"] < ALFA) & (t["diferencia"] > 0)
    return tabla, t


def entre_conjuntos(tablas, incluir_principal):
    medias = pd.concat(tablas).pivot(index="conjunto", columns="modelo", values="F1 macro")[MODELOS]
    if incluir_principal:
        principal = {m: np.mean([v[0]["f1_eva"] for s, v in AC.corridas(m).items() if s <= 6020]) for m in MODELOS}
        medias.loc["UAVIDS-2025"] = pd.Series(principal)
    M = medias.to_numpy()
    fr = E.friedman(M) if len(M) > 1 else {}
    rangos = dict(zip(MODELOS, E.rangos_medios(M).round(2).tolist()))
    pares = []
    for rival in FAMILIA:
        dif = medias["Original"] - medias[rival]
        p_w = stats.wilcoxon(dif, method="exact").pvalue if np.any(dif != 0) else 1.0
        pares.append({"rival": rival, "diferencias por conjunto": dif.round(4).to_dict(), "mejor en": int((dif > 0).sum()), "p Wilcoxon": float(p_w),
                      **E.bayes_rangos_signo(dif.to_numpy(), DELTA)})
    pares = pd.DataFrame(pares)
    pares["p Wilcoxon Holm"] = E.holm(pares["p Wilcoxon"])
    return {"conjuntos": medias.index.tolist(), "friedman": {k: float(v) for k, v in fr.items()}, "rangos medios (1 = mejor)": rangos,
            "pares": pares.to_dict(orient="records"),
            "p Wilcoxon mínimo alcanzable": float(2 / 2 ** len(medias)), "medias": medias.round(4).to_dict()}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    tablas, contrastes = [], []
    for c in CONJUNTOS:
        print(f"{CONJUNTOS[c]} ...", flush=True)
        tabla, t = por_conjunto(c)
        tabla.to_csv(SALIDA / f"tabla_{c}.csv", index=False, encoding="utf-8-sig")
        t.to_csv(SALIDA / f"contrastes_{c}.csv", index=False, encoding="utf-8-sig")
        tablas.append(tabla)
        contrastes.append(t)
    consistencia = {}
    for tabla, t in zip(tablas, contrastes):
        nombre = tabla["conjunto"].iloc[0]
        mejor_rival = tabla[tabla["modelo"] != "Original"].sort_values("F1 macro", ascending=False).iloc[0]
        f_orig = float(tabla.loc[tabla["modelo"] == "Original", "F1 macro"].iloc[0])
        consistencia[nombre] = {"F1 original": f_orig, "rango del original (de 13)": int(tabla.loc[tabla["modelo"] == "Original", "rango"].iloc[0]),
                                "mejor rival": mejor_rival["modelo"], "F1 mejor rival": float(mejor_rival["F1 macro"]),
                                "original mejor que todos en la media": bool(f_orig > mejor_rival["F1 macro"]),
                                "rivales significativamente mejores (T1 a T3)": t.loc[t["peor con significancia (T1 a T3)"], "rival"].tolist(),
                                "rivales significativamente mejores (T1 y T2)": t.loc[t["peor con significancia (T1 y T2)"], "rival"].tolist(),
                                "rivales significativamente peores (T1 y T2)": t.loc[t["mejor con significancia (T1 y T2)"], "rival"].tolist(),
                                "contrastes que supera (R3)": t.loc[t["supera"], "rival"].tolist()}
    n_mejor = sum(v["original mejor que todos en la media"] for v in consistencia.values())
    ninguno_peor = all(not v["rivales significativamente mejores (T1 a T3)"] for v in consistencia.values())
    resumen = {"consistencia por conjunto": consistencia, "conjuntos en que el original es el mejor en la media": n_mejor,
               "no es significativamente peor que ningún rival en ningún conjunto (T1 a T3)": ninguno_peor,
               "condición 4 de R3 (mejor en al menos 2 de 3 y nunca peor con significancia)": bool(n_mejor >= 2 and ninguno_peor),
               "entre conjuntos (3 adicionales)": entre_conjuntos(tablas, False),
               "entre conjuntos (3 adicionales + UAVIDS-2025, sensibilidad)": entre_conjuntos(tablas, True)}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    for tabla, t in zip(tablas, contrastes):
        print(tabla[["conjunto", "modelo", "F1 macro", "DE", "rango", "corridas en el tope de 30 épocas", "corridas en GPU"]].round(4).to_string(index=False))
        print(t[["rival", "diferencia", "p T1 t Holm", "p T2 Wilcoxon Holm", "p T3 bootstrap bloques Holm", "delta Cliff", "P(ROPE)",
                 "peor con significancia (T1 a T3)", "peor con significancia (T1 y T2)"]].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in resumen.items() if not k.startswith("entre")}, ensure_ascii=False, indent=1, default=float))
    for k in ("entre conjuntos (3 adicionales)", "entre conjuntos (3 adicionales + UAVIDS-2025, sensibilidad)"):
        print(k, json.dumps(resumen[k]["friedman"]), json.dumps(resumen[k]["rangos medios (1 = mejor)"], ensure_ascii=False))


if __name__ == "__main__":
    main()
