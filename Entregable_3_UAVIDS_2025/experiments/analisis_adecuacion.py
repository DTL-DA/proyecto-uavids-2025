"""Condiciones 5 y 6 de R3 (sección 8.4 del enunciado) y tablas A y B del modelo original (sección 6 de protocolo.md).

Condición 5: criterio de adecuación de la sección 7.2 del E2, con la MISMA implementación que el E2 aplicó a sus modelos
(herramientas/cuaderno/s7_ruta.py y s4b_evaluacion.py), sobre las semillas 6001 a 6020:
  (i)   supera al trivial: t de una muestra sobre F1_original − F1_trivial por semilla, p < 0,05 y mejora > δ;
  (ii)  brecha g = F1_ent − F1_val menor que δg = 0,05: Wilcoxon unilateral sobre g − δg (H1: g < δg);
  (iii) ECE tras temperature scaling ajustado en validación < 0,05 (umbral exploratorio, fijado en el E2);
  (iv)  validación consistente con prueba: el IC 95 % por bootstrap de bloques (2000 réplicas, bloques de validación y
        de prueba por separado, semillas 1 y 2 como en el E2) de F1_val − F1_prueba contiene al cero.
Además (tablas A y B): permutación de etiquetas (200), optimismo por bootstrap (20), CV entre semillas, curva de
aprendizaje (pendiente por log n), Hosmer-Lemeshow y pendiente de calibración por clase (uno contra todos).
Condición 6 (complemento): t de Nadeau-Bengio y F de 5×2 cv frente a LightGBM y CNN 1D en las 10 particiones del E2.
También: sensibilidad a τ_min, T y γ (validación) y curvas de aprendizaje de original, LightGBM y CNN-LSTM.

Uso: python experiments/analisis_adecuacion.py   ->   experiments/resultados/adecuacion/
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
from codigo import analisis as AN  # noqa: E402

E2R = AC.RAIZ_E2 / "resultados"
E2C = E2R / "corridas"
E3C = AQUI / "resultados" / "corridas"
SALIDA = AQUI / "resultados" / "adecuacion"
DELTA, ALFA, DELTA_G, UMBRAL_ECE = AC.DELTA, AC.ALFA, 0.05, 0.05
SEMILLAS = range(6001, 6021)


def leer(carpeta, patron):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(carpeta.glob(patron))]


def adecuacion():
    y = D.etiquetas()
    ids = D.heredados()["ids"]
    id_val, id_pru = ids["validacion"], ids["prueba"]
    bloques = D.crudo()[1]
    orig = AC.corridas("Original")
    triv = AC.corridas("Trivial")
    regs = [orig[s] for s in SEMILLAS]
    f1_eva = np.array([r["f1_eva"] for r, _ in regs])
    f1_val = np.array([r["f1_val"] for r, _ in regs])
    f1_ent = np.array([r["f1_ent"] for r, _ in regs])
    # (i) trivial
    f1_triv = np.array([triv[s][0]["f1_eva"] if s in triv else triv[min(triv)][0]["f1_eva"] for s in SEMILLAS])
    p_i = stats.ttest_1samp(f1_eva - f1_triv, 0).pvalue
    c1 = bool(p_i < ALFA and f1_eva.mean() - f1_triv.mean() > DELTA)
    # (ii) brecha
    g = f1_ent - f1_val
    p_g = stats.wilcoxon(g - DELTA_G, alternative="less").pvalue
    c2 = bool(p_g < ALFA)
    # (iii) calibración con temperatura, Hosmer-Lemeshow y pendiente por clase
    ece, ece_t, temps, hl, pend = [], [], [], [], []
    for r, base in regs:
        p_v, p_e = np.load(str(base) + "_val.npy"), np.load(str(base) + "_eva.npy")
        T = E.escalar_temperatura(E.logits_de(p_v), y[id_val])
        p_c = E.suavizar(p_e, T)
        temps.append(T)
        ece.append(E.ece(p_e, y[id_pru]))
        ece_t.append(E.ece(p_c, y[id_pru]))
        hl.append([E.hosmer_lemeshow(p_c[:, k], (y[id_pru] == k).astype(float))["p"] for k in range(5)])
        pend.append([E.pendiente_intercepto(p_c[:, k], (y[id_pru] == k).astype(float))["pendiente"] for k in range(5)])
    c3 = bool(np.mean(ece_t) < UMBRAL_ECE)
    # (iv) validación frente a prueba con bootstrap por bloques
    W_val = AN.pesos_bootstrap(len(np.unique(bloques[id_val])), semilla=1)
    W_pru = AN.pesos_bootstrap(len(np.unique(bloques[id_pru])), semilla=2)
    boot_val, boot_pru = [], []
    for r, base in regs:
        p_v, p_e = np.load(str(base) + "_val.npy"), np.load(str(base) + "_eva.npy")
        boot_val.append(AN.f1_desde_conteos(W_val @ AN.confusion_bloques(y[id_val], p_v.argmax(1), bloques[id_val])))
        boot_pru.append(AN.f1_desde_conteos(W_pru @ AN.confusion_bloques(y[id_pru], p_e.argmax(1), bloques[id_pru])))
    dif = np.mean(boot_val, 0) - np.mean(boot_pru, 0)
    ic_vp = E.ic_percentil(dif)
    c4 = bool(ic_vp[0] <= 0 <= ic_vp[1])
    fila = {"Modelo": "Original", "(i) Supera al trivial": "sí" if c1 else "no", "p (i)": p_i, "F1 trivial": f1_triv.mean(),
            "(ii) Brecha < δg": "sí" if c2 else "no", "Brecha g": g.mean(), "p Wilcoxon (g < δg)": p_g,
            "(iii) ECE con temperatura < 0,05": "sí" if c3 else "no", "ECE": np.mean(ece), "ECE con T": np.mean(ece_t), "T media": np.mean(temps),
            "Pendiente media": float(np.mean(pend)), "Hosmer-Lemeshow p por clase (mediana sobre semillas)": dict(zip(D.CLASES, np.median(hl, 0).round(4).tolist())),
            "(iv) Prueba consistente con validación": "sí" if c4 else "no", "Validación − prueba": float(f1_val.mean() - f1_eva.mean()),
            "IC 95 % por bloques": tuple(float(v) for v in ic_vp), "Adecuado": "sí" if all((c1, c2, c3, c4)) else "no",
            "CV entre semillas (%)": float(100 * f1_eva.std(ddof=1) / f1_eva.mean())}
    return fila


def permutacion():
    perm = np.array([r["f1_eva"] for r in leer(E3C / "permutacion", "permutacion_Original_*.json") if "f1_eva" in r])
    obs = AC.corridas("Original")[6001][0]["f1_eva"]
    return {"permutaciones": int(len(perm)), "F1 observado (semilla 6001)": obs, "F1 permutado medio": float(perm.mean()),
            "F1 permutado máximo": float(perm.max()), "p": float((1 + (perm >= obs).sum()) / (len(perm) + 1))}


def optimismo():
    filas = {}
    for modelo, carpeta in (("Original", E3C), ("LightGBM", E2C), ("CNN 1D", E2C), ("DNN", E2C)):
        regs = [r for r in leer(carpeta / "bootstrap", f"bootstrap_{modelo.replace(' ', '_')}_*.json") if "f1_eva" in r]
        o = np.array([r["f1_ent"] - r["f1_eva"] for r in regs])
        filas[modelo] = {"réplicas": len(o), "optimismo medio": float(o.mean()), "IC 95 %": tuple(np.percentile(o, [2.5, 97.5]).round(4).tolist())}
    return filas


def cv5x2():
    def por_particion(modelo, carpeta):
        return {r["particion"]: r["f1_eva"] for r in leer(carpeta / "cv5x2", f"cv5x2_{modelo.replace(' ', '_')}_*.json") if "f1_eva" in r}
    o = por_particion("Original", E3C)
    salida = {}
    for rival in ("LightGBM", "CNN 1D"):
        r = por_particion(rival, E2C)
        claves = sorted(set(o) & set(r))
        d = np.array([o[k] - r[k] for k in claves])
        part = np.load(E2R / "particiones" / f"{claves[0]}.npz")
        nb = E.nadeau_bengio(d, len(part["eva"]), len(part["ent"]))
        f = E.f_5x2cv(d) if len(d) == 10 else {"F": None, "p": None}
        salida[rival] = {"particiones": len(d), "diferencia media": float(d.mean()), "t Nadeau-Bengio": float(nb["t"]), "p Nadeau-Bengio": float(nb["p"]),
                         "F 5x2cv": f["F"] and float(f["F"]), "p F 5x2cv": f["p"] and float(f["p"])}
    return salida


def sensibilidad():
    filas = []
    for r in leer(E3C / "sensibilidad", "sensibilidad_Original_*.json"):
        filas.append({"variante": r["id"].split(f"_{r['semilla']}_")[-1], "semilla": r["semilla"], "f1_val": r.get("f1_val"), "f1_ent": r.get("f1_ent"), "epocas": r.get("epocas")})
    t = pd.DataFrame(filas).groupby("variante").agg(semillas=("semilla", "count"), f1_val=("f1_val", "mean"), de=("f1_val", "std"), f1_ent=("f1_ent", "mean"), epocas=("epocas", "mean"))
    return t.reset_index()


def curvas():
    filas = []
    for modelo, carpeta in (("Original", E3C), ("LightGBM", E3C), ("CNN-LSTM", E3C)):
        for r in leer(carpeta / "curvas", f"curvas_{modelo.replace(' ', '_')}_*.json"):
            filas.append({"modelo": modelo, "fraccion": r["fraccion_ent"], "semilla": r["semilla"], "f1_val": r.get("f1_val"), "f1_ent": r.get("f1_ent")})
    t = pd.DataFrame(filas)
    resumen = t.groupby(["modelo", "fraccion"]).agg(semillas=("semilla", "count"), f1_val=("f1_val", "mean"), de=("f1_val", "std"), f1_ent=("f1_ent", "mean")).reset_index()
    n_ent = len(D.heredados()["ids"]["entrenamiento"])
    pendientes = {}
    for m, sub in t.groupby("modelo"):
        ajuste = stats.linregress(np.log(sub["fraccion"] * n_ent), sub["f1_val"])
        pendientes[m] = {"pendiente por log n": float(ajuste.slope), "p": float(ajuste.pvalue)}
    return resumen, pendientes


def interno_externo():
    """Aproximación de la validación anidada usada en el E2 (tabla B): mejor F1 de validación del ajuste − F1 medio de prueba."""
    ajuste = max(r["f1_val"] for r in leer(E3C / "ajuste", "ajuste_Original_*.json") if "f1_val" in r)
    prueba = float(np.mean([AC.corridas("Original")[s][0]["f1_eva"] for s in SEMILLAS]))
    e2 = pd.read_csv(E2R / "tabla_b.csv", encoding="utf-8-sig").set_index("Modelo")["Interno (ajuste) − externo (prueba)"]
    return {"Original": {"mejor F1 de validación en el ajuste": ajuste, "F1 medio de prueba": prueba, "interno − externo": ajuste - prueba},
            "E2 (mismo cálculo)": e2.round(4).to_dict()}


def _conf_bloques(v, p, b, K):
    unicos, inversa = np.unique(b, return_inverse=True)
    M = np.zeros((len(unicos), K * K))
    np.add.at(M, (inversa, K * v + p), 1)
    return M


def _f1_conteos(C, K):
    C = C.reshape(*C.shape[:-1], K, K)
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    f1 = 2 * tp / np.maximum(C.sum(-2) + C.sum(-1), 1)
    presentes = C.sum(-1) > 0
    return (f1 * presentes).sum(-1) / np.maximum(presentes.sum(-1), 1)


def adecuacion_adicionales():
    """Los mismos diagnósticos en cada conjunto adicional (cuadro 4, fila "Varios conjuntos de datos")."""
    sys.path.insert(0, str(AQUI.parent / "training"))
    from conjuntos_adicionales import cargar

    carpeta = E3C / "diagnostico_adicionales"
    salida = {}
    for c, nombre in (("uav_nidd", "UAV-NIDD"), ("unsw_nb15", "UNSW-NB15"), ("unr_idd", "UNR-IDD")):
        d = cargar(c)
        y, K = d["y"], int(d["y"].max()) + 1
        yv, ye, bv, be = y[d["val"]], y[d["eva"]], d["bloques"][d["val"]], d["bloques"][d["eva"]]
        regs = [json.loads((carpeta / f"{c}_Original_{s}.json").read_text(encoding="utf-8")) for s in range(6001, 6011)]
        originales = [json.loads((E3C / "adicionales" / f"{c}_Original_{s}.json").read_text(encoding="utf-8")) for s in range(6001, 6011)]
        f1_ent, f1_val, f1_eva = (np.array([r[k] for r in regs]) for k in ("f1_ent", "f1_val", "f1_eva"))
        mayoritaria = np.bincount(y[d["ent"]]).argmax()
        f1_triv = float(np.mean(2 * (ye == mayoritaria).sum() / ((ye == mayoritaria).sum() + len(ye)) / len(np.unique(ye))))
        g = f1_ent - f1_val
        ece, ece_t, pend = [], [], []
        W_v, W_e = AN.pesos_bootstrap(len(np.unique(bv)), semilla=1), AN.pesos_bootstrap(len(np.unique(be)), semilla=2)
        bv_boot, be_boot = [], []
        for s in range(6001, 6011):
            pv, pe = np.load(carpeta / f"{c}_Original_{s}_val.npy"), np.load(carpeta / f"{c}_Original_{s}_eva.npy")
            T = E.escalar_temperatura(E.logits_de(pv), yv)
            pc = E.suavizar(pe, T)
            ece.append(E.ece(pe, ye))
            ece_t.append(E.ece(pc, ye))
            pend.append(np.mean([E.pendiente_intercepto(pc[:, k], (ye == k).astype(float))["pendiente"] for k in np.unique(ye)]))
            bv_boot.append(_f1_conteos(W_v @ _conf_bloques(yv, pv.argmax(1), bv, K), K))
            be_boot.append(_f1_conteos(W_e @ _conf_bloques(ye, pe.argmax(1), be, K), K))
        ic = E.ic_percentil(np.mean(bv_boot, 0) - np.mean(be_boot, 0))
        perm = np.array([json.loads(p.read_text(encoding="utf-8"))["f1_eva"] for p in carpeta.glob(f"{c}_Original_6001_perm*.json")])
        obs = regs[0]["f1_eva"]
        c1 = bool(stats.ttest_1samp(f1_eva - f1_triv, 0).pvalue < ALFA and f1_eva.mean() - f1_triv > DELTA)
        c2 = bool(stats.wilcoxon(g - DELTA_G, alternative="less").pvalue < ALFA)
        c3 = bool(np.mean(ece_t) < UMBRAL_ECE)
        c4 = bool(ic[0] <= 0 <= ic[1])
        salida[nombre] = {"reproducción: F1 de prueba idéntico al registrado": bool(all(a["f1_eva"] == b["f1_eva"] for a, b in zip(regs, originales))),
                          "F1 prueba": float(f1_eva.mean()), "F1 trivial (clase mayoritaria)": f1_triv, "(i) supera al trivial": c1,
                          "brecha g": float(g.mean()), "p Wilcoxon (g < δg)": float(stats.wilcoxon(g - DELTA_G, alternative="less").pvalue), "(ii) brecha < δg": c2,
                          "ECE": float(np.mean(ece)), "ECE con T": float(np.mean(ece_t)), "(iii) ECE con T < 0,05": c3, "pendiente media de calibración": float(np.mean(pend)),
                          "validación − prueba": float(f1_val.mean() - f1_eva.mean()), "IC 95 % por bloques": tuple(float(v) for v in ic), "(iv) prueba consistente con validación": c4,
                          "adecuado": bool(c1 and c2 and c3 and c4),
                          "permutación": {"B": int(len(perm)), "F1 observado (6001)": obs, "F1 permutado máximo": float(perm.max()) if len(perm) else None,
                                          "p": float((1 + (perm >= obs).sum()) / (len(perm) + 1)) if len(perm) else None}}
    return salida


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    fila = adecuacion()
    e2 = pd.read_csv(E2R / "adecuacion.csv", encoding="utf-8-sig")
    tabla = pd.concat([pd.DataFrame([{k: fila[k] for k in e2.columns if k in fila}]), e2], ignore_index=True)
    tabla.to_csv(SALIDA / "adecuacion_con_e2.csv", index=False, encoding="utf-8-sig")
    sens = sensibilidad()
    sens.to_csv(SALIDA / "sensibilidad.csv", index=False, encoding="utf-8-sig")
    cur, pend = curvas()
    cur.to_csv(SALIDA / "curvas.csv", index=False, encoding="utf-8-sig")
    resumen = {"adecuación del original (sección 7.2 del E2)": fila, "permutación de etiquetas": permutacion(), "optimismo por bootstrap": optimismo(),
               "Nadeau-Bengio y F 5x2cv (condición 6)": cv5x2(), "curva de aprendizaje": pend, "validación anidada (aproximación del E2)": interno_externo()}
    if len(list((E3C / "diagnostico_adicionales").glob("*_perm*.json"))) >= 600:
        resumen["diagnósticos en cada conjunto adicional"] = adecuacion_adicionales()
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tabla.to_string(index=False))
    print(sens.round(4).to_string(index=False))
    print(cur.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in resumen.items()}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
