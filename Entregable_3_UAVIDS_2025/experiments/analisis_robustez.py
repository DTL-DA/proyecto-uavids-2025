"""Robustez, cambio de distribución, estabilidad, eficiencia, errores, complejidad e interpretabilidad (cuadro 5; protocolo 10 y 11).

    python experiments/analisis_robustez.py [p4]

Solo inferencia sobre artefactos y probabilidades guardadas, salvo la complejidad (tiempos medidos) y P4, que usa las
corridas de sensibilidad de training/ejecutar_robustez.py. Resultados en experiments/resultados/robustez/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "training"))
sys.path.insert(0, str(RAIZ))
import registro  # noqa: E402
from ejecutar_original import CONFIGURACION  # noqa: E402
from codigo import datos as D  # noqa: E402
from codigo import modelos as M  # noqa: E402
from src import inference as I  # noqa: E402

SALIDA = AQUI / "resultados" / "robustez"
E2_DEF = RAIZ.parent / "Entregable_2_UAVIDS_2025" / "resultados" / "corridas" / "definitiva"
E3_DEF = AQUI / "resultados" / "corridas" / "definitiva"
FAMILIA = ["Original", "CNN-LSTM", "Autoencoder", "RNN", "Transformer", "BiLSTM", "ResNet", "GRU", "KAN", "TabNet", "DBN", "LightGBM", "CNN 1D"]
rng = np.random.default_rng(2026)


def f1(y, p):
    return float(f1_score(y, p, average="macro"))


def prob_desde_z(z, modelo):
    """Probabilidades sin calibrar a partir de la entrada ya transformada y estandarizada."""
    salidas = I.sesion(modelo).run(None, {"x": z.astype(np.float32)})
    if I.metadatos(modelo)["salida"]["nombre"] == "logits":
        a = np.asarray(salidas[0], dtype=np.float64)
        e = np.exp(a - a.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    return np.asarray(salidas[1] if len(salidas) > 1 else salidas[0])


def corridas(modelo):
    regs = {}
    for carpeta in (E2_DEF, E3_DEF):
        for p in carpeta.glob(f"definitiva_{modelo.replace(' ', '_')}_*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            if "f1_eva" in r:
                regs[r["semilla"]] = (r, p.with_suffix(""))
    return regs


def perturbaciones(df_prueba, y):
    crudo = df_prueba[I.ATRIBUTOS].to_numpy(dtype=np.float64)
    filas = []
    for m in FAMILIA:
        z0 = I.preprocesar(crudo, m).astype(np.float64)
        base = f1(y, prob_desde_z(z0, m).argmax(1))
        filas.append({"modelo": m, "perturbacion": "ninguna", "severidad": 0, "f1": base})
        for s in (0.1, 0.3, 0.5):
            z = z0 + rng.normal(0, s, z0.shape)
            filas.append({"modelo": m, "perturbacion": "ruido gaussiano (σ en el espacio estandarizado)", "severidad": s, "f1": f1(y, prob_desde_z(z, m).argmax(1))})
        for q in (0.10, 0.25, 0.50):
            z = z0.copy()
            z[rng.random(z.shape) < q] = 0.0  # faltante imputado con la media del entrenamiento
            filas.append({"modelo": m, "perturbacion": "atributos faltantes imputados con la media", "severidad": q, "f1": f1(y, prob_desde_z(z, m).argmax(1))})
        for q in (0.01, 0.05, 0.10):
            x = crudo.copy()
            filas_out = rng.random(len(x)) < q
            cols = rng.integers(0, x.shape[1], filas_out.sum())
            x[np.flatnonzero(filas_out), cols] *= 10
            filas.append({"modelo": m, "perturbacion": "valores atípicos (un atributo ×10)", "severidad": q, "f1": f1(y, prob_desde_z(I.preprocesar(x, m), m).argmax(1))})
    t = pd.DataFrame(filas)
    t["caída"] = t.groupby("modelo")["f1"].transform(lambda s: s.iloc[0]) - t["f1"]
    return t


def cambio_distribucion(y, fid):
    segmento = (fid >= 108071) & (fid <= 109045)
    filas = []
    for m in FAMILIA:
        aciertos = [float((np.load(str(base) + "_eva.npy").argmax(1)[segmento] == y[segmento]).mean()) for _, base in corridas(m).values()]
        resto = [f1(y[~segmento], np.load(str(base) + "_eva.npy").argmax(1)[~segmento]) for _, base in corridas(m).values()]
        filas.append({"modelo": m, "acierto en el escenario de Sybil aislado (media)": np.mean(aciertos), "DE": np.std(aciertos, ddof=1),
                      "F1 macro sin el escenario": np.mean(resto), "semillas": len(aciertos), "filas del escenario": int(segmento.sum())})
    return pd.DataFrame(filas)


def estabilidad():
    filas = [{"modelo": m, "semilla": s, "f1": r["f1_eva"]} for m in FAMILIA for s, (r, _) in corridas(m).items()]
    t = pd.DataFrame(filas)
    resumen = t.groupby("modelo")["f1"].agg(["mean", "std", "count"])
    resumen["CV (%)"] = 100 * resumen["std"] / resumen["mean"]
    kw = stats.kruskal(*[g["f1"].to_numpy() for _, g in t.groupby("modelo")])
    return t, resumen, {"H": float(kw.statistic), "p": float(kw.pvalue)}


def errores(y):
    preds = {}
    for m in FAMILIA:
        probs = [np.load(str(base) + "_eva.npy") for s, (_, base) in corridas(m).items() if s <= 6020]
        preds[m] = np.mean(probs, 0).argmax(1)
    err = {m: preds[m] != y for m in FAMILIA}
    compartidos = pd.DataFrame({"rival": FAMILIA[1:], "errores del original también cometidos por el rival (%)":
                                [100 * float((err["Original"] & err[m]).sum() / max(err["Original"].sum(), 1)) for m in FAMILIA[1:]],
                                "errores que comete solo el original (%)": [100 * float((err["Original"] & ~err[m]).sum() / max(err["Original"].sum(), 1)) for m in FAMILIA[1:]]})
    todos = np.all([err[m] for m in FAMILIA], axis=0)
    por_clase = pd.DataFrame({m: f1_score(y, preds[m], average=None, labels=range(5)) for m in FAMILIA}, index=D.CLASES).T
    return compartidos, por_clase, {"flujos que fallan los 13 modelos (%)": 100 * float(todos.mean()), "errores del original": int(err["Original"].sum())}


def complejidad():
    import torch

    registro.registrar(**CONFIGURACION)
    filas = []
    for d in (8, 16, 32, 64):
        m = M.CATALOGO["Original"]["clase"](d=d) if d != 16 else M.construir("Original")
        m.eval()
        for n in (1, 64, 1024, 16384):
            x = torch.randn(n, d)
            with torch.no_grad():
                for _ in range(3):
                    m(x)
                t0 = time.perf_counter()
                for _ in range(10):
                    m(x)
            filas.append({"d": d, "lote": n, "parametros": sum(p.numel() for p in m.parameters()), "ms por lote": 1000 * (time.perf_counter() - t0) / 10,
                          "µs por flujo": 1e6 * (time.perf_counter() - t0) / 10 / n})
    return pd.DataFrame(filas)


def shapley(modelo, x_expl, fondo, permutaciones=32):
    """Valores de Shapley por muestreo de permutaciones (Štrumbelj y Kononenko, 2014) de la probabilidad de la clase
    predicha, con distribución de referencia `fondo`; implementación propia, sin dependencias externas."""
    n, d = x_expl.shape
    p_obj = prob_desde_z(x_expl, modelo)
    c = p_obj.argmax(1)
    phi = np.zeros((n, d))
    for _ in range(permutaciones):
        orden = rng.permutation(d)
        ref = fondo[rng.integers(0, len(fondo), n)]
        actual = ref.copy()
        prev = prob_desde_z(actual, modelo)[np.arange(n), c]
        for j in orden:
            actual[:, j] = x_expl[:, j]
            nuevo = prob_desde_z(actual, modelo)[np.arange(n), c]
            phi[:, j] += nuevo - prev
            prev = nuevo
    return phi / permutaciones, c


def interpretabilidad(df, y_eva):
    h = D.heredados()
    prueba = df[df["particion"] == "prueba"]
    idx = np.concatenate([rng.choice(np.flatnonzero(prueba["clase"].to_numpy() == c), 60, replace=False) for c in D.CLASES])
    crudo = prueba[I.ATRIBUTOS].to_numpy()[idx]
    fondo_crudo = df[df["particion"] == "entrenamiento"][I.ATRIBUTOS].sample(50, random_state=0).to_numpy()
    filas, importancia = [], {}
    for m in ("Original", "LightGBM"):
        x, fondo = I.preprocesar(crudo, m).astype(np.float64), I.preprocesar(fondo_crudo, m).astype(np.float64)
        phi, c = shapley(m, x, fondo)
        importancia[m] = np.abs(phi).mean(0)
        # Fidelidad por borrado: reemplazar los k atributos más importantes de cada flujo por la mediana del fondo
        mediana = np.median(fondo, 0)
        p0 = prob_desde_z(x, m)[np.arange(len(x)), c]
        for k in (1, 3, 5):
            top = np.argsort(-np.abs(phi), 1)[:, :k]
            azar = np.array([rng.permutation(16)[:k] for _ in range(len(x))])
            xb, xa = x.copy(), x.copy()
            for i in range(len(x)):
                xb[i, top[i]] = mediana[top[i]]
                xa[i, azar[i]] = mediana[azar[i]]
            filas.append({"modelo": m, "k atributos borrados": k, "caída de prob. (más importantes)": float((p0 - prob_desde_z(xb, m)[np.arange(len(x)), c]).mean()),
                          "caída de prob. (al azar)": float((p0 - prob_desde_z(xa, m)[np.arange(len(x)), c]).mean())})
    imp = pd.DataFrame(importancia, index=I.ATRIBUTOS)
    rho = stats.spearmanr(imp["Original"], imp["LightGBM"])
    return imp, pd.DataFrame(filas), {"spearman_original_vs_lightgbm": float(rho.statistic), "p": float(rho.pvalue)}


def estabilidad_explicaciones(df):
    """Estabilidad entre semillas del original: correlación de Spearman de la importancia global (|φ| medio) entre 3 semillas."""
    import torch

    registro.registrar(**CONFIGURACION)
    prueba = df[df["particion"] == "prueba"].sample(150, random_state=1)
    x = I.preprocesar(prueba[I.ATRIBUTOS].to_numpy(), "Original").astype(np.float32)
    fondo = I.preprocesar(df[df["particion"] == "entrenamiento"][I.ATRIBUTOS].sample(50, random_state=0).to_numpy(), "Original").astype(np.float32)
    imps = []
    for s in (6001, 6002, 6003):
        m = M.construir("Original")
        m.load_state_dict(torch.load(E3_DEF / f"definitiva_Original_{s}.pt", map_location="cpu"))
        m.eval()
        f = lambda z: m(torch.from_numpy(z.astype(np.float32))).exp().detach().numpy()  # noqa: E731
        c = f(x).argmax(1)
        phi = np.zeros(x.shape)
        for _ in range(16):
            orden = rng.permutation(16)
            actual = fondo[rng.integers(0, len(fondo), len(x))].copy()
            prev = f(actual)[np.arange(len(x)), c]
            for j in orden:
                actual[:, j] = x[:, j]
                nuevo = f(actual)[np.arange(len(x)), c]
                phi[:, j] += nuevo - prev
                prev = nuevo
        imps.append(np.abs(phi).mean(0))
    rhos = [stats.spearmanr(imps[a], imps[b]).statistic for a, b in ((0, 1), (0, 2), (1, 2))]
    return {"spearman_entre_semillas": [float(r) for r in rhos], "media": float(np.mean(rhos))}


def p4():
    """P4: la variante con menor cota K se degrada menos ante ruido (corridas de sensibilidad de tau_min)."""
    import torch

    h = D.heredados()
    ids = h["ids"]["validacion"]
    x = torch.from_numpy(h["X"][ids])
    y = h["y"][ids]
    filas = []
    for p in sorted((AQUI / "resultados" / "corridas" / "sensibilidad").glob("sensibilidad_Original_*_tau_min*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        tau = float(p.stem.split("tau_min")[1])
        registro.registrar(**{**CONFIGURACION, "tau_min": tau})
        m = M.construir("Original")
        m.load_state_dict(torch.load(p.with_suffix(".pt"), map_location="cpu"))
        m.eval()
        g = torch.Generator().manual_seed(r["semilla"])
        with torch.no_grad():
            base = f1(y, m(x).argmax(1).numpy())
            fila = {"tau_min": tau, "semilla": r["semilla"], "cota K": m.cota_lipschitz(), "f1_val": base}
            for s in (0.1, 0.3, 0.5):
                fila[f"caída con ruido σ={s}"] = base - f1(y, m(x + s * torch.randn(x.shape, generator=g)).argmax(1).numpy())
        filas.append(fila)
    return pd.DataFrame(filas)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAIZ / "data" / "uavids.parquet")
    h = D.heredados()
    y = h["y"][h["ids"]["prueba"]]
    fid = D.crudo()[0]["FlowID"].to_numpy()[h["ids"]["prueba"]]
    prueba = df.set_index("flow_id").loc[fid].reset_index()
    if "p4" in sys.argv:
        p4().to_csv(SALIDA / "p4_sensibilidad_tau.csv", index=False, encoding="utf-8-sig")
        return
    pert = perturbaciones(prueba, y)
    pert.to_csv(SALIDA / "perturbaciones.csv", index=False, encoding="utf-8-sig")
    cambio_distribucion(y, fid).to_csv(SALIDA / "cambio_distribucion.csv", index=False, encoding="utf-8-sig")
    t, res, kw = estabilidad()
    t.to_csv(SALIDA / "estabilidad_por_semilla.csv", index=False, encoding="utf-8-sig")
    res.to_csv(SALIDA / "estabilidad_resumen.csv", encoding="utf-8-sig")
    comp, por_clase, extra = errores(y)
    comp.to_csv(SALIDA / "errores_compartidos.csv", index=False, encoding="utf-8-sig")
    por_clase.to_csv(SALIDA / "f1_por_clase.csv", encoding="utf-8-sig")
    complejidad().to_csv(SALIDA / "complejidad.csv", index=False, encoding="utf-8-sig")
    imp, fidelidad, rho = interpretabilidad(df, y)
    imp.to_csv(SALIDA / "importancia_shapley.csv", encoding="utf-8-sig")
    fidelidad.to_csv(SALIDA / "fidelidad_borrado.csv", index=False, encoding="utf-8-sig")
    est = estabilidad_explicaciones(df)
    resumen = {"kruskal_wallis": kw, **extra, "interpretabilidad": rho, "estabilidad_explicaciones": est}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    print(pert.pivot_table(index="modelo", columns=["perturbacion", "severidad"], values="caída").round(3).to_string())


if __name__ == "__main__":
    main()
