"""Precalcula todo lo que el dashboard lee (el dashboard no entrena ni usa PyTorch).

    python training/preparar_dashboard.py [--latencia]

Genera:
  data/uavids.parquet                    atributos crudos, etiqueta, partición, bloque y marca del escenario aislado
  artifacts/dashboard/calibracion.json   temperatura y cuantil conforme LAC al 90 % (ajustados en VALIDACIÓN), ECE antes y
                                         después, y la matriz de precisión y el umbral p99 de Mahalanobis (fuera de dominio)
  artifacts/dashboard/diagnosticos.json  matriz de confusión, F1 por clase y curva de confiabilidad en prueba (corrida exportada)
  artifacts/dashboard/lideres.csv        tabla de líderes ALIMENTADA DESDE LA TABLA MAESTRA (experiments/resultados/tabla_maestra.csv,
                                         fase definitiva de UAVIDS-2025): F1 macro medio ± DE de todas las semillas, IC 95 %, ECE,
                                         parámetros, FLOPs, latencia ONNX (con --latencia) y tiempo de entrenamiento
  artifacts/dashboard/adicionales.csv, adicionales_contrastes.csv, robustez.csv y balance_r3.json   conjuntos adicionales,
                                         perturbaciones y balance de las seis condiciones de R3 (capítulo 8 del libro)
Requiere haber corrido antes experiments/tabla_maestra.py y los análisis de experiments/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(RAIZ))
import exportar_onnx as X  # noqa: E402
from codigo import datos as D  # noqa: E402
from codigo import estadistica as E  # noqa: E402
from codigo import modelos as M  # noqa: E402
from src import inference as I  # noqa: E402

SALIDA = RAIZ / "artifacts" / "dashboard"
TABLA_MAESTRA = RAIZ / "experiments" / "resultados" / "tabla_maestra.csv"
GRUPO = {**{m: "Top 10" for m in ["CNN-LSTM", "Autoencoder", "RNN", "Transformer", "BiLSTM", "ResNet", "GRU", "KAN", "TabNet", "DBN"]},
         "LightGBM": "Baseline fuerte", "CNN 1D": "Baseline del E1", "DNN": "Base del E1", "LSTM": "Base del E1", "Logística": "Línea base", "Original": "Modelo original"}
FAMILIA = {**{k: v["familia"] for k, v in M.CATALOGO.items()}, "LightGBM": "Árboles potenciados", "Logística": "Lineal", "Original": "Mezcla con compuerta"}


def datos_parquet():
    datos, bloques, _ = D.crudo()
    h = D.heredados()
    parte = np.full(len(datos), "excluida (duplicada)", dtype=object)
    for nombre, clave in (("entrenamiento", "entrenamiento"), ("validación", "validacion"), ("prueba", "prueba")):
        parte[h["ids"][clave]] = nombre
    fid = datos["FlowID"].to_numpy()
    aislado = ((fid >= 94285) & (fid <= 94979)) | ((fid >= 108071) & (fid <= 109045))
    t = datos[D.ATRIBUTOS].astype("float32").copy()
    t["clase"] = [D.CLASES[i] for i in D.etiquetas()]
    t["particion"], t["bloque"], t["flow_id"], t["escenario_aislado"] = parte, bloques.astype("int32"), fid.astype("int32"), aislado
    (RAIZ / "data").mkdir(exist_ok=True)
    t.to_parquet(RAIZ / "data" / "uavids.parquet", index=False)
    return t


def corridas(modelo):
    """Todas las semillas definitivas del modelo: las del E2 y las agregadas en el E3 por el análisis de potencia."""
    regs = {}
    for carpeta in (X.E2_DEF, X.E3_DEF):
        for p in carpeta.glob(f"definitiva_{X.archivo(modelo)}_*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            if "f1_eva" in r:
                regs[r["semilla"]] = r
    return list(regs.values())


def calibracion_y_diagnosticos(modelos):
    h = D.heredados()
    y_val, y_eva = h["y"][h["ids"]["validacion"]], h["y"][h["ids"]["prueba"]]
    cal, diag = {}, {}
    for m in modelos:
        r, base = X.elegir_corrida(m)
        p_val, p_eva = np.load(str(base) + "_val.npy"), np.load(str(base) + "_eva.npy")
        T = float(E.escalar_temperatura(E.logits_de(p_val), y_val))
        pv, pe = E.suavizar(p_val, T), E.suavizar(p_eva, T)
        q = float(np.quantile(1 - pv[np.arange(len(y_val)), y_val], np.ceil((len(y_val) + 1) * 0.9) / len(y_val)))
        conj = (1 - pe) <= q
        cal[m] = {"temperatura": T, "cuantil_lac_90": q, "ece_antes": float(E.ece(p_eva, y_eva)), "ece_despues": float(E.ece(pe, y_eva)),
                  "cobertura_prueba": float(conj[np.arange(len(y_eva)), y_eva].mean()), "tamano_medio_conjunto": float(conj.sum(1).mean())}
        pred = pe.argmax(1)
        conf = pd.crosstab(pd.Series(y_eva, name="real"), pd.Series(pred, name="pred")).reindex(index=range(5), columns=range(5), fill_value=0)
        bins = np.linspace(0, 1, 16)
        conf_max = pe.max(1)
        acierto = pred == y_eva
        curva = [{"confianza": float(conf_max[(conf_max > a) & (conf_max <= b)].mean()), "exactitud": float(acierto[(conf_max > a) & (conf_max <= b)].mean()),
                  "n": int(((conf_max > a) & (conf_max <= b)).sum())} for a, b in zip(bins[:-1], bins[1:]) if ((conf_max > a) & (conf_max <= b)).any()]
        from sklearn.metrics import f1_score

        diag[m] = {"semilla": r["semilla"], "confusion": conf.to_numpy().tolist(), "f1_por_clase": f1_score(y_eva, pred, average=None, labels=range(5)).tolist(),
                   "confiabilidad": curva}
    # fuera de dominio: Mahalanobis en el espacio estandarizado del entrenamiento
    Z = h["X"][h["ids"]["entrenamiento"]].astype(np.float64)
    precision = np.linalg.pinv(np.cov(Z, rowvar=False))
    dist = np.sqrt(np.einsum("ij,jk,ik->i", Z, precision, Z))
    crudo = D.crudo()[0][D.ATRIBUTOS].to_numpy()[h["ids"]["entrenamiento"]]
    cal["_dominio"] = {"precision": precision.tolist(), "umbral_p99": float(np.percentile(dist, 99)), "mediana_entrenamiento": np.median(crudo, 0).tolist()}
    return cal, diag


def latencia(modelo, repeticiones=300):
    h = D.heredados()
    crudo = D.crudo()[0][D.ATRIBUTOS].to_numpy()[h["ids"]["prueba"][:repeticiones]]
    I.sesion.cache_clear()
    for fila in crudo[:20]:
        I.probabilidades(fila, modelo, calibrar=False)
    t = []
    for fila in crudo:
        t0 = time.perf_counter()
        I.probabilidades(fila, modelo, calibrar=False)
        t.append(1000 * (time.perf_counter() - t0))
    return float(np.median(t)), float(np.percentile(t, 95))


def corridas_maestra(modelo):
    """Filas de la tabla maestra con las semillas definitivas del modelo en UAVIDS-2025 (E2 y E3)."""
    t = pd.read_csv(TABLA_MAESTRA)
    return t[(t["fase"] == "definitiva") & (t["conjunto"] == "UAVIDS-2025") & (t["modelo"] == modelo) & t["f1_eva"].notna()]


def lideres(modelos, cal, con_latencia):
    filas = []
    for m in modelos:
        rs = corridas_maestra(m).to_dict("records")
        f1 = np.array([r["f1_eva"] for r in rs])
        boot = np.random.default_rng(0).choice(f1, (5000, len(f1))).mean(1)
        meta = I.metadatos(m)
        fila = {"Modelo": m, "Grupo": GRUPO.get(m, ""), "Familia": FAMILIA.get(m, ""), "Semillas": len(f1), "F1 macro": f1.mean(), "DE": f1.std(ddof=1),
                "IC95 inf": np.percentile(boot, 2.5), "IC95 sup": np.percentile(boot, 97.5), "ECE": cal[m]["ece_antes"], "ECE calibrado": cal[m]["ece_despues"],
                "Parámetros": meta.get("parametros"), "FLOPs": (2 * meta["mac"]) if meta.get("mac") else None,
                "Entrenamiento (s)": float(np.mean([r.get("segundos", np.nan) for r in rs]))}
        if con_latencia:
            fila["Latencia mediana (ms)"], fila["Latencia p95 (ms)"] = latencia(m)
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("F1 macro", ascending=False)


def copiar_analisis():
    """Lleva al dashboard los resultados confirmatorios, la diferencia crítica de Nemenyi y la ficha del original."""
    from scipy.stats import studentized_range

    conf = RAIZ / "experiments" / "resultados" / "confirmatorio"
    if (conf / "tabla_contrastes.csv").exists():
        pd.read_csv(conf / "tabla_contrastes.csv").to_csv(SALIDA / "contrastes.csv", index=False, encoding="utf-8-sig")
        r = json.loads((conf / "resumen.json").read_text(encoding="utf-8"))
        k, n = len(r["rangos medios (1 = mejor)"]), 20
        r["cd_nemenyi"] = float(studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n)))
        (SALIDA / "resumen.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    abl = RAIZ / "experiments" / "resultados" / "ablacion"
    if (abl / "predicciones_secundarias.json").exists():
        ficha = json.loads((abl / "predicciones_secundarias.json").read_text(encoding="utf-8"))
        ficha["pruebas"] = ("pruebas con pytest en integración continua: del modelo (sobreajuste de un lote, reproducibilidad, formas, gradientes, "
                            "Lipschitz, Jensen, guardado y ONNX) y del dashboard (páginas, inferencia, conformal, fuera de dominio, ida y vuelta y "
                            "consistencia con la tabla maestra).")
        (SALIDA / "ficha_original.json").write_text(json.dumps(ficha, ensure_ascii=False, indent=1), encoding="utf-8")
    res = RAIZ / "experiments" / "resultados"
    ad = res / "adicionales"
    if (ad / "resumen.json").exists():
        conj = ("uav_nidd", "unsw_nb15", "unr_idd")
        tab = pd.concat([pd.read_csv(ad / f"tabla_{c}.csv") for c in conj])
        tab[["conjunto", "modelo", "F1 macro", "DE", "rango", "corridas en GPU"]].to_csv(SALIDA / "adicionales.csv", index=False, encoding="utf-8-sig")
        con = pd.concat([pd.read_csv(ad / f"contrastes_{c}.csv") for c in conj])
        con[["conjunto", "rival", "diferencia", "p T1 t Holm", "p T2 Wilcoxon Holm", "p T3 bootstrap bloques Holm", "peor con significancia (T1 a T3)",
             "peor con significancia (T1 y T2)", "mejor con significancia (T1 y T2)", "supera"]].to_csv(SALIDA / "adicionales_contrastes.csv", index=False, encoding="utf-8-sig")
    if (res / "robustez" / "perturbaciones.csv").exists():
        pd.read_csv(res / "robustez" / "perturbaciones.csv").to_csv(SALIDA / "robustez.csv", index=False, encoding="utf-8-sig")
    rutas = [res / "confirmatorio" / "sensibilidad_cuadro4.json", ad / "resumen.json", res / "adecuacion" / "resumen.json"]
    if all(r.exists() for r in rutas):
        c4, rad, adec = (json.loads(r.read_text(encoding="utf-8")) for r in rutas)
        a = adec["adecuación del original (sección 7.2 del E2)"]
        nb = adec["Nadeau-Bengio y F 5x2cv (condición 6)"]["LightGBM"]
        balance = [
            {"condición": "1. Métrica media favorable frente a cada rival", "cumple": "no",
             "evidencia": f"F1 medio menor que el de {len(c4['rivales con media mayor que la del original'])} de los 12 rivales"},
            {"condición": "2. Tres pruebas con Holm y diferencia > δ", "cumple": "no",
             "evidencia": "ningún contraste rechaza con tres pruebas (preregistradas ni admisibles del cuadro 4)"},
            {"condición": "3. Efecto no despreciable e IC > 0", "cumple": "no", "evidencia": "solo frente a TabNet y RNN hay efecto a favor no despreciable"},
            {"condición": "4. Consistencia en los conjuntos adicionales", "cumple": "no",
             "evidencia": f"mejor en {rad['conjuntos en que el original es el mejor en la media']} de 3 conjuntos; peor con significancia en UNSW-NB15"},
            {"condición": "5. Adecuación (sección 7.2 del E2)", "cumple": "no",
             "evidencia": f"en UAVIDS-2025 cumple (i), (ii) y (iv) y falla la calibración (ECE con temperatura de {a['ECE con T']:.3f} frente a 0,05); adecuado en UNSW-NB15 y UNR-IDD, no en UAV-NIDD"},
            {"condición": "6. Robustez a la elección de la prueba", "cumple": "sí",
             "evidencia": f"ninguna prueba muestra superioridad del original; la ventaja de LightGBM desaparece en 5 × 2 (p de Nadeau-Bengio = {nb['p Nadeau-Bengio']:.2f})"}]
        (SALIDA / "balance_r3.json").write_text(json.dumps(balance, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    copiar_analisis()
    datos_parquet()
    modelos = I.modelos_disponibles()
    cal, diag = calibracion_y_diagnosticos(modelos)
    (SALIDA / "calibracion.json").write_text(json.dumps(cal), encoding="utf-8")
    (SALIDA / "diagnosticos.json").write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
    I.calibracion.cache_clear()
    t = lideres(modelos, cal, "--latencia" in sys.argv)
    previo = SALIDA / "lideres.csv"
    if previo.exists() and "--latencia" not in sys.argv:  # conserva la latencia medida en una corrida anterior sin contención
        viejo = pd.read_csv(previo)
        for c in ("Latencia mediana (ms)", "Latencia p95 (ms)"):
            if c in viejo:
                t[c] = t["Modelo"].map(dict(zip(viejo["Modelo"], viejo[c])))
    t.to_csv(previo, index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
