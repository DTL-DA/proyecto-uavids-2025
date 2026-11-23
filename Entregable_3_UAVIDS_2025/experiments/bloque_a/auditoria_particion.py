"""Auditoría de la partición por bloques (Bloque A del E3): ¿hay series que cruzan particiones y pesa eso en el F1?

Solo lectura: no entrena ni cambia la partición. Usa las predicciones guardadas de las corridas definitivas del E2.
Uso: python auditoria_particion.py <raiz_repo_E3> <carpeta_corridas_definitiva_E2> <carpeta_salida>
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

raiz, corridas, salida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
salida.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(raiz / "Entregable_2_UAVIDS_2025"))
from codigo import datos as D  # noqa: E402

datos, bloques, _ = D.crudo()
y = D.etiquetas()
h = D.heredados()
X, ids = h["X"], h["ids"]
N, K = len(datos), len(D.CLASES)
informe = {}

# A. Integridad: FlowID como posición, bloques exclusivos de una partición
fid = datos["FlowID"].to_numpy()
informe["flowid_es_posicion"] = bool((fid == np.arange(1, N + 1)).all())
parte = np.full(N, -1)
for k, nombre in enumerate(["entrenamiento", "validacion", "prueba"]):
    parte[ids[nombre]] = k
informe["filas"] = {"total": N, "entrenamiento": len(ids["entrenamiento"]), "validacion": len(ids["validacion"]),
                    "prueba": len(ids["prueba"]), "fuera (repetidas)": int((parte == -1).sum())}
cruces = {f"{a}-{b}": len(set(bloques[ids[a]]) & set(bloques[ids[b]]))
          for a, b in [("entrenamiento", "validacion"), ("entrenamiento", "prueba"), ("validacion", "prueba")]}
informe["bloques_compartidos_entre_particiones"] = cruces

# B. Tramos de una sola clase como aproximación del escenario de simulación
tramo = np.concatenate([[0], np.cumsum(y[1:] != y[:-1])])
tabla_tramos = pd.DataFrame({"tramo": tramo, "clase": y, "bloque": bloques, "parte": parte})
por_tramo = tabla_tramos.groupby("tramo").agg(largo=("clase", "size"), clase=("clase", "first"), bloques=("bloque", "nunique"),
                                              partes=("parte", lambda s: tuple(sorted(set(s) - {-1}))))
informe["tramos"] = {"numero": int(len(por_tramo)), "largo_mediano": float(por_tramo["largo"].median()),
                     "largo_maximo": int(por_tramo["largo"].max()),
                     "tramos_que_cruzan_bloques": int((por_tramo["bloques"] > 1).sum()),
                     "tramos_en_varias_particiones": int(por_tramo["partes"].map(len).gt(1).sum())}
for nombre, k in [("validacion", 1), ("prueba", 2)]:
    con_ent = por_tramo["partes"].map(lambda p: 0 in p and k in p)
    filas = tabla_tramos[tabla_tramos["parte"] == k]
    informe["tramos"][f"filas_{nombre}_en_tramo_compartido_con_entrenamiento_%"] = round(
        100 * filas["tramo"].map(con_ent).mean(), 2)
resumen_clase = por_tramo.groupby("clase").agg(tramos=("largo", "size"), largo_mediano=("largo", "median"), largo_max=("largo", "max"),
                                               cruzan_particiones=("partes", lambda s: int(s.map(len).gt(1).sum())))
resumen_clase.index = [D.CLASES[i] for i in resumen_clase.index]
resumen_clase.to_csv(salida / "tramos_por_clase.csv", encoding="utf-8-sig")

# C. Distancia en filas al flujo de entrenamiento más cercano y si comparte tramo con él
pos_ent = np.sort(ids["entrenamiento"])


def vecindad(filas):
    j = np.searchsorted(pos_ent, filas)
    izq = pos_ent[np.clip(j - 1, 0, len(pos_ent) - 1)]
    der = pos_ent[np.clip(j, 0, len(pos_ent) - 1)]
    d_izq, d_der = np.abs(filas - izq), np.abs(der - filas)
    vecino = np.where(d_izq <= d_der, izq, der)
    d = np.minimum(d_izq, d_der)
    return d, tramo[vecino] == tramo[filas]


CORTES = [0, 100, 300, 500, 1000, np.inf]
ETIQUETAS_D = ["1 a 100", "101 a 300", "301 a 500", "501 a 1000", "más de 1000"]
vec = {}
for nombre in ["validacion", "prueba"]:
    d, mismo = vecindad(ids[nombre])
    grupo = np.where(mismo & (d <= 1000), "contiguo", "lejano")
    vec[nombre] = {"d": d, "mismo": mismo, "grupo": grupo, "rango": pd.cut(d, CORTES, labels=ETIQUETAS_D)}
    informe[f"vecindad_{nombre}"] = {"distancia_mediana_filas": float(np.median(d)),
                                     "contiguo_%": round(100 * (grupo == "contiguo").mean(), 2),
                                     "por_rango_%": (100 * pd.Series(vec[nombre]["rango"]).value_counts(normalize=True)).round(2).reindex(ETIQUETAS_D).to_dict()}

# D. Parecido en atributos: distancia al vecino más cercano de entrenamiento, contiguos frente a lejanos
vecinos = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X[ids["entrenamiento"]])
for nombre in ["validacion", "prueba"]:
    dist = vecinos.kneighbors(X[ids[nombre]])[0][:, 0]
    g = vec[nombre]["grupo"]
    informe[f"vecino_atributos_{nombre}"] = {"mediana_contiguo": float(np.median(dist[g == "contiguo"])),
                                             "mediana_lejano": float(np.median(dist[g == "lejano"])),
                                             "gemelos_<0.01_contiguo_%": round(100 * float((dist[g == "contiguo"] < 0.01).mean()), 2),
                                             "gemelos_<0.01_lejano_%": round(100 * float((dist[g == "lejano"] < 0.01).mean()), 2)}


# E. Error de los modelos del E2 según la vecindad, estandarizado por clase, con bootstrap por bloques
def error_estandarizado(acierto, clases, mascara, pesos):
    tasas, w = [], []
    for c in range(K):
        sel = mascara & (clases == c)
        if sel.sum() >= 30:
            tasas.append(1 - acierto[sel].mean())
            w.append(pesos[c])
    return float(np.dot(tasas, w) / np.sum(w)) if w else np.nan


def comparar(acierto, nombre, rng, repeticiones=2000):
    filas = ids[nombre]
    clases, g, blq = y[filas], vec[nombre]["grupo"], bloques[filas]
    pesos = np.bincount(clases, minlength=K) / len(clases)
    # Solo clases con al menos 30 filas en los dos grupos, para que la diferencia compare lo mismo
    validas = [c for c in range(K) if ((g == "contiguo") & (clases == c)).sum() >= 30 and ((g == "lejano") & (clases == c)).sum() >= 30]
    w = np.array([pesos[c] if c in validas else 0.0 for c in range(K)])
    diferencia = lambda idx: (error_estandarizado(acierto[idx], clases[idx], g[idx] == "lejano", w)
                              - error_estandarizado(acierto[idx], clases[idx], g[idx] == "contiguo", w))
    obs = diferencia(np.arange(len(filas)))
    unicos = np.unique(blq)
    por_bloque = {b: np.flatnonzero(blq == b) for b in unicos}
    boot = []
    for _ in range(repeticiones):
        idx = np.concatenate([por_bloque[b] for b in rng.choice(unicos, len(unicos), replace=True)])
        boot.append(diferencia(idx))
    boot = np.array(boot)
    boot = boot[~np.isnan(boot)]
    return obs, np.percentile(boot, [2.5, 97.5]), float(np.mean(boot <= 0))


archivos = sorted(glob.glob(str(corridas / "*_eva.npy")))
modelos = sorted(set(re.sub(r"_\d+_eva\.npy$", "", os.path.basename(f)).replace("definitiva_", "") for f in archivos) - {"Trivial"})
rng = np.random.default_rng(42)
filas_resultado, discrepancias = [], 0
for modelo in modelos:
    aciertos = {"validacion": [], "prueba": []}
    f1 = {"validacion": [], "prueba": []}
    for f in sorted(glob.glob(str(corridas / f"definitiva_{modelo}_*_eva.npy"))):
        base = f[:-len("_eva.npy")]
        meta = json.load(open(base + ".json", encoding="utf-8"))
        for nombre, sufijo in [("prueba", "_eva.npy"), ("validacion", "_val.npy")]:
            if not os.path.exists(base + sufijo):
                continue
            pred = np.load(base + sufijo).argmax(1)
            verdad = y[ids[nombre]]
            f1[nombre].append(f1_score(verdad, pred, average="macro"))
            aciertos[nombre].append(pred == verdad)
        clave = "f1_eva"
        if clave in meta and abs(meta[clave] - f1["prueba"][-1]) > 1e-6:
            discrepancias += 1
    fila = {"Modelo": modelo.replace("_", " "), "Semillas": len(f1["prueba"]),
            "F1 validación": np.mean(f1["validacion"]) if f1["validacion"] else np.nan, "F1 prueba": np.mean(f1["prueba"])}
    for nombre, etiqueta in [("validacion", "val"), ("prueba", "prueba")]:
        if not aciertos[nombre]:
            continue
        acierto = np.mean(aciertos[nombre], axis=0)  # tasa de acierto por fila, promediada entre semillas
        obs, ic, p = comparar(acierto, nombre, rng)
        fila[f"Error lejano − contiguo ({etiqueta})"] = obs
        fila[f"IC 95 % ({etiqueta})"] = f"[{ic[0]:.4f}; {ic[1]:.4f}]"
        fila[f"p unilateral ({etiqueta})"] = p
    filas_resultado.append(fila)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()}, ensure_ascii=False), flush=True)

resultado = pd.DataFrame(filas_resultado)
resultado.to_csv(salida / "error_por_vecindad.csv", index=False, encoding="utf-8-sig")
informe["discrepancias_f1_registrado"] = discrepancias

# F. Error por rango de distancia (prueba), promedio de todos los modelos no triviales
por_rango = []
for nombre in ["validacion", "prueba"]:
    for modelo in modelos:
        for f in sorted(glob.glob(str(corridas / f"definitiva_{modelo}_*_eva.npy")))[:5]:
            base = f[:-len("_eva.npy")]
            ruta = base + ("_eva.npy" if nombre == "prueba" else "_val.npy")
            if not os.path.exists(ruta):
                continue
            acierto = np.load(ruta).argmax(1) == y[ids[nombre]]
            t = pd.DataFrame({"acierto": acierto, "rango": vec[nombre]["rango"], "mismo": vec[nombre]["mismo"], "clase": y[ids[nombre]]})
            for (r, m), sub in t.groupby(["rango", "mismo"], observed=True):
                por_rango.append({"particion": nombre, "modelo": modelo, "rango": r, "mismo_tramo": bool(m), "filas": len(sub), "error": 1 - sub["acierto"].mean()})
tabla_rango = pd.DataFrame(por_rango).groupby(["particion", "rango", "mismo_tramo"], observed=True).agg(filas=("filas", "mean"), error=("error", "mean")).reset_index()
tabla_rango.to_csv(salida / "error_por_rango.csv", index=False, encoding="utf-8-sig")

(salida / "auditoria_particion.json").write_text(json.dumps(informe, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(informe, ensure_ascii=False, indent=1, default=str))
print(resumen_clase.to_string())
print(tabla_rango.to_string())
