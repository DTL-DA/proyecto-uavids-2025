"""Análisis posteriores a la ejecución definitiva, con caché en resultados/analisis.

    python Entregable_2_UAVIDS_2025/codigo/analisis.py metricas|perturbaciones|atribuciones|mc_dropout

El cuaderno usa estas funciones y, si la caché existe, la lee en lugar de recalcular.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ_E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_E2))
from codigo import datos, estadistica, modelos, motor, resultados  # noqa: E402

ANALISIS = RAIZ_E2 / "resultados" / "analisis"
CORRIDAS = RAIZ_E2 / "resultados" / "corridas"
K = 5


def ids():
    h = datos.heredados()
    return h["ids"]["entrenamiento"], h["ids"]["validacion"], h["ids"]["prueba"]


def definitiva():
    return [r for r in resultados.registros("definitiva")]


def prob(r, conjunto="eva"):
    return np.load(CORRIDAS / r["fase"] / f"{r['id']}_{conjunto}.npy")


def confusion_bloques(verdad, prediccion, bloques):
    """Matriz (bloques únicos × K²) de conteos de confusión, para el bootstrap por bloques."""
    unicos, inversa = np.unique(bloques, return_inverse=True)
    M = np.zeros((len(unicos), K * K))
    np.add.at(M, (inversa, K * verdad + prediccion), 1)
    return M


def f1_desde_conteos(C):
    """F1 macro de una o varias matrices de confusión aplanadas (…, K²)."""
    C = C.reshape(*C.shape[:-1], K, K)
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    return (2 * tp / np.maximum(C.sum(-2) + C.sum(-1), 1)).mean(-1)


def pesos_bootstrap(n_bloques, repeticiones=2000, semilla=42):
    rng = np.random.default_rng(semilla)
    W = np.zeros((repeticiones, n_bloques))
    for i in range(repeticiones):
        np.add.at(W[i], rng.integers(0, n_bloques, n_bloques), 1)
    return W


def metricas_definitiva():
    """Métricas de prueba y de validación por corrida, más las matrices de confusión por bloque de prueba."""
    ruta = ANALISIS / "metricas_definitiva.csv"
    ruta_conf = ANALISIS / "confusion_bloques.npz"
    if ruta.exists() and ruta_conf.exists():
        return pd.read_csv(ruta), dict(np.load(ruta_conf))
    y = datos.heredados()["y"]
    _, i_val, i_pru = ids()
    bloques = datos.crudo()[1]
    filas, conf = [], {}
    for r in definitiva():
        p_eva, p_val = prob(r, "eva"), prob(r, "val")
        m = {f"prueba {k}": v for k, v in estadistica.metricas(p_eva, y[i_pru]).items()}
        m.update({f"validación {k}": v for k, v in estadistica.metricas(p_val, y[i_val]).items()})
        filas.append({"id": r["id"], "modelo": r["modelo"], "semilla": r["semilla"], "f1_ent": r.get("f1_ent"), "segundos": r.get("segundos"), "epocas": r.get("epocas"),
                      "mejor_epoca": r.get("mejor_epoca"), "parametros": r.get("parametros"), "mac": r.get("mac"), "ms_por_muestra": r.get("ms_por_muestra"),
                      "ms_por_muestra_lote": r.get("ms_por_muestra_lote"), "memoria_pico_mb": r.get("memoria_pico_mb"), **m})
        conf[f"{r['id']}|eva"] = confusion_bloques(y[i_pru], p_eva.argmax(1), bloques[i_pru])
        conf[f"{r['id']}|val"] = confusion_bloques(y[i_val], p_val.argmax(1), bloques[i_val])
    tabla = pd.DataFrame(filas)
    ANALISIS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    np.savez_compressed(ruta_conf, **conf)
    return tabla, conf


def cargar_modelo(r):
    """Reconstruye el modelo de una corrida definitiva a partir de sus pesos guardados."""
    if r["modelo"] in ("LightGBM", "Logística"):
        return pickle.loads((CORRIDAS / r["fase"] / f"{r['id']}.pkl").read_bytes())
    m = modelos.construir(r["modelo"], r.get("arquitectura"))
    m.load_state_dict(torch.load(CORRIDAS / r["fase"] / f"{r['id']}.pt", weights_only=True))
    return m.eval()


def predecir_matriz(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)
    with torch.no_grad():
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        return torch.cat([torch.softmax(modelo(Xt[i:i + 4096]), 1) for i in range(0, len(Xt), 4096)]).numpy()


NIVELES = {"Ruido gaussiano": [0.1, 0.25, 0.5, 1.0], "Atributos faltantes": [0.1, 0.25, 0.5], "Escalamiento multiplicativo": [0.1, 0.25, 0.5]}


def perturbar(tipo, nivel, semilla=0):
    """Prueba perturbada en la escala de entrada del modelo; la media de entrenamiento es 0 tras estandarizar."""
    h = datos.heredados()
    _, _, i_pru = ids()
    rng = np.random.default_rng(semilla)
    X = h["X"][i_pru].copy()
    if tipo == "Ruido gaussiano":
        return X + nivel * rng.standard_normal(X.shape).astype(np.float32)
    if tipo == "Atributos faltantes":
        X[rng.random(X.shape) < nivel] = 0.0
        return X
    crudo = datos.crudo()[0][datos.ATRIBUTOS].to_numpy(dtype=np.float64)[i_pru] * (1 + nivel)
    return ((datos.transformar(crudo) - h["media"]) / h["desviacion"]).astype(np.float32)


def perturbaciones(semillas_por_modelo=5):
    ruta = ANALISIS / "perturbaciones.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    y = datos.heredados()["y"]
    _, _, i_pru = ids()
    condiciones = [("Sin perturbar", 0.0)] + [(t, n) for t, v in NIVELES.items() for n in v]
    entradas = {c: (datos.heredados()["X"][i_pru] if c[0] == "Sin perturbar" else perturbar(*c)) for c in condiciones}
    filas = []
    por_modelo = {}
    for r in definitiva():
        if r["modelo"] == "Trivial":
            continue
        por_modelo.setdefault(r["modelo"], []).append(r)
    for modelo_nombre, lista in por_modelo.items():
        for r in sorted(lista, key=lambda r: r["semilla"])[:semillas_por_modelo]:
            m = cargar_modelo(r)
            for (tipo, nivel), X in entradas.items():
                p = predecir_matriz(m, X)
                filas.append({"modelo": modelo_nombre, "semilla": r["semilla"], "perturbacion": tipo, "nivel": nivel,
                              "f1": estadistica.f1_macro_rapido(y[i_pru], p.argmax(1)), "ece": estadistica.ece(p, y[i_pru])})
        print(modelo_nombre, flush=True)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(ruta, index=False)
    return tabla


def gradientes_integrados(modelo, X, objetivo, pasos=32):
    base = torch.zeros_like(X)
    total = torch.zeros_like(X)
    for a in torch.linspace(1 / pasos, 1, pasos):
        x = (base + a * (X - base)).requires_grad_(True)
        salida = torch.softmax(modelo(x), 1)
        seleccion = salida.gather(1, objetivo[:, None]).sum()
        total += torch.autograd.grad(seleccion, x)[0]
    return ((X - base) * total / pasos).detach()


def atribuciones(nombres, semillas_por_modelo=5, muestra=1500, semilla=0):
    """Gradientes integrados sobre una muestra estratificada de prueba, con base en la media de entrenamiento,
    y prueba de borrado: se anulan los atributos en el orden de la atribución y en un orden aleatorio."""
    ruta = ANALISIS / "atribuciones.npz"
    if ruta.exists():
        return dict(np.load(ruta, allow_pickle=True))
    h = datos.heredados()
    y = h["y"]
    _, _, i_pru = ids()
    rng = np.random.default_rng(semilla)
    elegidos = np.concatenate([rng.choice(np.flatnonzero(y[i_pru] == c), muestra // K, replace=False) for c in range(K)])
    X = torch.from_numpy(h["X"][i_pru][elegidos])
    salida = {"indices": elegidos}
    for nombre in nombres:
        lista = sorted([r for r in definitiva() if r["modelo"] == nombre], key=lambda r: r["semilla"])[:semillas_por_modelo]
        imp, borrado, aleatorio = [], [], []
        for r in lista:
            m = cargar_modelo(r)
            with torch.no_grad():
                pred = torch.softmax(m(X), 1)
            objetivo = pred.argmax(1)
            ig = gradientes_integrados(m, X, objetivo)
            imp.append(ig.abs().mean(0).numpy())
            curva_b, curva_a = [], []
            orden = ig.abs().argsort(1, descending=True)
            orden_al = torch.from_numpy(np.argsort(rng.random(X.shape), 1))
            for k in range(X.shape[1] + 1):
                for o, curva in ((orden, curva_b), (orden_al, curva_a)):
                    Xk = X.clone()
                    if k:
                        Xk.scatter_(1, o[:, :k], 0.0)
                    with torch.no_grad():
                        curva.append(float(torch.softmax(m(Xk), 1).gather(1, objetivo[:, None]).mean()))
            borrado.append(curva_b)
            aleatorio.append(curva_a)
        salida[f"{nombre}|importancia"] = np.array(imp)
        salida[f"{nombre}|borrado"] = np.array(borrado)
        salida[f"{nombre}|aleatorio"] = np.array(aleatorio)
        print(nombre, flush=True)
    np.savez(ruta, **salida)
    return salida


def mc_dropout(nombres, pasadas=30, semillas_por_modelo=5):
    """Promedio de `pasadas` predicciones con el dropout activo, por corrida."""
    ruta = ANALISIS / "mc_dropout.npz"
    if ruta.exists():
        return dict(np.load(ruta))
    h = datos.heredados()
    _, i_val, i_pru = ids()
    salida = {}
    for nombre in nombres:
        for r in sorted([r for r in definitiva() if r["modelo"] == nombre], key=lambda r: r["semilla"])[:semillas_por_modelo]:
            m = cargar_modelo(r)
            if not any(isinstance(c, torch.nn.Dropout) and c.p > 0 for c in m.modules()):
                continue
            for c in m.modules():
                if isinstance(c, torch.nn.Dropout):
                    c.train()
            for conjunto, indices in (("val", i_val), ("eva", i_pru)):
                X = torch.from_numpy(h["X"][indices])
                torch.manual_seed(r["semilla"])
                with torch.no_grad():
                    salida[f"{r['id']}|{conjunto}"] = np.mean([torch.softmax(m(X), 1).numpy() for _ in range(pasadas)], 0).astype(np.float32)
        print(nombre, flush=True)
    np.savez_compressed(ruta, **salida)
    return salida


if __name__ == "__main__":
    torch.set_num_threads(4)
    tarea = sys.argv[1]
    if tarea == "metricas":
        metricas_definitiva()
    elif tarea == "perturbaciones":
        perturbaciones()
    elif tarea == "atribuciones":
        atribuciones(sys.argv[2].split(","))
    elif tarea == "mc_dropout":
        mc_dropout(sys.argv[2].split(","))
