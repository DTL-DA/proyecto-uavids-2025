"""Motor de entrenamiento común a los modelos del benchmark.

Hereda el protocolo del Entregable 1: entropía cruzada ponderada, Adam, reducción de la tasa
por 10 si la pérdida de validación no mejora en 2 épocas y parada temprana sobre el F1 macro de
validación, con paciencia de 6 épocas. Cambios declarados en el Entregable 2: máximo de 30 épocas,
recorte del gradiente en norma 5 para todos los modelos y registro de divergencias (NaN).
"""
from __future__ import annotations

import json
import os
import platform
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn

from . import datos, modelos

warnings.filterwarnings("ignore", category=UserWarning)
MAX_EPOCAS = 30
PACIENCIA = 6
NORMA_GRADIENTE = 5.0
_CACHE = {}


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def datos_particion(nombre):
    """Matriz estandarizada, pesos e índices de una partición, en memoria del proceso."""
    if nombre not in _CACHE:
        ids = datos.cargar_particion(nombre)
        if nombre == "principal":
            h = datos.heredados()
            X, pesos = h["X"], h["pesos"]
        else:
            X, pesos = datos.preparar_pliegue(ids["ent"])
        _CACHE[nombre] = {"X": torch.from_numpy(X), "Y": torch.from_numpy(datos.heredados()["y"]), "pesos": torch.from_numpy(pesos),
                          "ids": {k: torch.from_numpy(v.astype(np.int64)) for k, v in ids.items()}}
    return _CACHE[nombre]


def predecir(modelo, X, indices, lote=4096):
    modelo.eval()
    with torch.no_grad():
        return torch.cat([modelo(X[b]) for b in indices.split(lote)])


def entrenar(nombre, arquitectura, hp, semilla, D, ids_ent, ids_val, max_epocas=MAX_EPOCAS, paciencia=PACIENCIA, etiquetas=None):
    fijar_semilla(semilla)
    X, pesos = D["X"], D["pesos"]
    Y = D["Y"] if etiquetas is None else etiquetas
    modelo = modelos.construir(nombre, arquitectura)
    generador = torch.Generator().manual_seed(semilla)
    inicio_pre = time.time()
    if hasattr(modelo, "preentrenar"):
        modelo.preentrenar(X[ids_ent], generador)
    segundos_pre = time.time() - inicio_pre
    optimizador = torch.optim.Adam(modelo.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, factor=0.1, patience=2)
    perdida = nn.CrossEntropyLoss(weight=pesos)
    mejor, mejor_estado, sin_mejora, curvas, divergente = -1.0, None, 0, [], False
    for epoca in range(1, max_epocas + 1):
        inicio = time.time()
        modelo.train()
        acumulado, n = 0.0, 0
        for lote in ids_ent[torch.randperm(len(ids_ent), generator=generador)].split(hp["batch"]):
            if len(lote) < 2:
                continue
            optimizador.zero_grad(set_to_none=True)
            valor = perdida(modelo(X[lote]), Y[lote])
            extra = getattr(modelo, "perdida_extra", None)
            if extra is not None:
                valor = valor + extra
            if not torch.isfinite(valor):
                divergente = True
                break
            valor.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), NORMA_GRADIENTE)
            optimizador.step()
            acumulado += valor.item() * len(lote)
            n += len(lote)
        if divergente:
            break
        logits = predecir(modelo, X, ids_val)
        perdida_val = perdida(logits, Y[ids_val]).item()
        f1_val = f1_score(Y[ids_val].numpy(), logits.argmax(1).numpy(), average="macro", zero_division=0)
        planificador.step(perdida_val)
        curvas.append({"epoca": epoca, "perdida_ent": acumulado / max(n, 1), "perdida_val": perdida_val, "f1_val": f1_val,
                       "lr": optimizador.param_groups[0]["lr"], "segundos": round(time.time() - inicio, 2)})
        if f1_val > mejor:
            mejor, sin_mejora = f1_val, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo, curvas, divergente, segundos_pre


def memoria_pico_mb():
    try:
        import psutil

        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2 ** 20, 1)
    except Exception:
        return None


def latencia_por_muestra(modelo, X, indices, repeticiones=200):
    """Milisegundos por muestra al predecir de a una, la latencia de un despliegue en línea."""
    modelo.eval()
    muestra = indices[:repeticiones]
    with torch.no_grad():
        for i in muestra[:20]:
            modelo(X[i:i + 1])
        inicio = time.perf_counter()
        for i in muestra:
            modelo(X[i:i + 1])
    return 1000 * (time.perf_counter() - inicio) / len(muestra)


def ejecutar(trabajo, carpeta):
    """Entrena y evalúa un trabajo; guarda un JSON por corrida y, si se pide, probabilidades y pesos."""
    carpeta = Path(carpeta)
    ruta = carpeta / f"{trabajo['id']}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    D = datos_particion(trabajo.get("particion", "principal"))
    ids = D["ids"]
    etiquetas = None
    if trabajo.get("permutar_etiquetas") is not None:
        g = np.random.default_rng(trabajo["permutar_etiquetas"])
        y_perm = D["Y"].clone()
        ent = ids["ent"].numpy()
        y_perm[ent] = D["Y"][ent[g.permutation(len(ent))]]
        etiquetas = y_perm
    ids_ent = ids["ent"]
    if trabajo.get("fraccion_ent") is not None:
        g = torch.Generator().manual_seed(trabajo["semilla"])
        ids_ent = ids_ent[torch.randperm(len(ids_ent), generator=g)[: int(round(trabajo["fraccion_ent"] * len(ids_ent)))]]
    inicio = time.time()
    modelo, curvas, divergente, segundos_pre = entrenar(trabajo["modelo"], trabajo.get("arquitectura"), trabajo["hp"], trabajo["semilla"], D, ids_ent, ids["val"],
                                                        trabajo.get("max_epocas", MAX_EPOCAS), trabajo.get("paciencia", PACIENCIA), etiquetas)
    segundos = time.time() - inicio
    Y = D["Y"] if etiquetas is None else etiquetas
    registro = {k: trabajo[k] for k in trabajo}
    registro.update({"curvas": curvas, "segundos": round(segundos, 2), "segundos_preentreno": round(segundos_pre, 2), "epocas": len(curvas), "divergente": divergente,
                     "mejor_epoca": (max(range(len(curvas)), key=lambda i: curvas[i]["f1_val"]) + 1) if curvas else None,
                     "parametros": modelos.parametros(modelo), "mac": modelos.contar_mac(modelo), "hilos": torch.get_num_threads(),
                     "equipo": f"{platform.processor()} | {platform.system()} {platform.release()} | Python {platform.python_version()} | torch {torch.__version__}",
                     "pid": os.getpid()})
    if not divergente and curvas:
        logits_val = predecir(modelo, D["X"], ids["val"])
        pred_val = logits_val.argmax(1).numpy()
        y_val = Y[ids["val"]].numpy()
        registro.update({"f1_val": f1_score(y_val, pred_val, average="macro", zero_division=0), "acc_val": accuracy_score(y_val, pred_val),
                         "bacc_val": balanced_accuracy_score(y_val, pred_val)})
        g = torch.Generator().manual_seed(trabajo["semilla"] + 1)
        muestra_ent = ids_ent[torch.randperm(len(ids_ent), generator=g)[:len(ids["val"])]]
        registro["f1_ent"] = f1_score(Y[muestra_ent].numpy(), predecir(modelo, D["X"], muestra_ent).argmax(1).numpy(), average="macro", zero_division=0)
        if trabajo.get("guardar_prob_val"):
            np.save(ruta.with_name(ruta.stem + "_val.npy"), torch.softmax(logits_val, 1).numpy().astype(np.float32))
        if trabajo.get("evaluar") and "eva" in ids:
            inicio_eva = time.time()
            logits_eva = predecir(modelo, D["X"], ids["eva"])
            registro["ms_por_muestra_lote"] = round(1000 * (time.time() - inicio_eva) / len(ids["eva"]), 6)
            pred = logits_eva.argmax(1).numpy()
            y_eva = Y[ids["eva"]].numpy()
            registro.update({"f1_eva": f1_score(y_eva, pred, average="macro", zero_division=0), "acc_eva": accuracy_score(y_eva, pred)})
            if trabajo.get("guardar_prob_eva", True):
                np.save(ruta.with_name(ruta.stem + "_eva.npy"), torch.softmax(logits_eva, 1).numpy().astype(np.float32))
            registro["ms_por_muestra"] = round(latencia_por_muestra(modelo, D["X"], ids["eva"]), 5)
        if trabajo.get("guardar_pesos"):
            torch.save(modelo.state_dict(), ruta.with_suffix(".pt"))
    registro["memoria_pico_mb"] = memoria_pico_mb()
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(registro, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta)
    return registro
