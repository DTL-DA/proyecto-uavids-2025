"""Bucle de entrenamiento del modelo original con el esquema cooperativo.

Es una copia fiel de ``codigo.motor.entrenar`` del E2 (mismo optimizador Adam, ReduceLROnPlateau con factor 0,1 y
paciencia 2 sobre la pérdida de validación, parada temprana con paciencia 6 sobre el F1 macro de validación,
máximo de 30 épocas, recorte del gradiente en norma 5, pérdida ponderada w_c = n / (K n_c) y registro de
divergencias). La única diferencia es la línea de la pérdida: en lugar de criterio(modelo(x), y) se usa
``modelo.perdida_cooperativa(x, y, criterio)``, que es el esquema de entrenamiento propio del modelo original.
La validación se mide igual que para los rivales: entropía cruzada y F1 macro sobre la salida del modelo.
"""
from __future__ import annotations

import time

import torch
from sklearn.metrics import f1_score
from torch import nn

import registro
from codigo import modelos, motor


def entrenar(arquitectura, hp, semilla, D, ids_ent, ids_val, max_epocas=motor.MAX_EPOCAS, paciencia=motor.PACIENCIA, etiquetas=None, variante=None,
             cooperativo: bool = True):
    variante = variante or {}
    motor.fijar_semilla(semilla)
    X, pesos = D["X"], D["pesos"]
    if not variante.get("ponderar", True):
        pesos = torch.ones_like(pesos)
    Y = D["Y"] if etiquetas is None else etiquetas
    modelo = modelos.construir(registro.NOMBRE, arquitectura)
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
            if cooperativo:
                valor = modelo.perdida_cooperativa(X[lote], Y[lote], perdida)
            else:  # entrenamiento conjunto clásico, idéntico al del motor del E2
                valor = perdida(modelo(X[lote]), Y[lote])
                if modelo.perdida_extra is not None:
                    valor = valor + modelo.perdida_extra
            if not torch.isfinite(valor):
                divergente = True
                break
            valor.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), motor.NORMA_GRADIENTE)
            optimizador.step()
            acumulado += valor.item() * len(lote)
            n += len(lote)
        if divergente:
            break
        logits = motor.predecir(modelo, X, ids_val)
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
