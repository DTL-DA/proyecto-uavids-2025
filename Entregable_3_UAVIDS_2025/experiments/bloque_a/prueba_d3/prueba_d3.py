"""Prueba rápida de D3: ¿la penalización invariante por escenario con riesgo condicional por clase (V-REx
adaptado) mejora el acierto en un escenario de simulación no visto? Solo usa filas de entrenamiento del E2.
La regla de decisión está en regla_decision.md, escrita antes de correr.

Uso: python prueba_d3.py <raiz_repo_E3> <carpeta_salida> [procesos]
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ESCENARIOS = [("S1", "Sybil", 1, 19000), ("S2", "Sybil", 51325, 54551), ("S3", "Sybil", 109001, 109046),
              ("B1", "Blackhole", 20001, 35000), ("B2", "Blackhole", 57961, 94000),
              ("W1", "Wormhole", 36001, 49000), ("W2", "Wormhole", 97428, 108000),
              ("F1", "Flooding", 51001, 51324), ("F2", "Flooding", 54552, 57960), ("F3", "Flooding", 95001, 97427),
              ("F4", "Flooding", 109047, 122171), ("N", "Normal", 57961, 94000)]
NOMBRES = [e[0] for e in ESCENARIOS]
PLIEGUES = ["S1", "S2", "B1", "B2", "W1", "W2", "F2", "F3", "F4"]
METODOS = {"ERM": 0.0, "VREx_1": 1.0, "VREx_10": 10.0, "VREx_100": 100.0}
SEMILLAS = [101, 102, 103]
HP = {"lr": 0.001, "weight_decay": 0.0, "batch": 256}
HP_LGBM = {"learning_rate": 0.1, "num_leaves": 127, "min_child_samples": 10, "reg_lambda": 1.0, "colsample_bytree": 0.7, "subsample": 1.0}
FILAS_POR_GRUPO, MINIMO_GRUPO = 32, 64
MAX_EPOCAS, PACIENCIA, NORMA = 30, 6, 5.0

_G = {}


def iniciar(raiz):
    import torch

    torch.set_num_threads(2)
    sys.path.insert(0, str(Path(raiz) / "Entregable_2_UAVIDS_2025"))
    _G["raiz"] = raiz


def base():
    """Etiquetas, bloques, matriz transformada, filas de entrenamiento y escenario de cada fila."""
    if "base" not in _G:
        from codigo import datos as D

        datos, bloques, x_log = D.crudo()
        y = D.etiquetas()
        ent = np.sort(D.heredados()["ids"]["entrenamiento"])
        fid = datos["FlowID"].to_numpy()
        en_ent = np.zeros(len(y), bool)
        en_ent[ent] = True
        env = np.full(len(y), -1)
        for k, (_, clase, a, b) in enumerate(ESCENARIOS):
            sel = en_ent & (y == D.CLASES.index(clase)) & (fid >= a) & (fid <= b)
            assert (env[sel] == -1).all(), "escenarios solapados"
            env[sel] = k
        assert (env[ent] >= 0).all(), "hay filas de entrenamiento sin escenario"
        _G["base"] = (y, bloques, x_log, ent, env)
    return _G["base"]


def pliegue(nombre):
    """Filas de ajuste, de parada (15 % de los bloques restantes) y del escenario que se deja fuera."""
    y, bloques, _, ent, env = base()
    fuera = ent[env[ent] == NOMBRES.index(nombre)]
    resto = ent[env[ent] != NOMBRES.index(nombre)]
    g = np.random.default_rng(1000 + PLIEGUES.index(nombre))
    unicos = np.unique(bloques[resto])
    parada = np.isin(bloques[resto], g.choice(unicos, int(round(0.15 * len(unicos))), replace=False))
    return resto[~parada], resto[parada], fuera


def matrices(nombre):
    clave = ("matrices", nombre)
    if clave not in _G:
        from codigo import datos as D

        ajuste, parada, fuera = pliegue(nombre)
        X, pesos = D.preparar_pliegue(ajuste, base()[2])
        _G[clave] = (X, pesos, ajuste, parada, fuera)
    return _G[clave]


def f1_presentes(verdad, pred):
    from sklearn.metrics import f1_score

    return float(f1_score(verdad, pred, labels=np.unique(verdad), average="macro", zero_division=0))


def entrenar_dnn(nombre, lam, semilla):
    import torch
    from torch import nn
    from codigo import modelos as M

    y, _, _, _, env = base()
    X_np, pesos_np, ajuste, parada, fuera = matrices(nombre)
    X, Y, pesos = torch.from_numpy(X_np), torch.from_numpy(y), torch.from_numpy(pesos_np)
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    modelo = M.construir("DNN")
    gen = torch.Generator().manual_seed(semilla)
    gen_pen = torch.Generator().manual_seed(semilla + 7)  # aparte, para que ERM y V-REx vean los mismos lotes
    opt = torch.optim.Adam(modelo.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=2)
    ce = nn.CrossEntropyLoss(weight=pesos)
    ce_fila = nn.CrossEntropyLoss(reduction="none")
    # Pares (escenario, clase) con filas suficientes, agrupados por clase; solo clases con >= 2 escenarios
    por_clase = {}
    for e in np.unique(env[ajuste]):
        filas = ajuste[env[ajuste] == e]
        if len(filas) >= MINIMO_GRUPO:
            por_clase.setdefault(int(y[filas[0]]), []).append(torch.from_numpy(filas))
    grupos = [(c, lst) for c, lst in sorted(por_clase.items()) if len(lst) >= 2]
    t_ajuste, t_parada = torch.from_numpy(ajuste), torch.from_numpy(parada)
    mejor, estado, sin_mejora, curvas = -1.0, None, 0, []
    for epoca in range(1, MAX_EPOCAS + 1):
        modelo.train()
        pen_acum, pasos = 0.0, 0
        for lote in t_ajuste[torch.randperm(len(t_ajuste), generator=gen)].split(HP["batch"]):
            if len(lote) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            valor = ce(modelo(X[lote]), Y[lote])
            if lam > 0 and epoca > 1 and grupos:
                muestras = [g[torch.randint(len(g), (FILAS_POR_GRUPO,), generator=gen_pen)] for _, lst in grupos for g in lst]
                filas = torch.cat(muestras)
                riesgo = ce_fila(modelo(X[filas]), Y[filas]).view(len(muestras), FILAS_POR_GRUPO).mean(1)
                varianzas, i = [], 0
                for _, lst in grupos:
                    varianzas.append(riesgo[i:i + len(lst)].var(unbiased=False))
                    i += len(lst)
                penal = torch.stack(varianzas).mean()
                valor = valor + lam * penal
                pen_acum += penal.item()
                pasos += 1
            valor.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), NORMA)
            opt.step()
        modelo.eval()
        with torch.no_grad():
            logits = modelo(X[t_parada])
        perdida = ce(logits, Y[t_parada]).item()
        f1 = f1_presentes(y[parada], logits.argmax(1).numpy())
        plan.step(perdida)
        curvas.append({"epoca": epoca, "f1_parada": round(f1, 5), "penal": round(pen_acum / max(pasos, 1), 6)})
        if f1 > mejor:
            mejor, sin_mejora = f1, 0
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    modelo.load_state_dict(estado)
    modelo.eval()
    with torch.no_grad():
        pred_fuera = modelo(X[torch.from_numpy(fuera)]).argmax(1).numpy()
        pred_parada = modelo(X[t_parada]).argmax(1).numpy()
    return pred_fuera, pred_parada, {"epocas": len(curvas), "curvas": curvas, "grupos_penalizados": {int(c): len(l) for c, l in grupos}}


def entrenar_lgbm(nombre, semilla):
    import lightgbm as lgb

    y = base()[0]
    X, _, ajuste, parada, fuera = matrices(nombre)
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, subsample_freq=1, class_weight="balanced", random_state=semilla,
                           n_jobs=2, verbose=-1, **HP_LGBM)
    m.fit(X[ajuste], y[ajuste], eval_set=[(X[parada], y[parada])], callbacks=[lgb.early_stopping(100, verbose=False)])
    it = int(m.best_iteration_ or m.n_estimators)
    return m.predict(X[fuera], num_iteration=it), m.predict(X[parada], num_iteration=it), {"arboles": it}


def trabajo(t, salida):
    ruta = Path(salida) / f"{t['id']}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    inicio = time.time()
    if t["metodo"] == "LightGBM":
        pred_fuera, pred_parada, extra = entrenar_lgbm(t["pliegue"], t["semilla"])
    else:
        pred_fuera, pred_parada, extra = entrenar_dnn(t["pliegue"], METODOS[t["metodo"]], t["semilla"])
    y = base()[0]
    _, _, ajuste, parada, fuera = matrices(t["pliegue"])
    registro = dict(t)
    registro.update(extra)
    registro.update({"filas_ajuste": len(ajuste), "filas_parada": len(parada), "filas_fuera": len(fuera),
                     "acierto_fuera": float((pred_fuera == y[fuera]).mean()),
                     "pred_fuera": np.bincount(pred_fuera.astype(int), minlength=5).tolist(),
                     "f1_vistos": f1_presentes(y[parada], pred_parada), "segundos": round(time.time() - inicio, 1)})
    ruta.write_text(json.dumps(registro, ensure_ascii=False), encoding="utf-8")
    return registro


def resumir(salida):
    import pandas as pd

    filas = [json.loads(p.read_text(encoding="utf-8")) for p in Path(salida).glob("*.json") if p.name[0] != "_"]
    r = pd.DataFrame(filas)
    por = r.groupby(["metodo", "pliegue"]).agg(acierto=("acierto_fuera", "mean"), f1_vistos=("f1_vistos", "mean"),
                                               sd_acierto=("acierto_fuera", "std"), epocas=("epocas", "mean")).reset_index()
    tabla = por.pivot(index="pliegue", columns="metodo", values="acierto").reindex(PLIEGUES)
    tabla.to_csv(Path(salida) / "_acierto_por_pliegue.csv", encoding="utf-8-sig")
    resumen, decision = [], {}
    erm = por[por.metodo == "ERM"].set_index("pliegue")
    for metodo in ["ERM", *[m for m in METODOS if m != "ERM"], "LightGBM"]:
        m = por[por.metodo == metodo].set_index("pliegue")
        if m.empty:
            continue
        dif = (m["acierto"] - erm["acierto"]).reindex(PLIEGUES)
        dif_f1 = float(m["f1_vistos"].mean() - erm["f1_vistos"].mean())
        fila = {"metodo": metodo, "acierto_fuera_medio": m["acierto"].mean(), "f1_vistos_medio": m["f1_vistos"].mean(),
                "mejora_media_vs_ERM": dif.mean(), "pliegues_que_mejoran": int((dif > 0).sum()), "cambio_f1_vistos": dif_f1}
        if metodo.startswith("VREx"):
            fila["cumple_regla"] = bool(dif.mean() >= 0.03 and (dif > 0).sum() >= 6 and dif_f1 >= -0.01)
            decision[metodo] = fila["cumple_regla"]
        resumen.append(fila)
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(Path(salida) / "_resumen.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(tabla.round(3).to_string())
    print(resumen.round(4).to_string(index=False))
    print("Regla: opción 2 favorecida" if any(decision.values()) else "Regla: ningún λ cumple; se mantiene la opción 1", decision)


def main():
    raiz, salida = sys.argv[1], Path(sys.argv[2])
    procesos = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    salida.mkdir(parents=True, exist_ok=True)
    iniciar(raiz)
    y, _, _, ent, env = base()
    conteo = {NOMBRES[k]: int((env[ent] == k).sum()) for k in range(len(NOMBRES))}
    print("Filas por escenario:", conteo, flush=True)
    (salida / "_escenarios.json").write_text(json.dumps(conteo), encoding="utf-8")
    trabajos = [{"id": f"{p}_{m}_{s}", "pliegue": p, "metodo": m, "semilla": s} for p in PLIEGUES for s in SEMILLAS for m in METODOS]
    trabajos += [{"id": f"{p}_LightGBM_101", "pliegue": p, "metodo": "LightGBM", "semilla": 101} for p in PLIEGUES]
    # Primero los V-REx (más lentos) para equilibrar la carga
    trabajos.sort(key=lambda t: (t["metodo"] in ("ERM", "LightGBM"), t["pliegue"]))
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=procesos, initializer=iniciar, initargs=(raiz,)) as ex:
        futuros = {ex.submit(trabajo, t, str(salida)): t for t in trabajos}
        for i, f in enumerate(as_completed(futuros), 1):
            r = f.result()
            print(f"[{i}/{len(trabajos)}] {r['id']}: acierto_fuera={r['acierto_fuera']:.3f} f1_vistos={r['f1_vistos']:.3f} "
                  f"({r['segundos']} s, {(time.time() - inicio) / 60:.1f} min)", flush=True)
    resumir(salida)


if __name__ == "__main__":
    main()
