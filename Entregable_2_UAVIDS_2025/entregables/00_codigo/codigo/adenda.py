"""Adenda del preregistro: aplica las reglas fijadas a los resultados del piloto y de la búsqueda.

    python Entregable_2_UAVIDS_2025/codigo/adenda.py piloto      # R y limitación del paper guía, tras el piloto
    python Entregable_2_UAVIDS_2025/codigo/adenda.py busqueda    # k (tras el estudio de ruido) y arquitecturas ganadoras (tras la fase 2)

Cada paso agrega su sección a resultados/preregistro_e2_adenda.json con la fecha y hora en que se calculó,
antes de lanzar la fase siguiente. Las reglas son las del preregistro; aquí no se decide nada nuevo.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

RAIZ_E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_E2))
from codigo import datos, estadistica, modelos, plan  # noqa: E402

RUTA = RAIZ_E2 / "resultados" / "preregistro_e2_adenda.json"
SIGMA_E1 = 0.01773092612361445


def cargar():
    return json.loads(RUTA.read_text(encoding="utf-8")) if RUTA.exists() else {}


def guardar(adenda):
    RUTA.write_text(json.dumps(adenda, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def paso_piloto():
    c = plan.CONFIG
    piloto = plan.leer("piloto")
    f1 = {}
    for r in piloto:
        f1.setdefault(r["modelo"], {})[r["semilla"]] = r["f1_val"]
    base = c["modelos"]["baseline_e1"]
    semillas = c["piloto"]["semillas"]
    pares = {}
    for m in f1:
        if m == base:
            continue
        d = np.array([f1[m][s] - f1[base][s] for s in semillas])
        pares[f"{m} - {base}"] = {"media": d.mean(), "sigma_d": d.std(ddof=1), "shapiro_p": stats.shapiro(d).pvalue}
    sigma_piloto = max(p["sigma_d"] for p in pares.values())
    sigma = max(sigma_piloto, SIGMA_E1)
    r_calc = estadistica.ejecuciones_necesarias(sigma, c["metricas"]["delta"], comparaciones=10)
    R = int(min(max(r_calc, c["definitiva"]["R_minimo"]), c["definitiva"]["R_tope"]))
    alfa_p = 0.05 / 10
    mdd = estadistica.minima_detectable(sigma, R, alfa=alfa_p)
    # Limitación del paper guía: confusión Blackhole–Wormhole en validación, por modelo del Top 10 del piloto
    y = datos.heredados()["y"]
    ids_val = datos.heredados()["ids"]["validacion"]
    bloques = datos.crudo()[1][ids_val]
    limitacion = {}
    for r in piloto:
        prob = np.load(RAIZ_E2 / "resultados" / "corridas" / "piloto" / f"{r['id']}_val.npy")
        pred = prob.argmax(1)
        limitacion.setdefault(r["modelo"], []).append(estadistica.tasa_ruteo(y[ids_val], pred))
    resumen_lim = {}
    for m, v in limitacion.items():
        r0 = next(x for x in piloto if x["modelo"] == m and x["semilla"] == semillas[0])
        pred0 = np.load(RAIZ_E2 / "resultados" / "corridas" / "piloto" / f"{r0['id']}_val.npy").argmax(1)
        boot = estadistica.bootstrap_bloques(lambda ix: estadistica.tasa_ruteo(y[ids_val][ix], pred0[ix]), bloques)
        resumen_lim[m] = {"media_semillas": float(np.mean(v)), "de_semillas": float(np.std(v, ddof=1)), "ic95_bloques_primera_semilla": estadistica.ic_percentil(boot)}
    adenda = cargar()
    adenda["piloto"] = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "f1_validacion": f1, "pares_frente_al_baseline": pares, "sigma_piloto": sigma_piloto, "sigma_e1_prueba": SIGMA_E1,
                        "sigma_usada": sigma, "R_calculado": r_calc, "R": R, "limitado_por_tope": r_calc > R, "diferencia_minima_detectable": mdd,
                        "limitacion_paper_guia": {"definicion": "(n_B→W + n_W→B) / (n_B + n_W) en validación", "por_modelo": resumen_lim}}
    adenda["R"] = R
    guardar(adenda)
    print(json.dumps(adenda["piloto"], ensure_ascii=False, indent=1, default=float)[:3000])


def paso_busqueda():
    adenda = cargar()
    k, razones = plan.k_finalistas()
    adenda["ruido_semilla"] = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "razon_dentro_entre": razones, "k": k}
    fase2 = plan.leer("fase2")
    if fase2:
        delta = plan.CONFIG["metricas"]["delta"]
        ganadoras, detalle = {}, {}
        for m in plan.CONFIG["modelos"]["con_busqueda"]:
            por_arq = {}
            for r in fase2:
                if r["modelo"] == m:
                    por_arq.setdefault(tuple(r["arquitectura"]), []).append(r["f1_val"])
            resumen = []
            for a, v in por_arq.items():
                v = np.array(v)
                margen = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                resumen.append({"arquitectura": a, "media": v.mean(), "ic": (v.mean() - margen, v.mean() + margen), "parametros": modelos.parametros(modelos.construir(m, a)),
                                "id": modelos.identificador(a), "valores": v.tolist()})
            resumen.sort(key=lambda x: (-x["media"], x["id"]))
            mejor, segunda = resumen[0], resumen[1]
            empate = (mejor["media"] - segunda["media"] < delta) or (mejor["ic"][0] <= segunda["ic"][1])
            if empate:
                candidatas = [x for x in resumen if mejor["media"] - x["media"] < delta or mejor["ic"][0] <= x["ic"][1]]
                elegida = min(candidatas, key=lambda x: (x["parametros"], x["id"]))
            else:
                elegida = mejor
            equivalentes = []
            for x in resumen:
                if x is elegida:
                    continue
                r = estadistica.comparar_par(np.array(x["valores"]), np.array(elegida["valores"]), delta)
                if r["p TOST"] < 0.05:
                    equivalentes.append(x["id"])
            ganadoras[m] = list(elegida["arquitectura"])
            detalle[m] = {"mejor_media": mejor["id"], "empate_tecnico": bool(empate), "elegida": elegida["id"], "parametros": elegida["parametros"],
                          "media_validacion": elegida["media"], "equivalentes_tost": equivalentes, "finalistas": [{k: v for k, v in x.items() if k != "valores"} for x in resumen]}
        adenda["fase2"] = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "detalle": detalle}
        adenda["arquitecturas_ganadoras"] = ganadoras
    guardar(adenda)
    print(json.dumps({k: v for k, v in adenda.items() if k in ("ruido_semilla", "arquitecturas_ganadoras")}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    {"piloto": paso_piloto, "busqueda": paso_busqueda}[sys.argv[1]]()
