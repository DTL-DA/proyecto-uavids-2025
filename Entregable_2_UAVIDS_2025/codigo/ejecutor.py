"""Ejecuta una lista de trabajos en paralelo, con reanudación.

    python -m codigo.ejecutor resultados/trabajos/<fase>.jsonl --trabajadores 6 --hilos 2

Cada trabajo es una línea JSON con `id`, `fase`, `modelo`, `hp`, `semilla` y, según el caso,
`arquitectura`, `particion`, `max_epocas`, `evaluar`, `guardar_pesos` o `guardar_prob_val`. El
resultado de cada uno se guarda en resultados/corridas/<fase>/<id>.json; si ya existe, el trabajo se salta,
de modo que una interrupción no obliga a repetir lo hecho. Este registro por corrida, con su configuración,
semilla, curvas, tiempo y equipo, es la herramienta de seguimiento de experimentos del proyecto.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

RAIZ_E2 = Path(__file__).resolve().parents[1]


def _iniciar(hilos):
    import torch

    torch.set_num_threads(hilos)
    global HILOS
    HILOS = hilos


def _correr(argumentos):
    trabajo, carpeta = argumentos
    from codigo import clasicos, motor

    inicio = time.time()
    try:
        if trabajo["modelo"] in clasicos.CLASICOS:
            trabajo = {**trabajo, "hilos": HILOS}
            r = clasicos.ejecutar_clasico(trabajo, carpeta)
        else:
            r = motor.ejecutar(trabajo, carpeta)
        return trabajo["id"], r.get("f1_val"), round(time.time() - inicio, 1), None
    except Exception:
        return trabajo["id"], None, round(time.time() - inicio, 1), traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trabajos")
    ap.add_argument("--trabajadores", type=int, default=4)
    ap.add_argument("--hilos", type=int, default=2)
    a = ap.parse_args()
    ruta = Path(a.trabajos)
    if not ruta.is_absolute():
        ruta = RAIZ_E2 / ruta
    trabajos = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    pendientes = []
    for t in trabajos:
        carpeta = RAIZ_E2 / "resultados" / "corridas" / t["fase"]
        carpeta.mkdir(parents=True, exist_ok=True)
        if not (carpeta / f"{t['id']}.json").exists():
            pendientes.append((t, str(carpeta)))
    # Los trabajos más largos primero, para repartir mejor la carga entre procesos
    pendientes.sort(key=lambda par: -par[0].get("costo", 1.0))
    registro = RAIZ_E2 / "resultados" / "corridas" / f"{ruta.stem}.log"
    print(f"{len(trabajos)} trabajos, {len(pendientes)} pendientes, {a.trabajadores} procesos de {a.hilos} hilos", flush=True)
    hechos, errores, inicio = 0, 0, time.time()
    with mp.get_context("spawn").Pool(a.trabajadores, initializer=_iniciar, initargs=(a.hilos,)) as pool, registro.open("a", encoding="utf-8") as log:
        for ident, f1, segundos, error in pool.imap_unordered(_correr, pendientes):
            hechos += 1
            if error:
                errores += 1
                log.write(f"ERROR {ident}\n{error}\n")
            linea = f"{time.strftime('%H:%M:%S')} {hechos}/{len(pendientes)} {ident} f1_val={f1} {segundos}s"
            log.write(linea + "\n")
            log.flush()
            if hechos % 10 == 0 or error or hechos == len(pendientes):
                print(linea, f"| transcurrido {(time.time() - inicio) / 60:.1f} min | errores {errores}", flush=True)
    print(f"Listo: {hechos} trabajos en {(time.time() - inicio) / 60:.1f} min, {errores} con error", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(RAIZ_E2))
    main()
