"""Ejecuta las fases del modelo original con el mismo motor, registro y formato de corridas que los rivales del E2.

    python training/ejecutar_original.py ajuste|piloto|definitiva [procesos]

- ajuste: las MISMAS 12 configuraciones (lr, weight_decay, batch) que recibieron los modelos profundos del E2
  (``plan.configuraciones_ajuste()``, semilla de muestreo 2026), con semilla de entrenamiento 4242 y 15 épocas;
  se elige la de mayor F1 macro de validación (regla del E2).
- piloto: semillas 6001 a 6005, solo validación (sin tocar la prueba), para estimar sigma_d frente a cada rival,
  cuyas corridas definitivas del E2 usan las mismas semillas.
- definitiva: semillas 6001 a 6020 (las del E2), evaluación única en prueba, probabilidades y pesos guardados.

``codigo.motor.ejecutar`` del E2 llama a ``entrenar`` por su nombre global; aquí se sustituye solo para el modelo
original por ``entrenar_original.entrenar`` (mismo protocolo, esquema de pérdida cooperativo). Para cualquier
otro modelo se sigue usando el motor original.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
CORRIDAS = AQUI.parent / "experiments" / "resultados" / "corridas"

# Diseño congelado (variante 10 de design_log.md)
CONFIGURACION = {"T": 8, "ocultas_umbral": (64,), "ocultas_suave": (64, 64), "ocultas_compuerta": (32,), "tau_min": 0.05, "gamma": 0.01,
                 "compuerta": "aprendida", "entrada_compuerta": "x"}
SEMILLAS_PILOTO = [6001, 6002, 6003, 6004, 6005]
SEMILLAS_DEFINITIVA = list(range(6001, 6021))


def preparar_proceso(hilos=2):
    import torch

    torch.set_num_threads(hilos)
    import entrenar_original
    import registro
    from codigo import motor

    registro.registrar(**CONFIGURACION)
    original = motor.entrenar if not hasattr(motor, "_entrenar_e2") else motor._entrenar_e2
    motor._entrenar_e2 = original

    def entrenar_despachado(nombre, arquitectura, hp, semilla, D, ids_ent, ids_val, max_epocas=motor.MAX_EPOCAS, paciencia=motor.PACIENCIA, etiquetas=None,
                            variante=None):
        if nombre == registro.NOMBRE:
            return entrenar_original.entrenar(arquitectura, hp, semilla, D, ids_ent, ids_val, max_epocas, paciencia, etiquetas, variante)
        return original(nombre, arquitectura, hp, semilla, D, ids_ent, ids_val, max_epocas, paciencia, etiquetas, variante)

    motor.entrenar = entrenar_despachado


def mejores_hp():
    filas = [json.loads(p.read_text(encoding="utf-8")) for p in (CORRIDAS / "ajuste").glob("ajuste_Original_*.json")]
    if len(filas) < 12:
        raise RuntimeError(f"El ajuste no está completo: {len(filas)} de 12 corridas")
    filas.sort(key=lambda r: (-r.get("f1_val", -1), r["intento"]))  # empate: la configuración que aparece primero
    return filas[0]["hp"]


def plan(fase):
    import registro
    from codigo import plan as plan_e2

    if fase == "ajuste":
        configs = plan_e2.configuraciones_ajuste()["profundos"]
        return [plan_e2.trabajo("ajuste", registro.NOMBRE, hp, 4242, f"_i{i:02d}", intento=i, max_epocas=15) for i, hp in enumerate(configs)]
    hp = mejores_hp()
    if fase == "piloto":
        return [plan_e2.trabajo("piloto", registro.NOMBRE, hp, s, guardar_prob_val=True) for s in SEMILLAS_PILOTO]
    if fase == "definitiva":
        return [plan_e2.trabajo("definitiva", registro.NOMBRE, hp, s, evaluar=True, guardar_prob_val=True, guardar_pesos=True) for s in SEMILLAS_DEFINITIVA]
    raise ValueError(fase)


def correr(trabajo, carpeta):
    from codigo import motor

    return motor.ejecutar(trabajo, carpeta)


def main():
    fase = sys.argv[1]
    procesos = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    preparar_proceso()
    carpeta = CORRIDAS / fase
    carpeta.mkdir(parents=True, exist_ok=True)
    trabajos = plan(fase)
    (carpeta / "_plan.json").write_text(json.dumps({"configuracion": CONFIGURACION, "trabajos": trabajos}, ensure_ascii=False, indent=1), encoding="utf-8")
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=procesos, initializer=preparar_proceso) as ex:
        futuros = {ex.submit(correr, t, str(carpeta)): t for t in trabajos}
        for i, f in enumerate(as_completed(futuros), 1):
            r = f.result()
            print(f"[{i}/{len(trabajos)}] {r['id']}: f1_val={r.get('f1_val', float('nan')):.4f} f1_eva={r.get('f1_eva', float('nan')):.4f} "
                  f"épocas={r.get('epocas')} ({r.get('segundos')} s; {(time.time() - inicio) / 60:.1f} min)", flush=True)
    if fase == "ajuste":
        print("Mejor configuración:", mejores_hp())


if __name__ == "__main__":
    main()
