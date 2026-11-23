"""Corridas de robustez y adecuación del protocolo (secciones 6 y 10 de protocolo.md), con el motor del E2.

    python training/ejecutar_robustez.py [procesos]

Fases (cada corrida se salta si su JSON ya existe):
  sensibilidad  tau_min en {0,02; 0,05; 0,1; 0,2}, T en {4; 8; 16}, gamma en {0; 0,01; 0,1}; semillas 6001-6003 (validación)
  curvas        10, 25, 50 y 100 % del entrenamiento; semillas 7101-7103; original, LightGBM y CNN-LSTM
  cv5x2         original en las 10 particiones 5 x 2 del E2 (para la t de Nadeau-Bengio frente a LightGBM y CNN 1D)
  permutacion   200 permutaciones de etiquetas del entrenamiento (prueba de permutación de la tabla A del E2)
  bootstrap     20 muestras bootstrap de bloques (optimismo de Efron y Gong), particiones del E2
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from ejecutar_definitivas import hp_e2  # noqa: E402
from ejecutar_original import CONFIGURACION, CORRIDAS, mejores_hp, preparar_proceso  # noqa: E402


def plan():
    from codigo import plan as plan_e2

    hp = mejores_hp()
    t = []
    variantes = [{"tau_min": v} for v in (0.02, 0.05, 0.1, 0.2)] + [{"T": v} for v in (4, 16)] + [{"gamma": v} for v in (0.0, 0.1)]
    for cambio in variantes:
        clave = "_".join(f"{k}{v}" for k, v in cambio.items())
        t += [plan_e2.trabajo("sensibilidad", "Original", hp, s, f"_{clave}", config_original=cambio, guardar_prob_val=True, guardar_pesos=True)
              for s in (6001, 6002, 6003)]
    for modelo in ("Original", "LightGBM", "CNN-LSTM"):
        h, arq = (hp, None) if modelo == "Original" else hp_e2(modelo)
        extra = {"arquitectura": arq} if arq else {}
        t += [plan_e2.trabajo("curvas", modelo, h, s, f"_p{int(100 * f):03d}", fraccion_ent=f, **extra) for f in (0.1, 0.25, 0.5, 1.0) for s in (7101, 7102, 7103)]
    cv = sorted(p.stem for p in (AQUI.parents[1] / "Entregable_2_UAVIDS_2025" / "resultados" / "particiones").glob("cv5x2_*.npz"))
    t += [plan_e2.trabajo("cv5x2", "Original", hp, 6001, "_" + n, particion=n, evaluar=True, guardar_prob_eva=False) for n in cv]
    boot = sorted(p.stem for p in (AQUI.parents[1] / "Entregable_2_UAVIDS_2025" / "resultados" / "particiones").glob("bootstrap_*.npz"))
    t += [plan_e2.trabajo("bootstrap", "Original", hp, 6001, "_" + n, particion=n, evaluar=True, guardar_prob_eva=False) for n in boot]
    t += [plan_e2.trabajo("permutacion", "Original", hp, 6001, f"_b{b:03d}", permutar_etiquetas=9000 + b, evaluar=True, guardar_prob_eva=False) for b in range(200)]
    return sorted(t, key=lambda x: -x["costo"])


def correr(trabajo):
    import registro
    from codigo import clasicos, motor

    carpeta = CORRIDAS / trabajo["fase"]
    carpeta.mkdir(parents=True, exist_ok=True)
    if trabajo["modelo"] in clasicos.CLASICOS:
        return clasicos.ejecutar_clasico({**trabajo, "hilos": 2}, carpeta)
    registro.registrar(**{**CONFIGURACION, **trabajo.get("config_original", {})})
    return motor.ejecutar(trabajo, carpeta)


def main():
    procesos = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    preparar_proceso()
    trabajos = plan()
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=procesos, initializer=preparar_proceso) as ex:
        futuros = {ex.submit(correr, t): t for t in trabajos}
        for i, f in enumerate(as_completed(futuros), 1):
            try:
                r = f.result()
                print(f"[{i}/{len(trabajos)}] {r['id']}: f1_val={r.get('f1_val', float('nan')):.4f} ({(time.time() - inicio) / 60:.1f} min)", flush=True)
            except Exception as e:
                print(f"[{i}/{len(trabajos)}] ERROR en {futuros[f]['id']}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
