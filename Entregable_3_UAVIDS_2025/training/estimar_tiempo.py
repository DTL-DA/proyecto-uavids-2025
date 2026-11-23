"""Hora estimada de fin de la cadena de corridas, con tiempos MEDIDOS (no con los costos relativos del E2).

    python training/estimar_tiempo.py

Método:
  1. Duración de cada corrida pendiente = media de los segundos reales ya medidos para ese (conjunto, modelo).
  2. Si ese par aún no tiene corridas: segundos del modelo en las definitivas de UAVIDS-2025 (E2/E3) por el factor del
     conjunto (mediana, sobre los modelos ya medidos en ese conjunto, de segundos_conjunto / segundos_UAVIDS). UNR-IDD,
     sin mediciones, toma el factor de UAV-NIDD (tamaño parecido; cota superior: tiene menos atributos).
  3. Simulación del reparto real: 6 procesos que toman las corridas en el orden del ejecutor (mayor costo primero).
     Las 6 en curso cuentan con la mitad de su duración por delante.
  4. Se repite con duraciones ×0,8 y ×1,3 para dar un rango, no una hora única.
"""
from __future__ import annotations

import heapq
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI), str(AQUI.parent), str(AQUI.parents[1] / "Entregable_2_UAVIDS_2025")]
import ejecutar_definitivas as ED  # noqa: E402

ADI = ED.CORRIDAS / "adicionales"
PROCESOS = 6


def leer(carpeta, patron="*.json"):
    for p in carpeta.glob(patron):
        r = json.loads(p.read_text(encoding="utf-8"))
        if not r.get("pendiente_gpu") and r.get("segundos"):
            yield r


def duraciones():
    uavids = defaultdict(list)
    for carpeta in (ED.E2_DEF, ED.CORRIDAS / "definitiva"):
        for r in leer(carpeta, "definitiva_*.json"):
            uavids[r["modelo"]].append(r["segundos"])
    uavids = {m: mean(v) for m, v in uavids.items()}
    medidas = defaultdict(list)
    for r in leer(ADI):
        medidas[(r["conjunto"], r["modelo"])].append(r["segundos"])
    medidas = {k: mean(v) for k, v in medidas.items()}
    factor = {}
    for c in ("uav_nidd", "unsw_nb15", "unr_idd"):
        cocientes = [s / uavids[m] for (cc, m), s in medidas.items() if cc == c and m in uavids]
        factor[c] = median(cocientes) if cocientes else None
    factor["unr_idd"] = factor["unr_idd"] or factor["uav_nidd"]

    def dur(c, m):
        return medidas.get((c, m)) or uavids[m] * factor[c]
    return dur, uavids, medidas, factor


def simular(duracion_trabajos, en_curso, escala, inicio=0.0):
    trabajadores = [inicio + escala * d / 2 for d in en_curso] + [inicio] * (PROCESOS - len(en_curso))
    heapq.heapify(trabajadores)
    for d in duracion_trabajos:
        t = heapq.heappop(trabajadores)
        heapq.heappush(trabajadores, t + escala * d)
    return max(trabajadores)


def main():
    dur, uavids, medidas, factor = duraciones()
    hechos = {p.stem for p in ADI.glob("*.json") if not json.loads(p.read_text(encoding="utf-8")).get("pendiente_gpu")}
    marcados = {p.stem for p in ADI.glob("*.json")} - hechos
    log = (ED.CORRIDAS.parent / "registros" / "adicionales.log").read_text(encoding="utf-8")
    vistos = sum(1 for l in log.splitlines() if l.startswith("["))
    orden = ED.plan_adicionales()  # mismo orden que el ejecutor
    pendientes_fase1 = [t for t in orden[vistos:] if t["id"] not in hechos and t["id"] not in marcados]
    en_curso = [dur(t["conjunto"], t["modelo"]) for t in orden[:vistos + PROCESOS] if t["id"] not in hechos and t["id"] not in marcados][:PROCESOS]
    fase1 = [dur(t["conjunto"], t["modelo"]) for t in pendientes_fase1[len(en_curso):]]
    fase2 = [dur(t["conjunto"], t["modelo"]) for t in ED.plan_adicionales(["BiLSTM", "GRU"]) if t["id"] in marcados]
    s_orig = uavids["Original"]
    robustez = [s_orig] * 24 + [s_orig * f for f in (0.1, 0.25, 0.5, 1.0) for _ in range(3)] + [s_orig * 0.5] * 10 + [s_orig] * 20 + [s_orig * 0.6] * 200
    robustez += [uavids["LightGBM"] * f for f in (0.1, 0.25, 0.5, 1.0) for _ in range(3)] + [uavids["CNN-LSTM"] * f for f in (0.1, 0.25, 0.5, 1.0) for _ in range(3)]
    ahora = datetime.now()
    print(f"Factores por conjunto (segundos / segundos en UAVIDS): { {k: round(v, 2) for k, v in factor.items()} }")
    print(f"Fase 1: {len(en_curso)} en curso + {len(fase1)} pendientes | fase 2 (BiLSTM y GRU): {len(fase2)} | fase 3 (robustez): {len(robustez)}")
    for nombre, escala in (("optimista ×0,8", 0.8), ("central", 1.0), ("pesimista ×1,3", 1.3)):
        f1 = simular(fase1, en_curso, escala)
        f2 = simular(sorted(fase2, reverse=True), [], escala, f1)
        f3 = simular(sorted(robustez, reverse=True), [], escala, f2)
        h = lambda s: (ahora + timedelta(seconds=s)).strftime("%H:%M")  # noqa: E731
        print(f"{nombre:>15}: otros modelos {h(f1)} | BiLSTM y GRU {h(f2)} | robustez {h(f3)}")


if __name__ == "__main__":
    main()
