"""Tabla maestra de experimentos (sección 12, entregable 8): una fila por corrida.

Incluye las corridas definitivas del problema principal (E2 y E3), la ablación y los tres conjuntos adicionales, con
modelo, conjunto, fase, semilla, métricas, épocas, tiempos y costo computacional.

Uso: python experiments/tabla_maestra.py   ->   experiments/resultados/tabla_maestra.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
E2 = AQUI.parents[1] / "Entregable_2_UAVIDS_2025" / "resultados" / "corridas" / "definitiva"
E3 = AQUI / "resultados" / "corridas"
CAMPOS = ["modelo", "semilla", "f1_val", "acc_val", "bacc_val", "f1_eva", "acc_eva", "f1_ent", "epocas", "mejor_epoca", "segundos", "parametros", "mac",
          "ms_por_muestra", "ms_por_muestra_lote", "memoria_pico_mb", "hilos", "equipo", "dispositivo", "gpu", "determinismo"]


def leer(carpeta, conjunto, fase, origen):
    filas = []
    for p in sorted(carpeta.glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("pendiente_gpu"):  # marcador de una corrida que aún corre en GPU (training/ejecutar_gpu.py)
            continue
        fila = {"id": r.get("id"), "conjunto": r.get("conjunto", conjunto), "fase": fase, "origen": origen, **{c: r.get(c) for c in CAMPOS}}
        if "variante_original" in r:
            fila["modelo"] = f"Original · {r['variante_original']}"
        fila["hp"] = json.dumps(r.get("hp"), ensure_ascii=False)
        filas.append(fila)
    return filas


def main():
    filas = leer(E2, "UAVIDS-2025", "definitiva", "E2")
    filas += leer(E3 / "definitiva", "UAVIDS-2025", "definitiva", "E3")
    filas += leer(E3 / "ablacion", "UAVIDS-2025", "ablacion", "E3")
    filas += leer(E3 / "adicionales", None, "adicionales", "E3")
    for fase in ("ajuste", "piloto", "sensibilidad", "curvas", "cv5x2", "bootstrap", "permutacion"):
        filas += leer(E3 / fase, "UAVIDS-2025", fase, "E3")
    filas += leer(E3 / "diagnostico_adicionales", None, "diagnostico_adicionales", "E3")
    t = pd.DataFrame(filas)
    t["FLOPs"] = 2 * pd.to_numeric(t["mac"], errors="coerce")
    t.to_csv(AQUI / "resultados" / "tabla_maestra.csv", index=False, encoding="utf-8-sig")
    print(t.groupby(["conjunto", "fase"]).size().to_string())
    print(len(t), "corridas")


if __name__ == "__main__":
    main()
