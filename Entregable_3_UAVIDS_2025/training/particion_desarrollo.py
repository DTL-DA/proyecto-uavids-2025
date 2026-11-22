"""Partición de desarrollo del modelo original (sección 7.2 del enunciado).

El diseño del modelo original se toma solo con filas de ENTRENAMIENTO del E2. La validación y la prueba
del E2 quedan para la confirmación y no influyen en ninguna decisión de diseño. Las filas de entrenamiento
se reparten por bloques de 1000 flujos (la misma unidad de la partición del E1 y el E2) en ajuste (70 %),
parada temprana (15 %) y evaluación del diseño (15 %). Hay tres reparticiones (semillas 1, 2 y 3) para
ver la variabilidad debida a la partición.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "Entregable_2_UAVIDS_2025"))
from codigo import datos as D  # noqa: E402  (datos, partición y transformación heredados del E1 y el E2)

SEMILLAS_PARTICION = (1, 2, 3)
FRACCIONES = (0.70, 0.15, 0.15)


def particion_desarrollo(semilla: int) -> dict[str, np.ndarray]:
    """Devuelve los índices de filas de 'ajuste', 'parada' y 'diseno' para una repartición de desarrollo."""
    _, bloques, _ = D.crudo()
    ent = np.sort(D.heredados()["ids"]["entrenamiento"])
    unicos = np.unique(bloques[ent])
    orden = np.random.default_rng(semilla).permutation(unicos)
    n_aj = int(round(FRACCIONES[0] * len(unicos)))
    n_pa = int(round(FRACCIONES[1] * len(unicos)))
    grupos = {"ajuste": orden[:n_aj], "parada": orden[n_aj:n_aj + n_pa], "diseno": orden[n_aj + n_pa:]}
    return {k: ent[np.isin(bloques[ent], v)] for k, v in grupos.items()}


def matrices(ids_ajuste: np.ndarray):
    """Matriz estandarizada con las estadísticas de las filas de ajuste y pesos de clase de esas filas."""
    X, pesos = D.preparar_pliegue(ids_ajuste)
    return X, pesos, D.etiquetas()


if __name__ == "__main__":
    for s in SEMILLAS_PARTICION:
        p = particion_desarrollo(s)
        y = D.etiquetas()
        print(s, {k: (len(v), np.bincount(y[v], minlength=5).tolist()) for k, v in p.items()})
