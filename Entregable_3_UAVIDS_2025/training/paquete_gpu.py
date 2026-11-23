"""Paquete de ida y vuelta para la máquina con GPU (BiLSTM y GRU de los conjuntos adicionales; ver ejecutar_gpu.py).

    python training/paquete_gpu.py armar       ->  ../paquete_gpu.zip   (código, matrices procesadas y marcadores)
    python training/paquete_gpu.py integrar    <-  resultados_gpu.zip   (solo reemplaza corridas aún marcadas como pendientes)

Al integrar, una corrida que la CPU terminó mientras tanto se conserva y la de GPU se descarta (se informa cuál).
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
E3 = AQUI.parent
RAIZ = E3.parent
E2 = RAIZ / "Entregable_2_UAVIDS_2025"
ADICIONALES = E3 / "experiments" / "resultados" / "corridas" / "adicionales"
MODELOS = ("BiLSTM", "GRU")


def pendiente(ruta):
    return not ruta.exists() or json.loads(ruta.read_text(encoding="utf-8")).get("pendiente_gpu", False)


def armar():
    rutas = sorted((E2 / "codigo").glob("*.py")) + [E2 / "configuracion" / "experimento.json", E2 / "resultados" / "preregistro_e2_adenda.json"]
    rutas += [E2 / "resultados" / "corridas" / "definitiva" / f"definitiva_{m}_6001.json" for m in MODELOS]
    rutas += sorted(AQUI.glob("*.py")) + sorted((E3 / "src").glob("*.py")) + [E3 / "requirements-gpu.txt"]
    rutas += [E3 / "data" / "raw" / c / "procesado.npz" for c in ("uav_nidd", "unsw_nb15", "unr_idd")]
    rutas += [p for m in MODELOS for p in sorted(ADICIONALES.glob(f"*_{m}_*.json"))]
    destino = RAIZ / "paquete_gpu.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for p in rutas:
            z.write(p, p.relative_to(RAIZ).as_posix())
    marcadas = sum(pendiente(p) for m in MODELOS for p in ADICIONALES.glob(f"*_{m}_*.json"))
    print(f"{destino} | {len(rutas)} archivos | {destino.stat().st_size / 1e6:.1f} MB | {marcadas} corridas pendientes de GPU")


def integrar(origen=E3 / "resultados_gpu.zip"):
    nuevas, conservadas = [], []
    with zipfile.ZipFile(origen) as z:
        nombres = set(z.namelist())
        for n in sorted(x for x in nombres if x.endswith(".json")):
            local = E3 / n
            if not pendiente(local):
                conservadas.append(local.stem)
                continue
            npy = n[:-5] + "_eva.npy"
            if npy not in nombres:
                raise FileNotFoundError(f"{npy} falta en {origen}")
            (E3 / npy).write_bytes(z.read(npy))
            local.write_bytes(z.read(n))
            nuevas.append(local.stem)
    faltan = [p.stem for m in MODELOS for p in ADICIONALES.glob(f"*_{m}_*.json") if pendiente(p)]
    print(json.dumps({"integradas desde GPU": len(nuevas), "conservadas de CPU (GPU descartada)": conservadas, "siguen pendientes": faltan},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"armar": armar, "integrar": integrar}[sys.argv[1]]()
