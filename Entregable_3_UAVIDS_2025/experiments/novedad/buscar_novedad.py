"""Revisión de novedad del modelo original (sección 6.3 del enunciado) en OpenAlex y arXiv.

Scopus y Web of Science se consultan desde su sitio con las mismas cadenas (cadenas_scopus_wos.md) y sus conteos se
anotan en exportaciones/. Las búsquedas reutilizan las funciones del E2 (literatura/buscar_literatura_e2.py).

Uso: python experiments/novedad/buscar_novedad.py
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[2] / "Entregable_2_UAVIDS_2025" / "literatura"))
import buscar_literatura_e2 as B  # noqa: E402

SALIDA = AQUI / "exportaciones"
SALIDA.mkdir(exist_ok=True)
HOY = date.today().isoformat()
DRON = '(UAV OR drone OR FANET OR "unmanned aerial")'
MEZCLA = '("mixture of experts" OR "gating network" OR "gated fusion" OR "dynamic ensemble selection")'
CADENAS = {
    # (i) mecanismo + (ii) problema de aplicación
    "N1": f'{MEZCLA} AND ("intrusion detection" OR "attack detection") AND {DRON}',
    "N2": '("mixture of experts" OR "gating network") AND "intrusion detection" AND ("decision tree" OR "gradient boosting" OR "tree-based" OR "soft tree")',
    "N3": '("piecewise linear encoding" OR "numerical embedding" OR "soft binning" OR "learnable threshold" OR "soft decision tree" OR "neural decision tree") '
          'AND ("intrusion detection" OR "network traffic")',
    "N5": f'{DRON} AND "intrusion detection" AND (hybrid OR ensemble OR fusion) AND ("gradient boosting" OR LightGBM OR XGBoost OR "random forest" OR "decision tree") '
          'AND ("neural network" OR "deep learning")',
    # mecanismo en OTRAS aplicaciones (origen)
    "N4": '("mixture of experts" OR "gating network") AND ("soft decision tree" OR "neural decision tree" OR "differentiable tree" OR "piecewise linear encoding" '
          'OR "numerical embeddings") AND (tabular OR vision OR language OR "time series")',
    "N7": '"Lipschitz" AND ("mixture of experts" OR "gating network")',
    # sinónimos en español
    "N6": '("mezcla de expertos" OR "red de compuerta" OR fusión) AND ("detección de intrusiones") AND (drones OR UAV OR "vehículos aéreos no tripulados")',
}
B.MAXIMOS = {c: 300 for c in CADENAS}
B.DESDE = {c: "2015-01-01" for c in CADENAS}

if __name__ == "__main__":
    resumen = []
    for nombre_fuente, fn in (("OpenAlex", B.openalex), ("arXiv", B.arxiv)):
        for c, q in CADENAS.items():
            try:
                total, regs = fn(c, q)
            except Exception as e:  # una fuente caída no detiene las demás
                print(nombre_fuente, c, "ERROR", e, flush=True)
                resumen.append({"fuente": nombre_fuente, "cadena": c, "total": None, "exportados": 0, "error": str(e)})
                continue
            (SALIDA / f"{HOY}_{nombre_fuente}_{c}.json").write_text(
                json.dumps({"fuente": nombre_fuente, "cadena": q, "fecha": HOY, "total": total, "registros": regs}, ensure_ascii=False, indent=1), encoding="utf-8")
            resumen.append({"fuente": nombre_fuente, "cadena": c, "total": total, "exportados": len(regs)})
            print(nombre_fuente, c, "total", total, "exportados", len(regs), flush=True)
            time.sleep(3)
    (SALIDA / f"{HOY}_resumen_novedad.json").write_text(json.dumps({"fecha": HOY, "cadenas": CADENAS, "resultados": resumen}, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
