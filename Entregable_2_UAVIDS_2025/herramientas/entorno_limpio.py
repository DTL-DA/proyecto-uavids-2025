"""Prueba de entorno limpio de la lista de verificación B.

Paso 1, con el Python del sistema: crea un entorno virtual nuevo e instala solo requirements.txt.

    python Entregable_2_UAVIDS_2025/herramientas/entorno_limpio.py preparar <carpeta_del_entorno>

Paso 2, con el Python de ese entorno: vuelve a entrenar una muestra de corridas de la ejecución definitiva, con sus
mismas semillas, configuraciones y número de hilos, recalcula el F1 macro medio de cada modelo a partir de las
probabilidades guardadas y escribe resultados/reproduccion_entorno_limpio.json.

    <carpeta_del_entorno>/Scripts/python Entregable_2_UAVIDS_2025/herramientas/entorno_limpio.py verificar

La muestra son las corridas de la primera semilla definitiva de los modelos cuyo entrenamiento original tomó a lo sumo
`--max-segundos`, para que la prueba quepa en unas horas; los modelos excluidos por costo se declaran en el resultado.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

RAIZ_E2 = Path(__file__).resolve().parents[1]
RAIZ = RAIZ_E2.parent
TRABAJOS = RAIZ_E2 / "resultados" / "trabajos"
CORRIDAS = RAIZ_E2 / "resultados" / "corridas"


def preparar(carpeta):
    carpeta = Path(carpeta)
    subprocess.run([sys.executable, "-m", "venv", "--clear", str(carpeta)], check=True)
    py = carpeta / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(RAIZ / "requirements.txt")], check=True)
    print("Entorno listo:", py)


def verificar(max_segundos, trabajadores, hilos):
    sys.path.insert(0, str(RAIZ_E2))
    import numpy as np
    from sklearn.metrics import f1_score

    from codigo import datos

    definitiva = [json.loads(l) for l in (TRABAJOS / "definitiva.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    registros = {t["id"]: json.loads((CORRIDAS / "definitiva" / f"{t['id']}.json").read_text(encoding="utf-8")) for t in definitiva}
    primera = min(t["semilla"] for t in definitiva)
    muestra, excluidos = [], []
    for t in definitiva:
        if t["semilla"] != primera:
            continue
        (muestra if registros[t["id"]].get("segundos", 0) <= max_segundos else excluidos).append(t)
    trabajos = [{**t, "fase": "entorno_limpio"} for t in muestra]
    ruta = TRABAJOS / "entorno_limpio.jsonl"
    ruta.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in trabajos), encoding="utf-8")
    inicio = time.time()
    subprocess.run([sys.executable, str(RAIZ_E2 / "codigo" / "ejecutor.py"), str(ruta), "--trabajadores", str(trabajadores), "--hilos", str(hilos)], check=True, cwd=RAIZ)
    corridas = []
    for t in trabajos:
        nuevo = json.loads((CORRIDAS / "entorno_limpio" / f"{t['id']}.json").read_text(encoding="utf-8"))
        original = registros[t["id"]]
        corridas.append({"Modelo": t["modelo"], "Semilla": t["semilla"], "F1 original": original["f1_eva"], "F1 en entorno limpio": nuevo["f1_eva"],
                         "Diferencia absoluta": abs(original["f1_eva"] - nuevo["f1_eva"])})
    # La tabla principal se recalcula desde las probabilidades guardadas, sin reentrenar
    h = datos.heredados()
    y_pru = h["y"][h["ids"]["prueba"]]
    por_modelo = {}
    for t in definitiva:
        p = np.load(CORRIDAS / "definitiva" / f"{t['id']}_eva.npy")
        por_modelo.setdefault(t["modelo"], []).append(f1_score(y_pru, p.argmax(1), average="macro", zero_division=0))
    import pandas as pd

    referencia = pd.read_csv(RAIZ_E2 / "resultados" / "analisis" / "metricas_definitiva.csv").groupby("modelo")["prueba F1 macro"].mean()
    discrepancia_tabla = max(abs(np.mean(v) - referencia[m]) for m, v in por_modelo.items())
    salida = {"python": platform.python_version(), "sistema": f"{platform.system()} {platform.release()}", "fecha": time.strftime("%Y-%m-%d %H:%M"),
              "horas": round((time.time() - inicio) / 3600, 2), "max_segundos": max_segundos, "hilos": hilos,
              "excluidos_por_costo": sorted({t["modelo"] for t in excluidos}), "corridas": corridas,
              "discrepancia_maxima": max(c["Diferencia absoluta"] for c in corridas), "discrepancia_tabla": float(discrepancia_tabla)}
    (RAIZ_E2 / "resultados" / "reproduccion_entorno_limpio.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in salida.items() if k != "corridas"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["preparar", "verificar"])
    ap.add_argument("carpeta", nargs="?")
    ap.add_argument("--max-segundos", type=float, default=900)
    ap.add_argument("--trabajadores", type=int, default=6)
    ap.add_argument("--hilos", type=int, default=2)
    a = ap.parse_args()
    if a.paso == "preparar":
        preparar(a.carpeta)
    else:
        verificar(a.max_segundos, a.trabajadores, a.hilos)
