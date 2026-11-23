"""Reproducción independiente del Entregable 3 (secciones 11, 12 ítem 13 y 14.5.7 del enunciado).

La ejecuta una persona ajena al código, dentro del contenedor (ver GUIA.md):

    docker run --name repro uavids-e3 python verificacion_externa/reproducir.py

Hace tres comprobaciones y escribe verificacion_externa/salida/resultado_reproduccion.json:

  1. Datos: SHA-256 del CSV de UAVIDS-2025 y de los índices de la prueba sellada del Entregable 1.
  2. Tabla principal: vuelve a correr experiments/analisis_confirmatorio.py en una carpeta temporal y compara, celda por
     celda, la tabla regenerada con la publicada (experiments/resultados/confirmatorio/tabla_contrastes.csv) y el resumen
     (Friedman y rangos). Reporta la discrepancia absoluta máxima y si cambia alguna decisión.
  3. Entrenamiento desde cero: reentrena tres corridas definitivas (Original, LightGBM y CNN 1D con la semilla 6001) y
     compara su F1 macro de prueba y sus probabilidades con las publicadas.

Opciones: --sin-reentrenar omite el paso 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
E3 = AQUI.parent
RAIZ = E3.parent
E2 = RAIZ / "Entregable_2_UAVIDS_2025"
sys.path[:0] = [str(E3 / "training"), str(E3 / "experiments"), str(E2)]
SALIDA = AQUI / "salida"
PUBLICADA = E3 / "experiments" / "resultados" / "confirmatorio"
E2_DEF = E2 / "resultados" / "corridas" / "definitiva"
E3_DEF = E3 / "experiments" / "resultados" / "corridas" / "definitiva"
DECISIONES = ["supera (R3)", "peor con significancia"]
REENTRENAR = [("Original", 6001, E3_DEF), ("LightGBM", 6001, E2_DEF), ("CNN 1D", 6001, E2_DEF)]
NUMERO = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan")


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def numeros(valor):
    """Lista de números de una celda: un número, un booleano o una tupla escrita como texto, p. ej. (np.float64(-0.01), ...)."""
    if isinstance(valor, (bool, np.bool_)):
        return [float(valor)]
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return [float(valor)]
    texto = str(valor).replace("np.float64", "")
    if texto in ("True", "False"):
        return [float(texto == "True")]
    return [float(x) for x in NUMERO.findall(texto)]


def discrepancia(a, b):
    """Diferencia absoluta máxima entre dos celdas; NaN contra NaN cuenta como 0."""
    x, y = numeros(a), numeros(b)
    if len(x) != len(y):
        return float("inf")
    d = [0.0 if (np.isnan(u) and np.isnan(v)) else abs(u - v) for u, v in zip(x, y)]
    return max(d) if d else 0.0


def hojas(objeto, prefijo=""):
    """Hojas numéricas de un JSON anidado, como {ruta: valor}."""
    if isinstance(objeto, dict):
        salida = {}
        for k, v in objeto.items():
            salida.update(hojas(v, f"{prefijo}/{k}"))
        return salida
    if isinstance(objeto, (int, float, bool)):
        return {prefijo: float(objeto)}
    return {}


def paso_datos():
    from codigo import datos as D

    sha_csv = sha256(D.CSV)
    h = D.heredados()  # verifica por dentro el SHA-256 de la prueba sellada
    sha_prueba = hashlib.sha256(np.sort(h["ids"]["prueba"]).tobytes()).hexdigest()
    return {"sha256 CSV": sha_csv, "CSV coincide": sha_csv == D.SHA_CSV, "sha256 prueba": sha_prueba, "prueba coincide": sha_prueba == D.SHA_PRUEBA,
            "flujos de prueba": int(len(h["ids"]["prueba"]))}


def paso_tabla(carpeta):
    import analisis_confirmatorio as AC

    AC.SALIDA = Path(carpeta) / "confirmatorio"
    AC.main()
    pub = pd.read_csv(PUBLICADA / "tabla_contrastes.csv", encoding="utf-8-sig").set_index("rival")
    reg = pd.read_csv(AC.SALIDA / "tabla_contrastes.csv", encoding="utf-8-sig").set_index("rival")
    reg.to_csv(SALIDA / "tabla_contrastes_regenerada.csv", encoding="utf-8-sig")
    por_columna, celdas = {}, 0
    for col in pub.columns:
        if col not in reg.columns or col == "familia":
            continue
        d = [discrepancia(pub.at[r, col], reg.at[r, col]) for r in pub.index]
        por_columna[col] = max(d)
        celdas += len(d)
    cambios = [{"rival": r, "columna": c, "publicada": bool(pub.at[r, c]), "regenerada": bool(reg.at[r, c])}
               for c in DECISIONES for r in pub.index if bool(pub.at[r, c]) != bool(reg.at[r, c])]
    res_pub = hojas(json.loads((PUBLICADA / "resumen.json").read_text(encoding="utf-8")))
    res_reg = hojas(json.loads((AC.SALIDA / "resumen.json").read_text(encoding="utf-8")))
    d_resumen = {k: abs(res_pub[k] - res_reg[k]) for k in res_pub if k in res_reg and not (np.isnan(res_pub[k]) and np.isnan(res_reg[k]))}
    return {"filas": int(len(pub)), "celdas comparadas": celdas, "discrepancia máxima tabla": max(por_columna.values()),
            "columna con la discrepancia máxima": max(por_columna, key=por_columna.get), "discrepancia por columna": por_columna,
            "decisiones que cambian": cambios, "discrepancia máxima resumen (Friedman, rangos, potencia)": max(d_resumen.values()) if d_resumen else None,
            "sha256 tabla publicada": sha256(PUBLICADA / "tabla_contrastes.csv")}


def paso_reentrenar(carpeta):
    import torch

    from codigo import clasicos, motor
    from codigo import plan as plan_e2
    from ejecutar_original import preparar_proceso

    filas = []
    for modelo, semilla, origen in REENTRENAR:
        nombre = f"definitiva_{modelo.replace(' ', '_')}_{semilla}"
        r0 = json.loads((origen / f"{nombre}.json").read_text(encoding="utf-8"))
        preparar_proceso(hilos=int(r0.get("hilos", 2)))
        extra = {k: r0[k] for k in ("arquitectura", "max_epocas", "paciencia", "variante", "particion") if r0.get(k) is not None}
        trabajo = plan_e2.trabajo("definitiva", modelo, r0["hp"], semilla, evaluar=True, **extra)
        inicio = time.time()
        if modelo in clasicos.CLASICOS:
            r = clasicos.ejecutar_clasico({**trabajo, "hilos": int(r0.get("hilos", 2))}, carpeta)
        else:
            r = motor.ejecutar(trabajo, carpeta)
        p_pub = np.load(origen / f"{nombre}_eva.npy")
        p_new = np.load(Path(carpeta) / f"{nombre}_eva.npy")
        filas.append({"modelo": modelo, "semilla": semilla, "F1 publicada": r0["f1_eva"], "F1 reproducida": r["f1_eva"],
                      "discrepancia F1": abs(r0["f1_eva"] - r["f1_eva"]), "discrepancia máxima de probabilidad": float(np.abs(p_pub - p_new).max()),
                      "clases iguales (%)": float(100 * (p_pub.argmax(1) == p_new.argmax(1)).mean()), "épocas publicadas": r0.get("epocas"),
                      "épocas reproducidas": r.get("epocas"), "segundos": round(time.time() - inicio, 1), "hilos": torch.get_num_threads()})
        print(f"  {modelo} {semilla}: F1 publicada {r0['f1_eva']:.6f} | reproducida {r['f1_eva']:.6f}", flush=True)
    return filas


def versiones():
    import importlib.metadata as md

    v = {"python": platform.python_version()}
    for p in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "jupyter-book"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-reentrenar", action="store_true")
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    resultado = {"fecha": datetime.now().isoformat(timespec="seconds"), "equipo": f"{platform.processor() or platform.machine()} | {platform.system()} {platform.release()}",
                 "versiones": versiones()}
    print("1/3 Verificando los datos ...", flush=True)
    resultado["datos"] = paso_datos()
    with tempfile.TemporaryDirectory() as tmp:
        print("2/3 Regenerando la tabla principal (unos minutos) ...", flush=True)
        resultado["tabla principal"] = paso_tabla(tmp)
        if not args.sin_reentrenar:
            print("3/3 Reentrenando tres corridas desde cero ...", flush=True)
            resultado["reentrenamiento"] = paso_reentrenar(tmp)
    resultado["minutos"] = round((time.time() - inicio) / 60, 1)
    t = resultado["tabla principal"]
    resultado["veredicto"] = {
        "datos íntegros": resultado["datos"]["CSV coincide"] and resultado["datos"]["prueba coincide"],
        "discrepancia máxima de la tabla principal": t["discrepancia máxima tabla"],
        "alguna decisión cambia": bool(t["decisiones que cambian"]),
        "discrepancia máxima de F1 al reentrenar": max((f["discrepancia F1"] for f in resultado.get("reentrenamiento", [])), default=None)}
    (SALIDA / "resultado_reproduccion.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n===== RESULTADO =====")
    print(json.dumps(resultado["veredicto"], ensure_ascii=False, indent=1))
    print(f"Guardado en {SALIDA / 'resultado_reproduccion.json'} ({resultado['minutos']} min)")


if __name__ == "__main__":
    main()
