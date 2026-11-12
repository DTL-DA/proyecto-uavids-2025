"""Encadena las fases del benchmark en el orden del preregistro, con un solo comando.

    python Entregable_2_UAVIDS_2025/herramientas/orquestar.py [--desde <paso>] [--hasta <paso>] [--trabajadores 6] [--hilos 2]

Pasos: ajuste, piloto_ruido, fase1, fase2, definitiva, complementarios, analisis. Cada paso genera sus trabajos con
codigo/plan.py, los ejecuta con codigo/ejecutor.py, que salta lo ya hecho, y, cuando el preregistro lo pide, aplica la
regla correspondiente con codigo/adenda.py y versiona la adenda antes de seguir.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RAIZ_E2 = Path(__file__).resolve().parents[1]
RAIZ = RAIZ_E2.parent
PY = sys.executable
PASOS = ["ajuste", "piloto_ruido", "fase1", "fase2", "definitiva", "complementarios", "analisis"]


def correr(*argumentos):
    print(time.strftime("%H:%M"), " ".join(str(a) for a in argumentos), flush=True)
    subprocess.run([PY, *map(str, argumentos)], check=True, cwd=RAIZ)


def ejecutar(nombre, a):
    correr(RAIZ_E2 / "codigo" / "ejecutor.py", RAIZ_E2 / "resultados" / "trabajos" / f"{nombre}.jsonl", "--trabajadores", a.trabajadores, "--hilos", a.hilos)


def versionar(mensaje, etiqueta):
    """Commit local de la adenda del preregistro, para fecharla antes de la fase siguiente."""
    archivos = ["Entregable_2_UAVIDS_2025/resultados/preregistro_e2_adenda.json"]
    subprocess.run(["git", "add", *archivos], cwd=RAIZ, check=False)
    r = subprocess.run(["git", "commit", "-m", mensaje + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *archivos], cwd=RAIZ, check=False)
    if r.returncode == 0:
        subprocess.run(["git", "tag", "-a", etiqueta, "-m", mensaje], cwd=RAIZ, check=False)


def unir(nombre, partes):
    lineas = []
    for p in partes:
        lineas += (RAIZ_E2 / "resultados" / "trabajos" / f"{p}.jsonl").read_text(encoding="utf-8").splitlines()
    (RAIZ_E2 / "resultados" / "trabajos" / f"{nombre}.jsonl").write_text("\n".join(lineas), encoding="utf-8")


def mejores_por_validacion(n=3):
    sys.path.insert(0, str(RAIZ_E2))
    from codigo import plan, resultados

    tabla = resultados.tabla_fase("definitiva")
    medias = tabla[tabla["modelo"].isin(plan.profundos())].groupby("modelo")["f1_val"].mean().sort_values(ascending=False)
    return medias.index[:n].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="ajuste", choices=PASOS)
    ap.add_argument("--hasta", default="analisis", choices=PASOS)
    ap.add_argument("--trabajadores", default="6")
    ap.add_argument("--hilos", default="2")
    a = ap.parse_args()
    plan = RAIZ_E2 / "codigo" / "plan.py"
    adenda = RAIZ_E2 / "codigo" / "adenda.py"
    for paso in PASOS[PASOS.index(a.desde):PASOS.index(a.hasta) + 1]:
        if paso == "ajuste":
            correr(plan, "ajuste")
            ejecutar("ajuste", a)
        elif paso == "piloto_ruido":
            correr(plan, "piloto")
            correr(plan, "ruido")
            unir("piloto_ruido", ["piloto", "ruido"])
            ejecutar("piloto_ruido", a)
            correr(adenda, "piloto")
            correr(adenda, "busqueda")
            versionar("Entregable 2: adenda del preregistro con R, la limitación del paper guía y k", "preregistro-e2-adenda-piloto")
        elif paso == "fase1":
            correr(plan, "fase1")
            ejecutar("fase1", a)
        elif paso == "fase2":
            correr(plan, "fase2")
            ejecutar("fase2", a)
            correr(adenda, "busqueda")
            versionar("Entregable 2: adenda del preregistro con las arquitecturas ganadoras de la fase 2", "preregistro-e2-adenda-arquitecturas")
        elif paso == "definitiva":
            correr(plan, "definitiva")
            ejecutar("definitiva", a)
        elif paso == "complementarios":
            top3 = mejores_por_validacion()
            baratos = [m for m in ["DNN", "Transformer", "KAN", "Autoencoder", "DBN", "ResNet", "LightGBM", "Logística"]]
            correr(plan, "cv5x2", ",".join([top3[0], "LightGBM"]))
            correr(plan, "ablacion", top3[0])
            correr(plan, "curvas", ",".join(dict.fromkeys([top3[0], "DNN", "LightGBM"])))
            correr(plan, "permutacion", ",".join(dict.fromkeys([m for m in top3 if m in baratos] + ["DNN", "LightGBM"])), "200")
            correr(plan, "bootstrap", ",".join(dict.fromkeys([top3[0], "DNN", "LightGBM"])), "20")
            unir("complementarios", ["cv5x2", "ablacion", "curvas", "permutacion", "bootstrap"])
            ejecutar("complementarios", a)
        elif paso == "analisis":
            top3 = mejores_por_validacion()
            correr(RAIZ_E2 / "codigo" / "analisis.py", "metricas")
            correr(RAIZ_E2 / "codigo" / "analisis.py", "perturbaciones")
            correr(RAIZ_E2 / "codigo" / "analisis.py", "atribuciones", ",".join(top3))
            correr(RAIZ_E2 / "codigo" / "analisis.py", "mc_dropout", ",".join(top3))
    print(time.strftime("%H:%M"), "Orquestación completa", flush=True)


if __name__ == "__main__":
    main()
