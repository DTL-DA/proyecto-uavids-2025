from .base import code, md

md("portada", r"""
# Entregable 2 · Benchmark de Deep Learning para detectar intrusiones en redes de drones

**Estado del arte sistemático, EDA orientado a decisiones, benchmark de 10 modelos, ruta metodológica para datos tabulares e inferencia estadística**

| | |
|---|---|
| **Autor** | Donny Torres |
| **Programa** | Maestría en Analítica de Datos, Universidad del Norte (Barranquilla) |
| **Asignatura** | Deep Learning |
| **Fecha** | Noviembre de 2026 |
| **Conjunto de datos** | *UAVIDS-2025*, versión 1 (`UAVIDS-2025.csv`) |
| **Fuente** | Kaggle: <https://www.kaggle.com/datasets/qinglizeng1997/uavids-2025>; Zenodo, licencia CC BY 4.0, DOI <https://doi.org/10.5281/zenodo.15336998> |
| **Tipo de problema** | Clasificación con datos tabulares, multiclase de cinco categorías (ruta F1 del enunciado) |
| **Preregistro** | `Entregable_2_UAVIDS_2025/preregistro_e2.md`, etiqueta `preregistro-e2` del repositorio, fechado el 12 de noviembre de 2026 antes del primer entrenamiento |
| **Repositorio** | Pendiente de publicación |
| **Jupyter Book** | Pendiente de publicación |

Este cuaderno desarrolla el segundo entregable del proyecto final de Deep Learning. El propósito es comparar, bajo un mismo protocolo y sin fuga de información, diez modelos de Deep Learning seleccionados de la literatura con un puntaje explícito, los tres modelos base del Entregable 1 y tres líneas base no profundas, para responder con inferencia estadística si alguna arquitectura supera de forma relevante el baseline del proyecto, cómo se comparan frente a los árboles potenciados y qué limitación del paper guía queda abierta para el modelo original del Entregable 3.

El entregable hereda sin cambios lo que fijó el Entregable 1: la métrica primaria (F1 macro), el margen δ = 0,01, la unidad de independencia (la serie de flujos, aproximada con bloques de 1000) y la partición congelada con su suma SHA-256. Lo que cambia es la escala: 16 modelos, 12 configuraciones de ajuste por modelo, la búsqueda de 1000 arquitecturas para los cuatro modelos que la literatura no define y la batería de pruebas de las tablas A, B y C del enunciado.
""")

md("resumen", r"""
## Resumen

**Problema.** Detectar y clasificar intrusiones en redes de drones con los 16 atributos de flujo de *UAVIDS-2025* (cinco clases), con el F1 macro como métrica primaria y una partición agrupada por series, para evitar la fuga entre flujos casi gemelos.

**Método.** Se comparan 16 modelos con el mismo pipeline, presupuesto de ajuste y 20 semillas: los diez más recurrentes en una revisión sistemática de 43 estudios, los tres base del Entregable 1, LightGBM, una regresión logística y un trivial. Los cuatro sin arquitectura definida pasan por una búsqueda de 1000 arquitecturas. El protocolo se preregistró, y las diferencias se prueban con *t* pareada, Wilcoxon, Holm, TOST y un *Model Confidence Set*.

**Resultados.** Ningún modelo del Top 10 supera al baseline, la CNN 1D (F1 = 0,8584), por más de δ = 0,01, y tres son equivalentes a él. LightGBM obtiene el mayor F1 (0,8748) y supera a la CNN 1D por 0,0164 (p < 0,001), pero en una validación cruzada 5 × 2 la diferencia es de −0,0003. La serie de prueba explica el 93,1 % de la varianza del F1, y la confusión entre Blackhole y Wormhole, de 12,3 % a 18,5 %, es compartida por las diez familias. El resultado clave del paper guía no se reproduce: 0,9285 frente a 0,94.

**Conclusión.** En este problema la serie pesa más que la arquitectura, y el margen de mejora está en separar los dos ataques de enrutamiento, que es el hueco de investigación para el Entregable 3.
""")

md("entorno-md", r"""
## Configuración del entorno

El cuaderno se ejecuta en CPU de principio a fin. Los entrenamientos no corren dentro del cuaderno, sino con el ejecutor de `Entregable_2_UAVIDS_2025/codigo/ejecutor.py`, que reparte los trabajos entre varios procesos, guarda cada corrida al terminar y salta las que ya existen; el cuaderno lee esas corridas, calcula las métricas y aplica las pruebas. Así, volver a ejecutar el cuaderno no reentrena nada y una interrupción no obliga a repetir lo hecho. Las reglas de reproducibilidad son tres: los datos se obtienen con un solo paso y se verifica su suma SHA-256; la partición, las semillas y la configuración están versionadas, y las versiones de las librerías quedan registradas.
""")

code("entorno-datos", r"""
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
RAIZ_E2 = Path("Entregable_2_UAVIDS_2025")
sys.path.insert(0, str(RAIZ_E2.resolve()))
from codigo import datos as D
from codigo import estadistica as E
from codigo import modelos as MOD
from codigo.informe import configurar, enumerar, interpretar, intervalo, num, p_exacto, pct, tabla, valor_p
from codigo.informe import GRIS_CLARO, GRIS_MEDIO, GRIS_OSCURO, LINEAS, MARCADORES, NEGRO

configurar()
URL = "https://www.kaggle.com/api/v1/datasets/download/qinglizeng1997/uavids-2025"
if not D.CSV.exists():
    D.CSV.parent.mkdir(parents=True, exist_ok=True)
    comprimido = D.CSV.parent / "archive.zip"
    subprocess.run(["curl", "-L", "-sS", "-o", str(comprimido), URL], check=True)
    with zipfile.ZipFile(comprimido) as z:
        z.extract("UAVIDS-2025.csv", D.CSV.parent)
    comprimido.unlink()
datos, bloques, x_log = D.crudo()
H = D.heredados()
y = H["y"]
CLASES = H["clases"]
K = len(CLASES)
ATRIBUTOS = H["atributos"]
ID_ENT, ID_VAL, ID_PRU = H["ids"]["entrenamiento"], H["ids"]["validacion"], H["ids"]["prueba"]
CONFIG = json.loads((RAIZ_E2 / "configuracion" / "experimento.json").read_text(encoding="utf-8"))
DELTA, ALFA, DELTA_G = CONFIG["metricas"]["delta"], CONFIG["metricas"]["alfa"], CONFIG["metricas"]["delta_g"]
RESULTADOS = RAIZ_E2 / "resultados"
ANALISIS = RESULTADOS / "analisis"
ANALISIS.mkdir(parents=True, exist_ok=True)
print("SHA-256 del CSV verificado:", D.SHA_CSV)
print("SHA-256 de la prueba verificado:", D.SHA_PRUEBA)
print(f"Filas: {num(len(datos), 0)} | entrenamiento {num(len(ID_ENT), 0)} | validación {num(len(ID_VAL), 0)} | prueba {num(len(ID_PRU), 0)}")
""")

code("entorno-versiones", r"""
PAQUETES = ["torch", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "lightgbm", "xgboost", "pytorch-tabnet", "rtdl_revisiting_models"]
tabla(pd.DataFrame({"Componente": ["Python", "Sistema operativo", *PAQUETES, "Procesador"],
                    "Versión": [platform.python_version(), f"{platform.system()} {platform.release()}", *map(version, PAQUETES), platform.processor()]}))
""")
