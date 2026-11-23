---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Dashboard y despliegue

El dashboard es una aplicación multipágina de Dash (`use_pages=True`) que se despliega en Render y hace inferencia en CPU con ONNX Runtime. Nunca entrena ni importa PyTorch: carga los artefactos de `artifacts/` y los precalculados de `artifacts/dashboard/`.

## Arquitectura

| Capa | Archivos | Qué hace |
|---|---|---|
| Interfaz | `app.py`, `pages/contexto.py`, `pages/eda.py`, `pages/modelos.py`, `assets/estilos.css` | Una página por sección, barra de navegación y pie con el enlace a este libro |
| Inferencia | `src/inference.py` | Carga bajo demanda de las sesiones ONNX (como mucho 6 vivas), preprocesamiento del E1, *temperature scaling*, conjunto conforme LAC al 90 %, aviso fuera de dominio y explicación local por oclusión |
| Datos y figuras | `src/data.py`, `src/plots.py` | Caché de los datos precalculados y una plantilla común de Plotly con la misma paleta por clase en todas las páginas |
| Precálculo | `training/exportar_onnx.py`, `training/preparar_dashboard.py` | Exportación con prueba de ida y vuelta, calibración y conformal ajustados en validación, diagnósticos, tabla de líderes **alimentada desde la tabla maestra** (`experiments/resultados/tabla_maestra.csv`), conjuntos adicionales, robustez y balance de R3 |
| Pruebas | `tests/test_dashboard.py` | Páginas, probabilidades válidas, coherencia con los metadatos, fuera de dominio, ida y vuelta y **consistencia automática con la tabla maestra y con el análisis de los conjuntos adicionales**, en integración continua |

## Contenido de cada sección

- **Contexto:** portada con la cifra clave, por qué importa, preguntas de investigación, ficha de datos, flujo metodológico, limitaciones, ética y autoría.
- **EDA:** filtros de atributo, clase, partición y escala que actualizan con *callbacks* el balance de clases, la distribución por clase, el panel de estructura por escenario (el específico de este problema), el cambio de distribución entre particiones y la correlación de Spearman. Cada figura tiene su interpretación.
- **Modelos:**
  - tabla de líderes con los 10 del Top 10, los *baselines* y el modelo original, ordenable y filtrable;
  - intervalos del F1 por modelo;
  - predicción en vivo con un flujo de prueba al azar o editado a mano, con probabilidades calibradas, conjunto conforme, aviso fuera de dominio y explicación local;
  - diagnósticos del modelo elegido: matriz de confusión, curva de confiabilidad y F1 por clase frente a LightGBM;
  - tabla de contrastes confirmatorios y diagrama de diferencia crítica;
  - F1 de los 13 modelos en los tres conjuntos adicionales, sin promediar conjuntos;
  - robustez ante ruido, atributos faltantes y valores atípicos, con selector de perturbación;
  - balance de las seis condiciones de R3;
  - ficha del modelo original con su pseudocódigo.

```{code-cell} ipython3
:tags: [hide-input]
from pathlib import Path
import pandas as pd
RAIZ = Path("..").resolve()
lideres = pd.read_csv(RAIZ / "artifacts/dashboard/lideres.csv")
tamanos = {p.parent.name: round(p.stat().st_size / 1024, 1) for p in (RAIZ / "artifacts").glob("*/model.onnx")}
lideres["ONNX (KB)"] = lideres["Modelo"].str.replace(" ", "_").str.replace("í", "i").map(tamanos)
print(f"Tamaño total de los modelos ONNX: {sum(tamanos.values()) / 1024:.1f} MB")
lideres[[c for c in ["Modelo", "Grupo", "Parámetros", "ONNX (KB)", "Latencia mediana (ms)", "Latencia p95 (ms)"] if c in lideres]]
```

## Memoria y rendimiento en el plan gratuito

Los 16 modelos ocupan unos 5 MB en ONNX, así que caben sin cuantizar. Aun así, las sesiones se cargan bajo demanda y se limitan a seis vivas, ONNX Runtime corre con un solo hilo, y los datos del EDA (6,7 MB en Parquet) se cargan una sola vez al iniciar. La aplicación no carga PyTorch, scikit-learn ni LightGBM: solo `onnxruntime`, `dash`, `plotly`, `pandas` y `pyarrow` (`requirements.txt`).

## Despliegue en Render

1. El repositorio de GitHub contiene los tres entregables; `render.yaml` apunta el servicio a la carpeta `Entregable_3_UAVIDS_2025` con `rootDir`.
2. Render instala `requirements.txt` y arranca con `gunicorn app:server --workers 1 --threads 4 --timeout 120`; `healthCheckPath: /` verifica que la portada responda.
3. Las variables `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` y `MKL_NUM_THREADS` se fijan en 1 para no competir por la CPU compartida del plan gratuito.

**Prueba de humo.** Después de cada despliegue se verifica que las tres páginas respondan con código 200, que la predicción en vivo devuelva probabilidades que sumen 1 y que un flujo absurdo active el aviso fuera de dominio. Es la misma batería de `tests/test_dashboard.py`, ejecutada contra la URL pública. El resultado se anexa en el capítulo 11.

## Seguridad y robustez de la interfaz

Las entradas numéricas se validan: un valor vacío no dispara la predicción y uno fuera de rango activa el aviso. La aplicación no guarda datos del usuario ni usa almacenamiento externo. Las explicaciones por oclusión cuestan 16 inferencias por predicción, unos pocos milisegundos en CPU.
