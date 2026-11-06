# Detección de intrusiones en redes de drones: baseline de Deep Learning

Primer entregable del proyecto final de Deep Learning (Maestría en Analítica de Datos, Universidad del Norte). Establece, con evidencia estadística, el nivel mínimo de desempeño que un modelo avanzado debe superar al clasificar flujos de red de un enjambre de drones en tráfico normal y cuatro ataques.

El informe completo es el cuaderno [`Entregable 1 - UAVIDS-2025.ipynb`](Entregable%201%20-%20UAVIDS-2025.ipynb), que se entrega ya ejecutado.

## Resultado

| Modelo | F1 macro en prueba (7 semillas) | IC 95 % |
|---|---|---|
| CNN 1D (baseline) | 0,8549 ± 0,0091 | [0,8466; 0,8633] |
| DNN | 0,8548 ± 0,0110 | [0,8446; 0,8649] |
| LSTM | 0,8423 ± 0,0135 | [0,8298; 0,8547] |

Ningún par de modelos difiere tras la corrección de Holm. Un modelo avanzado debe alcanzar un F1 macro de al menos 0,8775 para considerarse una mejora distinguible y relevante, con esta misma partición y semillas pareadas.

## Datos

Se usa [UAVIDS-2025](https://www.kaggle.com/datasets/qinglizeng1997/uavids-2025) (Zenodo, CC BY 4.0, DOI [10.5281/zenodo.15336998](https://doi.org/10.5281/zenodo.15336998)). El CSV no se versiona: si `Entregable_1_UAVIDS_2025/datos/UAVIDS-2025.csv` no existe, el cuaderno lo descarga de Kaggle, y en cualquier caso verifica su suma SHA-256.

## Cómo reproducirlo

```bash
pip install -r requirements.txt
```

Abra el cuaderno desde la raíz del repositorio y ejecute todas las celdas, en CPU. Las corridas ya están guardadas en `Entregable_1_UAVIDS_2025/resultados/corridas/`, de modo que el cuaderno las reutiliza en lugar de volver a entrenar y termina en unos minutos.

## Entregables

La tabla ubica cada producto que pide el enunciado, de modo que se pueda revisar punto por punto. Las rutas parten de `Entregable_1_UAVIDS_2025/`, salvo las del cuaderno y de `requirements.txt`, que están en la raíz.

| # | Entregable | Ubicación |
|---|---|---|
| 1 | Informe en Jupyter Book | `entregables/1_informe_jupyter_book/`, con las fuentes del libro e `Informe_Entregable_1.pdf`; el libro aún no tiene enlace público |
| 2 | Código ejecutable de extremo a extremo y versionado | `Entregable 1 - UAVIDS-2025.ipynb`, `herramientas/`, `literatura/` y `resultados/` |
| 3 | Dataset | `entregables/3_dataset/obtener_datos.py`, que descarga el CSV y verifica su suma SHA-256 |
| 4 | Tabla de síntesis de la literatura y diagrama de flujo | `entregables/4_literatura/`, con la tabla de síntesis, la frecuencia, las arquitecturas, los cuartiles, la verificación de cobertura de sinónimos y el diagrama PRISMA |
| 5 | Tabla comparativa, baseline e inferencia estadística | `entregables/5_benchmark/` |
| 6 | Dependencias, semillas y configuración de cada corrida | `entregables/6_reproducibilidad/` y, con el detalle de cada corrida, `resultados/corridas/` |
| 7 | Declaración a priori | `entregables/7_declaracion_a_priori/` y, con la partición, `resultados/particion.npz` |

Los archivos de `entregables/` se generan a partir del cuaderno, `resultados/` y `literatura/`, de modo que el análisis no depende de ellos. Para regenerarlos se ejecuta `python Entregable_1_UAVIDS_2025/herramientas/organizar_entregables.py`.

## Contenido

Las rutas parten de `Entregable_1_UAVIDS_2025/`.

| Ruta | Contenido |
|---|---|
| `herramientas/crear_cuaderno.py` | Genera el cuaderno; conserva las salidas de las celdas cuyo código no cambió |
| `herramientas/construir_libro.py` | Arma las fuentes del Jupyter Book a partir del cuaderno ejecutado |
| `herramientas/organizar_entregables.py` | Reúne los entregables y arma el ZIP para enviarlos |
| `herramientas/obtener_datos.py` | Descarga el conjunto de datos y verifica su suma SHA-256 |
| `literatura/` | Búsqueda, depuración, cribado, extracción y tabla de frecuencia. Las exportaciones crudas y los resúmenes son texto de terceros y no se versionan; `buscar_literatura.py` vuelve a consultar las fuentes. `verificar_cobertura_sinonimos.py` documenta la verificación de cobertura en OpenAlex, con su cribado en `cribado_cobertura_sinonimos.csv` y su extracción en `estudios_cobertura.json` |
| `resultados/declaracion_a_priori.json` | Métrica primaria, margen δ, unidad de independencia y partición |
| `resultados/particion.npz` y `prueba.sha256` | Partición y suma de verificación del conjunto de prueba congelado |
| `resultados/preregistro.json` | Piloto y número de ejecuciones, fechado antes de la fase definitiva |
| `resultados/corridas/` | Semilla, hiperparámetros, curvas, tiempo y equipo de cada corrida |
| `resultados/sensibilidad/` y `sensibilidad_dropout.csv` | Corridas y comparación con el *dropout* del código de los autores |
| `resultados/baseline.json` | Baseline y umbral de mejora para los Entregables 2 y 3 |
