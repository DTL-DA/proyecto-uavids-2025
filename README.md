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

## Contenido

Las rutas parten de `Entregable_1_UAVIDS_2025/`.

| Ruta | Contenido |
|---|---|
| `herramientas/crear_cuaderno.py` | Genera el cuaderno; conserva las salidas de las celdas cuyo código no cambió |
| `herramientas/construir_libro.py` | Arma las fuentes del Jupyter Book a partir del cuaderno ejecutado |
| `literatura/` | Búsqueda, depuración, cribado, extracción y tabla de frecuencia. Las exportaciones crudas y los resúmenes son texto de terceros y no se versionan; `buscar_literatura.py` vuelve a consultar las fuentes |
| `resultados/declaracion_a_priori.json` | Métrica primaria, margen δ, unidad de independencia y partición |
| `resultados/particion.npz` y `prueba.sha256` | Partición y suma de verificación del conjunto de prueba congelado |
| `resultados/preregistro.json` | Piloto y número de ejecuciones, fechado antes de la fase definitiva |
| `resultados/corridas/` | Semilla, hiperparámetros, curvas, tiempo y equipo de cada corrida |
| `resultados/baseline.json` | Baseline y umbral de mejora para los Entregables 2 y 3 |
