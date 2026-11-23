# Entregable 3 · Modelo original y validación rigurosa (UAVIDS-2025)

Proyecto Final de Deep Learning, Maestría en Analítica de Datos, Universidad del Norte. Autor: Donny Torres.

Este entregable propone un modelo original para detectar intrusiones en redes de drones: una **compuerta por flujo entre un experto de escalones suaves con umbrales aprendibles y un experto neuronal suave**, con entrenamiento cooperativo y una cota de Lipschitz demostrada. El modelo se compara con el Top 10 del Entregable 2, LightGBM y CNN 1D bajo un protocolo preregistrado, en la prueba sellada de UAVIDS-2025 y en tres conjuntos adicionales (UAV-NIDD, UNSW-NB15 y UNR-IDD), y se despliega en un dashboard junto con los demás modelos.

## Resultados principales

- **R3 no se cumple, y el tope de calificación es 4,0.** El original obtiene un F1 macro de 0,850 ± 0,012 con 48 semillas, menor que el de 8 de los 12 rivales. Ningún contraste rechaza con tres pruebas, ni con las preregistradas ni con las seis admisibles del cuadro 4 del enunciado. En los conjuntos adicionales no es el mejor en ninguno.
- **El original es el más robusto ante ruido y atributos faltantes.** Con ruido σ = 0,5 su F1 cae 0,182, frente a 0,324 de LightGBM. La cota de Lipschitz, sin embargo, no explica esa robustez (P4 falsa).
- **El escenario de simulación pesa más que la arquitectura.** LightGBM gana en la prueba sellada (0,875), pero en la validación cruzada 5 × 2 empata con el original y con CNN 1D (0,860). Además, ningún modelo reconoce un escenario no visto: el acierto es menor del 10 %.

El detalle está en el libro, publicado en **https://dtl-da.github.io/proyecto-uavids-2025/entregable-3/**, y el dashboard está desplegado en **https://uavids-2025-dashboard.onrender.com** (plan gratuito de Render: la primera visita tras 15 minutos sin uso tarda cerca de un minuto). Su fuente es `book/`, que se compila ejecutando todas las celdas sobre los resultados versionados, y se publica con `python book/publicar.py` en la rama `gh-pages`.

## Estructura

| Carpeta o archivo | Contenido |
|---|---|
| `src/original_model.py` | El modelo original (escalones suaves, expertos, compuerta, pérdida cooperativa y cota de Lipschitz) |
| `src/inference.py`, `src/data.py`, `src/plots.py`, `app.py`, `pages/`, `assets/` | Dashboard en Dash con inferencia en ONNX Runtime |
| `training/` | Registro del modelo en el motor del E2, ajuste, corridas definitivas, ablación, conjuntos adicionales, robustez, exportación a ONNX y preparación del dashboard |
| `experiments/` | Estándar, procedencia, protocolo preregistrado, bitácora de diseño, búsqueda de novedad y análisis (confirmatorio, cuadro 4, ablación, adicionales, robustez y adecuación) |
| `experiments/resultados/` | Registros JSON por corrida, tablas y resúmenes; `tabla_maestra.csv` tiene una fila por corrida |
| `artifacts/` | 16 modelos en ONNX con su `metadata.json`, y los precalculados del dashboard |
| `book/` | Jupyter Book: introducción y capítulos 1 a 11; `crear_cuaderno.py` arma con ellos el cuaderno ejecutado `Entregable 3 - UAVIDS-2025.ipynb` de la raíz |
| `tests/` | Pruebas unitarias del modelo y del dashboard |
| `verificacion_externa/` | Scripts de reproducción (`reproducir.py`), del paquete de reproducción (`empaquetar.py`) y de la prueba de humo del dashboard (`humo_dashboard.py`) |
| `MODEL_CARD.md`, `DATA_STATEMENT.md`, `CITATION.cff`, `LICENSE` | Ficha del modelo, declaración de los datos, cita y licencia |
| `Dockerfile`, `requirements.txt`, `requirements-train.txt`, `render.yaml` | Entorno del dashboard, entorno completo y despliegue en Render |

## Cómo reproducirlo

Todos los comandos se ejecutan en CPU, con Python 3.13.

1. **Datos.** `python Entregable_1_UAVIDS_2025/herramientas/obtener_datos.py` descarga `UAVIDS-2025.csv` y verifica su SHA-256. Los conjuntos adicionales se obtienen como indica `data/procedencia_conjuntos_adicionales.md`, y se preparan con `python training/conjuntos_adicionales.py preparar`.
2. **Entorno.** Hay dos opciones:
   - instalar las dependencias con `pip install -r Entregable_3_UAVIDS_2025/requirements-train.txt`;
   - construir el contenedor desde la raíz del repositorio con `docker build -t uavids-e3 -f Entregable_3_UAVIDS_2025/Dockerfile .`.
3. **Pruebas.** Ejecutar `python -m pytest tests` desde `Entregable_3_UAVIDS_2025/`.
4. **Reproducción rápida.** `python verificacion_externa/reproducir.py` verifica los datos, regenera la tabla principal del capítulo 7 y la compara celda por celda con la publicada, y reentrena tres corridas desde cero. Necesita las probabilidades de prueba de las corridas definitivas, que no se versionan por su tamaño. Vienen en `paquete_reproduccion_e3.zip`, publicado con la *release* `entregable-3` junto con su SHA-256, que trae además el repositorio y el CSV de UAVIDS-2025. Basta con descomprimirlo en una carpeta de ruta corta y, desde su raíz, ejecutar `docker build -t uavids-e3 -f Entregable_3_UAVIDS_2025/Dockerfile .` y luego `docker run --rm uavids-e3 python verificacion_externa/reproducir.py`, que tarda de 10 a 30 minutos en CPU y termina con un bloque `===== RESULTADO =====` con la discrepancia máxima.
5. **Libro.** `jupyter-book build book/` ejecuta todas las celdas y regenera tablas y figuras.
6. **Dashboard.** `python app.py`, o `gunicorn app:server`, lo levanta en http://localhost:8050. `python verificacion_externa/humo_dashboard.py` hace la prueba de humo contra la versión publicada, o contra la local con `--url http://localhost:8050`.
7. **Reentrenar todo.** Ejecutar `python training/ejecutar_definitivas.py principal|ablacion|adicionales`, `python training/ejecutar_robustez.py` y `python training/ejecutar_diagnosticos_adicionales.py`, y luego los análisis de `experiments/`. Cada corrida se salta si su JSON ya existe. En CPU toma del orden de un día.

## Protocolo y control de versiones

| Etiqueta de Git | Contenido |
|---|---|
| `e3-estandar` | Declaración del estándar de profundidad y de los compromisos |
| `e3-protocolo` | Protocolo preregistrado, congelado antes de cualquier corrida definitiva |
| `entregable-3` | Versión entregada |

Las desviaciones posteriores al congelamiento se declaran en el capítulo 7 del libro.

## Uso de herramientas de IA

Se usó Claude (Anthropic) como asistente de programación y de redacción. La declaración completa, firmada, está en el capítulo 11 del libro.
