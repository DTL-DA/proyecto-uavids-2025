# Entregable 2 · Benchmark de Deep Learning sobre UAVIDS-2025

Esta carpeta reúne, en el orden del enunciado, los productos del Entregable 2. Todo se genera con
`herramientas/organizar_entregables.py` a partir del cuaderno ejecutado y de `resultados/`, de modo que no se edita a mano.

| Carpeta | Entregable del enunciado |
|---|---|
| `00_codigo/` | Código completo y documentado: EDA, 10 modelos, baselines, ajuste y sobreajuste, análisis estadístico, robustez, incertidumbre e interpretabilidad |
| `01_informe_jupyter_book/` | Informe en Jupyter Book con ejecución reproducible de celdas (fuentes del libro e informe en PDF) |
| `02_dataset_y_contrato/` | Dataset o instrucciones verificables para obtenerlo, con la Data Card y el contrato de datos (secciones 1.4 y 1.5 del informe) |
| `03_preregistro/` | Preregistro fechado y versionado, con su adenda |
| `04_auditoria_de_fuga/` | Reporte de auditoría de fuga |
| `05_ficha_de_ruta/` | Ficha de ruta metodológica: tablas A, B y C y criterio de adecuación |
| `06_tabla_comparativa/` | Tabla comparativa con métricas, costos e inferencia estadística |
| `07_registro_y_fichas/` | Registro de experimentos y fichas de modelo |
| `08_techo_e_incertidumbre/` | Reporte de techo de desempeño e incertidumbre |
| `09_busqueda_de_arquitectura/` | Reporte de la búsqueda de arquitectura: las 1000 arquitecturas, las dos fases, los finalistas y el análisis de efectos |
| `10_paper_guia/` | Paper guía: réplica parcial y limitación identificada |
| `11_literatura/` | Revisión sistemática: hoja de extracción, puntaje del Top 10, mecanismos transferibles y consultas |
| `12_reproducibilidad/` | Lista de verificación de reproducibilidad (apéndice B), entorno y prueba de entorno limpio |
| `13_defensa/` | Presentación oral con defensa técnica de las decisiones metodológicas: guion con el texto por diapositiva, cifras de apoyo y preguntas probables del jurado |
| (cuaderno en la raíz) | `Entregable 2 - UAVIDS-2025.ipynb`, ejecutado de principio a fin |

El guion de `13_defensa/` se genera con `herramientas/crear_guion_defensa.py` y acompaña las 16 diapositivas de la
presentación oral, que se proyectan aparte.

## Cómo reproducir

1. Instalar las dependencias: `pip install -r requirements.txt` (Python 3.13, CPU).
2. Obtener los datos con `02_dataset_y_contrato/obtener_datos.py`, que descarga `UAVIDS-2025.csv` y verifica su suma SHA-256.
3. Reentrenar, si se quiere, con `python Entregable_2_UAVIDS_2025/herramientas/orquestar.py`, que encadena las fases del preregistro:
   ajuste, piloto, búsqueda de arquitectura, ejecución definitiva, experimentos complementarios y análisis. Cada corrida queda en
   `resultados/corridas/` y el ejecutor salta las que ya existen.
4. Ejecutar el cuaderno con `python Entregable_2_UAVIDS_2025/herramientas/ejecutar_cuaderno.py "Entregable 2 - UAVIDS-2025.ipynb"`.
   El cuaderno no reentrena: lee las corridas guardadas y recalcula todas las métricas, pruebas, tablas y gráficas.
5. Para verificar sin reentrenar, descomprimir `Corridas_E2_probabilidades_y_pesos.zip` en la raíz del repositorio, que agrega a
   `resultados/corridas/` las probabilidades y los pesos de cada corrida; por su tamaño, se entrega aparte del ZIP del correo.

La prueba de entorno limpio (`herramientas/entorno_limpio.py`) instala un entorno nuevo desde `requirements.txt`, reentrena la
primera semilla de cada modelo que tarda menos de 15 minutos y compara el F1; su resultado está en
`12_reproducibilidad/reproduccion_entorno_limpio.json` y en la sección 8 del informe.
