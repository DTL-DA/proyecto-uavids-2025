# Detección de intrusiones en redes de drones con Deep Learning (UAVIDS-2025)

Proyecto final de Deep Learning (Maestría en Analítica de Datos, Universidad del Norte). El objetivo es clasificar cada flujo de red de un enjambre de drones en tráfico normal o en uno de cuatro ataques (Blackhole, Flooding, Sybil y Wormhole) a partir de 16 atributos de tráfico. Cada entregable reutiliza la partición, el pipeline y las decisiones fijadas a priori en el anterior, de modo que los resultados se pueden comparar entre entregables.

| Entregable | Contenido | Dónde | Versión entregada |
|---|---|---|---|
| 1 | Estado del arte, EDA y baseline con tres modelos | `Entregable 1 - UAVIDS-2025.ipynb` y `Entregable_1_UAVIDS_2025/` | etiqueta `entregable-1` |
| 2 | Benchmark de 16 modelos con inferencia estadística | `Entregable 2 - UAVIDS-2025.ipynb` y `Entregable_2_UAVIDS_2025/` | etiqueta `entregable-2` y su [release](https://github.com/DTL-DA/proyecto-uavids-2025/releases/tag/entregable-2) |
| 3 | Modelo original MoE-ES, validación preregistrada en cuatro conjuntos de datos y dashboard desplegado | `Entregable 3 - UAVIDS-2025.ipynb` y `Entregable_3_UAVIDS_2025/` (ver su [README](Entregable_3_UAVIDS_2025/README.md)) | etiqueta `entregable-3`, su [release](https://github.com/DTL-DA/proyecto-uavids-2025/releases/tag/entregable-3) y DOI [10.5281/zenodo.24486475](https://doi.org/10.5281/zenodo.24486475) |

## Entregable 3: modelo original MoE-ES y validación preregistrada

El modelo original, MoE-ES (mezcla de expertos con escalones suaves), es una compuerta por flujo entre un experto de escalones suaves con umbrales aprendibles y un experto neuronal suave, con entrenamiento cooperativo y una cota de Lipschitz demostrada, comparada bajo un protocolo preregistrado con el Top 10 del Entregable 2, LightGBM y CNN 1D en UAVIDS-2025 y en tres conjuntos adicionales. El modelo original no supera a la familia (R3 no se cumple), es el más robusto ante ruido y atributos faltantes, y el hallazgo principal es que el escenario de simulación pesa más que la arquitectura. El libro está publicado en https://dtl-da.github.io/proyecto-uavids-2025/entregable-3/ y el dashboard en https://uavids-2025-dashboard.onrender.com, y el detalle, la forma de reproducirlo y el dashboard están en [`Entregable_3_UAVIDS_2025/README.md`](Entregable_3_UAVIDS_2025/README.md).

El cuaderno `Entregable 3 - UAVIDS-2025.ipynb` reúne el texto y todo el código del libro en un solo archivo y se entrega ya ejecutado, con todas las salidas. Se regenera desde las fuentes del libro con `python Entregable_3_UAVIDS_2025/book/crear_cuaderno.py`, en unos dos minutos.

## Entregable 2: benchmark de 16 modelos

El benchmark compara, con la misma partición por bloques de 1000 flujos, el mismo pipeline y el mismo presupuesto de ajuste, los tres modelos base del Entregable 1, el Top 10 de la revisión sistemática y tres líneas base: trivial, regresión logística y LightGBM. Cada modelo se entrena con 20 semillas; los cuatro modelos sin arquitectura definida en la literatura recorren además un espacio de 1000 arquitecturas en dos fases. Las preguntas, las pruebas y los márgenes se fijaron en el preregistro antes del primer entrenamiento.

Los resultados principales son los siguientes:

- **RQ1.** Ningún modelo del Top 10 supera a la CNN 1D del Entregable 1 (F1 macro de 0,8584 ± 0,0050) por más de δ = 0,01: tres son equivalentes a ella, tres quedan no concluyentes y cuatro quedan por debajo de forma significativa.
- **RQ2.** LightGBM obtiene el mayor F1 macro de prueba, 0,8748 ± 0,0012, y supera a la CNN 1D por 0,0164, una diferencia significativa y relevante. Sin embargo, en la validación cruzada 5 × 2 la diferencia es de 0,0003, de modo que la ventaja depende de las series de esta prueba.
- **La serie pesa más que la arquitectura.** El bloque de prueba explica el 93,1 % de la varianza del F1 por bloque, y el modelo, el 0,3 %.

El informe completo es el cuaderno `Entregable 2 - UAVIDS-2025.ipynb`, que se entrega ya ejecutado, y su versión en Jupyter Book está en `Entregable_2_UAVIDS_2025/entregables/01_informe_jupyter_book/`. La tabla de [`Entregable_2_UAVIDS_2025/entregables/README.md`](Entregable_2_UAVIDS_2025/entregables/README.md) ubica cada producto que pide el enunciado.

### Cómo reproducirlo

Todos los comandos se ejecutan desde la raíz del repositorio, con Python 3.13 y en CPU.

1. Descargar la versión entregada: `git clone --branch entregable-2 https://github.com/DTL-DA/proyecto-uavids-2025.git`.
2. Instalar las dependencias, con sus versiones exactas: `pip install -r requirements.txt`. También se puede construir el contenedor de referencia: `docker build -f Entregable_2_UAVIDS_2025/Dockerfile -t uavids-e2 .`.
3. Obtener los datos con `python Entregable_1_UAVIDS_2025/herramientas/obtener_datos.py`, que descarga `UAVIDS-2025.csv` y verifica su suma SHA-256.
4. Descargar `Corridas_E2_probabilidades_y_pesos.zip` de la [release](https://github.com/DTL-DA/proyecto-uavids-2025/releases/tag/entregable-2) y descomprimirlo en la raíz. Agrega a `Entregable_2_UAVIDS_2025/resultados/corridas/` las probabilidades y los pesos de cada corrida, que no se versionan por su tamaño.
5. Ejecutar el cuaderno: `python Entregable_2_UAVIDS_2025/herramientas/ejecutar_cuaderno.py "Entregable 2 - UAVIDS-2025.ipynb"`. No reentrena: lee las corridas guardadas y recalcula todas las tablas, pruebas y figuras en unos minutos.
6. Para reentrenar todo, `python Entregable_2_UAVIDS_2025/herramientas/orquestar.py` encadena las fases en el orden del preregistro. En CPU toma decenas de horas: solo la búsqueda de arquitectura consumió 55,1 horas de proceso.

Cada corrida tiene su registro en `Entregable_2_UAVIDS_2025/resultados/corridas/<fase>/<id>.json`, con la configuración, la semilla, las curvas, el tiempo y el equipo, y el registro consolidado está en `Entregable_2_UAVIDS_2025/entregables/07_registro_y_fichas/registro_experimentos.csv`.

### Preregistro y control de versiones

El preregistro (`Entregable_2_UAVIDS_2025/preregistro_e2.md`) y su adenda (`preregistro_e2_adenda.md`) están fechados con etiquetas de git:

| Etiqueta | Fecha | Contenido |
|---|---|---|
| `preregistro-e2` | 12/11/2026, 13:13 | Preregistro, configuración y código, antes del primer entrenamiento |
| `preregistro-e2-desviacion-1` | 12/11/2026, 13:59 | Desviación 1, antes de la fase 1 de la búsqueda |
| `preregistro-e2-adenda-piloto` | 12/11/2026, 16:58 | Número de ejecuciones R, limitación del paper guía y número de finalistas k |
| `preregistro-e2-adenda-arquitecturas` | 13/11/2026, 09:44 | Arquitecturas ganadoras de la fase 2, antes de la ejecución definitiva |

Hay que tener en cuenta que la historia del Entregable 2 se publicó en este repositorio el 15/11/2026, después de la ejecución definitiva, y que antes de publicarla se cambió el correo del autor de sus commits por la dirección privada de GitHub. El contenido, las fechas y los mensajes de los commits y de las etiquetas son los originales, pero sus identificadores cambiaron. El orden completo, con fechas, está también en `Entregable_2_UAVIDS_2025/entregables/03_preregistro/historial_git_preregistro.txt`.

## Entregable 1: baseline de Deep Learning

El Entregable 1 fijó la métrica primaria (F1 macro), el margen δ, la unidad de independencia y la partición, y estableció el baseline con tres modelos:

| Modelo | F1 macro en prueba (7 semillas) | IC 95 % |
|---|---|---|
| CNN 1D (baseline) | 0,8549 ± 0,0091 | [0,8466; 0,8633] |
| DNN | 0,8548 ± 0,0110 | [0,8446; 0,8649] |
| LSTM | 0,8423 ± 0,0135 | [0,8298; 0,8547] |

Ningún par de modelos difiere tras la corrección de Holm, y un modelo avanzado debe alcanzar un F1 macro de al menos 0,8775 para considerarse una mejora distinguible y relevante. Para revisarlo tal como se entregó, con su `README.md` y su `requirements.txt` originales, basta con `git checkout entregable-1`.

## Datos

Se usa [UAVIDS-2025](https://www.kaggle.com/datasets/qinglizeng1997/uavids-2025) (Zenodo, CC BY 4.0, DOI [10.5281/zenodo.15336998](https://doi.org/10.5281/zenodo.15336998)). El CSV no se versiona: `obtener_datos.py` lo descarga y verifica su suma SHA-256, y el Entregable 2 comprueba además la suma del conjunto de prueba sellado en el Entregable 1.
