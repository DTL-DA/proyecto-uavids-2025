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

# Benchmarks: Top 10, baselines y exportación

```{code-cell} ipython3
:tags: [hide-input]
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
RAIZ = Path("..").resolve()
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.precision", 4)
lideres = pd.read_csv(RAIZ / "artifacts/dashboard/lideres.csv")
```

## Criterio de selección de la mejor configuración

Para cada modelo del benchmark y cada *baseline* se usa la configuración elegida en el E2 por la **mayor F1 macro media de validación**, nunca por la prueba:

- **Modelos con hiperparámetros:** las 12 configuraciones de (lr, weight_decay, lote).
- **Modelos con búsqueda de arquitectura** (ResNet, Autoencoder, DBN y CNN-LSTM): la arquitectura ganadora de la fase 2, cuyo sesgo del ganador se reportó en el E2.
- **Modelo original:** las mismas 12 configuraciones, con la misma regla (capítulo 7).

Las configuraciones están congeladas con el SHA-256 del capítulo 1.

## Tabla maestra

La tabla maestra de experimentos (sección 12, entregable 8) es `experiments/resultados/tabla_maestra.csv`: una fila por corrida, con modelo, conjunto de datos, semilla, métricas de validación y prueba, épocas, tiempos y costo. Su resumen por modelo en el problema principal es la tabla de líderes:

```{code-cell} ipython3
:tags: [hide-input]
cols = [c for c in ["Modelo", "Grupo", "Familia", "Semillas", "F1 macro", "DE", "IC95 inf", "IC95 sup", "ECE", "Parámetros", "FLOPs",
                    "Latencia mediana (ms)", "Latencia p95 (ms)", "Entrenamiento (s)"] if c in lideres]
lideres[cols]
```

**LightGBM es el mejor modelo global (F1 macro de 0,875 ± 0,001 en 20 semillas), seguido de CNN-LSTM (0,859 ± 0,019 en 48 semillas) y de un grupo de redes entre 0,849 y 0,859. El modelo original queda en ese grupo (0,850 ± 0,012), por encima de KAN, GRU, BiLSTM, RNN, LSTM, TabNet y la regresión logística.**

Así mismo, la dispersión entre semillas de LightGBM es una décima de la de las redes, de modo que su ventaja no es un efecto de la semilla, y LightGBM tiene el mejor rango medio de la prueba de Friedman con los 13 modelos de la familia (capítulo 7).

Por esta razón, LightGBM es el *baseline* fuerte de referencia en el dashboard, y la tabla de líderes muestra todos los modelos con media ± DE e IC 95 %, no solo el ganador.

## Reentrenamiento final, exportación y prueba de ida y vuelta

**Política de reentrenamiento**, idéntica para todos los modelos, incluido el original:

- Cada corrida definitiva se entrenó sobre el conjunto de entrenamiento completo, con el número de épocas fijado en validación por la parada temprana.
- Se exporta la corrida cuya F1 de validación es la más cercana a la mediana de las 20 primeras semillas, lo que la hace representativa y la elige sin mirar la prueba.
- El *checkpoint* completo (pesos, configuración y curvas) queda en el registro de cada corrida.

**Exportación:**

- Las redes se exportan con `torch.onnx` (opset 17), LightGBM con `onnxmltools` y la regresión logística con `skl2onnx`.
- Cada modelo lleva su `metadata.json`, con versiones, semilla, hiperparámetros, preprocesamiento (transformación, medias y desviaciones), esquema de entrada con los rangos del entrenamiento, clases, métricas, temperatura y cuantil conforme.

```{code-cell} ipython3
:tags: [hide-input]
rt = pd.read_csv(RAIZ / "artifacts/reporte_ida_y_vuelta.csv")
rt
```

**Las 14 redes (las 13 del E2 y el original) y la regresión logística reproducen en ONNX las probabilidades del modelo en memoria con un error absoluto máximo de entre 2 × 10⁻⁷ y 3 × 10⁻⁵, por debajo de la tolerancia declarada de 10⁻⁴, sobre los 22 883 flujos de la prueba. Las clases coinciden en el 100 % de los flujos y la diferencia de F1 macro es exactamente 0.**

Hay que tener en cuenta la excepción de LightGBM: su error máximo es 0,028, aunque solo 5 de los 22 883 flujos (0,02 %) superan 10⁻⁴. El conversor de ONNX guarda los umbrales de los árboles en precisión simple, así que un flujo que cae justo en un umbral puede tomar la otra rama. Aun así, las clases coinciden en el 100 % de los flujos y el F1 es idéntico.

En consecuencia, se declara para LightGBM una tolerancia de 10⁻⁴ en el 99,98 % de los flujos, con la métrica primaria sin cambio, y no se cuantiza ningún modelo: juntos ocupan unos 5 MB y caben en el plan gratuito de Render.

## Costo de inferencia

La latencia por flujo se midió en las mismas condiciones para todos los modelos: ONNX Runtime con un hilo, 300 flujos de prueba de a uno y 20 de calentamiento. Los parámetros y los FLOPs se cuentan con los mismos ganchos del E2. La frontera de Pareto entre desempeño y costo está en el capítulo 8.
