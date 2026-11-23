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

# Correcciones del Entregable 2

El docente no hizo observaciones escritas al Entregable 2. Por tal motivo, este capítulo se construye con una autoauditoría del E2 frente a su propio enunciado y frente a los criterios del Bloque A de este entregable. Una observación sin fila en la tabla de trazabilidad se considera no corregida; si llegan observaciones del docente, se agregan como filas nuevas.

```{code-cell} ipython3
:tags: [hide-input]
import json
from pathlib import Path
import pandas as pd
RAIZ = Path("..").resolve()
A = RAIZ / "experiments/bloque_a"
pd.set_option("display.precision", 3)
pd.set_option("display.max_colwidth", 400)
```

## Tabla de trazabilidad

```{code-cell} ipython3
:tags: [hide-input]
texto = (A / "trazabilidad.md").read_text(encoding="utf-8")
bloque = texto.split("## Tabla de trazabilidad")[1].split("## Ranking")[0].strip().splitlines()
filas = [[c.strip() for c in l.strip("|").split("|")] for l in bloque if l.startswith("|") and not l.startswith("|---")]
trazabilidad = pd.DataFrame(filas[1:], columns=filas[0]).set_index("N.°")
trazabilidad
```

## Auditoría de fuga de información

La auditoría reutiliza la batería del E2 y la amplía con la taxonomía de {cite:t}`kapoor2023leakage`. Se corre en modo de solo lectura sobre las predicciones guardadas de las 20 semillas de los 15 modelos (`experiments/bloque_a/auditoria_particion.py`). El F1 recalculado coincide con el registrado en todas las corridas.

```{code-cell} ipython3
:tags: [hide-input]
auditoria = json.loads((A / "auditoria_particion/auditoria_particion.json").read_text(encoding="utf-8"))
resumen = pd.DataFrame([
    ["L1 · bloques compartidos entre particiones", sum(auditoria["bloques_compartidos_entre_particiones"].values())],
    ["L1 · filas duplicadas excluidas antes de partir", auditoria["filas"]["fuera (repetidas)"]],
    ["L3.2 · tramos de escenario presentes en varias particiones", auditoria["tramos"]["tramos_en_varias_particiones"]],
    ["L3.2 · filas de prueba en un escenario con filas en entrenamiento (%)", auditoria["tramos"]["filas_prueba_en_tramo_compartido_con_entrenamiento_%"]],
    ["L3.2 · filas de validación en un escenario con filas en entrenamiento (%)", auditoria["tramos"]["filas_validacion_en_tramo_compartido_con_entrenamiento_%"]],
    ["Gemelos en atributos (< 0,01) entre prueba contigua y entrenamiento (%)", auditoria["vecino_atributos_prueba"]["gemelos_<0.01_contiguo_%"]],
], columns=["Verificación", "Valor"])
resumen
```

**No hay fuga por separación (L1) ni por atributos ilegítimos (L2): ningún bloque se comparte entre particiones, los 8 219 duplicados exactos se excluyeron antes de partir y los identificadores (FlowID, IP, puertos y protocolo) no entran al modelo. Sí hay dependencia entre entrenamiento y prueba (L3.2): el 70,5 % de las filas de prueba pertenece a un escenario de simulación que también tiene filas en entrenamiento.**

Esto significa que la prueba mide sobre todo la generalización a segmentos nuevos de escenarios ya vistos, no a simulaciones nuevas. Sin embargo, la dependencia no se traduce en copias: solo el 1,8 % de las filas de prueba contiguas tiene un vecino de entrenamiento a menos de 0,01 en el espacio de atributos, y la cercanía en filas a un bloque de entrenamiento no baja el error de forma significativa en Blackhole, Normal, Flooding ni Wormhole.

En consecuencia, se decide conservar la partición congelada del E1, que es lo que permite comparar E1, E2 y E3, reportar la dependencia con esta cuantificación y medir su efecto en el ranking y en la generalización a escenarios nuevos, en lugar de rehacer la partición. Wormhole tiene solo dos escenarios, así que no se puede repartir por escenario en tres particiones.

## Ranking antes y después

El experimento natural son los dos escenarios de Sybil que quedaron aislados por azar: uno entero en validación y otro casi entero en prueba. Al quitarlos se obtiene el desempeño sobre escenarios ya vistos.

```{code-cell} ipython3
:tags: [hide-input]
f = pd.read_csv(A / "auditoria_particion/f1_sin_escenarios_sybil.csv")
f["Posición antes (prueba)"] = f["F1 pru"].rank(ascending=False).astype(int)
f["Posición después (prueba)"] = f["F1 pru sin escenario"].rank(ascending=False).astype(int)
f["Posición antes (validación)"] = f["F1 val"].rank(ascending=False).astype(int)
f["Posición después (validación)"] = f["F1 val sin escenario"].rank(ascending=False).astype(int)
f = f.rename(columns={"F1 pru": "F1 prueba antes", "F1 pru sin escenario": "F1 prueba después", "F1 val": "F1 validación antes",
                      "F1 val sin escenario": "F1 validación después", "acierto escenario pru": "Acierto en el escenario aislado de prueba"})
f[["Modelo", "F1 prueba antes", "F1 prueba después", "Posición antes (prueba)", "Posición después (prueba)", "F1 validación antes",
   "F1 validación después", "Posición antes (validación)", "Posición después (validación)", "Acierto en el escenario aislado de prueba"]]
```

**En la prueba, todos los modelos suben unos 0,04 de F1 al quitar el escenario aislado y el orden casi no cambia: LightGBM sigue primero (de 0,875 a 0,915) y CNN-LSTM segunda (de 0,863 a 0,904). En validación el orden sí cambia: LightGBM pasa de 0,887 a 0,936 y queda primero.**

Lo anterior indica que la ventaja de las redes en validación se debía por completo al escenario de 695 flujos aislado en esa partición, y que en el escenario aislado de prueba ningún modelo acierta más del 10 % de los flujos (el máximo es KAN, con 9,4 %). El cambio es desfavorable para las redes y se reporta sin ocultarlo.

Por esta razón, el modelo original se diseña contra la limitación que sí se puede atacar con los datos disponibles, la confusión Blackhole–Wormhole en escenarios vistos, y la falta de generalización a escenarios nuevos se trata como una predicción secundaria (P5) y como un estudio de cambio de distribución en el capítulo 8.

## Generalización a un escenario no visto

La prueba de la opción de invariancia (capítulo 5) dejó, además, una medición de la línea de referencia: con 9 pliegues que dejan fuera un escenario completo de las filas de entrenamiento, el DNN acierta en promedio el 47,5 % de los flujos del escenario no visto y LightGBM el 40,1 %. El Sybil nuevo se predice casi siempre como Flooding y el Wormhole nuevo como Blackhole.

```{code-cell} ipython3
:tags: [hide-input]
pd.read_csv(A / "prueba_d3/_acierto_por_pliegue.csv").set_index("pliegue")
```

## Determinismo y congelamiento de los benchmarks

Las semillas de Python, NumPy y PyTorch se fijan en cada corrida (`codigo/motor.fijar_semilla`); todo se entrenó en CPU con PyTorch 2.6.0, con 2 hilos por proceso. Los resultados de los benchmarks quedaron congelados el 22/11/2026, antes de cualquier corrida del E3, con el SHA-256 `a35f853544388d437ed9ba6755dec2ca2c77fe0a8b8b3b5be7fd3264d1f05f03` del conjunto ordenado de las 301 corridas definitivas del E2 (registros y probabilidades), fijado en `protocolo.md`. Desde ese punto no se reajustan; solo se agregaron las semillas que exigió el análisis de potencia, con los mismos hiperparámetros.
