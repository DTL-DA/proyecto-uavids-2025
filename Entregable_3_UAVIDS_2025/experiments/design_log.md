# Bitácora de diseño del modelo original (`design_log.md`)

Plantilla A.3 del enunciado. Cada fila es una variante del modelo original evaluada durante el diseño. *Nd* cuenta todas las variantes, también las descartadas, para estimar el sesgo de selección (sección 7.2).

**Datos de diseño.** Solo filas de entrenamiento del E2 (73 267). Hay dos particiones de desarrollo:

- **Por escenario** (`_trabajo_local/prueba_d3/`): deja fuera un escenario de simulación completo cada vez, en 9 pliegues.
- **Por bloques** (`training/particion_desarrollo.py`): 70 % de ajuste, 15 % de parada y 15 % de diseño, por bloques de 1000 flujos, con tres reparticiones (semillas 1, 2 y 3).

La validación y la prueba del E2 son el conjunto de **confirmación** y no se usan para diseñar. Los conjuntos adicionales tampoco.

**Declaración.** La auditoría de fuga del 18/11/2026 miró predicciones de prueba del E2, ya guardadas, para cuantificar la dependencia por escenario (L3.2). Esa auditoría motivó la prueba de la opción 2. Ninguna decisión de esta bitácora usa métricas de prueba o validación del E2.

## Análisis previos al diseño (no son variantes y no cuentan en *Nd*)

| Fecha | Análisis | Resultado | Consecuencia |
|---|---|---|---|
| 18/11/2026 | Evidencia del E2 (cuadro 3 del plan): LightGBM gana en prueba (0,875 frente a 0,863) y comete menos confusión Blackhole–Wormhole (0,100 frente a 0,123); las redes ganan en Sybil | Errores complementarios entre el sesgo de árbol y el neuronal | Candidata la opción 1 (compuerta por flujo) |
| 22/11/2026 | Línea de referencia de la partición por escenario: DNN entrenado con ERM y LightGBM | Acierto en el escenario no visto: DNN 0,475; LightGBM 0,401 | LightGBM no generaliza mejor a escenarios nuevos |
| 22/11/2026 | d01: techo del oráculo por flujo en la partición por bloques (`diseno/d01_techo_oraculo.py`) | F1 macro medio en 3 reparticiones: LightGBM 0,845; CNN-LSTM 0,842; DNN 0,837; promedio LightGBM+DNN 0,849; **oráculo LightGBM\|DNN 0,871**; discrepan en el 5,8 % de los flujos | Una compuerta perfecta ganaría +0,026; hay margen |
| 22/11/2026 | d01b: quién acierta cuando discrepan (mismas probabilidades) | La red acierta en Sybil y LightGBM en las demás clases. La regla fija "si la red predice Sybil, créele; si no, a LightGBM" da 0,856 (+0,011). Elegir por máxima confianza no aporta (0,848) | La confiabilidad de cada experto depende de la clase predicha; la compuerta debe ver las distribuciones de los expertos |
| 22/11/2026 | d01c: compuerta convexa y *stacking* entrenados en una mitad de los bloques de diseño y evaluados en la otra | Compuerta 0,837; *stacking* 0,474 (colapsa porque las mitades no tienen todas las clases); regla fija 0,856 | Una compuerta de mucha capacidad ajustada con pocos bloques no generaliza; hace falta una compuerta de baja capacidad ajustada con predicciones fuera de muestra de todo el entrenamiento |

## Variantes evaluadas

| N.° | Variante evaluada (cambio respecto a la anterior) | Métrica de desarrollo | Decisión | Motivo y fecha |
|---|---|---|---|---|
| 1 | Opción 2: DNN + penalización V-REx condicional por clase entre escenarios, λ = 1 | Acierto en escenario no visto 0,432 (ERM 0,475); F1 de escenarios vistos 0,866 (ERM 0,868) | Descartar | Regla fijada antes de correr (`regla_decision.md`): no mejora. 22/11/2026 |
| 2 | Igual, λ = 10 | 0,440; F1 de vistos 0,856 | Descartar | Igual. 22/11/2026 |
| 3 | Igual, λ = 100 | 0,443; F1 de vistos 0,841 | Descartar | Igual; además cae el F1 en escenarios vistos. 22/11/2026 |

| 4 | Opción 1, v1: escalones suaves (T = 8, tau_min = 0,05) + MLP; experto suave MLP 64-64; compuerta que ve x; entrenamiento conjunto (log-verosimilitud de la mezcla) + balance γ = 0,01 | F1 macro de diseño 0,832 (3 reparticiones) | Descartar | Peor que cada experto por separado; los expertos se especializan (F1 individual 0,41 y 0,56). 22/11/2026 |
| 5 | v1 con la compuerta viendo x y las distribuciones de ambos expertos | 0,826 | Descartar | Igual patología. 22/11/2026 |
| 6 | Solo el experto de umbrales (diagnóstico) | **0,843** (LightGBM 0,845; DNN 0,837) | Conservar el experto | El codificador de escalones suaves cierra casi toda la brecha con LightGBM. 22/11/2026 |
| 7 | Solo el experto suave (diagnóstico) | 0,824 | Referencia | 22/11/2026 |
| 8 | Promedio fijo de los dos expertos (diagnóstico) | 0,837 | Referencia | 22/11/2026 |
| 9 | Entrenamiento **cooperativo** (cada experto con su pérdida; la compuerta con gradiente detenido hacia los expertos), compuerta que ve x y los expertos | 0,839 (expertos 0,836 y 0,833) | Descartar | Los expertos siguen fuertes, pero la compuerta no aprende fuera de muestra. 22/11/2026 |
| 10 | Igual, compuerta que ve solo x | 0,842 | Descartar | No supera al experto de umbrales solo. 22/11/2026 |

*Nd* acumulado: **10**.
