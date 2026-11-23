# Prueba rápida de D3 · regla fijada antes de correr (22/11/2026)

**Pregunta.** ¿La penalización invariante por escenario con riesgo condicional por clase (opción 2 de D3, adaptación de V-REx) mejora el acierto en un escenario de simulación que el modelo no vio en entrenamiento?

**Datos.** Solo las 73 267 filas de entrenamiento del E2. No se usan la validación ni la prueba del E2. Los escenarios se identificaron con esas filas solamente (`explorar_escenarios.py`): tramos de una clase sobre las filas de entrenamiento ordenadas por FlowID, más la región intercalada de Blackhole y Normal (FlowID 57 961 a 94 000).

| Escenario | Clase | FlowID | Filas de entrenamiento |
|---|---|---|---|
| S1 | Sybil | 1 a 19 000 | 12 820 |
| S2 | Sybil | 51 325 a 54 551 | 2 219 |
| S3 | Sybil | 109 001 a 109 046 | 46 (nunca se deja fuera) |
| B1 | Blackhole | 20 001 a 35 000 | 10 000 |
| B2 | Blackhole | región intercalada, 57 961 a 94 000 | 7 342 |
| W1 | Wormhole | 36 001 a 49 000 | 8 000 |
| W2 | Wormhole | 97 428 a 108 000 | 8 573 |
| F1 | Flooding | 51 001 a 51 324 | 324 (nunca se deja fuera) |
| F2 | Flooding | 54 552 a 57 960 | 1 950 |
| F3 | Flooding | 95 001 a 97 427 | 1 427 |
| F4 | Flooding | 109 047 a 122 171 | 3 868 |
| N | Normal | región intercalada, 57 961 a 94 000 | 16 698 (nunca se deja fuera: es el único escenario de Normal) |

**Pliegues.** Nueve, dejando fuera un escenario cada vez: S1, S2, B1, B2, W1, W2, F2, F3 y F4. En B2 se dejan fuera solo las filas de Blackhole de la región intercalada, porque Normal no tiene otro escenario. De las filas restantes, el 15 % de los bloques de 1000 flujos (semilla fija por pliegue, la misma para todos los métodos) sirve para la parada temprana. La estandarización y los pesos de clase se ajustan solo con las filas de entrenamiento del pliegue.

**Métodos.** El DNN del E2 con sus hiperparámetros (lr 0,001, sin weight decay, lote 256), el mismo protocolo (máximo 30 épocas, paciencia 6 sobre el F1 macro de parada, ReduceLROnPlateau, norma del gradiente 5) y 3 semillas (101, 102 y 103):

1. **ERM:** entropía cruzada ponderada, como en el E2.
2. **V-REx condicional por clase, λ ∈ {1, 10, 100}:** ERM + λ · media, sobre las clases con al menos dos escenarios en el pliegue, de la varianza entre escenarios del riesgo medio de esa clase en ese escenario. El riesgo se estima en cada paso con 32 filas por par (escenario, clase) con al menos 64 filas. La primera época va sin penalización (calentamiento, como en V-REx).
3. **Referencia:** LightGBM con los hiperparámetros definitivos del E2 y parada a 100 rondas sobre el mismo conjunto de parada, una corrida por pliegue.

**Métricas.** Acierto en el escenario no visto (todas sus filas son de una clase) y F1 macro en el conjunto de parada (escenarios vistos).

**Regla de decisión (fijada antes de ver resultados).** La opción 2 se favorece si algún λ cumple las tres condiciones a la vez frente a ERM, con las mismas semillas:

1. Mejora media del acierto en el escenario no visto de al menos 0,03, promediando primero las semillas y luego los 9 pliegues.
2. Mejora en al menos 6 de los 9 pliegues.
3. El F1 macro en escenarios vistos no cae más de 0,01.

Si ningún λ cumple, se mantiene la opción 1 (decisión del 18/11). Elegir λ aquí es una iteración de diseño: las 3 variantes cuentan en el *Nd* de `design_log.md`.

**Nota agregada el 22/11/2026, con la corrida en curso y sin ver resultados.** El criterio de las tres condiciones no cambia. Lo que se corrige es la salida por defecto: si ningún λ cumple, **no** se vuelve de forma automática a la opción 1. Después de la auditoría de fuga, la recomendación de Claude ya no era la opción 1. Lo que siga se decide con Donny a la vista de esta prueba, que también mide si LightGBM (el sesgo de árbol del experto de la opción 1) generaliza mejor que el DNN a escenarios no vistos.

**Cuidado que se declara.** La motivación de esta prueba viene de la auditoría de fuga del 18/11, que miró predicciones de prueba del E2. Esta prueba no las usa.
