# Preregistro del Entregable 2

**Fecha:** 12 de noviembre de 2026, antes del primer entrenamiento del Entregable 2.
**Versión:** 1. Las cifras que dependen del estudio piloto (número de ejecuciones, k finalistas y limitación del paper guía) se fijan con las reglas de este documento y se registran en una adenda fechada antes de la ejecución definitiva, sin cambiar las reglas.
**Configuración asociada:** `configuracion/experimento.json`, cuya suma SHA-256 queda en `resultados/preregistro_e2.json`.

## 1. Tarea y unidad de independencia

Clasificación con datos tabulares, multiclase de cinco categorías (Blackhole, Flooding, Normal, Sybil, Wormhole), sobre flujos de red de UAVIDS-2025, versión 1. No hay horizonte de predicción. La unidad de independencia es la serie de flujos consecutivos según `FlowID`, aproximada con bloques de 1000 flujos, heredada del Entregable 1 sin cambios, al igual que la partición estratificada y agrupada por bloques (64, 16 y 20 %, semilla 42).

## 2. Métricas

- **Primaria:** F1 macro en el conjunto de prueba. Es la única que sustenta conclusiones confirmatorias.
- **Secundarias:** accuracy, balanced accuracy, precision, recall y F1 macro y ponderados, AUC-ROC macro uno contra todos, AUC-PR macro, MCC, log-loss, Brier, ECE con 15 bins y ECE adaptativo con 15 bins de igual tamaño.

## 3. Márgenes

- **δ = 0,01 en F1 macro**, heredado del Entregable 1: por la definición del promedio macro, equivale a que el F1 de una sola clase cambie 0,05 con las demás iguales.
- **δg = 0,05 en F1 macro** como brecha máxima aceptable entre entrenamiento y validación (tabla B del enunciado).

## 4. Modelos

- **Top 10**, según el puntaje de `literatura/puntaje_top10.csv`: CNN-LSTM, Autoencoder, RNN, Transformer (TSLT-Net), BiLSTM, ResNet, GRU, KAN, TabNet y DBN.
- **Modelos base del Entregable 1:** DNN, CNN 1D y LSTM, con el *dropout* del código de los autores (Zeng, 2025), lo que corrige la desviación declarada en el Entregable 1. El baseline del Entregable 1 es CNN 1D.
- **Líneas base no profundas:** LightGBM (baseline clásico fuerte), regresión logística penalizada y clasificador trivial de clase mayoritaria.
- **Arquitectura definida en la literatura:** RNN, GRU y TSLT-Net (Biswas et al., 2026), BiLSTM (Balgoon, 2026), KAN (clasificador de ataques de H-MicroKAN, Bakiri et al., 2026), TabNet (Arik y Pfister, 2021, con la configuración por defecto de pytorch-tabnet) y los tres modelos base (Zeng, 2025).
- **Sin arquitectura definida, con la búsqueda de la sección 14:** ResNet, Autoencoder, DBN y CNN-LSTM.

## 5. Preguntas de investigación y familias de comparaciones confirmatorias

| RQ | Tipo | Pregunta | Hipótesis | Familia y corrección |
|---|---|---|---|---|
| RQ1 | Confirmatoria | ¿Supera algún modelo del Top 10 al baseline del Entregable 1 (CNN 1D, reentrenado aquí con el mismo protocolo) en F1 macro de prueba, por más de δ? | Para cada modelo k: H0: μk − μCNN = 0; H1: μk − μCNN ≠ 0 | F1: 10 contrastes, Holm |
| RQ2 | Confirmatoria | ¿Difiere el mejor modelo profundo, elegido por la media de F1 macro de **validación** en la ejecución definitiva, del LightGBM ajustado con el mismo presupuesto? | H0: μDL − μLGBM = 0; H1: ≠ 0 | F2: 1 contraste |
| RQ3 | Confirmatoria | ¿Obtienen menor F1 macro los modelos que leen los atributos como secuencia (CNN 1D, LSTM, RNN, GRU, BiLSTM, CNN-LSTM) que los que no suponen un orden entre atributos (DNN, ResNet, Autoencoder, DBN, KAN, TabNet, Transformer)? | H0: μS − μN = 0, con el promedio de cada grupo por semilla; H1: ≠ 0 | F3: 1 contraste |
| RQ4 | Exploratoria | ¿Es compartida por los modelos del Top 10 la limitación cuantificada en el paper guía? | Generadora de hipótesis | Sin control estricto |
| RQ5 | Exploratoria | ¿Existe un modelo de menor costo equivalente al mejor dentro de ±δ? | TOST por pares | Sin control estricto |

**Tamaño de efecto mínimo de interés:** δ = 0,01 en todas las RQ confirmatorias.

**Criterio de decisión por contraste:**
- *Significativa y relevante:* las pruebas (i) y (ii) rechazan H0 con p ajustado por Holm < 0,05 dentro de su familia, la diferencia media supera δ en valor absoluto y el IC 95 % de la diferencia excluye el cero.
- *Significativa sin relevancia:* rechazan H0, pero la diferencia es menor o igual que δ.
- *Equivalente:* TOST con ±δ da p < 0,05.
- *No concluyente:* ninguna de las anteriores. Si (i) y (ii) discrepan, se reporta y se decide con la más conservadora.

## 6. Pruebas y supuestos

Cada contraste confirmatorio se evalúa con tres pruebas de la tabla C:

1. **Paramétrica o de remuestreo pareado:** *t* pareada sobre las R semillas, si Shapiro-Wilk sobre las diferencias no rechaza la normalidad (α = 0,05); si la rechaza, el *bootstrap* pareado de las diferencias con 10 000 remuestreos ocupa su lugar.
2. **No paramétrica:** Wilcoxon de rangos con signo, exacta, sobre las mismas R semillas; con R ≥ 10 su p mínimo bilateral es 0,002, de modo que puede rechazar tras Holm.
3. **Procedimiento global:** Friedman con la corrección de Iman y Davenport sobre los 16 modelos (13 profundos, LightGBM, logística y trivial), con las semillas como bloques, *post hoc* de Wilcoxon con Holm y diagrama de diferencia crítica de Nemenyi. En RQ3, el procedimiento global es un modelo lineal de efectos mixtos sobre el F1 macro por bloque de prueba, con efectos aleatorios de bloque y de semilla.

**Complementos y sensibilidad:** TOST con ±δ para equivalencia; prueba bayesiana de rangos con signo con ROPE de ±δ; IC 95 % de cada diferencia por *bootstrap* por bloques de prueba (2000 remuestreos); *t* con la corrección de Nadeau y Bengio y *F* de 5 × 2 cv de Alpaydin sobre una validación cruzada 5 × 2 agrupada por bloques en entrenamiento + validación; McNemar y DeLong en RQ2 sobre el promedio de probabilidades de las R semillas, declarando que suponen filas independientes; *Model Confidence Set* de Hansen con nivel 0,10 sobre la pérdida 1 − F1 macro por bloque de prueba.

## 7. Número de ejecuciones

R = techo(((z₁₋α′/₂ + z₀,₈₀) σ̂d / δ)²), con α′ = 0,05/10 y σ̂d el mayor entre los del piloto (diferencias pareadas frente a CNN 1D en validación) y el de prueba del Entregable 1, 0,0177, porque en el Entregable 1 la variabilidad en prueba superó a la del piloto. R se acota entre 10 y 20: el mínimo da potencia a Wilcoxon tras Holm y el tope es de cómputo, porque BiLSTM tarda unos 15 minutos por corrida en este equipo. Si R llega al tope, se reporta la diferencia mínima detectable. Las semillas definitivas son 6001, 6002, …, 6000 + R, las mismas para todos los modelos.

## 8. Reglas de exclusión

Una corrida es divergente si su pérdida de entrenamiento no es finita. Se reporta y se repite una sola vez con la semilla más 1000; si vuelve a divergir, el modelo se declara inestable y se analiza con las semillas disponibles. Ninguna corrida se excluye por su desempeño.

## 9. Ajuste de hiperparámetros y parada

- **Presupuesto:** 12 configuraciones por modelo, las mismas 12 para los 13 modelos profundos, muestreadas con semilla 2026 de la rejilla lr ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³}, *weight decay* ∈ {0, 10⁻⁵, 10⁻⁴} y lote ∈ {128, 256, 512}. LightGBM recibe 12 configuraciones de su propio espacio y la regresión logística 12 valores de C.
- **Cada intento:** una semilla (4242), máximo de 15 épocas, la misma parada temprana. Los modelos con búsqueda de arquitectura se ajustan con la arquitectura de referencia (64, 64, 64) y *dropout* (0, 1, 0).
- **Selección:** mayor F1 macro de validación. La prueba no interviene.
- **Entrenamiento común:** Adam, entropía cruzada ponderada, reducción de la tasa por 10 tras 2 épocas sin mejora de la pérdida de validación, recorte del gradiente en norma 5, parada temprana con paciencia de 6 épocas sobre el F1 macro de validación y máximo de 30 épocas. Ningún modelo se reentrena con entrenamiento + validación.

## 10. Espacio de arquitectura (sección 14)

- **Espacio:** 3 bloques, anchos W = {16, 32, 64, 128, 256} en todas las combinaciones y *dropout* de 0,2 tras el bloque i según b ∈ {0, 1}³: 1000 arquitecturas por modelo, enumeradas por código.
- **Correspondencia por familia:** ResNet, ancho del bloque residual i; Autoencoder, ancho de la capa i del codificador, con decodificador espejo y *dropout* también en el espejo; DBN, unidades ocultas de la RBM i; CNN-LSTM, filtros de las dos convoluciones y unidades de la LSTM (correspondencia aproximada).
- **Fase 1:** las 1000 arquitecturas, semilla 5000, 6 épocas (20 % del máximo), la misma parada temprana; métrica, F1 macro de validación.
- **k:** se estima el ruido por semilla con 5 arquitecturas por modelo y 3 semillas a presupuesto de fase 1; k = 20 si la desviación entre semillas es al menos la mitad de la desviación entre arquitecturas, y k = 10 en otro caso. A los finalistas se agrega la referencia (64, 64, 64), *dropout* (0, 1, 0).
- **Fase 2:** cada finalista con 5 semillas (5101 a 5105) y presupuesto completo.
- **Selección:** mayor media de F1 macro de validación en la fase 2; si la diferencia con la segunda es menor que δ o sus IC 95 % se solapan, la de menos parámetros; si persiste el empate, el identificador menor. Se reporta el conjunto de finalistas equivalentes a la ganadora por TOST con ±δ.
- **La fase 1 es un cribado:** el ganador es el mejor entre los finalistas, no necesariamente el mejor de las 1000.
- **Cómputo estimado:** C_total = m |A| t̄₁ + m (k + 1) R t̄₂ ≈ 126 000 s de proceso en la fase 1 y entre 22 550 (k = 10) y 43 050 s (k = 20) en la fase 2, unas 41 a 47 horas de proceso, que con 6 procesos en paralelo se estiman en 14 a 19 horas de reloj.

## 11. Paper guía y réplica

- **Paper guía:** Zeng et al. (2025), que presenta UAVIDS-2025 y publica su código.
- **Resultado clave:** F1 de 94 % de la LSTM en la tabla IV, con el protocolo de los autores: 16 atributos, estandarización ajustada en entrenamiento, partición aleatoria estratificada 80/20 y su entrenamiento. Se replican también DNN (93 %), CNN (93 %), regresión logística (82 %) y XGBoost (96 %).
- **Réplica:** 10 particiones y semillas, de 42 a 51, como su opción `--seeds`, con el código de este repositorio, que reproduce el de los autores.
- **Criterio de éxito:** la diferencia absoluta entre la media de la réplica y el valor reportado es menor que δ, o el valor reportado cae dentro del IC 95 % de la media. La métrica es el F1 macro y, como sensibilidad, el ponderado, porque el artículo no declara el promedio de esa columna.
- **Limitación candidata:** confusión entre Blackhole y Wormhole, cuantificada como (n_B→W + n_W→B) / (n_B + n_W), con IC 95 % por *bootstrap* por bloques. Se verifica en el piloto con tres modelos del Top 10 (Transformer, RNN y KAN), sobre validación, y se fija en la adenda antes de la ejecución definitiva.

## 12. Análisis exploratorios permitidos

Robustez ante ruido gaussiano (σ ∈ {0,1; 0,25; 0,5; 1,0} en la escala estandarizada), atributos faltantes imputados con la media de entrenamiento (p ∈ {0,1; 0,25; 0,5}) y escalamiento multiplicativo de los atributos crudos (×(1 + ε), ε ∈ {0,1; 0,25; 0,5}), con el índice de robustez del área bajo la curva de degradación normalizada; calibración y *temperature scaling* ajustado en validación; incertidumbre con ensambles de semillas, *MC dropout* y predicción conforme calibrada en validación; interpretabilidad con gradientes integrados y permutación, con pruebas de borrado y estabilidad entre semillas; ablación del mejor modelo; techo de desempeño; análisis por clase, por tamaño de flujo y por bloque; taxonomía de errores; curvas de aprendizaje; validación anidada, optimismo por *bootstrap* y prueba de permutación de etiquetas en los modelos cuyo costo lo permita. Ninguno se usa para elegir el modelo final.

## 13. Congelamiento de datos

- SHA-256 del CSV: `d50d339f68be7b23f0bf089dd438b20a1835c13182d8641220538121440164d0`.
- SHA-256 de los índices de prueba: `2a2dab99c08ed1f19f24add07f5c6b0375e5ce0df7951a99090f54f4f2cd32b4`.
- SHA-256 de `particion.npz` del Entregable 1: `4c6239a9dc529249af8fc1f009c7197fa9aeaf41dab9d3327eadaf7e9051789e`.

## 14. Lo que se ejecutó antes de este documento

Para dimensionar el cómputo y probar el código, antes de este preregistro se corrieron dos cosas, ninguna de las cuales fija un hiperparámetro ni una decisión de este documento: una medición de velocidad, de unos 6 segundos de entrenamiento por modelo sobre lotes de entrenamiento, sin evaluar, y una prueba de humo de 2 épocas por tipo de modelo, que calculó predicciones sobre validación y prueba para verificar el ejecutor. Sus resultados se borraron sin revisar las métricas de prueba. Se declaran porque el conjunto de prueba se tocó en esa prueba de humo.
