# Protocolo experimental preregistrado del Entregable 3 (`protocolo.md`)

Fecha de congelamiento: 22/11/2026. Etiqueta de Git: `e3-protocolo`. Se congela **antes** de toda corrida definitiva del modelo original, de las semillas adicionales de los rivales y de los conjuntos adicionales. Cualquier desviación posterior se declara en el informe (capítulo 7) con su justificación.

## 1. Estándar

El estándar teórico, declarado en `experiments/estandar.md` (etiqueta `e3-estandar`), sin relajaciones posteriores.

## 2. Hipótesis

**Confirmatoria (R3).** Para cada rival k de la familia (sección 6): H0: μ_original = μ_k frente a H1: μ_original ≠ μ_k, con μ el F1 macro medio en la prueba congelada del E1. El modelo original supera a k si se cumplen las seis condiciones de la sección 8.4 del enunciado, entre ellas que la diferencia media sea mayor que δ = 0,01.

**Hipótesis del mecanismo (falsable).** El codificador de escalones suaves da al modelo el sesgo de umbral por eje de los árboles y reduce la confusión Blackhole–Wormhole frente a las redes de sesgo suave. La compuerta por flujo combina ese experto con uno suave, que reconoce mejor Sybil. Por tanto, el F1 macro del original supera al de la mejor red del Top 10 en al menos δ, y la ventaja se concentra en Blackhole y Wormhole.

**Predicciones secundarias, declaradas a priori (exploratorias):**

- **P1.** Sin el codificador de escalones (variante A2), la confusión Blackhole–Wormhole sube.
- **P2.** El peso medio de la compuerta hacia el experto de umbrales es mayor en los flujos Blackhole y Wormhole que en los Sybil.
- **P3.** La ventaja del original frente a la mejor red se concentra en el F1 de Blackhole y Wormhole.
- **P4.** La cota de Lipschitz importa: entre las variantes del barrido de τ_min, la de menor K se degrada menos ante ruido gaussiano en las entradas.
- **P5.** Como ningún modelo generaliza a escenarios de simulación no vistos (prueba de D3), el original tampoco mejora en el segmento de prueba del escenario de Sybil aislado (filas 108 071 a 109 045).

**Criterio de éxito y de fracaso del diseño.** Éxito: R3 frente a los 12 rivales. Fracaso parcial: si no se cumple R3, el tope es 4,0 y el trabajo se sostiene en el hallazgo alternativo de la sección 12.

## 3. Métricas

- **Primaria:** F1 macro, heredada del E1, con δ = 0,01 (margen del E1).
- **Secundarias:** exactitud, exactitud balanceada, MCC, F1 por clase, confusión Blackhole–Wormhole, log-loss, Brier y ECE con 15 intervalos (antes y después del *temperature scaling* ajustado en validación).
- **Eficiencia:** parámetros, MAC, latencia en CPU (mediana y percentil 95) y memoria pico.

## 4. Fuente del mecanismo

Ver `experiments/procedencia.md`: es un mecanismo propio, con piezas transferidas de Jacobs et al. (1991), Gorishniy et al. (2022) y Shazeer et al. (2017), y cuatro modificaciones algorítmicas.

## 5. Datos, particiones, semillas y determinismo

### 5.1. Problema principal (UAVIDS-2025)

- Se usa la partición congelada del E1 y el E2: bloques de 1000 flujos, 73 267 filas de entrenamiento, 17 802 de validación y 22 883 de prueba, con SHA-256 de la prueba `2a2dab99…cd32b4` verificado por `codigo/datos.py`. La transformación es log(1 + x), con PacketDropRate recortado en 1, y la estandarización se ajusta con el entrenamiento.
- **Benchmarks congelados (Bloque A):** las 301 corridas definitivas del E2 (JSON y probabilidades de validación y de prueba). SHA-256 del conjunto ordenado de archivos `definitiva_*.json` y `definitiva_*.npy`: `a35f853544388d437ed9ba6755dec2ca2c77fe0a8b8b3b5be7fd3264d1f05f03`. No se reajustan.
- **Semillas del original:** 6001 a 6048. **Semillas adicionales de rivales:** CNN-LSTM, de 6021 a 6048, y TabNet, de 6021 a 6038, con los mismos hiperparámetros y la misma arquitectura del E2, porque el análisis de potencia (sección 7) lo exige. Los demás rivales conservan sus 20 semillas (6001 a 6020).
- **Determinismo:** CPU, con `torch.manual_seed`, `numpy.random.seed` y `random.seed` fijados en cada corrida (`codigo/motor.fijar_semilla`), 2 hilos por proceso y PyTorch 2.6.0+cpu. Las corridas que se hagan en GPU (solo BiLSTM y GRU de los conjuntos adicionales, si se usa RunPod) fijan `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True` y `CUBLAS_WORKSPACE_CONFIG=:4096:8`, y se declaran.

### 5.2. Partición de desarrollo y de confirmación (sección 7.2)

- **Diseño:** solo filas de entrenamiento del E2 (`training/particion_desarrollo.py` y la partición por escenario de `_trabajo_local/prueba_d3`).
- **Confirmación:** la validación y la prueba del E2.
- *Nd* = 10 variantes (`experiments/design_log.md`). El diseño congelado es la variante 10.

### 5.3. Tres conjuntos adicionales (sección 8.3)

Elegidos el 22/11/2026, antes de entrenar cualquier modelo con ellos, con estos criterios: fuente pública, misma tarea (clasificación multiclase de intrusiones) y diferencia en al menos dos ejes. Procedencia y SHA-256 en `data/procedencia_conjuntos_adicionales.md`.

| Conjunto | Datos que se usan | Ejes en que difiere de UAVIDS-2025 |
|---|---|---|
| UAV-NIDD | Caso 3 (GCS), el único con registros **por flujo**: 149 434 filas, 44 estadísticos de flujo y 10 clases. Los casos 1 y 2 son registros por trama de radio, con identificadores, y no se usan | Banco real frente a simulación, 44 atributos, desbalance |
| UNSW-NB15 | Los 4 CSV completos, ordenados por `Stime`; `attack_cat` vacío se toma como Normal | Red empresarial, 2,5 millones de registros, desbalance fuerte (87 % Normal) |
| UNR-IDD | `UNR-IDD.csv`, etiqueta `Label` (6 clases) | Estadísticos de puertos SDN en lugar de flujos, tamaño menor, clase minoritaria del 2,7 % |

**Preprocesamiento común (fijo antes de ver resultados):**

1. Se excluyen identificadores (IP, puertos, `Switch ID`, `Port Number`, `Connection Point`), marcas de tiempo, columnas categóricas (`proto`, `state`, `service`), etiquetas binarias y columnas constantes.
2. Se eliminan las filas duplicadas exactas (atributos y etiqueta), como en el E1.
3. Se eliminan las clases con menos de 300 filas tras la deduplicación.
4. Transformación sign(x) · log(1 + |x|), la extensión a valores negativos de la del E1, y estandarización con el entrenamiento.
5. **Partición agrupada y estratificada:** bloques de 250 filas contiguas **dentro de cada clase**, en el orden de la fuente, repartidos al azar (semilla 2027) en 70/15/15 dentro de cada clase, con al menos un bloque por clase en validación y en prueba. Filas vecinas de la misma ejecución del ataque nunca quedan en particiones distintas. Ningún bloque se comparte entre particiones; se verifica en `data/<conjunto>_resumen.json`. Las fuentes vienen ordenadas por ejecución de ataque. Una primera versión con bloques de 1000 filas sin estratificar dejaba clases enteras fuera de la prueba (UAV-NIDD quedaba con 3 de 6), por lo que se corrigió antes de entrenar cualquier modelo.
6. Para acotar el cómputo, dentro de cada partición se submuestrea con un tope por clase repartido por llenado (*water-filling*): 28 000 filas de entrenamiento, 6 000 de validación y 6 000 de prueba, conservando el orden de la fuente (semilla 2027).
7. El F1 macro de prueba se calcula sobre las clases presentes en la prueba.

**Resultado del preprocesamiento** (solo conteos; ningún modelo entrenado):

| Conjunto | Filas fuente | Duplicadas eliminadas | Clases que quedan | Atributos | Entrenamiento / validación / prueba |
|---|---|---|---|---|---|
| UAV-NIDD | 149 434 | 112 628 | 6 (Bruteforce, DDoS, DoS, Normal, Reco, Scanning) | 43 | 24 705 / 5 808 / 5 804 |
| UNSW-NB15 | 2 540 047 | 516 060 | 9 (Worms descartada por tener menos de 300 filas) | 38 | 28 000 / 6 000 / 6 000 |
| UNR-IDD | 37 411 | 27 | 6 | 18 | 26 153 / 5 615 / 5 616 |

**Modelos y presupuesto en los conjuntos adicionales:** los 12 rivales y el original, cada uno con sus hiperparámetros y su arquitectura del problema principal. **No hay búsqueda nueva para ningún modelo**: el presupuesto de ajuste adicional es el mismo, cero, para todos. Mismo protocolo de entrenamiento (máximo 30 épocas y parada temprana sobre el F1 de validación). **10 semillas** (6001 a 6010). LightGBM se detiene a las 100 rondas sobre la validación.

## 6. Familia de contrastes y pruebas

- **Familia confirmatoria (m = 12):** CNN-LSTM, Autoencoder, RNN, Transformer, BiLSTM, ResNet, GRU, KAN, TabNet, DBN, LightGBM y CNN 1D. La regla de composición está en `estandar.md` y no depende del desempeño. La afirmación "supera a todos" es una prueba de unión-intersección: hay que rechazar los 12 contrastes.
- **Exploratorios:** frente a DNN, LSTM y regresión logística.
- **Tres pruebas por contraste** (cuadro 4, fila F1 tabular):
  - **T1.** *t* pareado sobre las semillas compartidas, con el F1 macro de prueba de cada semilla. Supuesto de normalidad de las diferencias verificado con Shapiro-Wilk; si se rechaza, se reporta y se confía en T2 y T3.
  - **T2.** Wilcoxon de rangos con signo sobre las mismas semillas (20 o más pares, por encima del mínimo de 9 para α′ = 0,05/12).
  - **T3.** *Bootstrap* pareado por bloques de 1000 flujos de prueba (10 000 réplicas) de la diferencia de F1 macro entre los promedios de probabilidades sobre las semillas compartidas, con IC 95 % BCa y *p* bilateral. Respeta la unidad de independencia de la partición.
- **Control de la familia:** Holm sobre los 12 contrastes, aparte para cada prueba. Para cada contraste, la decisión usa el **máximo** de los *p* ajustados de T1, T2 y T3 (sección 8.7.1).
- **Procedimiento global:** Friedman con post hoc de Holm sobre los 13 modelos (bloques = semillas 6001 a 6020).
- **Tamaño del efecto:** δ de Cliff sobre las semillas (umbral |δ| ≥ 0,147) y d de Cohen pareado, con IC 95 % de la diferencia.
- **Sensibilidad de la conclusión (8.4.6 y 8.7.3):**
  - prueba bayesiana con ROPE = ±δ;
  - *t* con corrección de Nadeau-Bengio sobre la validación cruzada 5×2 del E2 (solo hay corridas 5×2 de LightGBM y CNN 1D; el original se corre en las mismas 10 particiones);
  - métrica secundaria: exactitud balanceada;
  - subconjunto de semillas: las 20 primeras.
- **Ajuste y sobreajuste del original (tablas A y B del E2):** permutación de etiquetas (200 permutaciones), brecha entre entrenamiento y validación con margen δ, consistencia entre validación y prueba y frente al trivial, calibración (Hosmer-Lemeshow por clase uno contra todos) y curva de aprendizaje.

## 7. Potencia (piloto del 22/11/2026)

Piloto con 5 semillas (6001 a 6005), solo en validación. F1 de validación del original: 0,912; 0,919; 0,927; 0,915 y 0,923. Con R = ⌈((z_{1−α′/2} + z_{0,90}) σ̂_d / δ)²⌉, α′ = 0,05/12 y δ = 0,01 (`experiments/potencia.py` y `experiments/resultados/potencia.json`):

| Rival | σ̂_d | R necesarias | R usadas |
|---|---|---|---|
| CNN-LSTM | 0,0167 | 48 | **48** |
| TabNet | 0,0148 | 38 | **38** |
| BiLSTM | 0,0103 | 19 | 20 |
| GRU | 0,0100 | 18 | 20 |
| RNN y Transformer | 0,0096 y 0,0095 | 16 | 20 |
| CNN 1D | 0,0081 | 12 | 20 |
| Autoencoder, ResNet, KAN, DBN y LightGBM | 0,0052 a 0,0063 | 10 | 20 |

Con estas R, la potencia por contraste es de al menos 0,90. La potencia conjunta prevista es de alrededor de 0,90^12 ≈ 0,28, y se reportará la observada. Si un contraste resulta no concluyente, se reporta su diferencia mínima detectable; la ausencia de significancia no se interpreta como equivalencia.

## 8. Presupuesto de ajuste y de cómputo

- **Hiperparámetros del original:** las mismas 12 configuraciones de (lr, weight_decay, batch) que recibieron los 13 modelos profundos del E2, con semilla de entrenamiento 4242 y 15 épocas, elegidas por el mayor F1 de validación. Resultado: lr = 0,003, weight_decay = 10⁻⁴ y lote 512 (F1 de validación 0,924). Los hiperparámetros propios (T = 8, τ_min = 0,05, γ = 0,01 y los anchos) se fijaron en la partición de desarrollo y cuentan en *Nd*.
- El presupuesto se mide en **número de configuraciones**, igual para todos (12). También se reporta el tiempo de cómputo total por modelo.

## 9. Ablación (familia secundaria propia, Holm dentro de ella)

Semillas 6001 a 6020. Cada variante cambia una sola cosa frente al modelo completo (A0):

| Variante | Qué cambia |
|---|---|
| **A1** | Sin compuerta: promedio fijo 1/2 de los dos expertos |
| **A2** | Sin escalones suaves: el experto de umbrales se reemplaza por un MLP sobre x de capacidad comparable |
| **A3** | Sin entrenamiento cooperativo: entrenamiento conjunto clásico de la mezcla |
| **A4** | Sin regularizador de balance: γ = 0 |
| **A5** | Modelo base sin componentes propios: solo el experto suave |
| **A6** | Solo el experto de umbrales |

Las variantes de ablación comparten la configuración de optimización del original.

## 10. Robustez, calibración y eficiencia (cuadro 5)

Se aplica el mismo protocolo a la familia en todo lo que solo requiere inferencia sobre las probabilidades o los pesos guardados.

- **Sensibilidad a hiperparámetros:** τ_min ∈ {0,02; 0,05; 0,1; 0,2}, T ∈ {4; 8; 16} y γ ∈ {0; 0,01; 0,1}, con 3 semillas, en validación.
- **Estabilidad por semilla:** cajas y CV del F1 en todas las semillas; Kruskal-Wallis entre modelos.
- **Perturbaciones:** ruido gaussiano en el espacio estandarizado (σ ∈ {0,1; 0,3; 0,5}), atributos faltantes imputados con la media (10, 25 y 50 % de los atributos) y valores atípicos (1, 5 y 10 % de filas con un atributo multiplicado por 10), con la curva de degradación de cada modelo.
- **Curva de aprendizaje:** 10, 25, 50 y 100 % del entrenamiento, con 3 semillas, para el original, LightGBM y CNN-LSTM.
- **Cambio de distribución:** el segmento de prueba del escenario de Sybil aislado y la partición por escenario de la prueba de D3. Ambos quedaron fuera de toda decisión de diseño.
- **Calibración:** curva de confiabilidad, ECE, Brier y log-loss antes y después del *temperature scaling* ajustado solo en validación.
- **Incertidumbre:** predicción conforme dividida (APS, ajustada en validación) con cobertura nominal del 90 %; cobertura empírica, cobertura por clase y tamaño medio del conjunto. Se declara la limitación de la intercambiabilidad por la dependencia entre bloques.
- **Eficiencia:** parámetros, MAC, latencia en CPU (mediana y p95), tiempo de entrenamiento y frontera de Pareto.
- **Análisis de errores:** por clase, por escenario y errores compartidos con los rivales.

## 11. Interpretabilidad

SHAP (KernelExplainer con fondo de 100 flujos de entrenamiento) para el original y para el mejor *benchmark*, sobre 500 flujos de prueba. Fidelidad por borrado de los atributos más importantes, estabilidad entre 3 semillas y lectura de los pesos de la compuerta por clase (P2).

## 12. Decisión previa al congelamiento (sección 14.5.9)

Si el original no cumple R3, el hallazgo alternativo que sostiene el trabajo es triple:

1. Un *benchmark* riguroso y preregistrado de 13 modelos con potencia por contraste reportada.
2. La explicación de por qué el escenario de simulación pesa más que la arquitectura: la auditoría de fuga L3.2, la prueba de D3 por escenario no visto y P5.
3. La medición de cuánto cierra la brecha entre árboles y redes un experto con escalones suaves y cuándo la compuerta ayuda (P1 a P3).

En ese caso el tope de calificación es 4,0 (sección 14.1).
