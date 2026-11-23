# Revisión de novedad del modelo original (sección 6.3)

**Primera consulta:** 22/11/2026, en OpenAlex y arXiv, por programa (`buscar_novedad.py`; exportaciones en `exportaciones/`). Scopus y Web of Science se consultan con acceso institucional (Universidad del Norte) con las mismas cadenas; ver más abajo. **Repetición antes de la entrega:** 23/11/2026, con el mismo programa (`exportaciones/2026-11-23_resumen_novedad.json`). Los conteos coinciden con los del 22/11 salvo OpenAlex N7, que pasa de 22 a 23 por BRIDGE (ICML 2025), una mezcla de expertos en un modelo fundacional de grafos con cotas de generalización; es otro problema y no cambia la declaración.

## Cadenas

| Clave | Propósito | Cadena |
|---|---|---|
| N1 | Mecanismo + problema (UAV) | `("mixture of experts" OR "gating network" OR "gated fusion" OR "dynamic ensemble selection") AND ("intrusion detection" OR "attack detection") AND (UAV OR drone OR FANET OR "unmanned aerial")` |
| N2 | Mecanismo + intrusiones con árboles | `("mixture of experts" OR "gating network") AND "intrusion detection" AND ("decision tree" OR "gradient boosting" OR "tree-based" OR "soft tree")` |
| N3 | Codificación por umbrales + intrusiones | `("piecewise linear encoding" OR "numerical embedding" OR "soft binning" OR "learnable threshold" OR "soft decision tree" OR "neural decision tree") AND ("intrusion detection" OR "network traffic")` |
| N4 | El mecanismo en **otras aplicaciones** (origen) | `("mixture of experts" OR "gating network") AND ("soft decision tree" OR "neural decision tree" OR "differentiable tree" OR "piecewise linear encoding" OR "numerical embeddings") AND (tabular OR vision OR language OR "time series")` |
| N5 | Híbridos árbol-red en intrusiones de UAV | `(UAV OR drone OR FANET OR "unmanned aerial") AND "intrusion detection" AND (hybrid OR ensemble OR fusion) AND ("gradient boosting" OR LightGBM OR XGBoost OR "random forest" OR "decision tree") AND ("neural network" OR "deep learning")` |
| N6 | Sinónimos en **español** | `("mezcla de expertos" OR "red de compuerta" OR fusión) AND ("detección de intrusiones") AND (drones OR UAV OR "vehículos aéreos no tripulados")` |
| N7 | Origen de la **propiedad** (Lipschitz en mezclas) | `"Lipschitz" AND ("mixture of experts" OR "gating network")` |

## Matriz de novedad (cuadro 2 del enunciado)

"Revis." son los registros revisados por título y resumen. "Relev." son los que se clasifican como trabajo cercano.

| Base | Cadena | Result. | Revis. | Relev. | Observación |
|---|---|---|---|---|---|
| OpenAlex | N1 | 3 | 3 | 1 | SkyVehiGuard-Net (2026): Mamba + Transformer + mezcla dispersa de expertos para IoV en redes satelitales; no son flujos de UAV |
| OpenAlex | N2 | 0 | 0 | 0 | Sin resultados |
| OpenAlex | N3 | 3 | 3 | 1 | EDND (2026): ensamble de árboles de decisión neuronales para intrusiones en salud industrial; sin compuerta ni experto suave |
| OpenAlex | N4 | 1 | 1 | 1 | Autoencoders híbridos para tablas (2025): sesgo espectral de las redes frente a los árboles. Motivación compartida, otro mecanismo |
| OpenAlex | N5 | 13 | 13 | 3 | (a) XAI y análisis estadístico en UAVIDS-2025 (CITS 2026): árboles, redes y *stacking* con validación cruzada estratificada; documenta el solapamiento Blackhole–Wormhole. (b) Ensambles explicables en redes de drones (2026). (c) Evaluación multiconjunto en UAV (IEEE IoT J., 2026) |
| OpenAlex | N6 | 0 | 0 | 0 | Sin resultados en español |
| OpenAlex | N7 | 22 | 22 | 2 | Puigcerver et al. (2022), robustez adversarial de la mezcla de expertos con argumento de Lipschitz; L2R (2026), enrutamiento con control de Lipschitz. Ninguno en intrusiones |
| arXiv | N1 | 0 | 0 | 0 | Sin resultados |
| arXiv | N2 | 1 | 1 | 1 | CNN + mezcla dispersa de expertos para intrusiones en 5G (arXiv 2412.03483, 2024) |
| arXiv | N3 a N5 | 3 | 3 | 0 | Duplicados de OpenAlex |
| arXiv | N6 | 0 | 0 | 0 | Sin resultados |
| arXiv | N7 | 13 | 13 | 0 | Duplicados de OpenAlex |
| IEEE Xplore | N1 | 2 | 2 | 2 | (a) Switch IDS (Wang et al., IEEE IoT J., 2025): mezcla de expertos en un IDS para computación de borde asistida por aeronaves, con expertos homogéneos y pérdida de reparto de tráfico entre nodos. (b) Fusión con compuerta de Transformer y BiLSTM en aprendizaje federado (ICIMA 2026), sobre Edge-IIoTset y ToN-IoT |
| OpenAlex, citación hacia adelante | Paper guía y 5 trabajos cercanos | 56 | 56 | 4 | PS-CAHE (2026), fusión heterogénea consciente de los pares que se confunden, sobre UAVIDS-2025; fusión XGBoost–MLP para 5G (2026); VG-MoE (2026); y, como evidencia de fuga, Aksu y Gümüş (2026) |
| OpenAlex, citación hacia atrás | Referencias de los trabajos cercanos | 181 | 32 (por título) | 0 | Fuentes ya conocidas (Jacobs et al., 1991; Shazeer et al., 2017), ensambles de árboles para drones sin compuerta por flujo y una fusión de DNN en varios niveles para UAV en 6G (Expert Systems with Applications, 2024), revisada solo por título porque su resumen no está disponible |
| GitHub | 6 consultas | 27 repositorios | 27 | 0 | Mezclas de expertos para intrusiones con expertos homogéneos (dmoe_nids, sparse MoE para IoT, hypergate-moe-ids, IDS-Using-MOE) y un repositorio de aprendizaje profundo sobre UAVIDS-2025; ninguno implementa un experto de escalones suaves |
| Hugging Face Hub | 4 consultas | 1 | 1 | 0 | Sin modelos ni conjuntos de intrusiones; Papers with Code cerró en 2025 y redirige a Hugging Face |
| Scopus | N1 | 4 | 4 | 3 | 23/11/2026. SkyVehiGuard-Net (duplicado de OpenAlex) y la fusión con compuerta de ICIMA 2026 (duplicado de IEEE Xplore). Dos registros son actas de congreso que coinciden por términos de trabajos distintos: las de PRCV 2025 no son pertinentes; las de ICCUBEA 2026 contienen un trabajo nuevo, una mezcla de expertos distribuida para intrusiones entre conjuntos de datos con una representación universal del flujo, clasificado solo por título |
| Scopus | N2 | 2 | 2 | 2 | 23/11/2026. (a) U-Flow MoE-IDS (Bhaskaran et al., ICCUBEA 2026), el trabajo de las actas de N1, ahora con registro y resumen propios. (b) ACGE (Abdelnaby, *Symmetry*, 2026), nuevo: compuerta aprendida por muestra entre LightGBM y un Transformer para intrusiones en fog-IoT; es el trabajo más cercano en el mecanismo |
| Scopus | N5 | 28 | 28 | 4 | 23/11/2026. 18 registros son actas de congreso que coinciden por términos de trabajos distintos; se revisan por la lista de títulos y solo traen una revisión de ciberseguridad en UAV y un IDS para UAV con aprendizaje por refuerzo, sin fusión de árboles y redes. De los 10 trabajos restantes, 7 ya estaban en OpenAlex N5, entre ellos los tres relevantes. Nuevo y relevante: Hedfi et al. (IC3IT 2026), detectores en paralelo de trayectoria, Random Forest y LSTM para enjambres de UAV, sin compuerta aprendida |
| Web of Science | N1 | 1 | 1 | 1 | 23/11/2026. SkyVehiGuard-Net, duplicado de OpenAlex y Scopus |
| Web of Science | N2 | 1 | 1 | 1 | 23/11/2026. ACGE, duplicado de Scopus; U-Flow MoE-IDS no está indexado |
| Web of Science | N5 | 3 | 3 | 1 | 23/11/2026. Los tres ya estaban en OpenAlex y Scopus; el relevante es el de los ensambles explicables en redes de drones. Web of Science no agrega trabajos nuevos |

## Clasificación de la diferencia frente a cada trabajo cercano

| Trabajo cercano | Qué hace | Diferencia con la propuesta | Categoría |
|---|---|---|---|
| Fuente: Jacobs et al. (1991), mezcla de expertos | Expertos homogéneos con compuerta; entrenamiento conjunto | Expertos de **sesgo inductivo distinto** (escalones suaves frente a suave), **entrenamiento cooperativo** con gradiente detenido, mezcla de probabilidades con **cota de Lipschitz** | Mecanismo y entrenamiento |
| Fuente: Gorishniy et al. (2022), codificación de atributos numéricos | Codificación por tramos lineales con cortes fijos | **Escalones sigmoides** con umbrales y temperaturas **aprendibles**, y τ ≥ τ_min para acotar la pendiente | Mecanismo |
| CNN + MoE para intrusiones en 5G (2024) | Expertos densos homogéneos sobre una representación CNN, enrutamiento disperso, entrenamiento conjunto; 5G-NIDD | Expertos heterogéneos, compuerta densa sobre el flujo, entrenamiento cooperativo, cota demostrada; flujos de UAV | Mecanismo y entrenamiento |
| SkyVehiGuard-Net (2026) | Mamba + Transformer + MoE disperso + destilación; IoV | Otra arquitectura y otro problema | Mecanismo |
| XAI en UAVIDS-2025, de árboles a híbridos (2026) | Compara árboles, redes y *stacking* (regla estándar) | Compuerta **por flujo** aprendida, con un experto neuronal de umbrales en lugar de un árbol externo; el *stacking* es una de nuestras reglas de comparación | Mecanismo |
| EDND en salud industrial (2026) | Ensamble de árboles de decisión neuronales | Sin compuerta entre sesgos ni experto suave; otro problema | Mecanismo |
| **PS-CAHE (Ma et al., 2026), sobre UAVIDS-2025** | Ensamble heterogéneo (XGBoost y otros) con atributos semánticos de protocolo y dos pesos de fusión elegidos fuera de muestra para los pares de clases que se confunden; F1 macro de 95,6 % en una partición por filas, y sin ventaja en una evaluación con nodos disjuntos | Compuerta **aprendida por flujo** entre un experto neuronal de escalones suaves y uno suave, entrenada de forma cooperativa y con cota de Lipschitz; PS-CAHE fija pesos por par de clases sobre modelos externos. Misma motivación (los pares que se confunden), otro mecanismo | Mecanismo y entrenamiento |
| Switch IDS (Wang et al., 2025) | Mezcla de expertos homogéneos en un IDS por paquete para borde asistido por aeronaves, con pérdida de reparto entre nodos | La compuerta de la propuesta combina sesgos inductivos distintos para mejorar la clasificación, no para repartir carga; flujos de UAV | Mecanismo y entrenamiento |
| Fusión XGBoost–MLP para 5G (Al-Balasmeh, 2026) | Fusión de probabilidades de un árbol y un MLP con un peso global elegido en validación | Peso **por flujo** aprendido por una compuerta, con el sesgo de umbral dentro de la red (escalones suaves) en lugar de un árbol externo | Mecanismo |
| VG-MoE (AL-Essa et al., 2026) | Compuerta ajustada sobre predicciones fuera de muestra de expertos ya entrenados; NSL-KDD y CICIDS17 | Entrenamiento cooperativo de extremo a extremo con gradiente detenido, expertos de sesgo distinto, cota de Lipschitz; flujos de UAV | Entrenamiento y mecanismo |
| **ACGE (Abdelnaby, 2026), en fog-IoT** | Regresión logística sobre cuatro rasgos de la salida del LightGBM (probabilidad máxima, entropía, indicador de clase minoritaria y distancia a los centroides de sus hojas) que da un peso continuo para mezclar las probabilidades del LightGBM y de un Transformer que recibe 5 flujos consecutivos. Se ajusta una sola vez en la partición de validación, después de entrenar los dos modelos, para predecir cuál de ellos tiene menor entropía cruzada, y en la práctica es casi binaria (82 % de las muestras con peso ≤ 0,10 y 18 % con peso ≥ 0,90). El autor la presenta como una adaptación de mecanismos conocidos (*stacking*, mezcla de expertos), no como un mecanismo nuevo. IoTID20 y NSL-KDD, con transferencia a CICIoT2023 y Edge-IIoTset; sin datos de UAV ni cota teórica (texto completo, sección 3.7) | El experto de umbrales de la propuesta es neuronal y diferenciable (escalones suaves con umbrales aprendibles), no un LightGBM externo; la compuerta ve el flujo y no rasgos de confianza de un modelo, y se entrena con los expertos bajo el esquema cooperativo, no después y por separado; la mezcla admite una cota de Lipschitz que un árbol, discontinuo, no permite. Un flujo por decisión; flujos de UAV | Mecanismo y entrenamiento |
| U-Flow MoE-IDS (Bhaskaran et al., ICCUBEA 2026) | Un experto por conjunto de datos (NSL-KDD, CICIDS2017, Bot-IoT) sobre un esquema común de atributos del flujo, entrenados en Apache Spark, y un metaaprendiz de regresión logística por *stacking* que fusiona sus predicciones; Macro-F1 entre 0,477 y 0,902 al dejar un conjunto por fuera | La compuerta de la propuesta elige por flujo entre sesgos inductivos distintos dentro de un mismo conjunto y se entrena de forma cooperativa con los expertos; el *stacking* es una de nuestras reglas de comparación; cota de Lipschitz; flujos de UAV | Mecanismo y entrenamiento |
| Hedfi et al. (IC3IT 2026), enjambres de UAV | Tres detectores en paralelo: desvío de trayectoria, Random Forest sobre el tráfico y LSTM para ataques de varios pasos; el Random Forest es el mejor (94,4 % de exactitud) | No hay compuerta ni fusión aprendida entre los detectores; la propuesta pondera por flujo dos expertos de sesgo distinto | Mecanismo |
| Puigcerver et al. (2022); L2R (2026) | Robustez de la mezcla de expertos mediante Lipschitz de la compuerta y los expertos | La cota se deriva para una mezcla de probabilidades con codificador de escalones de temperatura mínima, y se aplica a intrusiones en UAV | Mecanismo (propiedad) |

## Búsqueda por citación y en repositorios de código

Hecha el 23/11/2026 con `experiments/novedad/citacion_y_repositorios.py` (resultados en `exportaciones/2026-11-23_citacion_repositorios_resumen_novedad.json`).

- **Hacia adelante:** obras que citan el paper guía, los tres trabajos más cercanos (*stacking* sobre UAVIDS-2025, CNN + MoE en 5G y Switch IDS) y las dos fuentes del mecanismo, filtradas por intrusiones (y por UAV en Jacobs et al., que tiene miles de citas): 56 obras. Aparecen cuatro trabajos cercanos nuevos, clasificados arriba.
- **Hacia atrás:** referencias de esos trabajos con términos del mecanismo en el título: 32 de 181. No aparece ninguna compuerta por flujo entre sesgos distintos.
- **Repositorios:** 27 repositorios en GitHub con seis consultas y ninguno pertinente en Hugging Face Hub. Las mezclas de expertos para intrusiones que aparecen usan expertos homogéneos y ninguna está aplicada a UAV.
- **Evidencia convergente para el hallazgo alternativo:** PS-CAHE pierde su ventaja con nodos disjuntos, y Aksu y Gümüş (2026) muestran que las cifras altas en un benchmark público de UAV dependen de la forma de evaluar y de fugas de identificadores. Las dos refuerzan la conclusión de que en UAVIDS-2025 la partición pesa más que el modelo.

## Declaración de novedad (confirmada en Scopus, Web of Science y con la búsqueda por citación)

Al mejor de nuestro conocimiento y según la búsqueda descrita, no se encontraron estudios que apliquen a la detección de intrusiones en redes de UAV una compuerta por flujo entre un experto de escalones suaves con umbrales aprendibles y un experto neuronal suave, entrenada con un esquema cooperativo y con cota de Lipschitz demostrada. Los trabajos más cercanos son:

- PS-CAHE (2026), la fusión consciente de los pares que se confunden sobre UAVIDS-2025;
- ACGE (2026), la compuerta aprendida por muestra entre LightGBM y un Transformer en fog-IoT, el más cercano en el mecanismo;
- Switch IDS (2025), la mezcla de expertos homogéneos en un IDS para borde asistido por aeronaves;
- la mezcla de expertos homogéneos para intrusiones en 5G (2024);
- el estudio de *stacking* sobre UAVIDS-2025 (2026);
- las fuentes del mecanismo (Jacobs et al., 1991; Gorishniy et al., 2022).

Frente a todos ellos, la diferencia es de mecanismo o de esquema de entrenamiento, no solo de aplicación.

## Verificación por un tercero (D7)

Los profesores del curso indicaron el 23/11/2026 que esta verificación queda a su cargo, así que se omite en el informe. Las cadenas exactas están en `buscar_novedad.py` (OpenAlex y arXiv) y en `cadenas_scopus_wos.md` (Scopus, Web of Science y los enlaces directos de OpenAlex para N1 y N5).
