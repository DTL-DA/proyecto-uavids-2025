# Revisión de novedad del modelo original (sección 6.3)

**Primera consulta:** 22/11/2026, en OpenAlex y arXiv, por programa (`buscar_novedad.py`; exportaciones en `exportaciones/`). Scopus y Web of Science se consultan con acceso institucional (Universidad del Norte) con las mismas cadenas; ver más abajo. **Repetición antes de la entrega:** 26/11/2026.

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
| Scopus | N1, N2, N5 | *(pendiente, Donny)* | | | |
| Web of Science | N1, N2, N5 | *(pendiente, Donny)* | | | |

## Clasificación de la diferencia frente a cada trabajo cercano

| Trabajo cercano | Qué hace | Diferencia con la propuesta | Categoría |
|---|---|---|---|
| Fuente: Jacobs et al. (1991), mezcla de expertos | Expertos homogéneos con compuerta; entrenamiento conjunto | Expertos de **sesgo inductivo distinto** (escalones suaves frente a suave), **entrenamiento cooperativo** con gradiente detenido, mezcla de probabilidades con **cota de Lipschitz** | Mecanismo y entrenamiento |
| Fuente: Gorishniy et al. (2022), codificación de atributos numéricos | Codificación por tramos lineales con cortes fijos | **Escalones sigmoides** con umbrales y temperaturas **aprendibles**, y τ ≥ τ_min para acotar la pendiente | Mecanismo |
| CNN + MoE para intrusiones en 5G (2024) | Expertos densos homogéneos sobre una representación CNN, enrutamiento disperso, entrenamiento conjunto; 5G-NIDD | Expertos heterogéneos, compuerta densa sobre el flujo, entrenamiento cooperativo, cota demostrada; flujos de UAV | Mecanismo y entrenamiento |
| SkyVehiGuard-Net (2026) | Mamba + Transformer + MoE disperso + destilación; IoV | Otra arquitectura y otro problema | Mecanismo |
| XAI en UAVIDS-2025, de árboles a híbridos (2026) | Compara árboles, redes y *stacking* (regla estándar) | Compuerta **por flujo** aprendida, con un experto neuronal de umbrales en lugar de un árbol externo; el *stacking* es una de nuestras reglas de comparación | Mecanismo |
| EDND en salud industrial (2026) | Ensamble de árboles de decisión neuronales | Sin compuerta entre sesgos ni experto suave; otro problema | Mecanismo |
| Puigcerver et al. (2022); L2R (2026) | Robustez de la mezcla de expertos mediante Lipschitz de la compuerta y los expertos | La cota se deriva para una mezcla de probabilidades con codificador de escalones de temperatura mínima, y se aplica a intrusiones en UAV | Mecanismo (propiedad) |

## Búsqueda por citación y en repositorios de código

*(Se completa el 23/11 con OpenAlex `cites:` y `referenced_works` de los tres trabajos más cercanos, y con búsquedas en GitHub, Papers with Code y Hugging Face Hub.)*

## Declaración de novedad (provisional; se confirma con Scopus, Web of Science, la búsqueda por citación y el verificador externo)

Al mejor de nuestro conocimiento y según la búsqueda descrita, no se encontraron estudios que apliquen a la detección de intrusiones en redes de UAV una compuerta por flujo entre un experto de escalones suaves con umbrales aprendibles y un experto neuronal suave, entrenada con un esquema cooperativo y con cota de Lipschitz demostrada. Los trabajos más cercanos son:

- la mezcla de expertos homogéneos para intrusiones en 5G (2024);
- el estudio de *stacking* sobre UAVIDS-2025 (2026);
- las fuentes del mecanismo (Jacobs et al., 1991; Gorishniy et al., 2022).

Frente a todos ellos, la diferencia es de mecanismo o de esquema de entrenamiento, no solo de aplicación.

## Verificación por un tercero (D7)

Imera Logreira, colega ajena al diseño, repite **N1 y N5** en OpenAlex (https://openalex.org, filtro "title and abstract") o en Scopus, anota la fecha y el número de resultados, revisa los títulos y confirma o refuta la declaración. Su concepto firmado se anexa al capítulo 4 del libro.
