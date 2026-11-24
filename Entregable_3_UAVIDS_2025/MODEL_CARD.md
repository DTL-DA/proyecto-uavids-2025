# Ficha del modelo (`MODEL_CARD.md`)

Formato de Mitchell et al. (2019), *Model Cards for Model Reporting*. Todas las cifras salen de `experiments/resultados/` y se regeneran al compilar el libro.

## Detalles del modelo

- **Nombre:** MoE-ES (mezcla de expertos con escalones suaves), el modelo original del Entregable 3: una mezcla de dos expertos con compuerta por flujo. En el código es `ModeloOriginal` y en las tablas y el dashboard, "Original".
- **Autor:** Donny Torres, Maestría en Analítica de Datos, Universidad del Norte. Noviembre de 2026. Versión 1.0.0.
- **Arquitectura:**
  - **Experto de umbrales:** un codificador de escalones suaves, s = σ((x − θ)/τ), con T = 8 umbrales aprendibles por atributo y temperatura τ ≥ τ_min = 0,05, seguido de un MLP(64).
  - **Experto suave:** un MLP(64, 64) sobre los atributos estandarizados.
  - **Compuerta:** un MLP(32) que, para cada flujo, da el peso de cada experto. La salida es la mezcla de las probabilidades de los dos expertos.
- **Entrenamiento:**
  - **Pérdida cooperativa:** entropía cruzada de la mezcla con los expertos detenidos, más la entropía cruzada de cada experto y γ · KL(media de la compuerta ‖ uniforme), con γ = 0,01.
  - **Optimización:** Adam con lr = 0,003, weight_decay = 10⁻⁴ y lote de 512, elegidos entre las 12 configuraciones del E2 por el F1 de validación.
  - **Parada:** máximo de 30 épocas, parada temprana con paciencia 6 sobre el F1 macro de validación, pérdida ponderada por clase y recorte del gradiente en 5.
- **Tamaño:** 14 908 parámetros y 14 528 multiplicaciones-acumulaciones (29 056 FLOPs) por flujo.
- **Propiedad demostrada:** cota de Lipschitz K = √2 · L_g + máx(L_u, L_s) de la mezcla (capítulo 5 del libro).
- **Artefacto desplegado:** `artifacts/Original/model.onnx` (opset 17, 64 KB), entrenado con la semilla 6008. Es la corrida cuya F1 de validación está más cerca de la mediana de las semillas 6001 a 6020. Su `metadata.json` registra el preprocesamiento, el esquema de entrada con los rangos del entrenamiento, la temperatura y el cuantil conforme. La prueba de ida y vuelta da un error máximo de 6,6 × 10⁻⁷ y el 100 % de las clases coincide.
- **Licencia:** MIT. **Cita:** `CITATION.cff`.

## Uso previsto

- **Uso principal:** apoyar a un analista de seguridad en la clasificación de flujos de una red de UAV en cinco clases: Normal, Blackhole, Flooding, Sybil y Wormhole. Recibe 16 estadísticos de tráfico por flujo.
- **Usuarios previstos:** investigadores y analistas que trabajan con datos del mismo tipo que UAVIDS-2025.
- **Usos no previstos:**
  - bloquear tráfico de forma automática sin revisión humana;
  - aplicarlo a redes o escenarios de ataque distintos de los de entrenamiento sin una validación nueva;
  - usarlo como defensa ante un atacante que controle los contadores del flujo.

## Factores

- **Escenario de simulación.** Es el factor que más pesa. El 70,5 % de las filas de prueba comparte escenario con el entrenamiento, y en un escenario de Sybil no visto el acierto es del 9,4 %.
- **Clase.** El F1 de prueba por clase, con el ensamble de las semillas 6001 a 6020, es de 0,970 en Normal, 0,855 en Sybil, 0,834 en Blackhole, 0,820 en Wormhole y 0,796 en Flooding.
- **Calidad de la entrada.** Ruido, atributos faltantes o valores atípicos degradan el desempeño (ver "Análisis cuantitativo").

## Métricas

- **Métrica primaria:** F1 macro, con un margen de relevancia δ = 0,01.
- **Métricas secundarias:** exactitud balanceada, ECE antes y después del *temperature scaling*, cobertura y tamaño de los conjuntos conformes (LAC, 90 %), latencia en ONNX Runtime con un hilo y degradación ante perturbaciones.
- **Unidad de independencia:** la semilla para las pruebas entre modelos y el bloque de 1000 flujos para los intervalos sobre la prueba.

## Datos de evaluación

- **Problema principal:** la prueba sellada de UAVIDS-2025 (22 883 flujos), con SHA-256 `2a2dab99…32b4`, verificado en cada carga.
- **Conjuntos adicionales** (`DATA_STATEMENT.md`): UAV-NIDD (5 804 flujos de prueba), UNSW-NB15 (6 000) y UNR-IDD (5 616).

## Datos de entrenamiento

El entrenamiento de UAVIDS-2025 (73 267 flujos) de la partición sellada del E1, en bloques de 1000 flujos. El preprocesamiento es log(1 + x), con recorte de PacketDropRate en 1, y estandarización con los estadísticos del entrenamiento. El esquema y los rangos están en `metadata.json`.

## Análisis cuantitativo

| Medida | Original | Referencia |
|---|---|---|
| F1 macro de prueba (48 semillas) | 0,850 ± 0,012 | LightGBM: 0,875 ± 0,001; CNN-LSTM: 0,859 ± 0,019 |
| Rango medio de Friedman (13 modelos) | 8,6 | LightGBM: 1,65 |
| F1 macro en validación cruzada 5 × 2 | 0,860 | LightGBM: 0,860; CNN 1D: 0,860 |
| UAV-NIDD / UNSW-NB15 / UNR-IDD | 0,550 / 0,665 / 0,765 (puestos 11, 6 y 3 de 13) | El mejor de cada conjunto: DBN 0,617 / LightGBM 0,709 / LightGBM 0,829 |
| Caída del F1 con ruido gaussiano σ = 0,5 | 0,182 (la menor de los 13) | LightGBM: 0,324 |
| Caída del F1 con el 50 % de atributos faltantes | 0,330 (la menor de los 13) | LightGBM: 0,408 |
| ECE antes y después de la temperatura (semillas 6001 a 6020) | 0,048 y 0,053 | Umbral exploratorio del E2: 0,05 |
| Cobertura conforme en prueba (nominal del 90 %) | 0,846 | De 0,817 a 0,890 en los 16 modelos |
| Latencia mediana y p95 por flujo (ONNX, 1 hilo) | 0,185 ms y 0,313 ms | LightGBM: 0,109 ms; BiLSTM: 1,315 ms |
| Permutación de etiquetas (200) | *p* = 0,005 | — |

**Veredicto (sección 8.4 del enunciado):** el modelo no supera de forma significativa a la familia de 12 rivales (R3 no se cumple). Es más robusto que todos ante ruido y atributos faltantes, pero no está en la frontera de Pareto entre F1 y latencia.

## Consideraciones éticas

- **Datos:** son simulados y no contienen datos personales; las IP pertenecen a una red simulada.
- **Errores con consecuencias:** un falso negativo en Blackhole o Wormhole deja un ataque sin detectar, y un falso positivo puede llevar a aislar un dron legítimo. El modelo debe usarse con un humano en el circuito.
- **Exceso de confianza:** los conjuntos conformes cubren menos que lo declarado, por la dependencia de escenario, y el dashboard muestra la cobertura medida.

## Advertencias y recomendaciones

- No desplegar sin validar por escenarios en la red de destino. El detector de entradas fuera de dominio del dashboard (Mahalanobis) avisa cuando un flujo se aleja del entrenamiento, pero no es infalible.
- **La cota de Lipschitz no explica la robustez observada:** al bajarla de 39 a 21 la degradación ante ruido no cambió (P4 falsa).
- Las desviaciones del protocolo, las amenazas a la validez y los riesgos están en los capítulos 7 y 10 del libro.
