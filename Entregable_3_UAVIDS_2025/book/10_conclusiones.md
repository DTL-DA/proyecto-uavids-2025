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

# Conclusiones, limitaciones y trabajo futuro

```{code-cell} ipython3
:tags: [hide-input]
import json
from pathlib import Path
import pandas as pd
RAIZ = Path("..").resolve()
R = RAIZ / "experiments/resultados"
conf = json.loads((R / "confirmatorio/resumen.json").read_text(encoding="utf-8"))
c4 = json.loads((R / "confirmatorio/sensibilidad_cuadro4.json").read_text(encoding="utf-8"))
ad = json.loads((R / "adicionales/resumen.json").read_text(encoding="utf-8"))
adec = json.loads((R / "adecuacion/resumen.json").read_text(encoding="utf-8"))
```

## Hallazgos

1. **El modelo original no supera a la familia de contrastes.** Con 48 semillas obtiene un F1 macro de 0,850 ± 0,012 en la prueba sellada, menor que el de 8 de los 12 rivales, y ningún contraste rechaza con tres pruebas, ni con las preregistradas ni con las seis admisibles del cuadro 4 (capítulo 7). En los conjuntos adicionales no es el mejor en ninguno y en UNSW-NB15 pierde con significancia frente a tres rivales (capítulo 8).
2. **LightGBM es el modelo más fuerte en la prueba sellada, pero su ventaja no es robusta a la partición.** Gana en UAVIDS-2025 (0,875), en UNSW-NB15 y en UNR-IDD, y tiene el mejor rango medio entre conjuntos. Sin embargo, en la validación cruzada 5 × 2 su F1 medio (0,860) es igual al del original y al de CNN 1D (*p* de Nadeau-Bengio de 0,99 y 0,97).
3. **El escenario de simulación pesa más que la arquitectura.** El 70,5 % de las filas de prueba pertenece a escenarios vistos en el entrenamiento (dependencia L3.2, capítulo 1), ningún modelo reconoce el escenario de Sybil aislado (acierto entre 4,8 % y 9,4 %, P5 cierta), la predicción conforme pierde su cobertura nominal en todos los modelos (de 0,82 a 0,89 frente al 0,90 declarado) y la ventaja de LightGBM desaparece al cambiar la partición.
4. **El mecanismo no funciona como se predijo.** La compuerta sí especializa a los expertos (P2: el 88 % del peso de Wormhole y el 69 % de Blackhole van al experto de umbrales), pero ese experto es el que peor separa esas dos clases: quitar el codificador de escalones mejora el F1 de 0,848 a 0,862 (P1 falsa) y la desventaja frente a CNN-LSTM se concentra justo en Blackhole y Wormhole (P3 falsa).
5. **El original es el modelo más robusto ante ruido y atributos faltantes**, con una caída de 0,182 frente a 0,324 de LightGBM con ruido σ = 0,5 y de 0,330 frente a 0,408 con la mitad de los atributos imputados. **Pero la cota de Lipschitz demostrada no explica esa robustez:** al bajar K de 39 a 21 la degradación no cambia (P4 falsa).
6. **El original es adecuado en tres de las cuatro condiciones del E2:** supera al trivial, no sobreajusta (brecha de 0,004) y su prueba es consistente con la validación; falla la calibración por 0,003 (ECE con temperatura de 0,053 frente a 0,05), como 9 de los 15 modelos del E2. En los conjuntos adicionales es adecuado en UNSW-NB15 y UNR-IDD, y no en UAV-NIDD, donde memoriza las ejecuciones de ataque del entrenamiento (F1 de 0,89 en entrenamiento frente a 0,56 en validación). Los 30 reentrenamientos de control reproducen bit a bit el F1 registrado.

## Veredicto sobre R1, R2 y R3

| Condición | Veredicto | Evidencia |
|---|---|---|
| **R1. Novedad para el problema** | Se cumple, sujeta a la verificación del tercero | La búsqueda en OpenAlex y arXiv (cadenas N1 a N7, capítulo 4) no encontró estudios que apliquen a intrusiones en redes de UAV una compuerta por flujo entre un experto de escalones suaves y uno neuronal, entrenada de forma cooperativa. La diferencia con los trabajos más cercanos (mezcla de expertos homogéneos en 5G y *stacking* sobre UAVIDS-2025) es de mecanismo y de entrenamiento, no solo de aplicación. |
| **R2. Mecanismo propio, sin `import` ni código de terceros** | Se cumple | El mecanismo está programado desde su formulación en `src/original_model.py`, con PyTorch solo como motor tensorial, y lo cubren 14 pruebas unitarias (capítulo 6). La propiedad (cota de Lipschitz de la mezcla) está demostrada y verificada numéricamente (capítulo 5). |
| **R3. Superioridad estadística sobre la familia** | **No se cumple** | Fallan las condiciones 1 a 5 de la sección 8.4 del enunciado y la condición 6 confirma que la conclusión no depende de la prueba (capítulo 8). |

**Tope de calificación: 4,0** (sección 14.1 del enunciado: R1 sí, R2 sí, R3 no).

## Hallazgo alternativo

Como se fijó en la sección 12 del protocolo antes de las corridas definitivas, el trabajo se sostiene en tres aportes que no dependen de que el original gane:

1. **Un *benchmark* riguroso y preregistrado de 13 modelos** en el problema principal y en tres conjuntos adicionales, con el mismo presupuesto de ajuste, potencia por contraste, Holm por familia, seis pruebas admisibles y una tabla maestra con una fila por corrida (capítulo 11). Los reentrenamientos de control dieron el mismo F1, bit a bit, que las corridas registradas.
2. **La explicación de por qué el escenario pesa más que la arquitectura en UAVIDS-2025**, con tres evidencias independientes: la auditoría de fuga L3.2, la prueba de D3 por escenario no visto (ningún mecanismo de invariancia lo resolvió) y la desaparición de la ventaja de LightGBM en 5 × 2. La consecuencia práctica es que los *rankings* publicados sobre una sola partición de este conjunto, incluido el del E2, miden en buena parte qué escenarios cayeron en la prueba.
3. **La medición de cuánto cierra la brecha entre árboles y redes un experto con escalones suaves y cuándo ayuda la compuerta.** El oráculo que elige flujo por flujo entre un árbol y una red llega a 0,871 en desarrollo, frente a 0,845 de LightGBM, de modo que la selección por flujo tiene margen real; pero la compuerta aprendida no lo alcanza y el codificador de escalones, tal como se diseñó, resta. Es un resultado negativo bien medido, que acota el espacio de diseño para el siguiente intento.

## Amenazas a la validez

- **Validez interna.** La prueba sellada comparte escenarios de simulación con el entrenamiento, de modo que las diferencias entre modelos combinan capacidad y composición de escenarios; las pruebas sobre flujos (McNemar, DeLong) los tratan como independientes y son anticonservadoras. El diseño del original se hizo en una partición de desarrollo tomada del entrenamiento del E2, sin ver la prueba, pero dentro de los mismos escenarios.
- **Validez externa.** UAVIDS-2025 es una simulación (NS-3.24 con movilidad BOID), no tráfico real de drones. Los conjuntos adicionales cubren un banco real de UAV, una red empresarial y estadísticos de puertos SDN, pero con los hiperparámetros del problema principal y sin reajuste, lo que es equitativo y a la vez puede penalizar a los modelos más sensibles al ajuste.
- **Validez de constructo.** El F1 macro se calcula sobre las clases presentes en cada prueba, y el umbral de calibración (ECE < 0,05) es exploratorio desde el E2.
- **Validez de la conclusión.** La potencia conjunta de la familia es de apenas 0,115 y solo 6 de los 12 contrastes alcanzan potencia de 0,90 para δ. Una conclusión de "no supera" con baja potencia sería débil si el original estuviera cerca de ganar, pero su F1 medio es menor que el de 8 rivales y la condición 1 falla sin necesidad de pruebas.

## Limitaciones

- **Desviaciones declaradas** (capítulo 7): las pruebas T1 y T3 no coincidían al pie de la letra con la fila tabular del cuadro 4, lo que se corrigió con un análisis de sensibilidad posterior; 31 corridas de BiLSTM y GRU de los conjuntos adicionales se hicieron en GPU, y con la misma semilla el hardware movió el F1 unas dos desviaciones estándar entre semillas.
- **Tope de épocas.** En UNSW-NB15, varias redes llegaron a las 30 épocas todavía mejorando (8 de 10 corridas de CNN-LSTM, GRU y DBN).
- **Diseño fuera del óptimo de validación.** El barrido de sensibilidad posterior encontró dos variantes mejores en validación (T = 4 y τ_min = 0,2); no se adoptaron porque el diseño estaba congelado.
- **La cota de Lipschitz es holgada**: es un peor caso global que no ordena las variantes por su robustez real.
- **La verificación por un tercero** de la novedad y de la reproducción se hace en los días previos a la entrega (capítulos 4 y 11).

## Riesgos de uso

- **Falsos negativos en Blackhole y Wormhole.** Son las clases con menor F1 del original (0,834 y 0,820): un ataque de agujero negro o de gusano puede pasar como otra clase.
- **Fallos silenciosos ante escenarios nuevos.** Ante un escenario de red no visto, el acierto cae por debajo del 10 % sin que el modelo lo advierta; el dashboard mitiga este riesgo con un detector de entradas fuera de dominio (distancia de Mahalanobis), que no es infalible.
- **Exceso de confianza.** Los conjuntos conformes cubren menos que lo declarado, y la pendiente de calibración (0,73) indica probabilidades algo extremas; el dashboard reporta la cobertura medida, no la nominal.
- **Manipulación adversaria.** Un atacante que controle los contadores del flujo puede desplazar la entrada; la robustez ante ruido aleatorio que se midió no es robustez ante perturbaciones dirigidas.

Por estas razones, el modelo es una herramienta de apoyo para un analista de seguridad y no debe bloquear tráfico de forma automática.

## Trabajo futuro

1. **Un diseño con menos escalones y más suaves** (T = 4, τ_min ≥ 0,2), sugerido por la sensibilidad y la ablación, evaluado con un protocolo nuevo y preregistrado.
2. **Un protocolo de evaluación por escenarios** (dejar escenarios completos fuera) como partición principal de UAVIDS-2025, que es la pregunta relevante para un despliegue.
3. **Atributos que separen Blackhole de Wormhole**, el núcleo del 7,7 % de flujos que fallan los 13 modelos.
4. **Una cota de robustez local o certificada** (por ejemplo, por propagación de intervalos), más fina que la global, para contrastar de nuevo P4.
5. **Predicción conforme bajo cambio de covariables**, ponderada o estratificada por escenario, para recuperar la cobertura declarada.

## Declaración de uso de herramientas de IA

Se usó Claude (Anthropic), un asistente de programación, como apoyo en la escritura y depuración del código, la ejecución de los experimentos según el protocolo preregistrado, las búsquedas en OpenAlex y arXiv y la redacción de borradores de este libro. Todas las decisiones metodológicas quedaron registradas con fecha en la bitácora de diseño, el protocolo y el historial de Git, y el autor revisó el código y el texto. La declaración completa de autoría, contribuciones y uso de herramientas, firmada, está en el capítulo 11, que es el último del libro, como pide la sección 12 del enunciado.
