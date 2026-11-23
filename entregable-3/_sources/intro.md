# Introducción

**Entregable 3 · Proyecto Final de Deep Learning** · Donny Torres · Maestría en Analítica de Datos, Universidad del Norte · Noviembre de 2026

## Resumen ejecutivo

Este entregable propone, implementa y valida un modelo original para detectar intrusiones en redes de drones con UAVIDS-2025: una compuerta que, flujo por flujo, combina un experto de escalones suaves con umbrales aprendibles y un experto neuronal suave, con entrenamiento cooperativo y una cota de Lipschitz demostrada. Se comparó con los 10 mejores modelos del Entregable 2, LightGBM y CNN 1D, con un protocolo preregistrado y el mismo presupuesto de ajuste, en la prueba sellada y en tres conjuntos adicionales.

Con 48 semillas, el original obtiene un F1 macro de 0,850 ± 0,012, frente a 0,875 de LightGBM y 0,859 de CNN-LSTM. Su media es menor que la de 8 de los 12 rivales y ningún contraste rechaza con tres pruebas, de modo que no cumple la condición de superioridad R3 y el tope de calificación es 4,0. Tampoco es el mejor en ningún conjunto adicional. Sí es el más robusto ante ruido (caída de 0,18 frente a 0,32 de LightGBM con σ = 0,5) y ante atributos faltantes.

El hallazgo principal es otro: en UAVIDS-2025 el escenario de simulación pesa más que la arquitectura. El 70,5 % de la prueba comparte escenario con el entrenamiento, ningún modelo reconoce un escenario no visto (acierto menor del 10 %) y la ventaja de LightGBM desaparece en la validación cruzada 5 × 2. Los 16 modelos se publican en un dashboard con ONNX y en este libro reproducible.

## Problema

Las redes de vehículos aéreos no tripulados (UAV) se comunican por enlaces inalámbricos ad hoc, sin infraestructura fija, y por eso son vulnerables a ataques de enrutamiento. UAVIDS-2025 registra flujos de red de una simulación con NS-3 en cinco clases: tráfico normal y los ataques Blackhole, Flooding, Sybil y Wormhole. La tarea es clasificar cada flujo a partir de 16 estadísticos de tráfico, con el F1 macro como métrica primaria, porque un ataque poco frecuente importa tanto como uno frecuente.

## Motivación

El Entregable 2 mostró que LightGBM, un modelo de árboles, supera a las 13 redes profundas evaluadas, y midió la confusión entre Blackhole y Wormhole que el paper guía señala como la principal del conjunto: LightGBM confunde esas dos clases en el 10,0 % de sus flujos y las redes del Top 10 entre el 12,3 y el 18,5 %. Los árboles parten cada atributo por umbrales, y ese sesgo parece útil para separar esas dos clases; las redes, en cambio, reconocen mejor Sybil. La pregunta de este entregable es si un mecanismo que combine los dos sesgos, flujo por flujo, puede superar a ambos.

## Contribuciones

1. **Un modelo original** (R1 y R2): una mezcla de dos expertos con compuerta por flujo, programada desde su formulación, con 14 pruebas unitarias y exportada a ONNX (capítulos 5 y 6).
2. **Una propiedad demostrada**: la cota de Lipschitz de la mezcla, con verificación numérica y su contraste empírico, que resultó holgado (P4 falsa, capítulos 5 y 8).
3. **Un *benchmark* preregistrado** de 13 modelos en el problema principal y en tres conjuntos adicionales, con potencia por contraste, Holm por familia, seis pruebas admisibles y la tabla maestra de corridas (capítulos 3, 7 y 8).
4. **La evidencia de que el escenario pesa más que la arquitectura** en UAVIDS-2025, con la auditoría de fuga, la prueba por escenario no visto y la validación cruzada 5 × 2 (capítulos 1, 7 y 8).
5. **Un dashboard desplegado** con los 16 modelos, incertidumbre calibrada, predicción conforme, detección de entradas fuera de dominio y explicaciones locales (capítulo 9).

## Objetivos

**General.** Diseñar, implementar y validar con rigor estadístico un modelo original de aprendizaje profundo para detectar intrusiones en redes de UAV, y determinar si supera al Top 10 y a los *baselines* fuertes del Entregable 2.

**Específicos.**

1. Corregir y auditar el Entregable 2, incluida la fuga de información entre particiones (capítulo 1).
2. Documentar la novedad del modelo frente a la literatura, con verificación por un tercero (capítulo 4).
3. Demostrar una propiedad formal del modelo y verificarla numéricamente (capítulo 5).
4. Comparar el modelo con la familia completa de contrastes bajo un protocolo preregistrado y en tres conjuntos adicionales (capítulos 7 y 8).
5. Desplegar los modelos en un dashboard reproducible y documentado (capítulos 9 y 11).

## Mapa del libro

| Capítulo | Contenido |
|---|---|
| 1. Trazabilidad de correcciones | Autoauditoría del E2, fuga de información, prueba de D3 y *ranking* antes y después |
| 2. Datos | EDA, auditoría, ficha de los datos y conjuntos adicionales |
| 3. Benchmarks | Top 10 y *baselines* fuertes, tabla maestra, exportación y prueba de ida y vuelta |
| 4. Literatura y novedad | Cadenas de búsqueda, matriz de novedad, declaración y verificación por un tercero |
| 5. Modelo original: diseño | Limitación, hipótesis, formulación, pseudocódigo, demostración y bitácora |
| 6. Modelo original: implementación | Código, pruebas, complejidad, ablación e interpretabilidad |
| 7. Validación | Protocolo, potencia, contrastes, Friedman, pruebas admisibles y desviaciones |
| 8. Evaluación ampliada | Conjuntos adicionales, robustez, P4, adecuación, calibración, eficiencia y balance de R3 |
| 9. Dashboard | Arquitectura, despliegue, ficha del modelo y prueba de humo |
| 10. Conclusiones | Hallazgos, veredicto sobre R1 a R3, amenazas, limitaciones, riesgos y trabajo futuro |
| 11. Reproducibilidad | Entorno, semillas, lista de verificación, reproducción independiente, autoría y bibliografía |

## Dashboard

El dashboard se describe en el capítulo 9, donde está su URL pública.
