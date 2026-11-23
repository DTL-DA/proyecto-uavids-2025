# Declaración de los datos (`DATA_STATEMENT.md`)

## Conjunto principal: UAVIDS-2025

- **Procedencia:** Zeng, Q., Bashir, A. y Naït-Abdesselam, F. (2025). UAVIDS-2025. Zenodo, DOI 10.5281/zenodo.15336998. Artículo: *UAVIDS-2025: A Benchmark Dataset for Intrusion Detection in UAV Networks Using Machine Learning Techniques*, IEEE CNS 2025, DOI 10.1109/cns66487.2025.11194990.
- **Licencia:** CC BY 4.0. El CSV no se versiona: se descarga y se verifica con SHA-256 `d50d339f68be7b23f0bf089dd438b20a1835c13182d8641220538121440164d0`.
- **Contenido:** 122 171 flujos simulados en NS-3.24 con movilidad de enjambre BOID; 16 estadísticos por flujo y la etiqueta (Blackhole, Flooding, Normal, Sybil o Wormhole). FlowID, IP, puertos y protocolo son identificadores y se excluyen como atributos.
- **Preprocesamiento:**
  - Se excluyen 8 219 duplicados exactos.
  - PacketDropRate se recorta en 1 y se aplica log(1 + x).
  - Se estandariza con los estadísticos del entrenamiento.
- **Partición:** bloques de 1000 flujos contiguos, sellada en el E1, con SHA-256 de la prueba `2a2dab99c08ed1f19f24add07f5c6b0375e5ce0df7951a99090f54f4f2cd32b4`.
- **Sesgos y límites conocidos:**
  - Los datos son simulados, no tráfico real.
  - Los escenarios de ataque son tramos largos de una sola clase.
  - El 70,5 % de las filas de prueba pertenecen a un escenario con filas en entrenamiento (dependencia L3.2, capítulo 1 del libro). Ningún modelo generaliza bien a un escenario de simulación no visto.
- **Datos personales:** no contiene datos personales. Las IP son de una red simulada.

## Conjuntos adicionales

Procedencia, licencias, versiones y SHA-256 en `data/procedencia_conjuntos_adicionales.md`. Resumen:

| Conjunto | Fuente | Licencia | Uso en el proyecto |
|---|---|---|---|
| UAV-NIDD (caso 3, flujos de la estación de tierra) | Hadi et al. (2025), Figshare DOI 10.6084/m9.figshare.25486462 y GitHub CyberSaR-KAUST | CC BY 4.0 | 6 clases y 43 atributos tras quitar duplicados y clases con menos de 300 filas |
| UNSW-NB15 (4 CSV completos) | Moustafa y Slay (2015), UNSW Canberra | Uso académico con cita de los cinco artículos de los autores | 9 clases y 38 atributos numéricos. La fuente tiene 2 540 047 registros, 3 más que los 2 540 044 que declara la página oficial |
| UNR-IDD | Das et al. (2023), Universidad de Nevada | Redistribución permitida con cita | 6 clases y 18 atributos de estadísticos de puertos |

UAV-CAS (Mishra et al., 2026) se descartó porque exige una suscripción de pago en IEEE DataPort.

## Usos inadecuados

Desplegar un modelo entrenado con estos datos en una red real de drones sin validarlo con tráfico real; usar sus predicciones para aislar nodos de forma automática; o extrapolar el desempeño de la prueba a escenarios de simulación o configuraciones de red distintas.
