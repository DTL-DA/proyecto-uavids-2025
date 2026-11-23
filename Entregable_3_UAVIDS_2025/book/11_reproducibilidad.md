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

# Reproducibilidad, autoría y bibliografía

```{code-cell} ipython3
:tags: [hide-input]
import json, platform, sys
from pathlib import Path
import pandas as pd
RAIZ = Path("..").resolve()
```

## Lista de verificación de reproducibilidad

| Ítem | Estado | Dónde |
|---|---|---|
| Versiones del software | Python 3.13.14; PyTorch 2.6.0 (CPU); versiones exactas en `requirements.txt` (dashboard) y `requirements-train.txt` (entrenamiento) | Raíz del Entregable 3 |
| Hardware | Intel Core i5-1235U (10 núcleos, 12 hilos), 15,7 GB de RAM, Windows 11, sin GPU; 2 hilos por proceso y hasta 6 procesos en paralelo | Registro de cada corrida (`equipo`, `hilos`) |
| Semillas | Definitivas: 6001 a 6048 (original y CNN-LSTM), 6001 a 6038 (TabNet) y 6001 a 6020 (demás); ajuste: 4242; piloto: 6001 a 6005; conjuntos adicionales: 6001 a 6010; partición de los conjuntos adicionales: 2027 | `protocolo.md`, sección 5 |
| Determinismo | `random`, `numpy` y `torch` fijados en cada corrida; todo en CPU. Ninguna corrida usó GPU | `codigo/motor.fijar_semilla` |
| Rangos de hiperparámetros | lr ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³}, weight_decay ∈ {0, 10⁻⁵, 10⁻⁴}, lote ∈ {128, 256, 512}: las mismas 12 configuraciones para todos los modelos profundos | `configuracion/experimento.json` del E2 |
| Número de ejecuciones y dispersión | De 20 a 48 semillas por modelo en el problema principal y 10 en cada conjunto adicional; media ± DE e IC 95 % | Capítulos 3, 7 y 8 |
| Datos | UAVIDS-2025 (Zenodo, CC BY 4.0) verificado con SHA-256; tres conjuntos adicionales con procedencia, licencia y SHA-256 | `data/procedencia_conjuntos_adicionales.md` |
| Código | Repositorio de GitHub `DTL-DA/proyecto-uavids-2025`, rama `entregable-3`, etiquetas `e3-estandar` y `e3-protocolo` | Raíz del repositorio |
| Contenedor | `Dockerfile` (python:3.13-slim con `requirements-train.txt`) | `Entregable_3_UAVIDS_2025/Dockerfile` |
| Integración continua | GitHub Actions ejecuta `pytest` en cada *push* | `.github/workflows/entregable3-pruebas.yml` |

```{code-cell} ipython3
:tags: [hide-input]
corridas = []
for carpeta in ["definitiva", "ablacion", "adicionales", "ajuste", "piloto"]:
    for p in (RAIZ / "experiments/resultados/corridas" / carpeta).glob("*.json"):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        corridas.append({"Fase": carpeta, "Modelo": r.get("modelo"), "Segundos": r.get("segundos", 0)})
c = pd.DataFrame(corridas)
tiempo = c.groupby("Fase").agg(Corridas=("Segundos", "size"), Horas_CPU=("Segundos", lambda s: s.sum() / 3600))
tiempo.loc["Total"] = tiempo.sum()
print("Tiempo de cómputo del Entregable 3 (horas de proceso, 2 hilos cada uno):")
tiempo.round(2)
```

**Consumo energético estimado.** Con el TDP de 15 W del procesador, una cota superior del consumo es 15 W por las horas de reloj del cómputo, del orden de 0,1 kWh para todo el Entregable 3. Con una intensidad de la red de 0,1 a 0,4 kg de CO₂ equivalente por kWh, las emisiones quedan entre 10 y 50 g de CO₂ equivalente.

## Archivo versionado con identificador persistente

La *release* `entregable-3` del repositorio de GitHub se archiva en Zenodo, que le asigna un DOI. El DOI se cita aquí cuando se publique la *release* (sección 12, entregable 11).

## Reproducción independiente

La reproducción la hace **Imera Logreira**, colega que no participó en el diseño ni en el código, en su propio equipo y sin ayuda técnica del autor. Recibe el paquete `paquete_reproduccion_e3.zip`, que contiene el repositorio, el CSV de UAVIDS-2025 y las probabilidades de prueba de las corridas definitivas, junto con su SHA-256, y sigue la guía `verificacion_externa/GUIA.md`:

1. construye el contenedor con el `Dockerfile`;
2. ejecuta `python verificacion_externa/reproducir.py`, que verifica el SHA-256 de los datos y de la prueba sellada, vuelve a correr el análisis confirmatorio y compara celda por celda la tabla regenerada con la publicada (capítulo 7). También reentrena desde cero tres corridas definitivas (Original, LightGBM y CNN 1D con la semilla 6001) y compara su F1 y sus probabilidades;
3. corre las pruebas unitarias, compila este libro desde cero y hace la prueba de humo del dashboard.

Su reporte firmado (`verificacion_externa/reporte_reproduccion.md`), con la discrepancia máxima observada, y su concepto sobre la novedad (`verificacion_externa/concepto_novedad.md`, capítulo 4) se anexan aquí.

## Declaración de autoría, contribuciones y uso de herramientas

**Autoría y contribuciones.** Donny Torres es el autor único del proyecto: definió el problema, aprobó cada decisión del protocolo (conjuntos adicionales, mecanismo, familia de contrastes y fechas), revisó los resultados y responde por ellos en la sustentación.

**Uso de herramientas de IA.** Se usó Claude (Anthropic), un asistente de programación, como apoyo para:

- escribir y depurar el código del modelo, las pruebas, el análisis y el dashboard;
- ejecutar los experimentos según el protocolo preregistrado;
- buscar en OpenAlex y arXiv;
- redactar borradores de este libro.

Todas las decisiones metodológicas quedaron registradas con fecha en `design_log.md`, `protocolo.md` y el historial de Git. Las cifras del libro se regeneran desde los resultados al compilarlo. El autor revisó el código y el texto, y puede explicar y modificar en vivo cualquier parte del algoritmo, como exige la sección 6.5 del enunciado.

Firma: ______________________ Donny Torres

## Bibliografía

```{bibliography}
```
