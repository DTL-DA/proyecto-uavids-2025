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
| Hardware | Intel Core i5-1235U (10 núcleos, 12 hilos), 15,7 GB de RAM, Windows 11; 2 hilos por proceso y hasta 6 procesos en paralelo. Además, una GPU NVIDIA RTX PRO 6000 Blackwell (porción MIG de 24 GB, 8 vCPU AMD EPYC 9554, RunPod) para 31 corridas | Registro de cada corrida (`equipo`, `hilos`, `dispositivo`, `gpu`) |
| Semillas | Definitivas: 6001 a 6048 (original y CNN-LSTM), 6001 a 6038 (TabNet) y 6001 a 6020 (demás); ajuste: 4242; piloto: 6001 a 6005; conjuntos adicionales: 6001 a 6010; partición de los conjuntos adicionales: 2027; sensibilidad: 6001 a 6003; curvas de aprendizaje: 7101 a 7103; permutación de etiquetas: 9000 a 9199 | `protocolo.md`, sección 5, y `training/ejecutar_robustez.py` |
| Determinismo | `random`, `numpy` y `torch` fijados en cada corrida, en CPU. Los reentrenamientos de control (3 corridas del problema principal y 30 de los conjuntos adicionales) reproducen el F1 bit a bit. Las 31 corridas en GPU (BiLSTM y GRU de los conjuntos adicionales, opción prevista en el protocolo) usaron el determinismo estricto de CUDA; con la misma semilla, el cambio de hardware movió el F1 unas dos desviaciones estándar entre semillas (capítulo 8) | `codigo/motor.fijar_semilla`, `training/ejecutar_gpu.py` |
| Rangos de hiperparámetros | lr ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³}, weight_decay ∈ {0, 10⁻⁵, 10⁻⁴}, lote ∈ {128, 256, 512}: las mismas 12 configuraciones para todos los modelos profundos | `configuracion/experimento.json` del E2 |
| Número de ejecuciones y dispersión | De 20 a 48 semillas por modelo en el problema principal y 10 en cada conjunto adicional; media ± DE e IC 95 % | Capítulos 3, 7 y 8 |
| Datos | UAVIDS-2025 (Zenodo, CC BY 4.0) verificado con SHA-256; tres conjuntos adicionales con procedencia, licencia y SHA-256 | `data/procedencia_conjuntos_adicionales.md` |
| Código | Repositorio de GitHub `DTL-DA/proyecto-uavids-2025`, rama `entregable-3`, etiquetas `e3-estandar`, `e3-protocolo` y `entregable-3`; archivo en Zenodo con el DOI 10.5281/zenodo.24486475. Licencia MIT, confirmada por el autor el 23/11/2026; los datos conservan sus licencias (`DATA_STATEMENT.md`) | Raíz del repositorio, `LICENSE` |
| Contenedor | `Dockerfile` (python:3.13-slim con `requirements-train.txt`) | `Entregable_3_UAVIDS_2025/Dockerfile` |
| Integración continua | GitHub Actions ejecuta `pytest` en cada *push* | `.github/workflows/entregable3-pruebas.yml` |

```{code-cell} ipython3
:tags: [hide-input]
corridas = []
for carpeta in ["ajuste", "piloto", "definitiva", "ablacion", "adicionales", "sensibilidad", "curvas", "cv5x2", "bootstrap", "permutacion", "diagnostico_adicionales"]:
    for p in (RAIZ / "experiments/resultados/corridas" / carpeta).glob("*.json"):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("pendiente_gpu"):
            continue
        corridas.append({"Fase": carpeta, "Modelo": r.get("modelo"), "Segundos": r.get("segundos") or 0, "GPU": r.get("dispositivo", "cpu") != "cpu"})
c = pd.DataFrame(corridas)
tiempo = c.groupby("Fase").agg(Corridas=("Segundos", "size"), En_GPU=("GPU", "sum"), Horas_de_proceso=("Segundos", lambda s: s.sum() / 3600))
tiempo.loc["Total"] = tiempo.sum()
print("Tiempo de cómputo del Entregable 3 (horas de proceso, 2 hilos cada uno; las fases corren en paralelo con hasta 6 procesos):")
tiempo.round(2)
```

**Consumo energético estimado.** Las corridas del Entregable 3 ocuparon unas 9,5 horas de reloj en la CPU (los registros de `experiments/resultados/registros/` dan 303 minutos para los conjuntos adicionales, 37 para las definitivas, 17 para la ablación, 30 para la robustez y cerca de una hora para los diagnósticos, más el diseño y el ajuste). Con el TDP de 15 W del procesador, eso es como mucho 0,15 kWh. La GPU estuvo encendida unos 25 minutos; aun suponiendo 150 W para la porción MIG, suma menos de 0,07 kWh. El total es del orden de 0,2 kWh y, con una intensidad de la red de 0,1 a 0,4 kg de CO₂ equivalente por kWh, las emisiones quedan entre 20 y 90 g de CO₂ equivalente.

## Archivo versionado con identificador persistente

La *release* `entregable-3` del repositorio de GitHub, publicada el 23/11/2026, está archivada en Zenodo con el DOI [10.5281/zenodo.24486475](https://doi.org/10.5281/zenodo.24486475), versión 1.0.0 {cite:p}`torres2026uavidse3`. El DOI [10.5281/zenodo.24486474](https://doi.org/10.5281/zenodo.24486474) reúne todas las versiones y apunta siempre a la más reciente. La *release* trae además el paquete de reproducción `paquete_reproduccion_e3.zip` con su SHA-256 (sección 12, entregable 11).

## Reproducción independiente

La reproducción por una persona ajena al código, que piden las secciones 11 y 14.5.7 del enunciado, no se hizo, y se declara como limitación (capítulo 10). El procedimiento queda listo para que lo repita cualquier persona, incluidos los profesores al calificar, con los pasos de la sección "Cómo reproducirlo" del `README.md` del Entregable 3:

1. construir el contenedor con el `Dockerfile`;
2. ejecutar `python verificacion_externa/reproducir.py`, que verifica el SHA-256 de los datos y de la prueba sellada, vuelve a correr el análisis confirmatorio y compara celda por celda la tabla regenerada con la publicada (capítulo 7). También reentrena desde cero tres corridas definitivas (Original, LightGBM y CNN 1D con la semilla 6001) y compara su F1 y sus probabilidades;
3. correr las pruebas unitarias, compilar este libro desde cero y hacer la prueba de humo del dashboard.

El script necesita las probabilidades de prueba de las corridas definitivas, que no se versionan por su tamaño. Vienen, junto con el repositorio y el CSV de UAVIDS-2025, en `paquete_reproduccion_e3.zip`, que se publica con la *release* `entregable-3`.

La corrida del autor con ese script (22/11/2026, Python 3.13.14 y PyTorch 2.6.0 en CPU, 10,4 minutos) encontró los datos íntegros, una discrepancia máxima de 0 en las 465 celdas de la tabla principal, ninguna decisión distinta y el mismo F1, bit a bit, en las tres corridas reentrenadas. Así mismo, el contenedor se construyó en frío desde el paquete de reproducción el 23/11/2026, y en él pasaron las 36 pruebas unitarias y arrancó el dashboard. Esto muestra que el procedimiento funciona de principio a fin, pero no reemplaza la verificación de un tercero.

## Declaración de autoría, contribuciones y uso de herramientas

**Autoría y contribuciones.** Donny Torres es el autor único del proyecto: definió el problema, aprobó cada decisión del protocolo (conjuntos adicionales, mecanismo, familia de contrastes y fechas), revisó los resultados y responde por ellos en la sustentación.

**Uso de herramientas de IA.** Se usó Claude (Anthropic), un asistente de programación, como apoyo para:

- escribir y depurar el código del modelo, las pruebas, el análisis y el dashboard;
- ejecutar los experimentos según el protocolo preregistrado;
- buscar en OpenAlex, arXiv e IEEE Xplore, y hacer la búsqueda por citación y en repositorios de código;
- redactar borradores de este libro.

**Otras herramientas y servicios.** RunPod (GPU en la nube) para 31 corridas de BiLSTM y GRU de los conjuntos adicionales; GitHub Actions para la integración continua; GitHub Pages para publicar este libro y Render para el dashboard.

Todas las decisiones metodológicas quedaron registradas con fecha en `design_log.md`, `protocolo.md` y el historial de Git. Las cifras del libro se regeneran desde los resultados al compilarlo. El autor revisó el código y el texto, y puede explicar y modificar en vivo cualquier parte del algoritmo, como exige la sección 6.5 del enunciado.

Firma: **Donny Torres**, autor, aprobada el 23/11/2026.

## Bibliografía

```{bibliography}
```
