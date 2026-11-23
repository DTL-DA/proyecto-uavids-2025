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

# Datos: EDA, auditoría y ficha de datos

```{code-cell} ipython3
:tags: [hide-input]
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RAIZ = Path("..").resolve()
df = pd.read_parquet(RAIZ / "data/uavids.parquet")
CLASES = ["Blackhole", "Flooding", "Normal", "Sybil", "Wormhole"]
COLOR = dict(zip(CLASES, ["#2A78D6", "#EDA100", "#8A94A6", "#1BAF7A", "#EB6834"]))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
```

## Ficha de datos

| Campo | Contenido |
|---|---|
| Fuente | UAVIDS-2025 {cite:p}`zeng2025uavids,zeng2025uavidsx`, Zenodo, DOI 10.5281/zenodo.15336998 |
| Licencia | CC BY 4.0 |
| Periodo y generación | Simulación de enjambres de UAV en NS-3.24 con movilidad BOID, publicada en 2025 |
| Observaciones | 122 171 flujos; 8 219 duplicados exactos excluidos antes de partir |
| Variables | 16 estadísticos por flujo; FlowID, IP, puertos y protocolo se excluyen por ser identificadores |
| Objetivo y tarea | Clase del flujo (Blackhole, Flooding, Normal, Sybil y Wormhole): clasificación multiclase |
| Métrica primaria | F1 macro, con margen de relevancia δ = 0,01 heredado del E1 |
| Partición | Bloques de 1000 flujos: 73 267 de entrenamiento, 17 802 de validación y 22 883 de prueba, sellada en el E1 con SHA-256 |
| Sesgos conocidos | Escenarios de simulación largos y de una sola clase; dependencia L3.2 entre particiones (capítulo 1) |
| Usos inadecuados | Desplegar el modelo en una red real sin validarlo con tráfico real; tomar decisiones automáticas de aislamiento de nodos |

```{code-cell} ipython3
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(8, 3.6))
t = df[df["particion"] != "excluida (duplicada)"].groupby(["particion", "clase"]).size().unstack().reindex(columns=CLASES)
t = t.loc[["entrenamiento", "validación", "prueba"]]
t.plot(kind="bar", ax=ax, color=[COLOR[c] for c in CLASES], width=0.8)
ax.set_xlabel("Partición"); ax.set_ylabel("Flujos"); ax.set_title("Flujos por clase y partición"); ax.tick_params(axis="x", rotation=0)
ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper right")
plt.tight_layout(); plt.show()
t
```

**Las cinco clases tienen entre 7 569 (Flooding) y 17 342 (Blackhole) flujos en entrenamiento: un desbalance moderado, con Flooding como clase minoritaria en las tres particiones.**

Esto quiere decir que la exactitud global sobrestimaría el desempeño en Flooding, que es menos frecuente, mientras que el F1 macro da el mismo peso a cada clase.

Por tal motivo, se mantiene el F1 macro como métrica primaria y la pérdida ponderada por clase ($w_c = n / (K n_c)$) para todos los modelos, como en el E2.

## Estructura por escenario: el panel específico del problema

```{code-cell} ipython3
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(9, 3.4))
m = df.iloc[::20]
for i, c in enumerate(CLASES):
    s = m[m["clase"] == c]
    ax.scatter(s["flow_id"], np.full(len(s), i), s=2, color=COLOR[c], alpha=0.6)
for a, b in ((94285, 94979), (108071, 109045)):
    ax.axvspan(a, b, color="#EB6834", alpha=0.18)
ax.set_yticks(range(5), CLASES); ax.set_xlabel("FlowID (orden de la simulación)"); ax.set_title("Clase de cada flujo a lo largo de la simulación (sombreado: escenarios de Sybil aislados)")
plt.tight_layout(); plt.show()
```

**La clase de cada flujo no se reparte al azar: forma tramos largos y contiguos, que son los escenarios de simulación (por ejemplo, Sybil de FlowID 1 a 19 000 y Wormhole de 36 001 a 49 000), y Blackhole y Normal aparecen intercaladas en una región de unos 36 000 flujos.**

En ese sentido, un bloque de 1000 flujos casi siempre contiene una sola clase y pertenece a un escenario que tiene otros bloques en entrenamiento. Por eso la partición por bloques evita mezclar vecinos inmediatos pero no aísla los escenarios, que es la dependencia L3.2 del capítulo 1.

De ahí que se reporte el desempeño con y sin los escenarios aislados, y que la generalización a un escenario nuevo se trate como un estudio de cambio de distribución y no como el resultado principal.

## Distribución de los atributos

```{code-cell} ipython3
:tags: [hide-input]
atributos = ["PacketDropRate", "MeanDelay/s", "Throughput/Kbps", "TxPacketRate/s"]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
ent = df[df["particion"] == "entrenamiento"]
for ax, a in zip(axes, atributos):
    datos = [np.log1p(np.maximum(ent.loc[ent["clase"] == c, a], 0)) for c in CLASES]
    partes = ax.violinplot(datos, showmedians=True, showextrema=False)
    for cuerpo, c in zip(partes["bodies"], CLASES):
        cuerpo.set_facecolor(COLOR[c]); cuerpo.set_alpha(0.6)
    ax.set_xticks(range(1, 6), [c[:4] for c in CLASES], fontsize=8); ax.set_title(a, fontsize=10); ax.set_ylabel("log(1 + x)")
plt.suptitle("Distribución de cuatro atributos por clase (entrenamiento)", fontsize=11); plt.tight_layout(); plt.show()
```

**Flooding y Sybil se separan del resto por sus tasas: la mediana de TxPacketRate/s es 3,35 en Flooding y 1,43 en Sybil, frente a menos de 0,2 en las otras tres clases. La tasa de pérdida separa a Blackhole (mediana 0,69) y Wormhole (0,55) de Normal, Sybil y Flooding (por debajo de 0,2), pero no a esas dos clases entre sí: el rango intercuartílico de Wormhole (0,29 a 0,73) contiene casi todo el de Blackhole (0,59 a 0,80), y en MeanDelay/s ocurre lo mismo.**

Lo anterior indica que un umbral sobre un solo atributo distingue los ataques de saturación y de suplantación, pero no los dos ataques de enrutamiento. La frontera entre Blackhole y Wormhole depende de combinaciones de atributos, así que un modelo con umbrales por eje y uno de funciones suaves pueden equivocarse en flujos distintos.

Por ello, se diseña un modelo que combine los dos sesgos y no uno que solo afine un umbral.

## Conjuntos de datos adicionales

Los tres conjuntos adicionales se eligieron el 22/11/2026, antes de entrenar cualquier modelo con ellos, y no se usaron para diseñar. Su procedencia, licencia y SHA-256 están en `data/procedencia_conjuntos_adicionales.md`; la partición es agrupada por bloques de 250 filas contiguas dentro de cada clase, porque las fuentes vienen ordenadas por ejecución de ataque.

```{code-cell} ipython3
:tags: [hide-input]
filas = []
for nombre, cita in (("uav_nidd", "Hadi et al. (2025)"), ("unsw_nb15", "Moustafa y Slay (2015)"), ("unr_idd", "Das et al. (2023)")):
    r = json.loads((RAIZ / f"data/{nombre}_resumen.json").read_text(encoding="utf-8"))
    filas.append({"Conjunto": nombre, "Fuente": cita, "Filas en la fuente": r["filas_fuente"], "Duplicadas eliminadas": r["duplicadas_eliminadas"],
                  "Clases": len(r["clases"]), "Atributos": r["n_atributos"],
                  "Entrenamiento / validación / prueba": " / ".join(str(sum(v)) for v in r["conteo_por_particion"].values()),
                  "Bloques compartidos": r["bloques_compartidos"]})
pd.DataFrame(filas)
```

Los tres conjuntos difieren de UAVIDS-2025 en al menos dos ejes:

- **UAV-NIDD** {cite:p}`hadi2025uavnidd`: proviene de un banco real con drones PX4 y DJI, tiene 43 atributos y otros tipos de ataque.
- **UNSW-NB15** {cite:p}`moustafa2015unsw`: es tráfico de una red empresarial de laboratorio, tiene 2,5 millones de registros en la fuente y un desbalance fuerte.
- **UNR-IDD** {cite:p}`das2023unridd`: usa estadísticos de puertos de una red definida por software en lugar de flujos, es más pequeño y tiene una clase del 2,7 %.

UAV-CAS {cite:p}`mishra2026uavcas`, la primera opción, se descartó porque IEEE DataPort exige una suscripción de pago para descargarlo, lo que impide la reproducción independiente.
