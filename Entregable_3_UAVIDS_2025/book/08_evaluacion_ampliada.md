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

# Evaluación ampliada: conjuntos adicionales, robustez y adecuación

```{code-cell} ipython3
:tags: [hide-input]
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
RAIZ = Path("..").resolve()
R = RAIZ / "experiments/resultados"
A = R / "adicionales"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.precision", 4)
NARANJA, AZUL, GRIS = "#EB6834", "#2A78D6", "#8A94A6"
CONJ = {"uav_nidd": "UAV-NIDD", "unsw_nb15": "UNSW-NB15", "unr_idd": "UNR-IDD"}
tablas = pd.concat([pd.read_csv(A / f"tabla_{c}.csv") for c in CONJ])
contrastes = pd.concat([pd.read_csv(A / f"contrastes_{c}.csv") for c in CONJ])
res_ad = json.loads((A / "resumen.json").read_text(encoding="utf-8"))
adec = json.loads((R / "adecuacion/resumen.json").read_text(encoding="utf-8"))
```

Este capítulo responde a las condiciones 4, 5 y 6 de la definición operativa de superioridad (sección 8.4 del enunciado) y al cuadro 5 de robustez, calibración y eficiencia. Las condiciones 1 a 3 están en el capítulo 7; el balance de las seis condiciones, al final de este capítulo.

## Conjuntos adicionales

Los tres conjuntos se eligieron el 22/11/2026, antes de entrenar cualquier modelo con ellos, y no se usaron para ninguna decisión de diseño (sección 5.3 del protocolo). Los 13 modelos de la familia se entrenaron con sus hiperparámetros y su arquitectura del problema principal, sin búsqueda nueva para ninguno, con 10 semillas (6001 a 6010), el mismo tope de 30 épocas y la misma parada temprana. Son 390 corridas, sin errores.

### Desempeño por conjunto

```{code-cell} ipython3
:tags: [hide-input]
t = tablas.assign(**{"F1 ± DE": tablas["F1 macro"].map("{:.3f}".format) + " ± " + tablas["DE"].map("{:.3f}".format)})
pd.concat({c: t[t["conjunto"] == c].set_index("modelo")[["F1 ± DE", "rango"]] for c in CONJ.values()}, axis=1)
```

```{code-cell} ipython3
:tags: [hide-input]
fig, ejes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=False)
for ax, c in zip(ejes, CONJ.values()):
    s = tablas[tablas["conjunto"] == c].sort_values("F1 macro")
    ax.barh(s["modelo"], s["F1 macro"], xerr=s["DE"], color=[NARANJA if m == "Original" else AZUL for m in s["modelo"]], ecolor=GRIS, height=0.65)
    ax.set_xlim(s["F1 macro"].min() - 0.08, s["F1 macro"].max() + 0.03)
    ax.set_title(c); ax.set_xlabel("F1 macro de prueba (media ± DE, 10 semillas)")
plt.tight_layout(); plt.show()
```

**El modelo original no es el mejor en ningún conjunto adicional: queda en el puesto 11 de 13 en UAV-NIDD (F1 de 0,550 frente a 0,617 de DBN), en el 6 en UNSW-NB15 (0,665 frente a 0,709 de LightGBM) y en el 3 en UNR-IDD (0,765 frente a 0,829 de LightGBM y 0,798 de BiLSTM).**

Así mismo, el modelo que domina no cambia de un conjunto a otro tanto como el original: LightGBM es el mejor en UNSW-NB15 y en UNR-IDD, como en UAVIDS-2025, mientras que en UAV-NIDD queda séptimo y gana DBN, en un conjunto donde todos los modelos están entre 0,49 y 0,62 de F1 macro y las diferencias entre semillas son grandes (TabNet tiene una DE de 0,12).

Por esta razón, la tabla se lee conjunto por conjunto, sin promediar F1 de escalas distintas, como pide la sección 8.3 del enunciado, y la comparación formal se hace con los contrastes pareados de cada conjunto y con la prueba entre conjuntos.

### Contrastes del original en cada conjunto

```{code-cell} ipython3
:tags: [hide-input]
def marca(f):
    if f["peor con significancia (T1 a T3)"]:
        return "peor (T1, T2 y T3)"
    if f["peor con significancia (T1 y T2)"]:
        return "peor (T1 y T2)"
    if f["supera"]:
        return "supera (R3)"
    if f["mejor con significancia (T1 y T2)"]:
        return "mejor (T1 y T2)"
    return "no concluyente"
contrastes["lectura"] = contrastes.apply(marca, axis=1)
contrastes["celda"] = contrastes["diferencia"].map("{:+.3f}".format) + " · " + contrastes["lectura"]
contrastes.pivot(index="rival", columns="conjunto", values="celda")[list(CONJ.values())]
```

```{code-cell} ipython3
:tags: [hide-input]
pd.DataFrame({c: {"bloques de prueba": int(g["bloques de prueba"].iloc[0]), "potencia media para δ": round(g["potencia para delta"].mean(), 2),
                  "diferencia mínima detectable media": round(g["MDE"].mean(), 4)} for c, g in contrastes.groupby("conjunto")}).T
```

**El original es significativamente peor que CNN-LSTM, BiLSTM y LightGBM en UNSW-NB15 con las tres pruebas, y en UNR-IDD supera con significancia a seis rivales (CNN-LSTM, Autoencoder, RNN, GRU, TabNet y DBN) según la *t* y Wilcoxon, a RNN con las tres pruebas, pero pierde frente a BiLSTM y LightGBM según esas dos. En UAV-NIDD ninguna diferencia resiste las tres pruebas; según la *t* y Wilcoxon, RNN y DBN son mejores.**

Hay que tener en cuenta que la potencia cambia mucho entre conjuntos. Con 10 semillas, la potencia media para detectar δ = 0,01 es de 0,21 en UAV-NIDD, de 0,71 en UNSW-NB15 y de 0,37 en UNR-IDD, y la tercera prueba, el *bootstrap* por bloques, solo tiene 24 bloques de prueba en UAV-NIDD y 23 en UNR-IDD, frente a 687 en UNSW-NB15. Por eso las tres pruebas solo coinciden en UNSW-NB15, que es también el único conjunto donde el original pierde con todas.

Por lo anterior, conviene leer UNR-IDD como el caso favorable al original (gana a la mitad de la familia con dos pruebas) y UNSW-NB15 como el desfavorable, sin generalizar ninguno de los dos.

### Prueba entre conjuntos de datos

```{code-cell} ipython3
:tags: [hide-input]
filas = []
for clave, nombre in (("entre conjuntos (3 adicionales)", "3 adicionales"), ("entre conjuntos (3 adicionales + UAVIDS-2025, sensibilidad)", "3 adicionales + UAVIDS-2025")):
    e = res_ad[clave]
    filas.append({"conjuntos": nombre, "χ² de Friedman": e["friedman"]["chi2"], "p Iman-Davenport": e["friedman"]["p Iman-Davenport"],
                  "W de Kendall": e["friedman"]["W de Kendall"], "rango del original": e["rangos medios (1 = mejor)"]["Original"],
                  "mejor modelo (rango)": min(e["rangos medios (1 = mejor)"].items(), key=lambda kv: kv[1])})
display(pd.DataFrame(filas))
pares = pd.DataFrame(res_ad["entre conjuntos (3 adicionales)"]["pares"])[["rival", "mejor en", "p Wilcoxon", "p Wilcoxon Holm", "P(A > B)", "P(ROPE)", "P(A < B)"]]
pares.rename(columns={"mejor en": "conjuntos donde el original es mejor", "P(A > B)": "P(original > rival)", "P(A < B)": "P(original < rival)"})
```

**La prueba de Friedman con la corrección de Iman-Davenport no rechaza la igualdad de los 13 modelos sobre los tres conjuntos adicionales (*p* = 0,18), pero sí al incluir UAVIDS-2025 (*p* = 0,013). En los dos casos LightGBM tiene el mejor rango medio (3,0 y 2,5) y el original queda en la mitad (6,7 y 7,25).**

Esto significa que, con tres o cuatro conjuntos, la prueba global tiene muy poca potencia, como advierte la sección 8.3 del enunciado: con tres conjuntos el *p* más pequeño que puede dar Wilcoxon es 0,25, de modo que ningún contraste por pares entre conjuntos puede rechazar, y la prueba bayesiana, con tan pocas observaciones, reparte la probabilidad entre las tres regiones.

Por tal motivo, la conclusión entre conjuntos se apoya en la consistencia por conjunto, que es la condición 4 de R3.

### Consistencia (condición 4 de R3)

```{code-cell} ipython3
:tags: [hide-input]
pd.DataFrame(res_ad["consistencia por conjunto"]).T[["F1 original", "rango del original (de 13)", "mejor rival", "F1 mejor rival", "original mejor que todos en la media",
                                                     "rivales significativamente mejores (T1 a T3)", "rivales significativamente peores (T1 y T2)"]]
```

**La condición 4 no se cumple: el original no es el mejor en la media en ninguno de los tres conjuntos (se exige en al menos dos), y en UNSW-NB15 es significativamente peor que tres rivales de la familia.**

En ese sentido, los conjuntos adicionales confirman lo que mostró el problema principal: el original queda en el grupo intermedio de las redes, y su ventaja relativa aparece en el conjunto de menor dimensión (UNR-IDD, 18 atributos), donde los escalones suaves sobre pocos ejes compiten mejor con las redes que en UAV-NIDD (43 atributos) o UNSW-NB15 (38).

Así las cosas, R3 falla también por consistencia, independientemente de las pruebas del capítulo 7, y el tope de 4,0 no depende de una sola partición ni de un solo conjunto.

### Dos fuentes de variación que se declaran

```{code-cell} ipython3
:tags: [hide-input]
tope = tablas.pivot(index="modelo", columns="conjunto", values="corridas en el tope de 30 épocas")[list(CONJ.values())]
gpu = tablas.pivot(index="modelo", columns="conjunto", values="corridas en GPU")[list(CONJ.values())]
cpu_gpu = {k: json.loads((R / "corridas" / p).read_text(encoding="utf-8"))["f1_eva"]
           for k, p in (("CPU", "verificacion_dispositivo/unr_idd_GRU_6001_cpu.json"), ("GPU", "adicionales/unr_idd_GRU_6001.json"))}
print("GRU, UNR-IDD, semilla 6001: F1 de prueba en CPU = {CPU:.4f}, en GPU = {GPU:.4f}".format(**cpu_gpu))
pd.concat({"corridas en el tope de 30 épocas (de 10)": tope, "corridas en GPU (de 10)": gpu}, axis=1)
```

**Treinta y una corridas de BiLSTM y GRU se hicieron en GPU, y con la misma semilla la GRU de UNR-IDD dio un F1 de prueba de 0,722 en CPU y de 0,696 en GPU. Además, en UNSW-NB15 varias redes tocaron el tope de 30 épocas: CNN-LSTM, GRU y DBN en 8 de sus 10 corridas, frente a 2 del original.**

Hay que tener en cuenta que la diferencia entre CPU y GPU, de 0,026, equivale a unas dos desviaciones estándar entre semillas de esa GRU (0,014): el hardware es una fuente de variación del mismo orden que la semilla, como reconoce la sección 8.2 del enunciado. Sin embargo, la conclusión no depende de esas corridas: el mejor modelo de cada conjunto (DBN, LightGBM y LightGBM) y los tres rivales que superan al original con las tres pruebas en UNSW-NB15 (CNN-LSTM, BiLSTM y LightGBM) corrieron en CPU.

Por lo anterior, se declara el dispositivo de cada corrida en la tabla maestra y el tope de épocas como limitación: los rivales que lo tocan pudieron quedar algo cortos de entrenamiento, lo que, si acaso, favorece al original.

## Robustez

### Perturbaciones de las entradas

```{code-cell} ipython3
:tags: [hide-input]
pert = pd.read_csv(R / "robustez/perturbaciones.csv")
fig, ejes = plt.subplots(1, 3, figsize=(11, 3.6))
titulos = {"ruido gaussiano (σ en el espacio estandarizado)": "Ruido gaussiano (σ)", "atributos faltantes imputados con la media": "Atributos faltantes (fracción)",
           "valores atípicos (un atributo ×10)": "Valores atípicos (fracción de flujos)"}
for ax, (p, titulo) in zip(ejes, titulos.items()):
    for m, g in pert[pert["perturbacion"] == p].groupby("modelo"):
        color, ancho = (NARANJA, 2.4) if m == "Original" else ((AZUL, 2.0) if m == "LightGBM" else ("#C9CED6", 1.0))
        ax.plot(g["severidad"], g["caída"], color=color, lw=ancho, marker="o", ms=3, label=m if m in ("Original", "LightGBM") else None)
    ax.set_title(titulo); ax.set_ylabel("Caída del F1 macro")
ejes[0].legend(frameon=False)
plt.tight_layout(); plt.show()
pert.pivot_table(index="modelo", columns=["perturbacion", "severidad"], values="caída").round(3)
```

**El original es el modelo más robusto ante el ruido gaussiano y ante los atributos faltantes: con σ = 0,5 su F1 cae 0,182, frente a 0,324 de LightGBM, el más frágil, y con la mitad de los atributos imputados cae 0,330, frente a 0,408 de LightGBM y 0,453 de CNN-LSTM. Ante valores atípicos, en cambio, LightGBM es el más estable (caída de 0,003 frente a 0,007 del original con el 10 % de los flujos alterados).**

Esto significa que la ventaja de LightGBM en la prueba sellada se paga con fragilidad ante ruido: sus umbrales duros cambian de rama con perturbaciones pequeñas, mientras que los escalones suaves y el experto suave del original degradan la salida de forma gradual. A la inversa, un valor atípico solo mueve a un árbol hacia una hoja extrema, que ya existía, y por eso LightGBM lo absorbe mejor.

Por esta razón, en un despliegue donde los contadores de flujo lleguen con ruido o incompletos, el original es una alternativa razonable a LightGBM, y el dashboard ofrece las dos con su incertidumbre.

### Cambio de distribución: escenario de simulación no visto (P5)

```{code-cell} ipython3
:tags: [hide-input]
pd.read_csv(R / "robustez/cambio_distribucion.csv")
```

**Ningún modelo reconoce el escenario de Sybil aislado de la prueba (909 flujos de un escenario que no aparece en el entrenamiento): el acierto va de 4,8 % (CNN-LSTM) a 9,4 % (el original y KAN). La predicción secundaria P5 se cumple: el original tampoco generaliza a ese escenario.**

En ese sentido, el resultado confirma la prueba de D3 del capítulo 1 y la dependencia L3.2: en UAVIDS-2025 el escenario de simulación pesa más que la arquitectura, y ningún mecanismo de los 13 modelos cierra esa brecha.

Por lo anterior, se recomienda no desplegar ningún modelo de este benchmark sin una validación por escenarios en la red de destino, y el detector de entradas fuera de dominio del dashboard advierte cuando un flujo se aleja del entrenamiento.

### Estabilidad entre semillas y errores

```{code-cell} ipython3
:tags: [hide-input]
est = pd.read_csv(R / "robustez/estabilidad_resumen.csv").sort_values("CV (%)")
rob = json.loads((R / "robustez/resumen.json").read_text(encoding="utf-8"))
print(f"Kruskal-Wallis entre modelos: H = {rob['kruskal_wallis']['H']:.1f}, p = {rob['kruskal_wallis']['p']:.1e}")
print(f"Flujos que fallan los 13 modelos: {rob['flujos que fallan los 13 modelos (%)']:.1f} %")
display(est)
display(pd.read_csv(R / "robustez/errores_compartidos.csv"))
pd.read_csv(R / "robustez/f1_por_clase.csv", index_col=0)
```

**El coeficiente de variación del F1 del original entre sus 48 semillas es de 1,4 %, en el rango de las redes (0,6 % a 2,8 %) y diez veces el de LightGBM (0,14 %). El 7,7 % de los flujos de prueba lo fallan los 13 modelos a la vez, y el 61,5 % de los errores del original los comete también LightGBM.**

Así mismo, el F1 por clase muestra dónde pierde el original: en Blackhole (0,834) y Wormhole (0,820) queda por debajo de LightGBM (0,887 y 0,876) y de CNN-LSTM (0,874 y 0,863), mientras que en Sybil (0,855) supera a LightGBM (0,836). Son las mismas clases en las que la ablación del capítulo 6 encontró que la compuerta dirige Blackhole y Wormhole al experto de umbrales, que es el que peor los separa.

Por tal motivo, la mejora pendiente del mecanismo está en la separación Blackhole–Wormhole, y el núcleo de errores comunes (7,7 %) marca un techo que ningún modelo de esta familia supera con estos atributos.

## La cota de Lipschitz frente al ruido (P4)

```{code-cell} ipython3
:tags: [hide-input]
p4 = pd.read_csv(R / "robustez/p4_sensibilidad_tau.csv")
display(p4.groupby("tau_min").mean(numeric_only=True).drop(columns="semilla"))
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.scatter(p4["cota K"], p4["caída con ruido σ=0.5"], c=[AZUL] * len(p4))
for s in (0.1, 0.3, 0.5):
    r = stats.spearmanr(p4["cota K"], p4[f"caída con ruido σ={s}"])
    print(f"σ = {s}: Spearman entre K y la caída = {r.statistic:+.2f} (p = {r.pvalue:.2f})")
ax.set_xlabel("Cota de Lipschitz K (teorema del capítulo 5)"); ax.set_ylabel("Caída del F1 con σ = 0,5")
plt.tight_layout(); plt.show()
```

**P4 es falsa: al subir τ_min de 0,02 a 0,2 la cota K baja de 39,3 a 20,6, como predice el teorema, pero la caída del F1 con ruido σ = 0,5 no cambia (se mantiene entre 0,166 y 0,171, sin tendencia), y la correlación de Spearman entre K y la caída no es significativa en ningún nivel de ruido (*p* entre 0,23 y 0,86).**

Esto significa que la cota es válida, porque se verifica numéricamente en el capítulo 5, pero es demasiado holgada para ordenar las variantes por su robustez real: K es un peor caso global, producto de normas espectrales, y el ruido gaussiano explora sobre todo la región donde la red es mucho más suave que ese peor caso. La robustez del original frente a LightGBM, que sí es clara, viene de la suavidad de toda la arquitectura y no de las diferencias de K dentro del barrido.

En consecuencia, la propiedad demostrada se presenta como una garantía teórica sin consecuencia práctica medible en este problema, y una cota más fina (local o certificada por intervalos) queda como trabajo futuro.

## Sensibilidad del diseño (validación)

```{code-cell} ipython3
:tags: [hide-input]
sens = pd.read_csv(R / "adecuacion/sensibilidad.csv")
sens["diseño congelado"] = sens["variante"].eq("tau_min0.05")
sens
```

**Sobre la validación y con tres semillas, el diseño congelado (T = 8, τ_min = 0,05, γ = 0,01) obtiene un F1 de 0,919, y dos variantes lo superan en más de δ: T = 4 escalones (0,931) y τ_min = 0,2 (0,928). El término de balance γ no cambia el resultado (0,920 con γ = 0 y con γ = 0,1).**

Hay que tener en cuenta que son solo tres semillas y la partición de validación, y que el diseño se congeló antes en la partición de desarrollo (capítulo 5); este barrido es posterior y no se usa para cambiarlo, porque eso sería diseñar mirando resultados que ya incluyen la prueba. La dirección, sin embargo, coincide con la ablación: menos escalones y más suaves funcionan mejor, y el codificador de escalones, tal como se congeló, resta en lugar de sumar.

Con base en esto, se propone como trabajo futuro un diseño con T = 4 y τ_min ≥ 0,2, evaluado con un protocolo nuevo y preregistrado.

## Adecuación del modelo original (condición 5)

```{code-cell} ipython3
:tags: [hide-input]
pd.read_csv(R / "adecuacion/adecuacion_con_e2.csv")
```

```{code-cell} ipython3
:tags: [hide-input]
a = adec["adecuación del original (sección 7.2 del E2)"]
pm, ob, ie = adec["permutación de etiquetas"], adec["optimismo por bootstrap"], adec["validación anidada (aproximación del E2)"]
pd.DataFrame({
    "valor": [f"{a['F1 trivial']:.3f} (p = {a['p (i)']:.1e})", f"{a['Brecha g']:.4f} (p = {a['p Wilcoxon (g < δg)']:.1e})",
              f"{a['ECE']:.4f} → {a['ECE con T']:.4f} (T = {a['T media']:.3f})", f"{a['Pendiente media']:.3f}",
              f"{a['Validación − prueba']:.4f}, IC [{a['IC 95 % por bloques'][0]:.3f}; {a['IC 95 % por bloques'][1]:.3f}]",
              f"{pm['F1 observado (semilla 6001)']:.3f} frente a máx. {pm['F1 permutado máximo']:.3f} (p = {pm['p']:.3f}, {pm['permutaciones']} permutaciones)",
              f"{ob['Original']['optimismo medio']:.4f} {tuple(ob['Original']['IC 95 %'])}; LightGBM {ob['LightGBM']['optimismo medio']:.4f}",
              f"{ie['Original']['interno − externo']:.4f} (E2: de {min(ie['E2 (mismo cálculo)'].values()):.3f} a {max(ie['E2 (mismo cálculo)'].values()):.3f})",
              f"{adec['curva de aprendizaje']['Original']['pendiente por log n']:+.5f} (p = {adec['curva de aprendizaje']['Original']['p']:.2f})"]},
    index=["(i) F1 del trivial", "(ii) Brecha entrenamiento − validación", "(iii) ECE antes → después de la temperatura", "Pendiente de calibración (media de clases)",
           "(iv) Validación − prueba", "Permutación de etiquetas", "Optimismo por bootstrap (20 réplicas)", "Validación anidada: ajuste − prueba",
           "Curva de aprendizaje: pendiente por log n"])
```

```{code-cell} ipython3
:tags: [hide-input]
cur = pd.read_csv(R / "adecuacion/curvas.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for m, g in cur.groupby("modelo"):
    ax.errorbar(g["fraccion"], g["f1_val"], yerr=g["de"], marker="o", capsize=3, label=m,
                color=NARANJA if m == "Original" else (AZUL if m == "LightGBM" else GRIS))
ax.set_xscale("log"); ax.set_xlabel("Fracción del entrenamiento (escala log)"); ax.set_ylabel("F1 macro de validación")
ax.legend(frameon=False); plt.tight_layout(); plt.show()
```

**El original cumple tres de las cuatro condiciones de adecuación del E2: supera al trivial (0,850 frente a 0,070), su brecha entre entrenamiento y validación es de 0,004, muy por debajo de δg = 0,05, y la diferencia entre validación y prueba tiene un IC por bloques que contiene al cero. Falla la calibración por un margen mínimo: su ECE con temperatura es de 0,053 frente al umbral de 0,05.**

Así mismo, la prueba de permutación confirma que aprende señal real (F1 de 0,855 frente a un máximo de 0,434 con etiquetas permutadas, *p* = 0,005, el mínimo posible con 200 permutaciones), su optimismo por *bootstrap* (0,023) es del orden del de las redes y la mitad del de LightGBM (0,043), y su curva de aprendizaje es plana: con el 10 % del entrenamiento ya alcanza el F1 de validación que obtiene con todo, mientras que LightGBM sigue mejorando con más datos (pendiente significativa, *p* < 0,001). Hosmer-Lemeshow rechaza en las cinco clases, pero con 22 883 flujos rechaza cualquier desviación mínima, por lo que se reporta junto con la pendiente de calibración (0,73), que indica probabilidades algo extremas. La temperatura ajustada en validación es menor que 1 y no corrige ese sesgo en la prueba.

Por tal motivo, el original se declara no adecuado según el criterio del E2, con la misma salvedad que ese entregable: el umbral del ECE es exploratorio, y 9 de los 15 modelos del E2 fallan esa misma condición. La condición 5 de R3 tampoco se cumpliría, aun si el original superara a la familia.

### Calibración y predicción conforme

```{code-cell} ipython3
:tags: [hide-input]
cal = json.loads((RAIZ / "artifacts/dashboard/calibracion.json").read_text(encoding="utf-8"))
pd.DataFrame({m: v for m, v in cal.items() if not m.startswith("_")}).T[["temperatura", "ece_antes", "ece_despues", "cobertura_prueba", "tamano_medio_conjunto"]].sort_values("cobertura_prueba")
```

**Con predicción conforme LAC calibrada en validación para una cobertura nominal del 90 %, ningún modelo la alcanza en la prueba: la cobertura va de 0,817 (ResNet y la regresión logística) a 0,890 (LightGBM), y la del original es de 0,846 con conjuntos de 0,96 clases en promedio.**

Esto significa que la garantía de la predicción conforme, que supone que validación y prueba son intercambiables, se rompe en UAVIDS-2025 por la misma dependencia de escenario que explica la brecha entre validación y prueba. No es un defecto de un modelo en particular, sino de la partición.

Por lo anterior, el dashboard muestra los conjuntos conformes con la advertencia de que su cobertura real medida es menor que la nominal, y reporta la cobertura observada de cada modelo.

### Los mismos diagnósticos en cada conjunto adicional

```{code-cell} ipython3
:tags: [hide-input]
diag = adec.get("diagnósticos en cada conjunto adicional")
if diag:
    tabla_diag = pd.DataFrame({c: {**{k: v for k, v in d.items() if k != "permutación"}, "p de permutación": d["permutación"]["p"],
                                   "F1 permutado máximo": d["permutación"]["F1 permutado máximo"]} for c, d in diag.items()}).T
    display(tabla_diag)
else:
    print("Pendiente: training/ejecutar_diagnosticos_adicionales.py")
```

## Robustez a la elección de la prueba (condición 6)

```{code-cell} ipython3
:tags: [hide-input]
pd.DataFrame(adec["Nadeau-Bengio y F 5x2cv (condición 6)"]).T
```

**Con la *t* de Nadeau-Bengio y la *F* de 5 × 2 cv sobre las diez particiones de la validación cruzada del E2, el original no se distingue de LightGBM (diferencia media de +0,0002, *p* = 0,99) ni de CNN 1D (+0,0006, *p* = 0,97): los tres obtienen un F1 medio de 0,860.**

Esto quiere decir que la conclusión principal es robusta a la elección de la prueba, porque ninguna, paramétrica o no paramétrica, con o sin la corrección de Nadeau-Bengio, muestra que el original supere a un rival. En cambio, la ventaja de LightGBM sobre el original, que en la prueba sellada es de 0,027 y respaldan las seis pruebas admisibles del capítulo 7, desaparece cuando la partición cambia.

Por tal motivo, y como exige la condición 6 de la sección 8.4 del enunciado, la superioridad de LightGBM sobre el original se reporta como un resultado no robusto a la partición: depende de qué escenarios de simulación quedaron en la prueba sellada, igual que la brecha entre validación y prueba.

## Escalabilidad, incertidumbre e interpretabilidad

Tres contenidos del cuadro 7 para este capítulo están desarrollados en otros lugares del libro y aquí solo se enlazan:

- **Escalabilidad.** El costo del original crece de forma lineal con el número de atributos y su tiempo por flujo se estabiliza en microsegundos con lotes grandes (capítulo 6, sección de complejidad). La curva de aprendizaje de este capítulo muestra la otra cara: con el 10 % del entrenamiento ya alcanza su F1 de validación.
- **Incertidumbre.** Además de la calibración y la predicción conforme de este capítulo, la dispersión entre semillas se reporta con media, DE e IC 95 % en la tabla de líderes (capítulo 3) y en cada conjunto adicional.
- **Interpretabilidad.** Los valores de Shapley, la fidelidad por borrado y la estabilidad de las explicaciones entre semillas están en el capítulo 6, junto con el peso de la compuerta por clase.

## Eficiencia y frontera de Pareto

```{code-cell} ipython3
:tags: [hide-input]
lid = pd.read_csv(RAIZ / "artifacts/dashboard/lideres.csv")
lid = lid.dropna(subset=["Latencia mediana (ms)"]).sort_values("Latencia mediana (ms)")
lid["Pareto"] = lid["F1 macro"] > lid["F1 macro"].cummax().shift(fill_value=-1)
fig, ax = plt.subplots(figsize=(7, 4.2))
for _, f in lid.iterrows():
    ax.scatter(f["Latencia mediana (ms)"], f["F1 macro"], color=NARANJA if f["Modelo"] == "Original" else (AZUL if f["Pareto"] else GRIS), zorder=3)
    ax.annotate(f["Modelo"], (f["Latencia mediana (ms)"], f["F1 macro"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
frontera = lid[lid["Pareto"]]
ax.plot(frontera["Latencia mediana (ms)"], frontera["F1 macro"], color=AZUL, lw=1, ls="--")
ax.set_xscale("log"); ax.set_xlabel("Latencia mediana por flujo en ONNX Runtime, 1 hilo (ms, escala log)"); ax.set_ylabel("F1 macro de prueba")
plt.tight_layout(); plt.show()
lid[["Modelo", "F1 macro", "Latencia mediana (ms)", "Latencia p95 (ms)", "Parámetros", "FLOPs", "Entrenamiento (s)", "Pareto"]]
```

**La frontera de Pareto entre F1 y latencia la forman DBN (0,080 ms), DNN (0,091 ms) y LightGBM (0,109 ms, el mejor F1). El original, con 0,185 ms por flujo, queda dominado: hay modelos más rápidos y con mayor F1.**

Hay que tener en cuenta que todas las latencias están por debajo de 2 ms por flujo en un solo hilo de CPU, incluida la BiLSTM, la más lenta (1,3 ms), de modo que cualquiera de los 16 modelos procesa varios miles de flujos por segundo y la latencia no es una restricción para el dashboard. El original tiene 14 908 parámetros y 29 056 FLOPs por flujo, del orden de las redes densas y 36 veces menos parámetros que la BiLSTM.

Por esta razón, la elección entre modelos se decide por el desempeño y la robustez, no por el costo de inferencia, y el capítulo 10 recoge esa elección.

## Balance de las seis condiciones de R3

```{code-cell} ipython3
:tags: [hide-input]
c4 = json.loads((R / "confirmatorio/sensibilidad_cuadro4.json").read_text(encoding="utf-8"))
conf = json.loads((R / "confirmatorio/resumen.json").read_text(encoding="utf-8"))
pd.DataFrame([
    {"condición": "1. Métrica media favorable frente a cada rival", "cumple": "no", "evidencia": f"F1 medio menor que el de {len(c4['rivales con media mayor que la del original'])} de los 12 rivales"},
    {"condición": "2. Tres pruebas con Holm y diferencia > δ", "cumple": "no", "evidencia": "ningún contraste con tres pruebas, ni con las preregistradas ni con las admisibles del cuadro 4"},
    {"condición": "3. Efecto no despreciable e IC > 0", "cumple": "no", "evidencia": "solo frente a TabNet y RNN hay efecto a favor no despreciable"},
    {"condición": "4. Consistencia en los conjuntos adicionales", "cumple": "no", "evidencia": f"mejor en {res_ad['conjuntos en que el original es el mejor en la media']} de 3; peor con significancia en UNSW-NB15"},
    {"condición": "5. Adecuación del modelo (E2, sección 7.2)", "cumple": "no", "evidencia": "cumple (i), (ii) y (iv); falla (iii) por ECE de 0,053 frente a 0,05"},
    {"condición": "6. Robustez a la elección de la prueba", "cumple": "sí (para la conclusión principal)", "evidencia": "ninguna prueba muestra superioridad del original; la ventaja de LightGBM no es robusta a la partición"},
])
```

**El modelo original no cumple R3: falla las condiciones 1 a 5, y la condición 6 confirma que esa conclusión no depende de la prueba elegida. El tope de calificación es 4,0 (sección 14.1 del enunciado).**

En síntesis, la evidencia de los cuatro conjuntos y de las seis pruebas admisibles apunta en la misma dirección: el original es una red del grupo intermedio, más robusta que el resto ante ruido y atributos faltantes, sin una ventaja de desempeño que resista la comparación con la familia completa.

Por lo anterior, el trabajo se sostiene en el hallazgo alternativo preregistrado en la sección 12 del protocolo, que el capítulo 10 desarrolla.
