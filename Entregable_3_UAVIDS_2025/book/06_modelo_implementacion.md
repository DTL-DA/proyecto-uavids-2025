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

# Modelo original (MoE-ES): implementación, pruebas, ablación e interpretabilidad

```{code-cell} ipython3
:tags: [hide-input]
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RAIZ = Path("..").resolve()
R = RAIZ / "experiments/resultados"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.precision", 4)
CLASES = ["Blackhole", "Flooding", "Normal", "Sybil", "Wormhole"]
```

## Implementación propia (R2)

El mecanismo está programado desde su formulación en `src/original_model.py`, sin `import` de un modelo de librería ni código de terceros. PyTorch se usa solo como motor tensorial: capas `nn.Linear`, `sigmoid`, `softmax`, `logsumexp` y diferenciación automática.

| Pieza | Archivo | Qué hace |
|---|---|---|
| `EscalonesSuaves` | `src/original_model.py` | Codificador $s_{j,t} = \sigma((x_j - \theta_{j,t})/\tau_j)$ con umbrales y temperaturas aprendibles, $\tau_j \ge \tau_{\min}$, e inicialización por cuantiles |
| `ModeloOriginal` | `src/original_model.py` | Expertos, compuerta y mezcla de probabilidades; devuelve $\log p(y \mid x)$; `cota_lipschitz()` calcula $K$; `perdida_cooperativa()` es el esquema de entrenamiento |
| Bucle de entrenamiento | `training/entrenar_original.py` | Copia fiel de `codigo.motor.entrenar` del E2, con una sola línea distinta: la pérdida cooperativa |
| Registro en el catálogo | `training/registro.py` | Agrega el modelo al catálogo del E2 sin modificar su código: mismos datos, transformación, optimizador, planificador, parada temprana, conteo de MAC y registro de corridas |

El modelo usa los ganchos que el motor del E2 ya ofrecía: `preentrenar(X, generador)`, con el que la red de creencia profunda inicializaba sus RBM y que aquí inicializa los umbrales en cuantiles, y la salida en log-probabilidades, que hace que la entropía cruzada estándar sea la log-verosimilitud de la mezcla.

## Pruebas unitarias

`tests/test_original_model.py` cubre los seis puntos de la sección 6.5 del enunciado y corre en integración continua (`.github/workflows/entregable3-pruebas.yml`) junto con las pruebas del dashboard.

| Punto | Prueba | Qué verifica |
|---|---|---|
| (i) | `test_sobreajusta_lote_pequeno` | Pérdida menor que 0,1 en 300 pasos sobre 64 ejemplos sintéticos con 4 clases definidas por umbrales en dos ejes |
| (ii) | `test_reproducible_con_misma_semilla` | Mismas probabilidades con la misma semilla |
| (iii) | `test_formas_y_probabilidades`, `test_entrada_invalida` | Salida (n, 5), probabilidades que suman 1, error claro con 7 atributos en lugar de 16 y con opciones desconocidas |
| (iv) | `test_todos_los_parametros_reciben_gradiente` | Gradiente finito y no nulo en todos los parámetros, incluidos umbrales y temperaturas |
| (v) | `test_cota_lipschitz_no_se_excede` (5 configuraciones), `test_codificador_lipschitz`, `test_desigualdad_de_jensen_de_la_mezcla` | El cociente empírico nunca supera $K$; la cota del codificador; la desigualdad de Jensen flujo por flujo |
| (vi) | `test_guardar_y_cargar_conserva_salidas`, `test_exportacion_onnx_ida_y_vuelta` | Mismas salidas tras guardar y cargar, y error menor que $10^{-4}$ tras exportar a ONNX |

## Complejidad

Para un flujo con $d$ atributos, $T$ umbrales y anchos $h_u$, $h_{s1}$, $h_{s2}$ y $h_g$, una pasada hacia adelante cuesta $O(dT + dT\,h_u + d\,h_{s1} + h_{s1}h_{s2} + d\,h_g)$ operaciones, es decir, $O(d\,(T h_u + h_{s1} + h_g))$: es lineal en el número de atributos y no tiene término cuadrático, a diferencia de la atención del Transformer. El entrenamiento cuesta $O(N\,E)$ veces eso, con $N$ flujos y $E$ épocas. Con los valores congelados ($d = 16$, $T = 8$, $h_u = 64$, $h_s = (64, 64)$ y $h_g = 32$) son 14 908 parámetros y 14 528 multiplicaciones-acumulaciones por flujo.

```{code-cell} ipython3
:tags: [hide-input]
comp = pd.read_csv(R / "robustez/complejidad.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for d, g in comp.groupby("d"):
    axes[0].plot(g["lote"], g["µs por flujo"], marker="o", label=f"d = {d}")
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("Tamaño del lote"); axes[0].set_ylabel("µs por flujo"); axes[0].legend(frameon=False)
axes[0].set_title("Tiempo de inferencia por flujo")
g = comp[comp["lote"] == 16384]
axes[1].plot(g["d"], g["ms por lote"], marker="o", label="tiempo (lote de 16 384)")
axes[1].set_xlabel("Número de atributos d"); axes[1].set_ylabel("ms por lote")
ax2 = axes[1].twinx(); ax2.plot(g["d"], g["parametros"], marker="s", color="#EB6834", label="parámetros"); ax2.set_ylabel("Parámetros")
axes[1].set_title("Escalamiento con d"); plt.tight_layout(); plt.show()
comp.pivot(index="d", columns="lote", values="µs por flujo")
```

**El tiempo por flujo cae con el tamaño del lote hasta estabilizarse en pocos microsegundos, y tanto los parámetros como el tiempo con lotes grandes crecen de forma aproximadamente lineal con el número de atributos $d$, como predice el análisis asintótico.**

Esto quiere decir que el costo del modelo lo dominan las capas lineales del experto de umbrales, cuya entrada tiene $dT$ dimensiones, y que el modelo puede servir en tiempo real en un nodo con CPU modesta.

En consecuencia, no hace falta cuantizarlo para el dashboard (64 KB en ONNX) y, si se aplicara a conjuntos con más atributos, conviene reducir $T$ antes que los anchos.

## Ablación

Cada variante cambia un solo componente frente al modelo completo y usa las mismas 20 semillas (6001 a 6020), los mismos hiperparámetros y el mismo protocolo. La familia de la ablación tiene su propio Holm (protocolo, sección 9).

```{code-cell} ipython3
:tags: [hide-input]
abl = pd.read_csv(R / "ablacion/contrastes.csv")
abl[["descripción", "F1 variante", "F1 completo", "completo − variante", "p t Holm", "p Wilcoxon Holm", "δ Cliff", "conf B–W variante", "conf B–W completo",
     "el componente aporta", "el componente resta"]]
```

```{code-cell} ipython3
:tags: [hide-input]
ps = pd.read_csv(R / "ablacion/por_semilla.csv")
orden = ["A0_completo", "A1_sin_compuerta", "A2_sin_escalones", "A3_conjunto", "A4_sin_balance", "A5_solo_suave", "A6_solo_umbral", "CNN-LSTM (mejor red)", "LightGBM"]
fig, ax = plt.subplots(figsize=(9, 3.8))
datos = [ps.loc[ps["variante"] == v, "f1"] for v in orden]
cajas = ax.boxplot(datos, vert=False, patch_artist=True, tick_labels=orden)
for c, v in zip(cajas["boxes"], orden):
    c.set_facecolor("#EB6834" if v == "A0_completo" else ("#2A78D6" if v in ("CNN-LSTM (mejor red)", "LightGBM") else "#D5DAE1"))
ax.set_xlabel("F1 macro en prueba (20 semillas)"); ax.set_title("Ablación del modelo original frente a la mejor red y LightGBM")
plt.tight_layout(); plt.show()
```

**Quitar el codificador de escalones suaves (A2) mejora el F1 macro de 0,848 a 0,862, una diferencia significativa después de Holm (*p* de 0,0001 en la *t* y 0,0002 en Wilcoxon; δ de Cliff = −0,81). Usar solo el experto suave (A5) también es mejor (0,859). Quitar la compuerta (A1), el entrenamiento cooperativo (A3) o el regularizador de balance (A4) no cambia el F1 de forma significativa. Y usar solo el experto de umbrales (A6) da 0,845, sin diferencia con el modelo completo.**

Esto significa que, en la prueba sellada, el componente central del mecanismo resta en lugar de sumar: los umbrales aprendidos en los escenarios de entrenamiento no transfieren a los segmentos de prueba, y la compuerta no compensa esa pérdida. Es lo contrario de lo observado en la partición de desarrollo, donde el experto de umbrales solo era el mejor diseño (0,843 frente a 0,824 del suave). La diferencia entre desarrollo y confirmación es otra manifestación de la dependencia por escenario: la partición de desarrollo comparte escenarios entre ajuste y evaluación, igual que la prueba, pero en otra proporción.

Por tal motivo, la ablación se reporta como evidencia en contra de la hipótesis del mecanismo y no se itera el diseño con estos resultados, porque usar la prueba para rediseñar invalidaría la confirmación (sección 7.2 del enunciado).

## Predicciones secundarias

```{code-cell} ipython3
:tags: [hide-input]
p = json.loads((R / "ablacion/predicciones_secundarias.json").read_text(encoding="utf-8"))
pesos = pd.read_csv(R / "ablacion/peso_compuerta_por_clase.csv")
pd.DataFrame({"Predicción": ["P1: sin escalones sube la confusión B–W", "P2: la compuerta pesa más el experto de umbrales en B y W que en Sybil",
                             "P3: la ventaja frente a CNN-LSTM se concentra en B y W"],
              "¿Se cumple?": [p["P1 (sin escalones sube la confusión B–W)"], p["P2 (más peso al experto de umbrales en B y W que en Sybil)"],
                              p["P3 (ventaja concentrada en B y W frente a CNN-LSTM)"]],
              "Evidencia": [f"conf. B–W completo {p['conf_BW']['A0_completo']:.3f}; sin escalones {p['conf_BW']['A2_sin_escalones']:.3f}",
                            "peso medio hacia el experto de umbrales: " + ", ".join(f"{c} {v:.2f}" for c, v in p["peso_umbral_por_clase"].items()),
                            "original − CNN-LSTM por clase: " + ", ".join(f"{k.replace('F1 ', '')} {v:+.3f}" for k, v in p["diferencia_por_clase_vs_CNN-LSTM"].items())]})
```

**P2 se cumple: la compuerta envía al experto de umbrales el 88 % del peso en los flujos Wormhole y el 69 % en los Blackhole, frente al 51 % en Sybil y el 34 % en Normal. P1 y P3 no se cumplen: sin el codificador la confusión Blackhole–Wormhole baja de 0,153 a 0,143, y frente a CNN-LSTM el original pierde 0,029 de F1 en Blackhole y 0,031 en Wormhole.**

En ese sentido, el mecanismo hace lo que se diseñó, porque la compuerta aprende a mandar los ataques de enrutamiento al experto de umbrales, pero la premisa causal falla: en la prueba, ese experto separa peor Blackhole de Wormhole que una red suave. El fallo de la hipótesis no es de implementación, sino de transferencia del sesgo.

Con base en esto, se concluye que el sesgo de árbol que hace fuerte a LightGBM no se obtiene con escalones aprendidos por gradiente, y la explicación de esa diferencia queda como trabajo futuro (capítulo 10).

## Interpretabilidad

Se calcularon valores de Shapley de la probabilidad de la clase predicha por muestreo de permutaciones {cite:p}`lundberg2017unified`, con implementación propia (`experiments/analisis_robustez.py`), para 300 flujos de prueba (60 por clase) frente a un fondo de 50 flujos de entrenamiento, en el modelo original y en LightGBM, el mejor *benchmark*.

```{code-cell} ipython3
:tags: [hide-input]
imp = pd.read_csv(R / "robustez/importancia_shapley.csv", index_col=0)
fid = pd.read_csv(R / "robustez/fidelidad_borrado.csv")
res = json.loads((R / "robustez/resumen.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(8, 4.4))
o = imp.sort_values("Original")
y = np.arange(len(o))
ax.barh(y - 0.2, o["Original"], 0.4, color="#EB6834", label="Original"); ax.barh(y + 0.2, o["LightGBM"], 0.4, color="#2A78D6", label="LightGBM")
ax.set_yticks(y, o.index); ax.set_xlabel("|φ| medio (probabilidad de la clase predicha)"); ax.legend(frameon=False)
ax.set_title(f"Importancia global por Shapley (ρ de Spearman entre modelos = {res['interpretabilidad']['spearman_original_vs_lightgbm']:.2f})")
plt.tight_layout(); plt.show()
print("Estabilidad entre semillas del original (ρ de Spearman de la importancia):", [round(r, 3) for r in res["estabilidad_explicaciones"]["spearman_entre_semillas"]])
fid
```

**Los dos modelos se apoyan en atributos parecidos (ρ de Spearman de 0,79 entre sus importancias globales), y la importancia del original es estable entre semillas (ρ medio de 0,89). La prueba de borrado confirma la fidelidad: reemplazar por la mediana del fondo los atributos más importantes de cada flujo baja la probabilidad de la clase predicha mucho más que borrar el mismo número de atributos al azar.**

Lo anterior indica que las explicaciones no son ruido de una semilla y que el original no depende de un atajo distinto al de LightGBM: usa la información de tasas, pérdida y rendimiento que el dominio asocia con cada ataque.

Por ello, las explicaciones locales del dashboard (oclusión por atributo) se presentan como una ayuda para el analista y se acompañan del aviso fuera de dominio, porque la estabilidad de la importancia no garantiza que la predicción sea correcta.
