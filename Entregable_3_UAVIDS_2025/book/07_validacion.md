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

# Validación: protocolo y pruebas de hipótesis

```{code-cell} ipython3
:tags: [hide-input]
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RAIZ = Path("..").resolve()
C = RAIZ / "experiments/resultados/confirmatorio"
t = pd.read_csv(C / "tabla_contrastes.csv")
resumen = json.loads((C / "resumen.json").read_text(encoding="utf-8"))
potencia = json.loads((RAIZ / "experiments/resultados/potencia.json").read_text(encoding="utf-8"))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.precision", 4)
```

## Protocolo preregistrado

El protocolo (`experiments/protocolo.md`) se congeló el 22/11/2026 con la etiqueta de Git `e3-protocolo`, antes de cualquier corrida definitiva, y la declaración del estándar lleva la etiqueta anterior `e3-estandar`. Fija las hipótesis, las predicciones secundarias, la métrica primaria y δ, la fuente del mecanismo, los tres conjuntos adicionales, las particiones, las semillas, el determinismo, la familia de contrastes, las pruebas, la corrección de Holm, el presupuesto de ajuste, el análisis de potencia y *Nd*. Las desviaciones posteriores al congelamiento se declaran, con su justificación, en la última sección de este capítulo.

**Equidad del presupuesto.** El modelo original recibió las mismas 12 configuraciones de (lr, weight_decay, lote) que los 13 modelos profundos del E2, el mismo máximo de 30 épocas, la misma parada temprana sobre el F1 macro de validación con paciencia 6, la misma pérdida ponderada y el mismo recorte del gradiente, con el motor de entrenamiento del E2 (capítulo 6). El presupuesto se midió en número de configuraciones.

## Análisis de potencia

```{code-cell} ipython3
:tags: [hide-input]
pot = pd.DataFrame(potencia["contrastes"])
pot = pot[pot["familia"] == "confirmatoria"][["rival", "sigma_d", "R_necesarias"]]
pot["semillas usadas"] = pot["rival"].map(t.set_index("rival")["n_semillas"])
pot["potencia observada para δ"] = pot["rival"].map(t.set_index("rival")["potencia para delta"])
pot["MDE observada"] = pot["rival"].map(t.set_index("rival")["MDE"])
print(f"Potencia conjunta observada (producto de las potencias por contraste): {resumen['potencia conjunta observada']:.3f}")
pot
```

**El piloto pidió 48 semillas para el contraste con CNN-LSTM y 38 para TabNet, y con ellas se planearon todos los contrastes para una potencia de 0,90. La potencia observada, calculada con la dispersión real de las diferencias, alcanza 0,90 en 6 de los 12 contrastes (CNN-LSTM, Autoencoder, ResNet, DBN, LightGBM y CNN 1D) y se queda entre 0,76 y 0,86 en otros cinco. BiLSTM es el caso extremo, con 0,41 y una diferencia mínima detectable de 0,016. La potencia conjunta es de apenas el 11,5 %.**

Esto significa que el piloto de 5 semillas subestimó la dispersión de las diferencias en la mitad de los contrastes, algo previsible con tan pocas semillas, y que, aun si el modelo original fuera mejor que todos los rivales por exactamente δ, la probabilidad de rechazar los 12 contrastes a la vez sería baja. Es la consecuencia esperada de una prueba de unión-intersección con m = 12 (sección 8.5 del enunciado).

Por lo anterior, cada contraste se reporta con su diferencia mínima detectable, la potencia insuficiente se declara como limitación y se distingue entre "no concluyente" y "equivalente"; un resultado no concluyente no se interpreta como equivalencia.

## Resultados confirmatorios

```{code-cell} ipython3
:tags: [hide-input]
fam = t[t["familia"] == "confirmatoria"].copy()
fam["decisión"] = np.where(fam["supera (R3)"], "Supera (R3)",
                    np.where((fam[["p T1 t Holm", "p T2 Wilcoxon Holm"]].max(axis=1) < 0.05) & (fam["diferencia"] < 0), "Peor según T1 y T2; T3 no concluyente",
                    np.where((fam[["p T1 t Holm", "p T2 Wilcoxon Holm"]].max(axis=1) < 0.05) & (fam["diferencia"] > 0), "Mejor según T1 y T2; T3 no concluyente",
                             "No concluyente")))
fam[["rival", "n_semillas", "F1 original", "F1 rival", "diferencia", "IC95 t", "p T1 t Holm", "p T2 Wilcoxon Holm", "p T3 bootstrap bloques Holm",
     "delta Cliff", "P(original>rival) bayes", "P(ROPE)", "decisión"]]
```

```{code-cell} ipython3
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(8, 4.6))
f = fam.sort_values("diferencia")
ic = np.array([eval(s.replace("np.float64", "")) for s in f["IC95 t"]])
colores = ["#EB6834" if d < 0 else "#2A78D6" for d in f["diferencia"]]
ax.errorbar(f["diferencia"], f["rival"], xerr=[f["diferencia"] - ic[:, 0], ic[:, 1] - f["diferencia"]], fmt="none", ecolor="#8A94A6")
ax.scatter(f["diferencia"], f["rival"], c=colores, zorder=3)
for x0 in (-0.01, 0, 0.01):
    ax.axvline(x0, color="#2C3E50" if x0 == 0 else "#C9CED6", lw=1, ls="-" if x0 == 0 else "--")
ax.set_xlabel("F1 macro del original − F1 del rival (media sobre semillas, IC 95 % t)")
ax.set_title("Diferencia frente a cada rival de la familia (líneas punteadas: ±δ)")
plt.tight_layout(); plt.show()
```

**El modelo original obtiene un F1 macro medio de 0,850 en la prueba sellada (48 semillas) y no supera con significancia a ninguno de los 12 rivales: la condición R3 no se cumple.** Contraste por contraste:

- **Es mejor que TabNet** (+0,018) según la *t* pareada y Wilcoxon, y que RNN (+0,011) solo según la *t*.
- **Es peor que LightGBM** (−0,027), Autoencoder (−0,011), CNN 1D (−0,010), CNN-LSTM (−0,009) y DBN (−0,009) según esas dos pruebas, y que Transformer (−0,010) solo según la *t*.
- **No se distingue de BiLSTM, ResNet, GRU y KAN.** Frente a ResNet, GRU y KAN la prueba bayesiana pone más del 98 % de la probabilidad dentro de la región de equivalencia práctica (±δ).

Hay que tener en cuenta que la tercera prueba, el *bootstrap* por bloques de prueba, no rechaza en ningún contraste de la familia: la prueba sellada tiene solo 23 bloques de 1000 flujos, casi todos de una sola clase, y al remuestrearlos cambia la composición de clases, así que el intervalo de la diferencia es muy ancho (frente a LightGBM va de −0,067 a +0,039). Como la regla preregistrada exige que las tres pruebas rechacen, los contrastes en los que la *t* y Wilcoxon coinciden quedan como resultados de dos pruebas y no como decisiones confirmatorias. La diferencia con LightGBM es la más grande y la más estable: es negativa en las 20 semillas (δ de Cliff = −1).

Así las cosas, el tope de la calificación es 4,0 (sección 14.1 del enunciado) y el trabajo se sostiene en el hallazgo alternativo preregistrado: el *benchmark* riguroso, la explicación de por qué el escenario pesa más que la arquitectura y la medición de cuándo ayuda el mecanismo (capítulos 6, 8 y 10).

## Procedimiento global y diagrama de diferencia crítica

```{code-cell} ipython3
:tags: [hide-input]
rangos = pd.Series(resumen["rangos medios (1 = mejor)"]).sort_values()
cd = json.loads((RAIZ / "artifacts/dashboard/resumen.json").read_text(encoding="utf-8")).get("cd_nemenyi")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.scatter(rangos.values, rangos.index, color=["#EB6834" if m == "Original" else "#2A78D6" for m in rangos.index], zorder=3)
ax.axvspan(rangos.iloc[0], rangos.iloc[0] + cd, color="#2A78D6", alpha=0.08)
ax.invert_yaxis(); ax.set_xlabel("Rango medio sobre las semillas 6001 a 6020 (1 = mejor)")
ax.set_title(f"Friedman: χ² = {resumen['friedman']['chi2']:.1f}, p Iman-Davenport = {resumen['friedman']['p Iman-Davenport']:.1e}; CD de Nemenyi = {cd:.2f}")
plt.tight_layout(); plt.show()
```

**La prueba de Friedman rechaza que los 13 modelos sean iguales (χ² = 103,6; *p* de Iman-Davenport del orden de 10⁻²²; W de Kendall = 0,43). LightGBM tiene el mejor rango medio (1,65), seguido de CNN-LSTM (4,1); el modelo original queda en un rango medio de 8,6, dentro del grupo intermedio.**

De esta forma, la conclusión del procedimiento global coincide con la de los contrastes pareados: hay diferencias reales entre modelos, LightGBM está en el primer lugar con una distancia mayor que la diferencia crítica (4,08) frente a la mayoría, y el original no se separa del grupo de redes intermedias.

Por esta razón, el modelo original se presenta como una alternativa con ventajas en robustez (capítulo 8), no como un modelo superior en F1 macro.

## Sensibilidad de la conclusión

```{code-cell} ipython3
:tags: [hide-input]
fam[["rival", "P(original>rival) bayes", "P(ROPE)", "P(original<rival) bayes", "dif exactitud balanceada", "p t bacc Holm", "p t primeras 20"]]
```

La conclusión principal, que R3 no se cumple, se mantiene con las cuatro variantes preregistradas:

- la prueba bayesiana de rangos con signo con ROPE ±δ, que da una probabilidad alta de que el original sea peor que LightGBM;
- la exactitud balanceada como métrica secundaria;
- el subconjunto de las 20 primeras semillas;
- la *t* corregida de Nadeau-Bengio sobre la validación cruzada 5 × 2 (capítulo 8).

## Pruebas admisibles del cuadro 4 (sensibilidad)

```{code-cell} ipython3
:tags: [hide-input]
c4 = pd.read_csv(C / "sensibilidad_cuadro4.csv")
c4[["rival", "diferencia", "p A1 Wilcoxon Holm", "p A2 t Nadeau-Bengio Holm", "p A3 McNemar Holm", "p A4 Friedman post hoc Holm",
    "P(original>rival) bayes", "P(original<rival) bayes", "DeLong clases a favor", "DeLong clases en contra",
    "pruebas a favor del original", "pruebas en contra del original", "supera (R3 estricto)"]].fillna("")
```

La fila "F1. Tabular" del cuadro 4 del enunciado admite la *t* pareada con corrección de Nadeau-Bengio, la *F* de 5 × 2 cv, Wilcoxon, McNemar y DeLong, Friedman con Holm y la prueba bayesiana con ROPE. Dos de las tres pruebas preregistradas no coinciden al pie de la letra con esa lista: T1 es la *t* pareada sin la corrección y T3, el *bootstrap* por bloques, figura en la fila de problemas espaciales. Por eso se repite el análisis con seis pruebas admisibles, con Holm aparte para cada una:

- **A1.** Wilcoxon sobre las semillas.
- **A2.** *t* con corrección de Nadeau-Bengio sobre las semillas.
- **A3.** McNemar exacta sobre los aciertos del ensamble.
- **A4.** Friedman con *post hoc* de Holm.
- **A5.** Bayesiana con ROPE, que cuenta como rechazo cuando la probabilidad posterior supera 0,95.
- **A6.** DeLong sobre el AUC uno contra todos de cada clase, que cuenta como rechazo cuando al menos 3 de las 5 clases son significativas en el mismo sentido.

La *F* de 5 × 2 cv y Nadeau-Bengio sobre 5 × 2 están en el capítulo 8.

**Con las pruebas admisibles la conclusión no cambia: el modelo original no supera a ningún rival con tres pruebas, y cinco rivales son mejores que él con tres o más: LightGBM (con las seis), CNN-LSTM (con cuatro), Autoencoder, DBN y CNN 1D (con tres cada uno). Frente a TabNet solo lo favorecen Wilcoxon y la bayesiana, con P(original > TabNet) = 0,994.**

Hay que tener en cuenta que McNemar (A3) y DeLong (A6) tratan los 22 883 flujos como si fueran independientes, cuando están agrupados por escenario, de modo que son anticonservadoras. Si se cuentan solo las pruebas sobre semillas (A1, A2, A4 y A5), únicamente LightGBM es mejor con tres o más, con las cuatro. Además, la *t* con corrección de Nadeau-Bengio infla la varianza con el factor n_prueba/n_entrenamiento = 0,31 aunque el entrenamiento no cambie entre semillas, y solo rechaza frente a LightGBM.

Por tal motivo, se mantiene la decisión preregistrada (R3 no se cumple y el tope es 4,0) y la batería admisible se reporta como sensibilidad. En ninguna combinación de pruebas el original supera a un rival. La superioridad de LightGBM queda respaldada por todas, aunque el capítulo 8 muestra que desaparece en la validación cruzada 5 × 2 y la reporta como resultado no robusto a la partición, como pide la condición 6 de la sección 8.4 del enunciado.

## Ajuste y sobreajuste del modelo original

La prueba de permutación de etiquetas, el optimismo por *bootstrap* y la curva de aprendizaje se reportan en el capítulo 8 con las corridas de `training/ejecutar_robustez.py`. Su F1 medio de prueba (0,850) supera ampliamente al clasificador trivial (0,070). La brecha media entre validación (0,919 en el piloto) y prueba es de unos 0,07, del mismo orden que la de las demás redes del E2, porque se debe a la dependencia por escenario documentada en el capítulo 1 y no a un sobreajuste propio del modelo.

## Desviaciones del protocolo

1. **Pruebas del cuadro 4 (23/11/2026, después de ver los resultados).** El protocolo citó la fila tabular del cuadro 4 para T1 a T3, pero T3 pertenece a la fila espacial y T1 no lleva la corrección de Nadeau-Bengio. No se reemplazó la decisión preregistrada: se agregó la sensibilidad con las pruebas admisibles de la sección anterior (`experiments/analisis_cuadro4.py`). La conclusión es la misma, y el cambio no podía favorecer al original, porque su F1 medio ya es menor que el de 8 de los 12 rivales.
2. **GPU en 31 corridas de los conjuntos adicionales (23/11/2026).** Las GRU de UNSW-NB15, las BiLSTM y GRU de UNR-IDD y una GRU de UAV-NIDD corrieron en una GPU NVIDIA RTX PRO 6000 Blackwell (porción MIG de 24 GB, PyTorch 2.8.0 y CUDA 12.8), con determinismo estricto de CUDA. Es la opción prevista en la sección 5 del protocolo, solo para BiLSTM y GRU de esos conjuntos. Con la misma semilla, la GRU de UNR-IDD dio un F1 de prueba de 0,722 en CPU y de 0,696 en GPU, una variación del orden de dos desviaciones estándar entre semillas, que se discute en el capítulo 8.
3. **Partición de los conjuntos adicionales (22/11/2026, antes de etiquetar el protocolo y de entrenar cualquier modelo).** Los bloques de 1000 filas sin estratificar dejaban clases fuera de la prueba; se cambiaron por bloques de 250 filas estratificados por clase (sección 5.3 del protocolo).
