from .base import code, md

md("s4", r"""
## 4. Implementación de los modelos del benchmark

En esta sección se implementan los 13 modelos profundos y las tres líneas base no profundas bajo el mismo entorno, el mismo pipeline y el mismo presupuesto de ajuste, siguiendo la ruta F1 del enunciado y el protocolo de su sección 11. El orden es el del enunciado: diseño experimental (4.1), implementación y ajuste (4.2), requisitos experimentales, con el piloto, la potencia y la búsqueda de arquitectura (4.3), buenas prácticas contra la fuga (4.4), evaluación (4.5) y robustez, interpretabilidad y ablación (4.6).

### 4.1 Diseño experimental

| Elemento | Definición |
|---|---|
| Variables de entrada | Los 16 atributos de tráfico y desempeño, recortados, transformados con $\log(1 + x)$ y estandarizados con entrenamiento (sección 2.2.1) |
| Variable objetivo y tarea | Una de las 5 clases; clasificación multiclase de cada flujo, sin horizonte ni rezagos |
| División del dataset | Entrenamiento, validación y prueba heredados del Entregable 1 (64, 16 y 20 %), estratificados por clase y agrupados por bloques, idénticos para todos los modelos; la prueba se toca una sola vez por corrida |
| Pipeline | Un único pipeline por modelo, en `codigo/`: datos y transformación (`datos.py`), modelos (`modelos.py`), entrenamiento (`motor.py`), líneas base (`clasicos.py`) y ejecución en paralelo con reanudación (`ejecutor.py`) |
| Selección de hiperparámetros | Búsqueda aleatoria con las mismas 12 configuraciones para los 13 modelos profundos, 15 épocas y una semilla, elegida solo con el F1 macro de validación; LightGBM y la regresión logística reciben 12 configuraciones de su propio espacio |
| Arquitectura no predefinida | Búsqueda de la sección 14 para ResNet, Autoencoder, DBN y CNN-LSTM: 1000 arquitecturas en la fase 1 y k finalistas con 5 semillas en la fase 2 |
| Repeticiones | R semillas por modelo, con R fijado por el piloto (sección 4.3) y las mismas semillas para todos |
| Validación interna | Partición fija de validación para seleccionar, más una validación cruzada 5 × 2 agrupada por bloques sobre entrenamiento + validación para las pruebas de Nadeau y Bengio y de Alpaydin |
| Evaluación | Mismas métricas sobre la misma prueba (sección 4.5) |

La validación anidada completa, que repetiría el ajuste dentro de cada pliegue externo, multiplicaría por 12 el costo de todo el benchmark, de modo que se reemplaza por la comparación entre el puntaje de validación usado para seleccionar y el de prueba (tabla B, sección 7), que mide el mismo optimismo por selección.
""")

md("s4-2", r"""
### 4.2 Implementación de los modelos

Todos los modelos profundos se implementan en PyTorch, con la versión de la configuración del entorno; TabNet usa la red de la implementación de referencia pytorch-tabnet y los demás, capas estándar de PyTorch. Ningún modelo usa pesos preentrenados, de modo que no aplica la comparación con y sin preentrenamiento ni la fuga por modelos preentrenados. Las líneas base usan scikit-learn y LightGBM.

Hay una corrección respecto del Entregable 1: los tres modelos base usan ahora el *dropout* exacto del código de los autores (Zeng, 2025), que el Entregable 1 declaró como desviación y evaluó con un análisis de sensibilidad.
""")

code("s4-modelos-tabla", r"""
FUENTE = {"DNN": "Zeng (2025)", "CNN 1D": "Zeng (2025)", "LSTM": "Zeng (2025)", "RNN": "Biswas et al. (2026)", "GRU": "Biswas et al. (2026)", "BiLSTM": "Balgoon (2026)",
          "Transformer": "Biswas et al. (2026), TSLT-Net", "TabNet": "Arik y Pfister (2021); pytorch-tabnet 4.1.0", "KAN": "Bakiri et al. (2026)", "ResNet": "Búsqueda (sección 4.3)",
          "Autoencoder": "Búsqueda (sección 4.3)", "DBN": "Búsqueda (sección 4.3)", "CNN-LSTM": "Búsqueda (sección 4.3)"}
ADAPTACION = {"DNN": "Ninguna", "CNN 1D": "Ninguna", "LSTM": "Ninguna", "RNN": "Ninguna; la secuencia es la de las 16 columnas", "GRU": "Ninguna; la secuencia es la de las 16 columnas",
              "BiLSTM": "Cabeza lineal sobre los dos últimos estados, que el artículo no detalla", "Transformer": "PyTorch MultiheadAttention en lugar de Keras", "TabNet": "Entrenado con el motor común",
              "KAN": "Clasificador de 5 clases en lugar de 4 ataques", "ResNet": "Bloque residual denso", "Autoencoder": "Supervisado, λ = 1", "DBN": "CD-1, 3 épocas por RBM",
              "CNN-LSTM": "Correspondencia aproximada de 3 bloques"}
fuera_rango = float((np.abs(H["X"][ID_ENT]) > 5).mean())
tabla(complejidad_modelos.assign(Fuente=complejidad_modelos["Modelo"].map(FUENTE), Adaptación=complejidad_modelos["Modelo"].map(ADAPTACION))[["Modelo", "Familia", "Fuente", "Adaptación", "Parámetros", "FLOPs por muestra"]],
      {"Parámetros": "{:,.0f}", "FLOPs por muestra": "{:,.0f}"})
print(f"Valores estandarizados de entrenamiento fuera de [-5, 5], que la KAN recorta: {100 * fuera_rango:.2f} %")
""")

md("s4-ajuste", r"""
#### 4.2.1 Ajuste de hiperparámetros con el mismo presupuesto

Cada modelo profundo recibe las mismas 12 configuraciones de tasa de aprendizaje, *weight decay* y tamaño de lote, muestreadas con semilla 2026 de la rejilla del preregistro, con una semilla de entrenamiento y un máximo de 15 épocas. Los modelos con búsqueda de arquitectura se ajustan con la arquitectura de referencia (64, 64, 64) y *dropout* (0, 1, 0), como pide la sección 14.3, de modo que la búsqueda posterior solo cambia los anchos y el *dropout*. Los demás hiperparámetros de cada arquitectura, como el *dropout* de los modelos definidos, quedan en los valores de su fuente.
""")

code("s4-ajuste-tabla", r"""
from codigo import resultados as RES
from codigo import plan as PLAN

ajuste = RES.tabla_fase("ajuste")
PROFUNDOS = CONFIG["modelos"]["base_e1"] + CONFIG["modelos"]["top10"]
configs = PLAN.configuraciones_ajuste()["profundos"]
etiqueta_cfg = [f"{i}: lr {c['lr']:g}, wd {c['weight_decay']:g}, lote {c['batch']}" for i, c in enumerate(configs)]
pivote = ajuste[ajuste["modelo"].isin(PROFUNDOS)].pivot(index="intento", columns="modelo", values="f1_val")[PROFUNDOS]
pivote.index = etiqueta_cfg
tabla(pivote.reset_index(names="Configuración"), {m: "{:.4f}" for m in PROFUNDOS})
MEJORES_HP = PLAN.mejores_hp()
resumen_ajuste = pd.DataFrame([{"Modelo": m, "Mejor F1 de validación": ajuste.loc[ajuste["modelo"] == m, "f1_val"].max(), "Peor": ajuste.loc[ajuste["modelo"] == m, "f1_val"].min(),
                                "Rango": ajuste.loc[ajuste["modelo"] == m, "f1_val"].max() - ajuste.loc[ajuste["modelo"] == m, "f1_val"].min(),
                                "Configuración elegida": ", ".join(f"{k} {v:g}" if isinstance(v, float) else f"{k} {v}" for k, v in MEJORES_HP[m].items()),
                                "Minutos del ajuste": ajuste.loc[ajuste["modelo"] == m, "segundos"].sum() / 60, "Divergentes": int(ajuste.loc[ajuste["modelo"] == m, "divergente"].sum())}
                               for m in PROFUNDOS + ["LightGBM", "Logística"]])
tabla(resumen_ajuste, {"Mejor F1 de validación": "{:.4f}", "Peor": "{:.4f}", "Rango": "{:.4f}", "Minutos del ajuste": "{:.1f}"})
fig, eje = plt.subplots(figsize=(10, 3.6))
for j, m in enumerate(PROFUNDOS):
    v = pivote[m].to_numpy()
    eje.scatter(np.full(len(v), j) + np.linspace(-0.2, 0.2, len(v)), v, s=14, marker=MARCADORES[j % len(MARCADORES)], facecolors="none" if j % 2 else NEGRO, edgecolors=NEGRO, lw=0.7)
    eje.plot([j - 0.3, j + 0.3], [v.max()] * 2, color=GRIS_OSCURO, lw=1.2)
eje.set_xticks(range(len(PROFUNDOS)), PROFUNDOS, rotation=35, ha="right")
eje.set_ylabel("F1 macro de validación")
eje.set_title("Ajuste: las 12 configuraciones por modelo; la línea marca la elegida")
plt.tight_layout()
plt.show()
mas_sensible = resumen_ajuste.loc[resumen_ajuste["Rango"].idxmax()]
menos_sensible = resumen_ajuste.loc[resumen_ajuste.loc[resumen_ajuste["Modelo"].isin(PROFUNDOS), "Rango"].idxmin()]
lote_elegido = pd.Series({m: MEJORES_HP[m]["batch"] for m in PROFUNDOS}).value_counts()
tasa_elegida = pd.Series({m: MEJORES_HP[m]["lr"] for m in PROFUNDOS}).value_counts()
interpretar(f'''**Interpretación.** El modelo más sensible a los hiperparámetros es {mas_sensible["Modelo"]}, con un rango de {num(mas_sensible["Rango"], 4)} en F1 macro de validación entre sus 12 configuraciones, y el menos sensible es {menos_sensible["Modelo"]}, con {num(menos_sensible["Rango"], 4)}. Entre los profundos, la tasa de aprendizaje más elegida es {num(tasa_elegida.index[0], 4)} ({tasa_elegida.iloc[0]} de 13 modelos) y el lote más elegido, {lote_elegido.index[0]} ({lote_elegido.iloc[0]} de 13). {"Ninguna configuración divergió." if resumen_ajuste["Divergentes"].sum() == 0 else f"Divergieron {int(resumen_ajuste['Divergentes'].sum())} configuraciones, que se reportan y no se usan."}

Esto significa que el mismo presupuesto no produce el mismo beneficio en todos los modelos: los que tienen un rango amplio dependen más de acertar la configuración, de modo que con un presupuesto menor habrían quedado subestimados, y por eso se iguala el número de intentos y no el tiempo. Hay que tener en cuenta que el ajuste usa una sola semilla, de modo que diferencias entre configuraciones del orden del ruido por semilla, que mide el piloto, no son significativas.

Por tal motivo, se fijan para la ejecución definitiva las configuraciones de la tabla, elegidas solo con validación, y el resto del protocolo queda congelado.''')
""")

md("s4-hp", r"""
**Hiperparámetros y estrategia de entrenamiento.** La tabla reúne lo que pide la sección 4.2 del enunciado para cada modelo. Todo el entrenamiento es en precisión simple (float32) en CPU, sin precisión mixta, porque no hay GPU.
""")

code("s4-hp-tabla", r"""
DROPOUT = {"DNN": "0,2 tras las capas de 128 y 64", "CNN 1D": "0,2 antes de la densa", "LSTM": "0,2 antes de la salida", "RNN": "0,3", "GRU": "0,3 en tres posiciones", "BiLSTM": "0,3",
           "Transformer": "0,3 antes de la salida", "TabNet": "No usa; regularización de dispersión λ = 10⁻³", "KAN": "0,25 y 0,20", "ResNet": "Por búsqueda", "Autoencoder": "Por búsqueda",
           "DBN": "Por búsqueda", "CNN-LSTM": "Por búsqueda"}
hp_tabla = pd.DataFrame([{"Modelo": m, "Optimizador": "Adam", "Tasa de aprendizaje": MEJORES_HP[m]["lr"], "Planificador": "ReduceLROnPlateau (0,1; 2)", "Weight decay": MEJORES_HP[m]["weight_decay"],
                          "Lote": MEJORES_HP[m]["batch"], "Dropout": DROPOUT[m], "Pérdida": "CE ponderada" + (" + dispersión" if m == "TabNet" else " + reconstrucción" if m == "Autoencoder" else ""),
                          "Épocas máximas": CONFIG["entrenamiento"]["max_epocas"], "Parada temprana": f"Paciencia {CONFIG['entrenamiento']['paciencia']} sobre F1 de validación",
                          "Recorte de gradiente": f"Norma {CONFIG['entrenamiento']['norma_gradiente']:g}"} for m in PROFUNDOS])
tabla(hp_tabla, {"Tasa de aprendizaje": "{:g}", "Weight decay": "{:g}"})
""")

md("s4-3", r"""
### 4.3 Requisitos experimentales

**Semillas y determinismo.** Cada corrida fija las semillas de Python, NumPy y PyTorch, y el orden de los lotes sale de un generador con la misma semilla; las operaciones de PyTorch en CPU que se usan son deterministas con un número de hilos fijo, de modo que una corrida se puede repetir con la misma semilla. No se activa `torch.use_deterministic_algorithms`, porque no hay GPU. Las semillas son 4242 para el ajuste, 4101 a 4105 para el piloto, 5000 para la fase 1, 5101 a 5105 para la fase 2, 5201 a 5203 para el ruido por semilla y desde 6001 para la ejecución definitiva.

**Seguimiento de experimentos.** El ejecutor guarda cada corrida en un JSON con su configuración, semilla, curvas por época, tiempo, memoria, equipo y métricas, y la configuración del experimento está versionada en `configuracion/experimento.json`. Ese registro por corrida cumple la función de una herramienta de seguimiento como MLflow sin agregar una dependencia, y se consolida en `entregables/registro_experimentos.csv`.

**Cómputo.** Todo corre en CPU, de modo que no hay ventaja de GPU para ningún modelo; los trabajos se reparten entre procesos de 2 hilos y cada modelo recibe el mismo número de intentos, no el mismo tiempo.

#### 4.3.1 Estudio piloto y número de ejecuciones

El piloto entrena cuatro modelos representativos, el baseline del Entregable 1 (CNN 1D) y tres del Top 10 de familias distintas (Transformer, RNN y KAN), con cinco semillas y sobre validación, para estimar la desviación de las diferencias pareadas frente al baseline y fijar R con la regla del preregistro. El mismo piloto verifica la limitación del paper guía en tres modelos del Top 10 (sección 5).
""")

code("s4-piloto", r"""
ADENDA = json.loads((RESULTADOS / "preregistro_e2_adenda.json").read_text(encoding="utf-8"))
pil = ADENDA["piloto"]
tabla(pd.DataFrame(pil["f1_validacion"]).rename_axis("Semilla").reset_index(), {m: "{:.4f}" for m in pil["f1_validacion"]})
tabla(pd.DataFrame([{"Par": p, "Diferencia media": v["media"], "σ̂_d": v["sigma_d"], "Shapiro-Wilk, p": v["shapiro_p"]} for p, v in pil["pares_frente_al_baseline"].items()]),
      {"Diferencia media": "{:.4f}", "σ̂_d": "{:.4f}", "Shapiro-Wilk, p": "{:.3f}"})
R = ADENDA["R"]
SEMILLAS = [CONFIG["definitiva"]["semilla_inicial"] + i for i in range(R)]
interpretar(f'''**Interpretación.** En el piloto, la mayor desviación de las diferencias pareadas frente a CNN 1D es σ̂_d = {num(pil["sigma_piloto"], 4)}, y la regla del preregistro toma el máximo entre ella y la del Entregable 1 en prueba, {num(pil["sigma_e1_prueba"], 4)}, de modo que se usa σ̂_d = {num(pil["sigma_usada"], 4)}. Con α′ = 0,05/10 y potencia de 0,80, detectar δ exigiría R = {pil["R_calculado"]} ejecuciones; {"el tope de cómputo es " + str(CONFIG["definitiva"]["R_tope"]) + ", de modo que se usa R = " + str(R) if pil["limitado_por_tope"] else "se usa R = " + str(R)}, con una diferencia mínima detectable de {num(pil["diferencia_minima_detectable"], 4)} en cada contraste de RQ1.

Esto significa que, {"como en el Entregable 1, el diseño no alcanza a detectar con potencia de 0,80 una diferencia de magnitud δ tras la corrección de Bonferroni, y que solo diferencias de unas " + num(pil["diferencia_minima_detectable"] / DELTA, 1) + " veces δ quedarían detectadas" if pil["limitado_por_tope"] else "el diseño detecta δ con la potencia deseada"}. Hay que tener en cuenta que la cifra es conservadora, porque Holm es menos exigente que Bonferroni salvo en el primer contraste.

Por lo anterior, conviene declarar la diferencia mínima detectable como limitación y leer los resultados no significativos con TOST y con la prueba bayesiana, y no como ausencia de diferencias. Las semillas definitivas son {SEMILLAS[0]} a {SEMILLAS[-1]}, las mismas para los 16 modelos, y quedaron fijadas en la adenda del preregistro el {pil["fecha"]}.''')
""")

md("s4-busqueda", r"""
#### 4.3.2 Búsqueda de arquitectura para los modelos sin diseño definido

ResNet, Autoencoder, DBN y CNN-LSTM recorren el espacio de la sección 14: 3 bloques, anchos de {16, 32, 64, 128, 256} en todas las combinaciones y *dropout* de 0,2 tras cada bloque según b ∈ {0, 1}³, es decir, 1000 arquitecturas por modelo, generadas por código e identificadas como `w64-64-64_d010`. Los demás hiperparámetros se fijaron antes con la arquitectura de referencia (sección 4.2.1). El protocolo es el de dos fases del enunciado.

**Ruido por semilla y número de finalistas.** Antes de la fase 1 se entrenan 5 arquitecturas por modelo con 3 semillas a presupuesto de fase 1, para comparar la desviación entre semillas de una misma arquitectura con la desviación entre arquitecturas y fijar k con la regla del preregistro.
""")

code("s4-ruido", r"""
ruido = RES.tabla_fase("ruido")
tabla_ruido = ruido.groupby(["modelo", "arquitectura"])["f1_val"].agg(["mean", "std"]).reset_index()
tabla_ruido["arquitectura"] = tabla_ruido["arquitectura"].map(MOD.identificador)
tabla(tabla_ruido.rename(columns={"modelo": "Modelo", "arquitectura": "Arquitectura", "mean": "Media de F1 (3 semillas)", "std": "DE entre semillas"}),
      {"Media de F1 (3 semillas)": "{:.4f}", "DE entre semillas": "{:.4f}"})
rs = ADENDA["ruido_semilla"]
K_FINALISTAS = rs["k"]
interpretar(f'''**Interpretación.** La razón entre la desviación dentro de cada arquitectura (entre semillas) y la desviación entre arquitecturas va de {num(min(rs["razon_dentro_entre"].values()), 2)} en {min(rs["razon_dentro_entre"], key=rs["razon_dentro_entre"].get)} a {num(max(rs["razon_dentro_entre"].values()), 2)} en {max(rs["razon_dentro_entre"], key=rs["razon_dentro_entre"].get)}. Con la regla del preregistro, k = {K_FINALISTAS}.

Lo anterior indica que {"el ruido por semilla es comparable a la dispersión real entre arquitecturas en al menos un modelo, de modo que el orden de la fase 1 es ruidoso y el mejor verdadero puede quedar fuera de los finalistas; según la simulación del enunciado, con un ruido igual a la dispersión, la probabilidad de capturarlo con k = 10 es de cerca del 50 %" if K_FINALISTAS == 20 else "el ruido por semilla es pequeño frente a la dispersión entre arquitecturas, de modo que el orden de la fase 1 es informativo"}.

Por tal motivo, se conservan {K_FINALISTAS} finalistas por modelo, más la referencia (64, 64, 64) con *dropout* (0, 1, 0), y se reporta la correlación de rangos entre fases para medir cuánto informó el cribado.''')
""")

code("s4-fase1", r"""
f1 = RES.tabla_fase("fase1")
f1[["w1", "w2", "w3", "b1", "b2", "b3"]] = pd.DataFrame(f1["arquitectura"].tolist(), index=f1.index)
f1["patron"] = f1["b1"].astype(str) + f1["b2"].astype(str) + f1["b3"].astype(str)
f1.to_csv(RESULTADOS / "busqueda_fase1.csv", index=False)
CON_BUSQUEDA = CONFIG["modelos"]["con_busqueda"]
for m in CON_BUSQUEDA:
    sub = f1[f1["modelo"] == m]
    fig, ejes = plt.subplots(1, 8, figsize=(13, 3.1), sharey=True)
    lim = (sub["f1_val"].quantile(0.05), sub["f1_val"].max())
    for eje, patron in zip(ejes, sorted(sub["patron"].unique())):
        m_pat = sub[sub["patron"] == patron].pivot_table(index=["w1", "w2"], columns="w3", values="f1_val")
        imagen = eje.imshow(m_pat.to_numpy(), aspect="auto", vmin=lim[0], vmax=lim[1])
        eje.set_title(f"dropout {patron}", fontsize=8)
        eje.set_xticks(range(5), [str(w) for w in MOD.ANCHOS], fontsize=6)
        eje.set_yticks(range(25), [f"{a}-{b}" for a, b in m_pat.index], fontsize=5)
        eje.set_xlabel("w₃", fontsize=7)
        eje.grid(False)
    ejes[0].set_ylabel("w₁-w₂", fontsize=7)
    fig.colorbar(imagen, ax=ejes, fraction=0.015)
    fig.suptitle(f"{m}: F1 macro de validación en la fase 1 (1000 arquitecturas, 6 épocas, semilla 5000)", fontsize=9.5)
    plt.show()
tabla(f1.groupby("modelo")["f1_val"].describe()[["min", "25%", "50%", "75%", "max", "std"]].reset_index(names="Modelo"), {c: "{:.4f}" for c in ["min", "25%", "50%", "75%", "max", "std"]})
""")

md("s4-fase1-efectos", r"""
**Efectos principales e interacciones.** Sobre las 1000 arquitecturas de cada modelo se ajusta el modelo lineal de la sección 14.6, $y_a = \beta_0 + \sum_{i=1}^{3} \sum_{w \in W} \beta_{i,w} \mathbb{1}[w_i = w] + \sum_i \gamma_i b_i + \sum_{i<j} \lambda_{ij} b_i b_j + \varepsilon_a$, con una ANOVA de tipo II por factor y corrección de Holm por el número de factores, y la proporción de varianza que explica cada uno. Se agregan una prueba de tendencia del ancho, con la regresión sobre $\log_2 w_i$, y la comparación pareada de cada combinación de anchos sin *dropout* frente al promedio de sus siete patrones con *dropout*, con Wilcoxon sobre las 125 combinaciones. Como la fase 1 usa una sola semilla, el error del modelo mezcla el ruido por semilla con el efecto de la arquitectura.
""")

code("s4-fase1-efectos-calc", r"""
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

efectos = []
dropout_pareado = []
for m in CON_BUSQUEDA:
    sub = f1[f1["modelo"] == m].copy()
    ajuste_lm = smf.ols("f1_val ~ C(w1) + C(w2) + C(w3) + b1 + b2 + b3 + b1:b2 + b1:b3 + b2:b3", data=sub).fit()
    anova = anova_lm(ajuste_lm, typ=2)
    ss_total = ((sub["f1_val"] - sub["f1_val"].mean()) ** 2).sum()
    p_holm = multipletests(anova["PR(>F)"].iloc[:-1], method="holm")[1]
    for (factor, fila), ph in zip(anova.iloc[:-1].iterrows(), p_holm):
        efectos.append({"Modelo": m, "Factor": factor.replace("C(", "").replace(")", ""), "F": fila["F"], "p Holm": ph, "% de varianza": 100 * fila["sum_sq"] / ss_total})
    efectos.append({"Modelo": m, "Factor": "Residuo", "F": np.nan, "p Holm": np.nan, "% de varianza": 100 * anova["sum_sq"].iloc[-1] / ss_total})
    for i in (1, 2, 3):
        tendencia_w = smf.ols(f"f1_val ~ np.log2(w{i})", data=sub).fit()
        efectos.append({"Modelo": m, "Factor": f"tendencia log₂ w{i} (pendiente {tendencia_w.params.iloc[1]:+.4f})", "F": tendencia_w.fvalue, "p Holm": tendencia_w.f_pvalue, "% de varianza": 100 * tendencia_w.rsquared})
    sin = sub[sub["patron"] == "000"].set_index(["w1", "w2", "w3"])["f1_val"]
    con = sub[sub["patron"] != "000"].groupby(["w1", "w2", "w3"])["f1_val"].mean()
    d = (con - sin.reindex(con.index)).to_numpy()
    w = stats.wilcoxon(d)
    dropout_pareado.append({"Modelo": m, "Diferencia media con dropout − sin dropout": d.mean(), "Combinaciones donde ayuda": int((d > 0).sum()), "p Wilcoxon": w.pvalue,
                            "r biserial": float((stats.rankdata(np.abs(d))[d > 0].sum() - stats.rankdata(np.abs(d))[d < 0].sum()) / stats.rankdata(np.abs(d)).sum())})
efectos = pd.DataFrame(efectos)
tabla(efectos, {"F": "{:.2f}", "p Holm": "{:.2e}", "% de varianza": "{:.1f}"})
tabla(pd.DataFrame(dropout_pareado), {"Diferencia media con dropout − sin dropout": "{:+.4f}", "p Wilcoxon": "{:.2e}", "r biserial": "{:.3f}"})
""")

code("s4-fase2", r"""
f2 = RES.tabla_fase("fase2")
detalle_f2 = ADENDA["fase2"]["detalle"]
filas_f2 = []
for m in CON_BUSQUEDA:
    sub2 = f2[f2["modelo"] == m]
    agg = sub2.groupby("arquitectura").agg(media=("f1_val", "mean"), de=("f1_val", "std"), f1_ent=("f1_ent", "mean"), segundos=("segundos", "mean"), parametros=("parametros", "first")).reset_index()
    agg["brecha"] = agg["f1_ent"] - agg["media"]
    fase1_m = f1[f1["modelo"] == m].set_index("arquitectura")["f1_val"]
    agg["fase1"] = agg["arquitectura"].map(fase1_m)
    rho = stats.spearmanr(agg["fase1"], agg["media"])
    ganadora = tuple(ADENDA["arquitecturas_ganadoras"][m])
    borde = any(w in (16, 256) for w in ganadora[:3]) and len({w for w in ganadora[:3]}) == 1
    filas_f2.append({"Modelo": m, "Finalistas": len(agg), "Ganadora": MOD.identificador(ganadora), "F1 de validación (5 semillas)": float(agg.loc[agg["arquitectura"] == ganadora, "media"].iloc[0]),
                     "Parámetros": int(agg.loc[agg["arquitectura"] == ganadora, "parametros"].iloc[0]), "Mejor media": detalle_f2[m]["mejor_media"], "Empate técnico": "sí" if detalle_f2[m]["empate_tecnico"] else "no",
                     "Equivalentes por TOST": len(detalle_f2[m]["equivalentes_tost"]), "ρ de Spearman fase 1–fase 2": rho.statistic, "p": rho.pvalue,
                     "DE entre arquitecturas (fase 1)": fase1_m.std(), "DE entre semillas (fase 2)": float(np.sqrt((sub2.groupby("arquitectura")["f1_val"].var()).mean())),
                     "Brecha entrenamiento − validación": float(agg.loc[agg["arquitectura"] == ganadora, "brecha"].iloc[0])})
    fig, ejes = plt.subplots(1, 2, figsize=(9, 3.2))
    ejes[0].scatter(agg["fase1"], agg["media"], s=18, facecolors="none", edgecolors=NEGRO)
    ejes[0].set_xlabel("F1 de la fase 1 (1 semilla, 6 épocas)")
    ejes[0].set_ylabel("F1 de la fase 2 (media de 5 semillas)")
    ejes[0].set_title(f"{m}: cribado frente a confirmación, ρ = {rho.statistic:.2f}")
    orden = agg.sort_values("parametros")
    frente = orden[orden["media"] >= orden["media"].cummax()]
    ejes[1].scatter(orden["parametros"], orden["media"], s=18, facecolors="none", edgecolors=GRIS_MEDIO)
    ejes[1].plot(frente["parametros"], frente["media"], color=NEGRO, marker="o", ms=4, lw=1)
    ejes[1].set_xscale("log")
    ejes[1].set_xlabel("Parámetros")
    ejes[1].set_title("Frente de Pareto de los finalistas")
    plt.tight_layout()
    plt.show()
FASE2 = pd.DataFrame(filas_f2)
tabla(FASE2, {"F1 de validación (5 semillas)": "{:.4f}", "Parámetros": "{:,.0f}", "ρ de Spearman fase 1–fase 2": "{:.3f}", "p": "{:.3f}", "DE entre arquitecturas (fase 1)": "{:.4f}",
              "DE entre semillas (fase 2)": "{:.4f}", "Brecha entrenamiento − validación": "{:.4f}"})
costo_busqueda = pd.DataFrame({"Fase": ["Ruido por semilla", "Fase 1", "Fase 2"], "Corridas": [len(ruido), len(f1), len(f2)],
                               "Horas de proceso": [ruido["segundos"].sum() / 3600, f1["segundos"].sum() / 3600, f2["segundos"].sum() / 3600]})
tabla(costo_busqueda, {"Horas de proceso": "{:.1f}"})
""")
