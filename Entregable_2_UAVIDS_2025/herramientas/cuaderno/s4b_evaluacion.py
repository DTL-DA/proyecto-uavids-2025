from .base import code, md

md("s4-4", r"""
### 4.4 Buenas prácticas obligatorias contra la fuga de información

La tabla contrasta cada práctica de la sección 4.4 del enunciado con lo que se hace aquí; la evidencia de las verificaciones de datos está en la auditoría de la sección 2.1.4, y la prueba de sanidad con etiquetas permutadas, en la sección 7.

| Práctica | Cómo se cumple |
|---|---|
| Separar entrenamiento, validación y prueba | Partición heredada, sin bloques compartidos (sección 2.1.4) |
| Ajustar escaladores, imputadores y codificadores solo en entrenamiento | La estandarización y los pesos de clase usan solo entrenamiento, también en cada pliegue; prueba sintética aprobada |
| Evitar mezcla temporal o espacial | No aplica: no hay marca de tiempo ni coordenadas, y la partición agrupa por serie |
| Evitar fuga de duplicados entre particiones | Cero filas idénticas entre particiones; casi duplicados de prueba por debajo de la tolerancia y evaluados en la sección 4.5 |
| No usar la prueba para seleccionar hiperparámetros, arquitecturas ni umbrales | Toda selección usa validación; el umbral de decisión es el arg max de las probabilidades, sin ajustar |
| Aplicar la misma validación a todos los modelos | Misma partición, mismas semillas y mismo presupuesto de ajuste |
| Ejecutar la auditoría formal de la sección 10 | Secciones 2.1.4 y 7 |
""")

md("s4-5", r"""
### 4.5 Evaluación

Cada corrida definitiva evalúa la prueba una sola vez, con el modelo de la mejor época de validación, y guarda sus probabilidades. Aquí se calculan, para cada uno de los 16 modelos y sus R semillas, las métricas declaradas a priori, las curvas de entrenamiento, el costo, la brecha entre entrenamiento, validación y prueba, la calibración, los errores y el desempeño por segmentos. Las pruebas de hipótesis entre modelos están en la sección 6.
""")

code("s4-5-metricas", r"""
from codigo import analisis as AN

METRICAS, CONF_BLOQUES = AN.metricas_definitiva()
DEF = [r for r in RES.registros("definitiva")]
ORDEN = PROFUNDOS + ["LightGBM", "Logística", "Trivial"]
# El trivial es determinista: se repite su única corrida en todas las semillas para las pruebas pareadas
trivial = METRICAS[METRICAS["modelo"] == "Trivial"].iloc[0]
METRICAS = pd.concat([METRICAS[METRICAS["modelo"] != "Trivial"], pd.DataFrame([trivial.to_dict() | {"semilla": s} for s in SEMILLAS])], ignore_index=True)
METRICAS = METRICAS[METRICAS["semilla"].isin(SEMILLAS)]
assert METRICAS.groupby("modelo").size().eq(R).all(), "Faltan corridas definitivas"
def resumen(col):
    g = METRICAS.groupby("modelo")[col]
    media, de = g.mean(), g.std()
    margen = stats.t.ppf(0.975, R - 1) * de / np.sqrt(R)
    return media, de, media - margen, media + margen
NOMBRES_M = ["F1 macro", "Accuracy", "Balanced accuracy", "Precision macro", "Recall macro", "F1 ponderado", "AUC-ROC macro", "AUC-PR macro", "MCC", "Log-loss", "Brier", "ECE", "ECE adaptativo",
             "Confusión Blackhole–Wormhole"]
tabla_metricas = {}
for nombre in NOMBRES_M:
    media, de, bajo, alto = resumen(f"prueba {nombre}")
    tabla_metricas[nombre] = {m: f"{num(media[m], 4)} ± {num(de[m], 4)} [{num(bajo[m], 4)}; {num(alto[m], 4)}]" for m in ORDEN}
tabla(pd.DataFrame(tabla_metricas).loc[ORDEN].reset_index(names="Modelo"))
F1P = RES.matriz(METRICAS, "prueba F1 macro")[ORDEN]
F1V = RES.matriz(METRICAS, "validación F1 macro")[ORDEN]
TOP3 = F1V[PROFUNDOS].mean().sort_values(ascending=False).index[:3].tolist()
MEJOR_DL = TOP3[0]
print("Tres mejores modelos profundos por F1 macro medio de validación:", TOP3)
""")

code("s4-5-curvas", r"""
cur = RES.curvas("definitiva")
fig, ejes = plt.subplots(3, 5, figsize=(13, 7.2), sharey=False)
for eje, m in zip(ejes.ravel(), PROFUNDOS):
    c = cur[cur["modelo"] == m]
    agg = c.groupby("epoca")[["perdida_ent", "perdida_val", "f1_val"]].mean()
    eje.plot(agg.index, agg["perdida_ent"], color=NEGRO, lw=1, label="Pérdida de entrenamiento")
    eje.plot(agg.index, agg["perdida_val"], color=GRIS_OSCURO, ls="--", lw=1, label="Pérdida de validación")
    eje2 = eje.twinx()
    eje2.plot(agg.index, agg["f1_val"], color=GRIS_MEDIO, ls=":", lw=1.4, label="F1 de validación")
    eje2.tick_params(labelsize=6)
    eje.set_title(m, fontsize=8.5)
    eje.tick_params(labelsize=6)
for eje in ejes.ravel()[len(PROFUNDOS):]:
    eje.axis("off")
ejes[0, 0].legend(fontsize=6, loc="upper right")
fig.suptitle("Curvas medias de las R semillas: pérdida (eje izquierdo) y F1 macro de validación (eje derecho, línea punteada)", fontsize=10)
plt.tight_layout()
plt.show()
costo = METRICAS.groupby("modelo").agg(Parámetros=("parametros", "first"), MAC=("mac", "first"), Épocas=("epocas", "mean"), Mejor_época=("mejor_epoca", "mean"),
                                       Minutos=("segundos", lambda s: s.mean() / 60), ms_por_muestra=("ms_por_muestra", "median"), ms_lote=("ms_por_muestra_lote", "median"),
                                       Memoria_MB=("memoria_pico_mb", "median")).loc[ORDEN]
costo["FLOPs"] = 2 * costo["MAC"]
costo["Muestras por segundo"] = 1000 / costo["ms_lote"]
ENERGIA_W = 15.0
costo["kWh estimados (R corridas)"] = costo["Minutos"] * 60 * R * ENERGIA_W / 3.6e6 * 2 / 10
tabla(costo.reset_index(names="Modelo").rename(columns={"Mejor_época": "Mejor época", "ms_por_muestra": "Latencia por muestra (ms)", "ms_lote": "ms por muestra en lote", "Memoria_MB": "Memoria pico del proceso (MB)"}),
      {"Parámetros": "{:,.0f}", "MAC": "{:,.0f}", "FLOPs": "{:,.0f}", "Épocas": "{:.1f}", "Mejor época": "{:.1f}", "Minutos": "{:.2f}", "Latencia por muestra (ms)": "{:.3f}",
       "ms por muestra en lote": "{:.4f}", "Memoria pico del proceso (MB)": "{:.0f}", "Muestras por segundo": "{:,.0f}", "kWh estimados (R corridas)": "{:.4f}"})
print(f"Hardware: {platform.processor()}, {torch.get_num_threads()} hilos disponibles; cada corrida usó 2 hilos. Energía estimada con la potencia base del procesador ({ENERGIA_W:g} W) repartida entre los 10 núcleos.")
""")

md("s4-5-ajuste", r"""
**Ajuste, sobreajuste y consistencia entre validación y prueba.** Para cada modelo se calcula la brecha entre entrenamiento y validación, $g = s_{ent} - s_{val}$, y se prueba la hipótesis de no inferioridad de la tabla B: $H_0$: la mediana de $g$ es al menos δg = 0,05, frente a $H_1$: es menor, con Wilcoxon unilateral sobre $g - δ_g$ en las R semillas. La diferencia entre validación y prueba se estima con su IC 95 % por *bootstrap* por bloques, remuestreando por separado los bloques de cada conjunto, porque son series distintas.
""")

code("s4-5-brechas", r"""
NB_VAL = len(np.unique(bloques[ID_VAL]))
NB_PRU = len(np.unique(bloques[ID_PRU]))
W_VAL, W_PRU = AN.pesos_bootstrap(NB_VAL, semilla=1), AN.pesos_bootstrap(NB_PRU, semilla=2)
def f1_bootstrap(modelo, conjunto):
    filas_m = METRICAS[METRICAS["modelo"] == modelo]
    W = W_VAL if conjunto == "val" else W_PRU
    return np.mean([AN.f1_desde_conteos(W @ CONF_BLOQUES[f"{i}|{conjunto}"]) for i in filas_m["id"]], 0)
brechas = []
for m in ORDEN:
    sub_m = METRICAS[METRICAS["modelo"] == m]
    g = (sub_m["f1_ent"] - sub_m["validación F1 macro"]).to_numpy()
    p_g = stats.wilcoxon(g - DELTA_G, alternative="less").pvalue if np.any(g != DELTA_G) else 1.0
    dif_boot = f1_bootstrap(m, "val") - f1_bootstrap(m, "eva")
    brechas.append({"Modelo": m, "F1 de entrenamiento": sub_m["f1_ent"].mean(), "F1 de validación": sub_m["validación F1 macro"].mean(), "F1 de prueba": sub_m["prueba F1 macro"].mean(),
                    "Brecha g": g.mean(), "p Wilcoxon (g < δg)": p_g, "Validación − prueba": sub_m["validación F1 macro"].mean() - sub_m["prueba F1 macro"].mean(),
                    "IC 95 % por bloques": intervalo(E.ic_percentil(dif_boot))})
BRECHAS = pd.DataFrame(brechas)
tabla(BRECHAS, {"F1 de entrenamiento": "{:.4f}", "F1 de validación": "{:.4f}", "F1 de prueba": "{:.4f}", "Brecha g": "{:+.4f}", "p Wilcoxon (g < δg)": "{:.2e}", "Validación − prueba": "{:+.4f}"})
""")

code("s4-5-calibracion", r"""
calib = []
for m in ORDEN:
    if m == "Trivial":
        continue
    for i in METRICAS.loc[METRICAS["modelo"] == m, "id"]:
        r = next(x for x in DEF if x["id"] == i)
        p_val, p_eva = AN.prob(r, "val"), AN.prob(r, "eva")
        T = E.escalar_temperatura(E.logits_de(p_val), y[ID_VAL])
        p_cal = E.suavizar(p_eva, T)
        calib.append({"modelo": m, "semilla": r["semilla"], "T": T, "ECE": E.ece(p_eva, y[ID_PRU]), "ECE con T": E.ece(p_cal, y[ID_PRU]), "ECE adaptativo": E.ece(p_eva, y[ID_PRU], adaptativo=True),
                      "ECE adaptativo con T": E.ece(p_cal, y[ID_PRU], adaptativo=True), "Brier": E.brier(p_eva, y[ID_PRU]), "Brier con T": E.brier(p_cal, y[ID_PRU])})
CALIB = pd.DataFrame(calib)
tabla(CALIB.groupby("modelo").mean(numeric_only=True).drop(columns="semilla").loc[[m for m in ORDEN if m != "Trivial"]].reset_index(names="Modelo"), {c: "{:.4f}" for c in ["T", "ECE", "ECE con T", "ECE adaptativo", "ECE adaptativo con T", "Brier", "Brier con T"]})
fig, ejes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for eje, m in zip(ejes, TOP3):
    r = next(x for x in DEF if x["modelo"] == m and x["semilla"] == SEMILLAS[0])
    p_eva = AN.prob(r, "eva")
    T = E.escalar_temperatura(E.logits_de(AN.prob(r, "val")), y[ID_VAL])
    for p, estilo, etq in ((p_eva, dict(color=NEGRO, marker="o"), "Original"), (E.suavizar(p_eva, T), dict(color=GRIS_MEDIO, marker="s", ls="--"), f"Con T = {T:.2f}")):
        conf, acierto = p.max(1), p.argmax(1) == y[ID_PRU]
        cortes = np.quantile(conf, np.linspace(0, 1, 16))
        idx = np.clip(np.digitize(conf, cortes[1:-1]), 0, 14)
        eje.plot([conf[idx == b].mean() for b in range(15)], [acierto[idx == b].mean() for b in range(15)], ms=3.5, lw=1, label=etq, mfc="white", **estilo)
    eje.plot([0, 1], [0, 1], color=GRIS_CLARO, lw=1)
    eje.set(title=m, xlabel="Confianza media del bin", xlim=(0.2, 1), ylim=(0.2, 1))
    eje.legend(fontsize=7)
ejes[0].set_ylabel("Exactitud del bin")
fig.suptitle("Diagramas de confiabilidad en prueba de los tres mejores modelos por validación (primera semilla, 15 bins de igual tamaño)", fontsize=9.5)
plt.tight_layout()
plt.show()
""")

code("s4-5-errores", r"""
CATEGORIAS = {"Confusión Blackhole–Wormhole": lambda v, p: ((v == 0) & (p == 4)) | ((v == 4) & (p == 0)), "Ataque predicho como Normal": lambda v, p: (v != 2) & (p == 2),
              "Normal predicho como ataque": lambda v, p: (v == 2) & (p != 2)}
def categorias_error(v, p):
    error = v != p
    otras = error.copy()
    salida = {}
    for nombre, f in CATEGORIAS.items():
        salida[nombre] = f(v, p) & error
        otras &= ~salida[nombre]
    salida["Otras confusiones entre ataques"] = otras
    return salida, error
y_pru = y[ID_PRU]
b_pru = bloques[ID_PRU]
taxonomia, errores_por_modelo = [], {}
for m in TOP3 + ["LightGBM"]:
    r = next(x for x in DEF if x["modelo"] == m and x["semilla"] == SEMILLAS[0])
    p = AN.prob(r, "eva").argmax(1)
    cats, error = categorias_error(y_pru, p)
    errores_por_modelo[m] = error
    for nombre, mascara in cats.items():
        boot = E.bootstrap_bloques(lambda ix: mascara[ix].sum() / max(error[ix].sum(), 1), b_pru, repeticiones=2000)
        taxonomia.append({"Modelo": m, "Categoría": nombre, "Errores": int(mascara.sum()), "% de los errores": 100 * mascara.sum() / error.sum(), "IC 95 % por bloques": intervalo(100 * np.array(E.ic_percentil(boot)), 1)})
tabla(pd.DataFrame(taxonomia), {"% de los errores": "{:.1f}"})
todos = np.mean([errores_por_modelo[m] for m in TOP3], 0)
compartidos = float((todos == 1).sum() / max((todos > 0).sum(), 1))
correlacion = pd.DataFrame(np.corrcoef([errores_por_modelo[m].astype(float) for m in errores_por_modelo]), index=list(errores_por_modelo), columns=list(errores_por_modelo))
tabla(correlacion.reset_index(names="Correlación de los errores (φ)"), {c: "{:.3f}" for c in correlacion.columns})
promedio_top3 = np.mean([AN.prob(next(x for x in DEF if x["modelo"] == m and x["semilla"] == SEMILLAS[0]), "eva") for m in TOP3], 0)
f1_ensamble = E.f1_macro_rapido(y_pru, promedio_top3.argmax(1))
oraculo = 1 - (todos == 1).mean()
print(f"Errores compartidos por los tres mejores: {100 * compartidos:.1f} % de los flujos que falla al menos uno | F1 del promedio de los tres: {f1_ensamble:.4f} | exactitud del oráculo: {oraculo:.4f}")
fig, ejes = plt.subplots(1, 3, figsize=(11.5, 3.5))
for eje, m in zip(ejes, TOP3):
    conf_total = sum(np.bincount(K * y_pru + AN.prob(next(x for x in DEF if x["modelo"] == m and x["semilla"] == s), "eva").argmax(1), minlength=K * K).reshape(K, K) for s in SEMILLAS)
    normalizada = conf_total / conf_total.sum(1, keepdims=True)
    eje.imshow(normalizada, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            eje.text(j, i, f"{100 * normalizada[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if normalizada[i, j] > 0.5 else NEGRO)
    eje.set_xticks(range(K), CLASES, rotation=45, fontsize=7)
    eje.set_yticks(range(K), CLASES, fontsize=7)
    eje.set_title(f"{m} (% por fila, {R} semillas)", fontsize=8.5)
    eje.grid(False)
plt.tight_layout()
plt.show()
""")

code("s4-5-segmentos", r"""
from sklearn.metrics import f1_score

cortes_flujo = np.quantile(H["tx_packets"][ID_ENT], [0.25, 0.5, 0.75])
banda = np.digitize(H["tx_packets"][ID_PRU], cortes_flujo)
puerto654 = datos["SrcPort"].to_numpy()[ID_PRU] == 654
casi_dup = np.load(ANALISIS / "prueba_con_casi_duplicado.npy")
segmentos = []
for m in ORDEN:
    if m == "Trivial":
        continue
    preds = [AN.prob(next(x for x in DEF if x["modelo"] == m and x["semilla"] == s), "eva").argmax(1) for s in SEMILLAS]
    fila = {"Modelo": m}
    por_clase = np.mean([f1_score(y_pru, p, average=None, labels=range(K)) for p in preds], 0)
    for c in range(K):
        fila[f"F1 {CLASES[c]}"] = por_clase[c]
    for b, nombre in enumerate(["Muy cortos", "Cortos", "Medianos", "Largos"]):
        fila[nombre] = np.mean([E.f1_macro_rapido(y_pru[banda == b], p[banda == b]) for p in preds])
    fila["Con puerto 654"] = np.mean([(p[puerto654] == y_pru[puerto654]).mean() for p in preds])
    fila["Sin puerto 654"] = np.mean([(p[~puerto654] == y_pru[~puerto654]).mean() for p in preds])
    fila["Sin casi duplicado"] = np.mean([E.f1_macro_rapido(y_pru[~casi_dup], p[~casi_dup]) for p in preds])
    fila["Con casi duplicado"] = np.mean([E.f1_macro_rapido(y_pru[casi_dup], p[casi_dup]) for p in preds])
    segmentos.append(fila)
SEGMENTOS = pd.DataFrame(segmentos)
tabla(SEGMENTOS, {c: "{:.4f}" for c in SEGMENTOS.columns if c != "Modelo"})
f1_bloque = {m: np.array([[AN.f1_desde_conteos(CONF_BLOQUES[f"{i}|eva"][b]) for b in range(NB_PRU)] for i in METRICAS.loc[METRICAS["modelo"] == m, "id"]]).mean(0) for m in ORDEN if m != "Trivial"}
fig, eje = plt.subplots(figsize=(11, 3.4))
eje.boxplot([f1_bloque[m] for m in f1_bloque], labels=list(f1_bloque), medianprops=dict(color=NEGRO), flierprops=dict(marker="o", markersize=3, markerfacecolor="white"))
eje.set_ylabel("F1 macro por bloque de prueba")
eje.set_title(f"Dispersión del F1 macro entre los {NB_PRU} bloques de prueba (media de las R semillas)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
""")

md("s4-6", r"""
### 4.6 Robustez, interpretabilidad y ablación

**Robustez.** Cada modelo, con sus primeras cinco semillas, se evalúa sobre la prueba con tres perturbaciones controladas y al menos tres niveles de severidad, definidas en el preregistro: ruido gaussiano en la escala estandarizada, atributos faltantes imputados con la media de entrenamiento y escalamiento multiplicativo de los atributos crudos, que simula una deriva en la medición. El índice de robustez es el área bajo la curva de degradación del F1 macro, normalizada por el F1 sin perturbar y por el rango de severidades, de modo que 1 indica que el modelo no se degrada.
""")

code("s4-6-robustez", r"""
PERT = AN.perturbaciones()
limpio = PERT[PERT["perturbacion"] == "Sin perturbar"].groupby("modelo")["f1"].mean()
indices_rob = {}
fig, ejes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
for eje, tipo in zip(ejes, AN.NIVELES):
    for j, m in enumerate([x for x in ORDEN if x != "Trivial"]):
        sub_p = PERT[(PERT["perturbacion"] == tipo) & (PERT["modelo"] == m)].groupby("nivel")["f1"].mean()
        niveles = np.r_[0.0, sub_p.index.to_numpy()]
        curva = np.r_[limpio[m], sub_p.to_numpy()]
        indices_rob[(m, tipo)] = float(np.trapezoid(curva, niveles) / (limpio[m] * (niveles[-1] - niveles[0])))
        if m in TOP3 + ["LightGBM", "DNN"]:
            eje.plot(niveles, curva, ls=LINEAS[j % 4], marker=MARCADORES[j % 8], ms=4, color=[NEGRO, GRIS_OSCURO, GRIS_MEDIO][j % 3], mfc="white" if j % 2 else None, lw=1, label=m)
    eje.set(title=tipo, xlabel="Severidad")
    eje.legend(fontsize=6.5)
ejes[0].set_ylabel("F1 macro en prueba (media de 5 semillas)")
plt.tight_layout()
plt.show()
ROBUSTEZ = pd.DataFrame({tipo: {m: indices_rob[(m, tipo)] for m in limpio.index} for tipo in AN.NIVELES})
ROBUSTEZ["Índice medio"] = ROBUSTEZ.mean(1)
tabla(ROBUSTEZ.loc[[m for m in ORDEN if m in ROBUSTEZ.index]].reset_index(names="Modelo"), {c: "{:.4f}" for c in ROBUSTEZ.columns})
""")

md("s4-6-interp", r"""
**Interpretabilidad.** A los tres mejores modelos por validación se aplican gradientes integrados (Sundararajan et al., 2017), $IG_j(\mathbf{x}) = (x_j - x'_j) \int_0^1 \frac{\partial f_c(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_j} d\alpha$, con la base $\mathbf{x}'$ en la media de entrenamiento y 32 pasos, sobre 1500 flujos de prueba estratificados por clase y cinco semillas. La **fidelidad** se verifica con una prueba de borrado: se anulan los atributos en el orden de su atribución y se compara la caída de la probabilidad de la clase predicha con la de un orden aleatorio; una explicación fiel produce una caída más rápida. La **estabilidad** se mide con la correlación de Spearman entre los vectores de importancia de cada par de semillas.
""")

code("s4-6-interp-calc", r"""
ATR = AN.atribuciones(TOP3)
filas_i, estabilidad = [], {}
fig, ejes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
for eje, m in zip(ejes, TOP3):
    imp = ATR[f"{m}|importancia"]
    imp_n = imp / imp.sum(1, keepdims=True)
    rhos = [stats.spearmanr(imp[a], imp[b]).statistic for a in range(len(imp)) for b in range(a + 1, len(imp))]
    estabilidad[m] = (np.mean(rhos), np.min(rhos))
    borr, alea = ATR[f"{m}|borrado"].mean(0), ATR[f"{m}|aleatorio"].mean(0)
    k = np.arange(len(borr))
    eje.plot(k, borr, color=NEGRO, marker="o", ms=3, label="Orden de la atribución")
    eje.plot(k, alea, color=GRIS_MEDIO, ls="--", marker="s", ms=3, mfc="white", label="Orden aleatorio")
    aopc = float((alea - borr).mean())
    eje.set(title=f"{m}: área entre curvas {aopc:.3f}", xlabel="Atributos anulados")
    eje.legend(fontsize=7)
    orden_imp = np.argsort(-imp_n.mean(0))
    filas_i.append({"Modelo": m, "Estabilidad media (ρ)": estabilidad[m][0], "Estabilidad mínima (ρ)": estabilidad[m][1], "Área entre curvas": aopc,
                    "Tres atributos más importantes": ", ".join(ATRIBUTOS[j] for j in orden_imp[:3])})
ejes[0].set_ylabel("Probabilidad media de la clase predicha")
plt.tight_layout()
plt.show()
INTERP = pd.DataFrame(filas_i)
tabla(INTERP, {"Estabilidad media (ρ)": "{:.3f}", "Estabilidad mínima (ρ)": "{:.3f}", "Área entre curvas": "{:.3f}"})
importancia_tabla = pd.DataFrame({m: (ATR[f"{m}|importancia"] / ATR[f"{m}|importancia"].sum(1, keepdims=True)).mean(0) for m in TOP3}, index=ATRIBUTOS)
tabla(importancia_tabla.sort_values(TOP3[0], ascending=False).reset_index(names="Atributo"), {m: "{:.3f}" for m in TOP3})
""")

md("s4-6-ablacion", r"""
**Ablación.** Sobre el mejor modelo por validación se aíslan tres componentes del pipeline común, con las mismas R semillas: la ponderación de la pérdida por clase, la transformación $\log(1 + x)$ y el *dropout*. Cada variante cambia un solo componente, y la diferencia con el modelo completo se prueba con la *t* pareada, Wilcoxon y TOST sobre las semillas.
""")

code("s4-6-ablacion-calc", r"""
abl = RES.tabla_fase("ablacion")
completo = METRICAS[METRICAS["modelo"] == MEJOR_DL].set_index("semilla")["prueba F1 macro"]
filas_a = []
for variante in sorted(abl["variante_nombre"].unique()):
    sub_a = abl[abl["variante_nombre"] == variante].set_index("semilla")
    comunes = sorted(set(sub_a.index) & set(completo.index))
    r_ab = E.comparar_par(sub_a.loc[comunes, "f1_eva"].to_numpy(), completo.loc[comunes].to_numpy())
    filas_a.append({"Variante": variante, "Semillas": len(comunes), "F1 macro de prueba": sub_a.loc[comunes, "f1_eva"].mean(), "Diferencia con el completo": r_ab["Diferencia media"],
                    "IC 95 %": intervalo(r_ab["IC 95 % t"] if r_ab["Shapiro-Wilk p"] >= ALFA else r_ab["IC 95 % bootstrap"]), "p t": r_ab["p t pareada"], "p Wilcoxon": r_ab["p Wilcoxon"],
                    "p TOST": r_ab["p TOST"], "d de Cohen": r_ab["d de Cohen pareada"]})
ABLACION = pd.DataFrame(filas_a)
ABLACION[["p t Holm", "p Wilcoxon Holm"]] = np.column_stack([E.holm(ABLACION["p t"]), E.holm(ABLACION["p Wilcoxon"])])
tabla(ABLACION, {"F1 macro de prueba": "{:.4f}", "Diferencia con el completo": "{:+.4f}", "p t": "{:.4f}", "p Wilcoxon": "{:.4f}", "p TOST": "{:.4f}", "d de Cohen": "{:.2f}", "p t Holm": "{:.4f}", "p Wilcoxon Holm": "{:.4f}"})
""")
