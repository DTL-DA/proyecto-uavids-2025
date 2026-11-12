from .base import code, md

md("s6", r"""
## 6. Benchmark final

En esta sección se reúnen los resultados en la tabla de la plantilla A.2, se ordenan los modelos por desempeño y costo, se aplican las pruebas de hipótesis de las tres RQ confirmatorias, se estima el techo de desempeño y la incertidumbre, y se discuten los resultados. Las observaciones pareadas son las R semillas, que comparten partición, prueba y orden de los datos entre modelos.

### 6.1 Comparación tabular
""")

code("s6-tabla", r"""
W_PRU_PAREADO = AN.pesos_bootstrap(NB_PRU, semilla=7)
F1_BOOT = {m: np.mean([AN.f1_desde_conteos(W_PRU_PAREADO @ CONF_BLOQUES[f"{i}|eva"]) for i in METRICAS.loc[METRICAS["modelo"] == m, "id"]], 0) for m in ORDEN}
rangos = pd.Series(E.rangos_medios(F1P[ORDEN].to_numpy()), index=ORDEN)
ece_m = METRICAS.groupby("modelo")["prueba ECE"].mean()
tabla_a2 = pd.DataFrame({"Grupo": [("Trivial" if m == "Trivial" else "Clásico" if m in ("LightGBM", "Logística") else "Base E1" if m in CONFIG["modelos"]["base_e1"] else "Top 10") for m in ORDEN],
                         "F1 macro (media ± DE)": [f"{num(F1P[m].mean(), 4)} ± {num(F1P[m].std(), 4)}" for m in ORDEN],
                         "IC 95 % (bootstrap por bloques)": [intervalo(E.ic_percentil(F1_BOOT[m])) for m in ORDEN], "ECE": [ece_m[m] for m in ORDEN],
                         "Robustez (índice)": [ROBUSTEZ.loc[m, "Índice medio"] if m in ROBUSTEZ.index else np.nan for m in ORDEN],
                         "Parámetros": [costo.loc[m, "Parámetros"] for m in ORDEN], "FLOPs": [costo.loc[m, "FLOPs"] for m in ORDEN], "Latencia (ms)": [costo.loc[m, "ms_por_muestra"] for m in ORDEN],
                         "Rango promedio": [rangos[m] for m in ORDEN]}, index=ORDEN)
TABLA_A2 = tabla_a2.sort_values("Rango promedio")
TABLA_A2.to_csv(RESULTADOS / "tabla_benchmark.csv", encoding="utf-8-sig")
tabla(TABLA_A2.reset_index(names="Modelo"), {"ECE": "{:.4f}", "Robustez (índice)": "{:.3f}", "Parámetros": "{:,.0f}", "FLOPs": "{:,.0f}", "Latencia (ms)": "{:.3f}", "Rango promedio": "{:.2f}"})
""")

md("s6-2", r"""
### 6.2 Ranking y eficiencia

Se ordenan los modelos por su rango promedio sobre las semillas y se identifica la frontera de Pareto entre el F1 macro y cada medida de costo: un modelo es no dominado si ningún otro tiene mayor F1 con menor o igual costo. El costo por unidad de mejora relaciona el incremento de F1 frente al mejor baseline no profundo con el incremento de costo, $\eta_k = (s_k - s_{base}) / (c_k - c_{base})$, con su intervalo por *bootstrap* por bloques.
""")

code("s6-pareto", r"""
def no_dominados(score, costo_v):
    orden = np.argsort(costo_v)
    frente, mejor = [], -np.inf
    for i in orden:
        if score[i] > mejor:
            frente.append(i)
            mejor = score[i]
    return frente
modelos_p = [m for m in ORDEN if m != "Trivial"]
fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
pareto = {}
for eje, (col, titulo) in zip(ejes, (("Parámetros", "Parámetros"), ("FLOPs", "FLOPs por muestra"), ("Latencia (ms)", "Latencia por muestra (ms)"))):
    x = TABLA_A2.loc[modelos_p, col].astype(float).to_numpy()
    s = np.array([F1P[m].mean() for m in modelos_p])
    frente = no_dominados(s, x)
    pareto[col] = [modelos_p[i] for i in frente]
    eje.scatter(x, s, s=22, facecolors="white", edgecolors=GRIS_MEDIO)
    eje.plot(x[frente], s[frente], color=NEGRO, marker="o", ms=5, lw=1)
    for i, m in enumerate(modelos_p):
        eje.annotate(m, (x[i], s[i]), fontsize=6, xytext=(3, 2), textcoords="offset points", color=NEGRO if i in frente else GRIS_OSCURO)
    eje.set_xscale("log")
    eje.set(xlabel=titulo)
ejes[0].set_ylabel("F1 macro de prueba (media)")
fig.suptitle("Frente de Pareto entre desempeño y costo: los modelos sobre la línea no están dominados", fontsize=10)
plt.tight_layout()
plt.show()
mejor_base = max(["LightGBM", "Logística"], key=lambda m: F1P[m].mean())
eficiencia = []
for m in PROFUNDOS:
    ganancia = F1_BOOT[m] - F1_BOOT[mejor_base]
    d_lat = costo.loc[m, "ms_por_muestra"] - costo.loc[mejor_base, "ms_por_muestra"]
    eficiencia.append({"Modelo": m, "Ganancia de F1 frente a " + mejor_base: F1P[m].mean() - F1P[mejor_base].mean(), "IC 95 % de la ganancia": intervalo(E.ic_percentil(ganancia)),
                       "Δ latencia (ms)": d_lat, "η: F1 por ms adicional": (F1P[m].mean() - F1P[mejor_base].mean()) / d_lat if d_lat else np.nan})
tabla(pd.DataFrame(eficiencia), {"Ganancia de F1 frente a " + mejor_base: "{:+.4f}", "Δ latencia (ms)": "{:+.3f}", "η: F1 por ms adicional": "{:+.4f}"})
print({k: v for k, v in pareto.items()})
""")

md("s6-3", r"""
### 6.3 Análisis estadístico y pruebas de hipótesis

**Verificación de supuestos.** Antes de elegir la prueba se verifican la normalidad de las diferencias pareadas con Shapiro-Wilk, la homogeneidad de varianzas entre modelos con Brown-Forsythe y la esfericidad con Mauchly. La independencia se cumple a nivel de semilla, la unidad de las pruebas pareadas: cada semilla es una corrida independiente sobre la misma partición. Las R corridas comparten el conjunto de prueba, de modo que las inferencias valen para esta prueba y no para otra muestra de series; la variabilidad por partición se mide aparte con la validación cruzada 5 × 2.
""")

code("s6-supuestos", r"""
M_ALL = F1P[ORDEN].to_numpy()
levene = stats.levene(*[F1P[m].to_numpy() for m in ORDEN if m != "Trivial"], center="median")
mauchly = E.mauchly(F1P[[m for m in ORDEN if m != "Trivial"]].to_numpy())
base = CONFIG["modelos"]["baseline_e1"]
shapiro_pares = {m: stats.shapiro(F1P[m] - F1P[base]).pvalue for m in CONFIG["modelos"]["top10"]}
tabla(pd.DataFrame({"Supuesto": ["Normalidad de las diferencias frente a CNN 1D (Shapiro-Wilk)", "Homogeneidad de varianzas (Brown-Forsythe)", "Esfericidad (Mauchly)"],
                    "Resultado": [f"{sum(p >= ALFA for p in shapiro_pares.values())} de 10 contrastes no rechazan; p de {num(min(shapiro_pares.values()), 3)} a {num(max(shapiro_pares.values()), 3)}",
                                  f"W = {num(levene.statistic, 3)}, {valor_p(levene.pvalue)}", f"W = {num(mauchly['W'], 4)}, {valor_p(mauchly['p']) if np.isfinite(mauchly['p']) else 'no calculable con R semillas frente a 15 contrastes'}"],
                    "Consecuencia": ["Prueba (i): t pareada donde no se rechaza; bootstrap pareado donde sí", "Las pruebas pareadas no suponen varianzas iguales entre modelos",
                                     "La ANOVA de medidas repetidas se corrige con Greenhouse-Geisser"]}))
""")

md("s6-rq1", r"""
**RQ1: el Top 10 frente al baseline del Entregable 1.** Cada modelo del Top 10 se compara con CNN 1D, reentrenada en este entregable con el mismo protocolo, con las tres pruebas preregistradas y la corrección de Holm dentro de la familia de 10 contrastes. La tabla sigue la plantilla A.3 del enunciado.
""")

code("s6-rq1-calc", r"""
def contrastes(pares, familia):
    filas_c = []
    for a, b in pares:
        r_c = E.comparar_par(F1P[a].to_numpy(), F1P[b].to_numpy())
        boot = F1_BOOT[a] - F1_BOOT[b]
        filas_c.append({"Contraste": f"{a} − {b}", "a": a, "b": b, **r_c, "IC 95 % por bloques": E.ic_percentil(boot)})
    t = pd.DataFrame(filas_c)
    t["p (i)"] = np.where(t["Shapiro-Wilk p"] >= ALFA, t["p t pareada"], t["p bootstrap"])
    t["Prueba (i)"] = np.where(t["Shapiro-Wilk p"] >= ALFA, "t pareada", "bootstrap pareado")
    t["p (i) Holm"] = E.holm(t["p (i)"])
    t["p Wilcoxon Holm"] = E.holm(t["p Wilcoxon"])
    t["Decisión"] = [E.decision(fila, fila["p (i) Holm"], fila["p Wilcoxon Holm"]) for _, fila in t.iterrows()]
    t["Familia"] = familia
    return t
RQ1 = contrastes([(m, base) for m in CONFIG["modelos"]["top10"]], "F1")
def presentar(t):
    return pd.DataFrame({"Contraste": t["Contraste"], "Prueba (i)": t["Prueba (i)"], "Diferencia": t["Diferencia media"],
                         "IC 95 % (semillas)": [intervalo(f["IC 95 % t"] if f["Shapiro-Wilk p"] >= ALFA else f["IC 95 % bootstrap"]) for _, f in t.iterrows()],
                         "IC 95 % (bloques)": [intervalo(x) for x in t["IC 95 % por bloques"]], "d de Cohen": t["d de Cohen pareada"], "δ de Cliff": t["δ de Cliff"],
                         "p (i)": t["p (i)"].map(p_exacto), "p (i) Holm": t["p (i) Holm"].map(p_exacto), "p Wilcoxon": t["p Wilcoxon"].map(p_exacto), "p Wilcoxon Holm": t["p Wilcoxon Holm"].map(p_exacto),
                         "p TOST": t["p TOST"].map(p_exacto), "P(ROPE) bayesiana": t["P(ROPE)"], "Potencia para δ": t["Potencia para δ"], "Decisión frente a δ": t["Decisión"]})
tabla(presentar(RQ1), {"Diferencia": "{:+.4f}", "d de Cohen": "{:.2f}", "δ de Cliff": "{:.2f}", "P(ROPE) bayesiana": "{:.3f}", "Potencia para δ": "{:.3f}"})
RQ1.to_csv(RESULTADOS / "contrastes_rq1.csv", index=False)
""")

md("s6-global", r"""
**Procedimiento global.** Friedman, con la corrección de Iman y Davenport, compara los 16 modelos con las semillas como bloques; el *post hoc* de Nemenyi da la distancia crítica del diagrama, y el de Wilcoxon con Holm, las comparaciones por pares. Se complementa con la ANOVA de medidas repetidas con la corrección de Greenhouse-Geisser y con el *Model Confidence Set* de Hansen al 90 %, sobre la pérdida 1 − F1 macro por bloque de prueba.
""")

code("s6-global-calc", r"""
fr = E.friedman(M_ALL)
anova_rm = E.anova_repetidas(M_ALL)
rangos_n, p_nem, cd = E.nemenyi(M_ALL)
tabla(pd.DataFrame({"Prueba": ["Friedman", "Iman-Davenport", "ANOVA de medidas repetidas (Greenhouse-Geisser)"], "Estadístico": [fr["chi2"], fr["F Iman-Davenport"], anova_rm["F"]],
                    "p": [p_exacto(fr["p"]), p_exacto(fr["p Iman-Davenport"]), p_exacto(anova_rm["p GG"])],
                    "Tamaño de efecto": [f"W de Kendall = {num(fr['W de Kendall'], 3)}", "", f"η² parcial = {num(anova_rm['η² parcial'], 3)}; ε = {num(anova_rm['épsilon GG'], 3)}"]}),
      {"Estadístico": "{:.2f}"})
orden_r = np.argsort(rangos_n)
fig, eje = plt.subplots(figsize=(11, 3.0))
eje.set(xlim=(len(ORDEN) + 0.5, 0.5), ylim=(0, 1))
eje.yaxis.set_visible(False)
for i, j in enumerate(orden_r):
    lado = i < len(ORDEN) / 2
    yy = 0.85 - 0.09 * (i if lado else len(ORDEN) - 1 - i)
    eje.plot([rangos_n[j]] * 2, [1, yy], color=GRIS_OSCURO, lw=0.6)
    eje.text(rangos_n[j], yy - 0.02, f"{ORDEN[j]} ({rangos_n[j]:.1f})", ha="left" if lado else "right", va="top", fontsize=6.5)
grupos_cd = []
for i in range(len(orden_r)):
    fin = max([k for k in range(i, len(orden_r)) if rangos_n[orden_r[k]] - rangos_n[orden_r[i]] <= cd], default=i)
    if fin > i and not any(g[0] <= i and fin <= g[1] for g in grupos_cd):
        grupos_cd.append((i, fin))
for n_g, (i, fin) in enumerate(grupos_cd):
    eje.plot([rangos_n[orden_r[i]] - 0.05, rangos_n[orden_r[fin]] + 0.05], [0.95 - 0.03 * n_g] * 2, color=NEGRO, lw=2.2)
eje.plot([1, 1 + cd], [0.05] * 2, color=NEGRO, lw=1.5)
eje.text(1 + cd / 2, 0.08, f"CD = {cd:.2f}", ha="center", fontsize=7)
eje.set_xlabel("Rango promedio (1 = mejor)")
eje.set_title("Diagrama de diferencia crítica de Nemenyi; las barras unen modelos sin diferencia significativa", fontsize=9.5)
plt.tight_layout()
plt.show()
p_wh = np.ones((len(ORDEN), len(ORDEN)))
pares_todos = [(i, j) for i in range(len(ORDEN)) for j in range(i + 1, len(ORDEN))]
p_crudos = [stats.wilcoxon(M_ALL[:, i] - M_ALL[:, j]).pvalue if np.any(M_ALL[:, i] != M_ALL[:, j]) else 1.0 for i, j in pares_todos]
for (i, j), ph in zip(pares_todos, E.holm(p_crudos)):
    p_wh[i, j] = p_wh[j, i] = ph
WILCOXON_HOLM = pd.DataFrame(p_wh, index=ORDEN, columns=ORDEN)
print(f"Pares significativos con Wilcoxon-Holm: {int((np.triu(p_wh, 1)[np.triu_indices(len(ORDEN), 1)] < ALFA).sum())} de {len(pares_todos)}")
perdida_bloque = np.column_stack([1 - np.mean([[AN.f1_desde_conteos(CONF_BLOQUES[f"{i}|eva"][b]) for b in range(NB_PRU)] for i in METRICAS.loc[METRICAS["modelo"] == m, "id"]], 0)
                                  for m in modelos_p])
mcs = E.model_confidence_set(perdida_bloque, nivel=0.10)
MCS = [modelos_p[i] for i in mcs["conjunto"]]
print("Model Confidence Set al 90 %:", MCS)
""")

md("s6-rq2", r"""
**RQ2: el mejor modelo profundo frente a LightGBM.** El mejor modelo profundo se elige por su F1 macro medio de validación, para no sesgar el contraste con la propia prueba. Además de las tres pruebas preregistradas y TOST, se reportan McNemar y DeLong sobre el promedio de probabilidades de las R semillas, que suponen filas independientes, y la *F* de 5 × 2 cv de Alpaydin y la *t* con la corrección de Nadeau y Bengio sobre la validación cruzada 5 × 2 agrupada, que miden también la variabilidad por partición.
""")

code("s6-rq2-calc", r"""
RQ2 = contrastes([(MEJOR_DL, "LightGBM")], "F2")
tabla(presentar(RQ2), {"Diferencia": "{:+.4f}", "d de Cohen": "{:.2f}", "δ de Cliff": "{:.2f}", "P(ROPE) bayesiana": "{:.3f}", "Potencia para δ": "{:.3f}"})
ens = {m: np.mean([AN.prob(next(x for x in DEF if x["modelo"] == m and x["semilla"] == s), "eva") for s in SEMILLAS], 0) for m in (MEJOR_DL, "LightGBM")}
mc = E.mcnemar_exacta(ens[MEJOR_DL].argmax(1) == y_pru, ens["LightGBM"].argmax(1) == y_pru)
dl_clases = [E.delong(y_pru == c, ens[MEJOR_DL][:, c], ens["LightGBM"][:, c]) for c in range(K)]
cv = RES.tabla_fase("cv5x2")
def diferencias_cv(a, b):
    pa = cv[cv["modelo"] == a].set_index("particion")["f1_eva"]
    pb = cv[cv["modelo"] == b].set_index("particion")["f1_eva"]
    nombres_cv = sorted(pa.index)
    return (pa[nombres_cv] - pb[nombres_cv]).to_numpy()
d_cv = diferencias_cv(MEJOR_DL, "LightGBM")
nb = E.nadeau_bengio(d_cv, 1, 1)
f52 = E.f_5x2cv(d_cv)
tabla(pd.DataFrame({"Prueba complementaria": ["McNemar (promedio de semillas)", *[f"DeLong, AUC de {CLASES[c]} uno contra todos" for c in range(K)], "t de Nadeau-Bengio (5 × 2 cv)", "F de 5 × 2 cv (Alpaydin)"],
                    "Diferencia o estadístico": [f"b = {mc['b']}, c = {mc['c']}", *[f"{num(d['diferencia'], 4)}" for d in dl_clases], f"diferencia media {num(d_cv.mean(), 4)}, t = {num(nb['t'], 2)}", f"F = {num(f52['F'], 2)}"],
                    "p": [p_exacto(mc["p"]), *[p_exacto(ph) for ph in E.holm([d["p"] for d in dl_clases])], p_exacto(nb["p"]), p_exacto(f52["p"])],
                    "Nota": ["Supone filas independientes", *["p con Holm entre las 5 clases"] * K, "Corrige el solapamiento de entrenamiento", "Alternativa con pocas ejecuciones"]}))
""")

md("s6-rq3", r"""
**RQ3: modelos que leen los atributos como secuencia frente a los que no suponen orden.** Para cada semilla se promedia el F1 macro de los seis modelos secuenciales (CNN 1D, LSTM, RNN, GRU, BiLSTM y CNN-LSTM) y el de los siete que no suponen orden (DNN, ResNet, Autoencoder, DBN, KAN, TabNet y Transformer), y se comparan los promedios con la *t* pareada y Wilcoxon. Como procedimiento global, un modelo lineal de efectos mixtos sobre el F1 macro por bloque de prueba, $y_{ijr} = \mu + \alpha_{g(j)} + u_i + v_r + \varepsilon_{ijr}$, estima el efecto del grupo con efectos aleatorios de bloque $u_i$ y de semilla $v_r$.
""")

code("s6-rq3-calc", r"""
import statsmodels.formula.api as smf

SECUENCIA = ["CNN 1D", "LSTM", "RNN", "GRU", "BiLSTM", "CNN-LSTM"]
SIN_ORDEN = ["DNN", "ResNet", "Autoencoder", "DBN", "KAN", "TabNet", "Transformer"]
r3 = E.comparar_par(F1P[SECUENCIA].mean(1).to_numpy(), F1P[SIN_ORDEN].mean(1).to_numpy())
filas_mm = []
for m in PROFUNDOS:
    for i in METRICAS.loc[METRICAS["modelo"] == m, "id"]:
        s_ = int(METRICAS.loc[METRICAS["id"] == i, "semilla"].iloc[0])
        for b in range(NB_PRU):
            filas_mm.append({"f1": AN.f1_desde_conteos(CONF_BLOQUES[f"{i}|eva"][b]), "modelo": m, "grupo": "secuencia" if m in SECUENCIA else "sin orden", "bloque": b, "semilla": s_})
MM = pd.DataFrame(filas_mm)
mixto = smf.mixedlm("f1 ~ C(grupo, Treatment('sin orden'))", MM, groups=MM["bloque"], vc_formula={"semilla": "0 + C(semilla)"}).fit(reml=True)
coef = mixto.params.iloc[1]
ic_mm = mixto.conf_int().iloc[1].to_numpy()
tabla(pd.DataFrame({"Prueba": ["t pareada sobre semillas", "Wilcoxon sobre semillas", "Modelo mixto (bloque y semilla aleatorios)"],
                    "Diferencia secuencia − sin orden": [r3["Diferencia media"], r3["Diferencia media"], coef],
                    "IC 95 %": [intervalo(r3["IC 95 % t"]), intervalo(r3["IC 95 % bootstrap"]), intervalo(ic_mm)],
                    "p": [p_exacto(r3["p t pareada"]), p_exacto(r3["p Wilcoxon"]), p_exacto(mixto.pvalues.iloc[1])]}), {"Diferencia secuencia − sin orden": "{:+.4f}"})
RQ3_DECISION = E.decision(r3, r3["p t pareada"] if r3["Shapiro-Wilk p"] >= ALFA else r3["p bootstrap"], r3["p Wilcoxon"])
print("Decisión de RQ3:", RQ3_DECISION)
mixto_var = smf.mixedlm("f1 ~ C(modelo)", MM, groups=MM["bloque"], vc_formula={"semilla": "0 + C(semilla)"}).fit(reml=True)
var_bloque, var_semilla, var_resid = float(mixto_var.cov_re.iloc[0, 0]), float(mixto_var.vcomp[0]), float(mixto_var.scale)
efectos_fijos = mixto_var.fe_params.iloc[1:]
var_modelo = float(np.var(np.r_[0, efectos_fijos.to_numpy()]))
total_var = var_modelo + var_bloque + var_semilla + var_resid
DESCOMP = pd.DataFrame({"Componente": ["Modelo (efecto fijo)", "Bloque de prueba", "Semilla", "Residuo"], "Varianza": [var_modelo, var_bloque, var_semilla, var_resid]})
DESCOMP["% del total"] = 100 * DESCOMP["Varianza"] / total_var
tabla(DESCOMP, {"Varianza": "{:.6f}", "% del total": "{:.1f}"})
""")

md("s6-4", r"""
### 6.4 Techo de desempeño e incertidumbre

**Techo de desempeño.** Se estiman dos cotas del error irreducible: la del aprendizaje confiado de la sección 2.1.2, que acota la parte del error atribuible a etiquetas ruidosas o a clases superpuestas, y la de Cover y Hart (1967) con el vecino más cercano, que para $K$ clases acota el error de Bayes $R^*$ con el error del 1-NN, $R_{NN}$: $\frac{K-1}{K}\left(1 - \sqrt{1 - \frac{K}{K-1} R_{NN}}\right) \le R^* \le R_{NN}$. El 1-NN se calcula sobre validación con entrenamiento como referencia, de modo que incluye el efecto del cambio entre series. Con el techo en exactitud, la distancia normalizada es $\gamma = (s_{techo} - s_{mejor}) / (s_{techo} - s_{base})$, con el baseline del Entregable 1 como $s_{base}$.
""")

code("s6-techo", r"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(1, n_jobs=2).fit(H["X"][ID_ENT], y[ID_ENT])
r_nn = float((knn.predict(H["X"][ID_VAL]) != y[ID_VAL]).mean())
cota_inf = (K - 1) / K * (1 - np.sqrt(max(1 - K / (K - 1) * r_nn, 0)))
acc = METRICAS.groupby("modelo")["prueba Accuracy"].mean()
techos = {"Aprendizaje confiado": 1 - ruido_cl, "Cover y Hart, cota inferior del error de Bayes": 1 - cota_inf}
filas_t = []
for nombre, s_techo in techos.items():
    gamma = (s_techo - acc[MEJOR_DL]) / (s_techo - acc[base])
    filas_t.append({"Cota": nombre, "Exactitud máxima alcanzable": s_techo, f"Exactitud de {MEJOR_DL}": acc[MEJOR_DL], f"Exactitud de {base}": acc[base], "γ": gamma})
TECHO = pd.DataFrame(filas_t)
tabla(TECHO, {c: "{:.4f}" for c in TECHO.columns if c != "Cota"})
print(f"Error del 1-NN en validación: {r_nn:.4f}")
""")

md("s6-incertidumbre", r"""
**Incertidumbre predictiva.** A los tres mejores modelos por validación se aplican tres métodos: el ensamble profundo de sus R semillas, que promedia sus probabilidades (Lakshminarayanan et al., 2017); *MC dropout* con 30 pasadas y el *dropout* activo, en los modelos que lo tienen (Gal y Ghahramani, 2016), y la predicción conforme dividida, calibrada en validación, que es disjunta de entrenamiento y prueba, con los puntajes LAC y APS y niveles de 90 y 95 % (Angelopoulos y Bates, 2023). Para la predicción conforme se reportan la cobertura empírica frente a la nominal, el tamaño medio de los conjuntos y la cobertura condicional por clase y por bloque. Hay que tener en cuenta que la intercambiabilidad que supone la garantía conforme no se cumple entre series con desplazamiento (sección 2.1.5), de modo que la cobertura en prueba puede quedar por debajo de la nominal.
""")

code("s6-incertidumbre-calc", r"""
MC = AN.mc_dropout(TOP3)
filas_u, filas_conf = [], []
for m in TOP3:
    probs_eva = [AN.prob(next(x for x in DEF if x["modelo"] == m and x["semilla"] == s), "eva") for s in SEMILLAS]
    probs_val = [AN.prob(next(x for x in DEF if x["modelo"] == m and x["semilla"] == s), "val") for s in SEMILLAS]
    individual = E.metricas(probs_eva[0], y_pru)
    ensamble = E.metricas(np.mean(probs_eva, 0), y_pru)
    filas_u.append({"Modelo": m, "Método": "Una semilla", "F1 macro": individual["F1 macro"], "Log-loss": individual["Log-loss"], "ECE": individual["ECE"]})
    filas_u.append({"Modelo": m, "Método": f"Ensamble de {R} semillas", "F1 macro": ensamble["F1 macro"], "Log-loss": ensamble["Log-loss"], "ECE": ensamble["ECE"]})
    clave_mc = f"{next(x for x in DEF if x['modelo'] == m and x['semilla'] == SEMILLAS[0])['id']}|eva"
    if clave_mc in MC:
        mcd = E.metricas(MC[clave_mc], y_pru)
        filas_u.append({"Modelo": m, "Método": "MC dropout (30 pasadas)", "F1 macro": mcd["F1 macro"], "Log-loss": mcd["Log-loss"], "ECE": mcd["ECE"]})
    p_val_e, p_eva_e = np.mean(probs_val, 0), np.mean(probs_eva, 0)
    for metodo in ("LAC", "APS"):
        for alfa_c in (0.10, 0.05):
            conj = E.conforme(p_val_e, y[ID_VAL], p_eva_e, alfa_c, metodo)
            cob, tam = E.cobertura(conj, y_pru)
            por_clase = [conj[y_pru == c, c].mean() for c in range(K)]
            por_bloque = [conj[b_pru == b][np.arange((b_pru == b).sum()), y_pru[b_pru == b]].mean() for b in np.unique(b_pru)]
            filas_conf.append({"Modelo": m, "Puntaje": metodo, "Nominal": 1 - alfa_c, "Cobertura empírica": cob, "Tamaño medio": tam, "Cobertura mínima por clase": min(por_clase),
                               "Clase con menor cobertura": CLASES[int(np.argmin(por_clase))], "Cobertura mínima por bloque": min(por_bloque), "Bloques bajo la nominal": int(sum(c < 1 - alfa_c for c in por_bloque))})
tabla(pd.DataFrame(filas_u), {"F1 macro": "{:.4f}", "Log-loss": "{:.4f}", "ECE": "{:.4f}"})
CONFORME = pd.DataFrame(filas_conf)
tabla(CONFORME, {"Nominal": "{:.2f}", "Cobertura empírica": "{:.4f}", "Tamaño medio": "{:.3f}", "Cobertura mínima por clase": "{:.4f}", "Cobertura mínima por bloque": "{:.4f}"})
""")
