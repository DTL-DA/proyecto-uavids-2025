from .base import code, md

md("s7", r"""
## 7. Ruta metodológica: ficha de la familia F1, datos tabulares

El problema es de clasificación tabular, de modo que se sigue la ruta F1 de la sección 7.4 del enunciado y no aplica ninguna otra: no hay imágenes, texto, series ni coordenadas que combinar. La ficha indica dónde se cumple cada paso de la ruta y reúne la evidencia de cada prueba de las tablas A, B y C que aplica a este tipo de problema, con su valor p, su tamaño de efecto o su intervalo. Las pruebas propias de regresión, series de tiempo y datos espaciales se marcan como no aplicables, con la razón.

| Paso de la ruta F1 | Aplicación en este trabajo | Dónde |
|---|---|---|
| D1. Tarea y unidad de independencia | Clasificación multiclase; serie aproximada con bloques de 1000 flujos | Secciones 1.1 y 2.2.3 |
| D2. Partición estratificada o por grupos; validación cruzada repetida | Estratificada y agrupada por bloques; 5 × 2 cv agrupada | Secciones 2.1.4 y 6.3 |
| D3. Pipeline único ajustado dentro de cada pliegue | `codigo/`, con estandarización y pesos de clase por pliegue | Secciones 1.5, 2.1.4 y 4.1 |
| D4. Líneas base y árboles potenciados como comparador fuerte | Trivial, regresión logística y LightGBM, con el mismo presupuesto | Secciones 4.2 y 6.1 |
| D5. Mismo presupuesto de búsqueda | 12 configuraciones por modelo y búsqueda de 1000 arquitecturas para los cuatro sin diseño | Secciones 4.2.1 y 4.3.2 |
| D6. Umbral en validación y calibración solo con validación | Arg max sin umbral ajustado; *temperature scaling* con validación | Sección 4.5 |
| D7. Pruebas de ajuste, sobreajuste y comparaciones con corrección | Tablas A, B y C, abajo | Secciones 6.3 y 7 |
| D8. Interpretabilidad, subgrupos y coherencia con el dominio | Gradientes integrados, segmentos y diagnóstico de atajos | Secciones 4.5 y 4.6 |
""")

code("s7-tabla-a", r"""
r1 = {m: next(x for x in DEF if x["modelo"] == m and x["semilla"] == SEMILLAS[0]) for m in ORDEN}
p_mayoritaria = float(np.bincount(y[ID_ENT]).max() / len(ID_ENT))
perm = RES.tabla_fase("permutacion")
filas_a = []
for m in [x for x in ORDEN if x != "Trivial"]:
    p1 = AN.prob(r1[m], "eva")
    pred1 = p1.argmax(1)
    aciertos = int((pred1 == y_pru).sum())
    auc_b = E.auc_macro_bootstrap(y_pru, p1, b_pru, repeticiones=1000)
    mcc_b = E.mcc_desde_conteos(W_PRU_PAREADO @ CONF_BLOQUES[f"{r1[m]['id']}|eva"])
    hl = [E.hosmer_lemeshow(p1[:, c], (y_pru == c).astype(int)) for c in range(K)]
    sp = [E.spiegelhalter(p1[:, c], (y_pru == c).astype(int)) for c in range(K)]
    pi = [E.pendiente_intercepto(p1[:, c], (y_pru == c).astype(int)) for c in range(K)]
    obs = METRICAS.loc[(METRICAS["modelo"] == m) & (METRICAS["semilla"] == SEMILLAS[0]), "prueba F1 macro"].iloc[0]
    nulos = perm.loc[perm["modelo"] == m, "f1_eva"].to_numpy() if len(perm) else np.array([])
    filas_a.append({"Modelo": m,
                    "Permutación de etiquetas": f"p = {num((1 + (nulos >= obs).sum()) / (len(nulos) + 1), 4)} (B = {len(nulos)}; F1 nulo máximo {num(nulos.max(), 3)})" if len(nulos) else "No ejecutada por costo",
                    "Binomial frente a la mayoritaria": p_exacto(E.binomial_frente_mayoritaria(aciertos, len(y_pru), p_mayoritaria)),
                    "AUC macro, IC 95 % por bloques": intervalo(E.ic_percentil(auc_b), 3), "MCC, IC 95 % por bloques": intervalo(E.ic_percentil(mcc_b), 3),
                    "Hosmer-Lemeshow: clases con p < 0,05": f"{sum(h['p'] < ALFA for h in hl)} de 5", "Spiegelhalter: clases con p < 0,05": f"{sum(s_['p'] < ALFA for s_ in sp)} de 5",
                    "Pendiente de calibración (media de clases)": np.mean([x["pendiente"] for x in pi]), "Intercepto (media)": np.mean([x["intercepto"] for x in pi]),
                    "Pendiente = 1 e intercepto = 0: clases con p < 0,05": f"{sum(x['p'] < ALFA for x in pi)} de 5"})
TABLA_A = pd.DataFrame(filas_a)
tabla(TABLA_A, {"Pendiente de calibración (media de clases)": "{:.3f}", "Intercepto (media)": "{:.3f}"})
print("No aplican, por ser pruebas de regresión, series de tiempo o datos espaciales: Mincer-Zarnowitz, normalidad y homocedasticidad de residuos, Ljung-Box, ARCH, BDS, I de Moran sobre residuos y PIT.")
""")

code("s7-tabla-b", r"""
curvas_ap = RES.tabla_fase("curvas")
boot_opt = RES.tabla_fase("bootstrap")
filas_b = []
for m in [x for x in ORDEN if x != "Trivial"]:
    f1_s = F1P[m].to_numpy()
    cv_semilla = f1_s.std(ddof=1) / f1_s.mean()
    por_bloque = [[AN.f1_desde_conteos(CONF_BLOQUES[f"{i}|eva"][b]) for b in range(NB_PRU)] for i in METRICAS.loc[METRICAS["modelo"] == m, "id"]]
    kw = stats.kruskal(*por_bloque)
    br = BRECHAS.set_index("Modelo").loc[m]
    fila = {"Modelo": m, "Brecha g (p, H1: g < δg)": f"{num(br['Brecha g'], 4)} ({p_exacto(br['p Wilcoxon (g < δg)'])})",
            "Interno (ajuste) − externo (prueba)": float(ajuste.loc[ajuste["modelo"] == m, "f1_val"].max() - F1P[m].mean()),
            "Validación − prueba, IC por bloques": f"{num(br['Validación − prueba'], 4)} {br['IC 95 % por bloques']}",
            "CV entre semillas (%)": 100 * cv_semilla, "Kruskal-Wallis entre semillas": f"H = {num(kw.statistic, 2)}, {valor_p(kw.pvalue)}"}
    sub_c = curvas_ap[curvas_ap["modelo"] == m] if len(curvas_ap) else []
    if len(sub_c):
        pend = stats.linregress(np.log(sub_c["fraccion_ent"] * len(ID_ENT)), sub_c["f1_val"])
        fila["Curva de aprendizaje: pendiente por log n (p)"] = f"{num(pend.slope, 4)} ({p_exacto(pend.pvalue)})"
    else:
        fila["Curva de aprendizaje: pendiente por log n (p)"] = "No ejecutada"
    sub_b = boot_opt[boot_opt["modelo"] == m] if len(boot_opt) else []
    if len(sub_b):
        optimismo = (sub_b["f1_ent"] - sub_b["f1_eva"]).to_numpy()
        fila["Optimismo por bootstrap (IC 95 %)"] = f"{num(optimismo.mean(), 4)} {intervalo(np.percentile(optimismo, [2.5, 97.5]))}"
    else:
        fila["Optimismo por bootstrap (IC 95 %)"] = "No ejecutado"
    filas_b.append(fila)
TABLA_B = pd.DataFrame(filas_b)
tabla(TABLA_B, {"Interno (ajuste) − externo (prueba)": "{:+.4f}", "CV entre semillas (%)": "{:.2f}"})
interno = np.array([ajuste.loc[ajuste["modelo"] == m, "f1_val"].max() for m in PROFUNDOS])
externo = np.array([F1P[m].mean() for m in PROFUNDOS])
w_anidada = stats.wilcoxon(interno - externo)
print(f"Validación anidada aproximada, 13 modelos profundos: interno − externo = {np.mean(interno - externo):+.4f}, Wilcoxon p = {w_anidada.pvalue:.2e}")
print(f"Validación adversarial (sección 2.1.5): AUC entrenamiento frente a prueba {shift['auc_prueba']:.3f}; máximo nulo por bloques {max(shift['auc_nulo_prueba']):.3f}. Estabilidad temporal (Giacomini-Rossi): no aplica, no hay tiempo.")
""")

code("s7-tabla-c", r"""
resumen_c = pd.concat([RQ1, RQ2])[["Familia", "Contraste", "Prueba (i)", "p (i) Holm", "p Wilcoxon Holm", "p TOST", "P(ROPE)", "d de Cohen pareada", "Decisión"]]
resumen_c["Friedman e Iman-Davenport (global)"] = p_exacto(fr["p Iman-Davenport"])
resumen_c["Wilcoxon-Holm entre todos los pares"] = [p_exacto(WILCOXON_HOLM.loc[f.split(" − ")[0], f.split(" − ")[1]]) for f in resumen_c["Contraste"]]
tabla(resumen_c.assign(**{c: resumen_c[c].map(p_exacto) for c in ["p (i) Holm", "p Wilcoxon Holm", "p TOST"]}), {"P(ROPE)": "{:.3f}", "d de Cohen pareada": "{:.2f}"})
""")

md("s7-adecuacion", r"""
**Criterio de adecuación de la solución.** Según la sección 7.2 del enunciado, un modelo es adecuado solo si cumple a la vez cuatro condiciones: (i) supera al baseline trivial con p ajustado < 0,05 y una mejora mayor que δ; (ii) su brecha entre entrenamiento y validación es menor que δg con evidencia estadística; (iii) sus probabilidades están calibradas, lo que aquí se verifica con un ECE menor que 0,05 tras el *temperature scaling* ajustado en validación y con un IC del 95 % de la pendiente de calibración media que contenga a 1, y (iv) su desempeño en prueba es consistente con el de validación, es decir, el IC por bloques de la diferencia contiene al cero.
""")

code("s7-adecuacion-calc", r"""
p_trivial = E.holm([stats.ttest_1samp(F1P[m] - F1P["Trivial"], 0).pvalue for m in [x for x in ORDEN if x != "Trivial"]])
ece_t = CALIB.groupby("modelo")["ECE con T"].mean()
filas_ad = []
for j, m in enumerate([x for x in ORDEN if x != "Trivial"]):
    br = BRECHAS.set_index("Modelo").loc[m]
    c1 = p_trivial[j] < ALFA and F1P[m].mean() - F1P["Trivial"].mean() > DELTA
    c2 = br["p Wilcoxon (g < δg)"] < ALFA
    pend = TABLA_A.set_index("Modelo").loc[m, "Pendiente de calibración (media de clases)"]
    c3 = ece_t[m] < 0.05
    ic_vp = [float(v.replace(",", ".")) for v in br["IC 95 % por bloques"].strip("[]").split("; ")]
    c4 = ic_vp[0] <= 0 <= ic_vp[1]
    filas_ad.append({"Modelo": m, "(i) Supera al trivial": "sí" if c1 else "no", "(ii) Brecha < δg": "sí" if c2 else "no", "(iii) Calibrado (ECE con T < 0,05)": "sí" if c3 else "no",
                     "Pendiente media": pend, "(iv) Prueba consistente con validación": "sí" if c4 else "no", "Adecuado": "sí" if all((c1, c2, c3, c4)) else "no"})
ADECUACION = pd.DataFrame(filas_ad)
tabla(ADECUACION, {"Pendiente media": "{:.3f}"})
""")
