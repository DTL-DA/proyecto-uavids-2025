from .base import code, md

md("s7", r"""
## 7. Ruta metodológica: ficha de la familia F1, datos tabulares

El problema es de clasificación tabular, de modo que se sigue la ruta F1 de la sección 7.4 del enunciado y no aplica ninguna otra: no hay imágenes, texto, series ni coordenadas que combinar. La ficha indica dónde se cumple cada paso de la ruta y reúne la evidencia de cada prueba de las tablas A, B y C que aplica a este tipo de problema, con su valor p, su tamaño de efecto o su intervalo. En la tabla A, Hosmer-Lemeshow sigue a Hosmer et al. (2013) y la *Z* de calibración, a Spiegelhalter (1986), ambas en esquema uno contra todos por clase. Las pruebas propias de regresión, series de tiempo y datos espaciales se marcan como no aplicables, con la razón.

| Paso de la ruta F1 | Aplicación en este trabajo | Dónde |
|---|---|---|
| D1. Tarea y unidad de independencia | Clasificación multiclase; serie aproximada con bloques de 1000 flujos | Secciones 1.1 y 2.2.3 |
| D2. Partición estratificada o por grupos; validación cruzada repetida | Estratificada y agrupada por bloques; 5 × 2 cv agrupada | Secciones 2.1.4 y 6.3 |
| D3. Pipeline único ajustado dentro de cada pliegue | `codigo/`, con estandarización y pesos de clase por pliegue | Secciones 1.5, 2.1.4 y 4.1 |
| D4. Líneas base y árboles potenciados como comparador fuerte | Trivial, regresión logística y LightGBM, con el mismo presupuesto | Secciones 4.2 y 6.1 |
| D5. Mismo presupuesto de búsqueda | 12 configuraciones por modelo y búsqueda de 1000 arquitecturas para los cuatro sin diseño | Secciones 4.2.1 y 4.3.2 |
| D6. Umbral en validación y calibración solo con validación | Arg max sin umbral ajustado; *temperature scaling* con validación (Guo et al., 2017) | Sección 4.5 |
| D7. Pruebas de ajuste, sobreajuste y comparaciones con corrección | Tablas A, B y C, abajo | Secciones 6.3 y 7 |
| D8. Interpretabilidad, subgrupos y coherencia con el dominio | Gradientes integrados, segmentos y diagnóstico de atajos | Secciones 4.5 y 4.6 |

No aplican, por ser propias de regresión, series de tiempo o datos espaciales, Mincer-Zarnowitz, la normalidad y la homocedasticidad de los residuos, Ljung-Box, ARCH, BDS, la *I* de Moran sobre los residuos, el PIT y la estabilidad temporal de Giacomini y Rossi: no hay una salida continua, ni un orden temporal, ni coordenadas.
""")

code("s7-tabla-a", r"""
r1 = {m: next(x for x in DEF if x["modelo"] == m and x["semilla"] == SEMILLAS[0]) for m in ORDEN}
p_mayoritaria = float(np.bincount(y[ID_ENT]).max() / len(ID_ENT))
perm = RES.tabla_fase("permutacion")
filas_a, num_a = [], []
for m in [x for x in ORDEN if x != "Trivial"]:
    p1 = AN.prob(r1[m], "eva")
    pred1 = p1.argmax(1)
    aciertos = int((pred1 == y_pru).sum())
    auc_b = E.auc_macro_bootstrap(y_pru, p1, b_pru, repeticiones=2000)
    mcc_b = E.mcc_desde_conteos(W_PRU_PAREADO @ CONF_BLOQUES[f"{r1[m]['id']}|eva"])
    hl = [E.hosmer_lemeshow(p1[:, c], (y_pru == c).astype(int)) for c in range(K)]
    sp = [E.spiegelhalter(p1[:, c], (y_pru == c).astype(int)) for c in range(K)]
    pi = [E.pendiente_intercepto(p1[:, c], (y_pru == c).astype(int)) for c in range(K)]
    obs = METRICAS.loc[(METRICAS["modelo"] == m) & (METRICAS["semilla"] == SEMILLAS[0]), "prueba F1 macro"].iloc[0]
    nulos = perm.loc[perm["modelo"] == m, "f1_eva"].to_numpy() if len(perm) else np.array([])
    p_bin = E.binomial_frente_mayoritaria(aciertos, len(y_pru), p_mayoritaria)
    num_a.append({"Modelo": m, "p binomial": p_bin, "AUC inferior": E.ic_percentil(auc_b)[0], "MCC inferior": E.ic_percentil(mcc_b)[0], "HL": sum(h["p"] < ALFA for h in hl), "SP": sum(s_["p"] < ALFA for s_ in sp),
                  "p permutación": (1 + (nulos >= obs).sum()) / (len(nulos) + 1) if len(nulos) else np.nan, "B": len(nulos)})
    filas_a.append({"Modelo": m,
                    "Permutación de etiquetas": f"p = {num((1 + (nulos >= obs).sum()) / (len(nulos) + 1), 4)} (B = {len(nulos)}; F1 nulo máximo {num(nulos.max(), 3)})" if len(nulos) else "No ejecutada por costo",
                    "Binomial frente a la mayoritaria": p_exacto(p_bin),
                    "AUC macro, IC 95 % por bloques": intervalo(E.ic_percentil(auc_b), 3), "MCC, IC 95 % por bloques": intervalo(E.ic_percentil(mcc_b), 3),
                    "Hosmer-Lemeshow: clases con p < 0,05": f"{sum(h['p'] < ALFA for h in hl)} de 5", "Spiegelhalter: clases con p < 0,05": f"{sum(s_['p'] < ALFA for s_ in sp)} de 5",
                    "Pendiente de calibración (media de clases)": np.mean([x["pendiente"] for x in pi]), "Intercepto (media)": np.mean([x["intercepto"] for x in pi]),
                    "Pendiente = 1 e intercepto = 0: clases con p < 0,05": f"{sum(x['p'] < ALFA for x in pi)} de 5"})
TABLA_A = pd.DataFrame(filas_a)
TABLA_A.to_csv(RESULTADOS / "tabla_a.csv", index=False, encoding="utf-8-sig")
tabla(TABLA_A, {"Pendiente de calibración (media de clases)": "{:.3f}", "Intercepto (media)": "{:.3f}"})
NA = pd.DataFrame(num_a)
n_a = len(NA)
n_bin, n_auc, n_mcc = int((NA["p binomial"] < ALFA).sum()), int((NA["AUC inferior"] > 0.5).sum()), int((NA["MCC inferior"] > 0).sum())
perm_ok = NA["p permutación"].dropna()
pend_a = TABLA_A["Pendiente de calibración (media de clases)"]
frac_hl, frac_sp = NA["HL"].sum() / (5 * n_a), NA["SP"].sum() / (5 * n_a)
if len(perm_ok):
    b_perm = int(NA["B"].max())
    lectura_perm = (f"; la permutación de etiquetas, ejecutada en {enumerar(NA.loc[perm_ok.index, 'Modelo'])} con B = {b_perm}, da p = {num(perm_ok.max(), 4)}"
                    + (", el mínimo posible: ningún F1 con etiquetas permutadas alcanza al observado" if np.isclose(perm_ok.max(), 1 / (b_perm + 1)) else ""))
else:
    lectura_perm = ""
lectura_pend = ("todas las pendientes medias son mayores que 1, es decir, los modelos son subconfiados en promedio, coherente con la pérdida ponderada por clase, que reparte probabilidad hacia las clases minoritarias" if (pend_a > 1).all()
                else "todas las pendientes medias son menores que 1, es decir, los modelos son sobreconfiados en promedio" if (pend_a < 1).all()
                else (f"{enumerar(TABLA_A.loc[pend_a > 1, 'Modelo'].tolist())} {'tiene' if (pend_a > 1).sum() == 1 else 'tienen'} pendiente media mayor que 1, "
                      f"{'subconfiado' if (pend_a > 1).sum() == 1 else 'subconfiados'}, y los otros {int((pend_a < 1).sum())}, menor que 1, sobreconfiados"
                      + (". En validación, en cambio, la temperatura que corrige a los modelos profundos queda cerca de 1 (sección 4.5), de modo que es el paso a series nuevas el que vuelve las probabilidades demasiado extremas" if (pend_a < 1).mean() > 0.5 else "")))
interpretar(f'''**Interpretación.** En la tabla A, {"los " + str(n_a) + " modelos superan" if n_bin == n_a else f"{n_bin} de los {n_a} modelos superan"} a la clase mayoritaria con la prueba binomial exacta, {"todos" if n_auc == n_a else str(n_auc)} tienen un IC del AUC macro por bloques por encima de 0,5 y {"todos" if n_mcc == n_a else str(n_mcc)}, un IC del MCC por encima de 0, de modo que capturan relación entre los atributos y la clase{lectura_perm}. En calibración, Hosmer-Lemeshow rechaza en el {num(100 * frac_hl, 0)} % de los pares de modelo y clase y la *Z* de Spiegelhalter en el {num(100 * frac_sp, 0)} %, y la pendiente de calibración media va de {num(pend_a.min(), 3)} a {num(pend_a.max(), 3)}.

Hay que tener en cuenta que, con 22 883 flujos de prueba, las pruebas de calibración detectan desviaciones muy pequeñas, de modo que su rechazo se lee junto con el ECE de la sección 4.5, que mide su magnitud. Una pendiente menor que 1 indica sobreconfianza y una mayor que 1, subconfianza: aquí, {lectura_pend}.

Por tal motivo, la condición de calibración del criterio de adecuación se evalúa con el ECE tras el *temperature scaling*, y estas pruebas se reportan como evidencia de dirección.''')
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
TABLA_B.to_csv(RESULTADOS / "tabla_b.csv", index=False, encoding="utf-8-sig")
tabla(TABLA_B, {"Interno (ajuste) − externo (prueba)": "{:+.4f}", "CV entre semillas (%)": "{:.2f}"})
interno = np.array([ajuste.loc[ajuste["modelo"] == m, "f1_val"].max() for m in PROFUNDOS])
externo = np.array([F1P[m].mean() for m in PROFUNDOS])
w_anidada = stats.wilcoxon(interno - externo)
desplazamiento = shift["auc_prueba"] > max(shift["auc_nulo_prueba"])
tabla(pd.DataFrame({"Prueba complementaria": ["Validación anidada aproximada (13 modelos profundos)", "Validación adversarial, entrenamiento frente a prueba (sección 2.1.5)"],
                    "Resultado": [f"interno − externo = {num(np.mean(interno - externo), 4)}; Wilcoxon {valor_p(w_anidada.pvalue)}",
                                  f"AUC = {num(shift['auc_prueba'], 3)}; máximo del nulo por bloques = {num(max(shift['auc_nulo_prueba']), 3)}"]}))
bp_b = BRECHAS.set_index("Modelo").loc[[x for x in ORDEN if x != "Trivial"]]
n_brecha = int((bp_b["p Wilcoxon (g < δg)"] < ALFA).sum())
pend_curva = {m: stats.linregress(np.log(sub["fraccion_ent"] * len(ID_ENT)), sub["f1_val"]) for m, sub in curvas_ap.groupby("modelo")} if len(curvas_ap) else {}
suben = [m for m, r_ in pend_curva.items() if r_.slope > 0 and r_.pvalue < ALFA]
if pend_curva:
    lectura_curva = (f"La curva de aprendizaje, ejecutada en {enumerar(list(pend_curva))}, " + (("sube con el tamaño de entrenamiento en los " + str(len(suben)) + " modelos" if len(suben) == len(pend_curva) else f"sube con el tamaño de entrenamiento en {enumerar(suben)}") + ", de modo que más datos de las mismas series todavía ayudarían" if suben else "no sube de forma significativa con el tamaño de entrenamiento, de modo que más datos de las mismas series no ayudarían") + ".")
else:
    lectura_curva = ""
interpretar(f'''**Interpretación.** En la tabla B, la prueba de no inferioridad confirma una brecha entre entrenamiento y validación menor que δg en {n_brecha} de los {len(bp_b)} modelos, y el puntaje interno del ajuste supera al externo de prueba en {num(np.mean(interno - externo), 4)} en promedio sobre los 13 modelos profundos ({valor_p(w_anidada.pvalue)} en Wilcoxon), que es el optimismo conjunto de la selección de hiperparámetros y del cambio entre series. El coeficiente de variación entre semillas es de {num(TABLA_B["CV entre semillas (%)"].min(), 2)} % a {num(TABLA_B["CV entre semillas (%)"].max(), 2)} %.

Esto significa que {"no hay un sobreajuste del entrenamiento que invalide los modelos" if n_brecha == len(bp_b) else "salvo en " + enumerar(bp_b.index[bp_b["p Wilcoxon (g < δg)"] >= ALFA].tolist()) + (f", cuya brecha entre entrenamiento y validación es de {num(bp_b.loc[bp_b['p Wilcoxon (g < δg)'] >= ALFA, 'Brecha g'].iloc[0], 4)}" if len(bp_b) - n_brecha == 1 else "") + (", una brecha propia de los árboles potenciados, que se ajustan casi por completo al entrenamiento y aun así son los que menos caen en prueba" if bp_b.index[bp_b["p Wilcoxon (g < δg)"] >= ALFA].tolist() == ["LightGBM"] else "") + ", no hay un sobreajuste del entrenamiento que invalide los modelos"}, {"pero sí una distancia sistemática entre validación y prueba" if np.mean(interno - externo) > DELTA and w_anidada.pvalue < ALFA else "y que la distancia entre validación y prueba es pequeña"}{(", que la validación adversarial no atribuye a un desplazamiento propio de la partición (sección 2.1.5) y que, por tanto, se explica sobre todo por la variación entre series" if not desplazamiento else ", coherente con el desplazamiento entre entrenamiento y prueba que detecta la validación adversarial (sección 2.1.5)") if np.mean(interno - externo) > DELTA and w_anidada.pvalue < ALFA else ""}. {lectura_curva}

Por ello, se recomienda reportar el desempeño esperado en series nuevas con el intervalo por bloques y no con el F1 de validación.''')
""")

code("s7-tabla-c", r"""
resumen_c = pd.concat([RQ1, RQ2])[["Familia", "Contraste", "Prueba (i)", "p (i) Holm", "p Wilcoxon Holm", "p TOST", "P(ROPE)", "d de Cohen pareada", "Decisión"]]
resumen_c["Friedman e Iman-Davenport (global)"] = p_exacto(fr["p Iman-Davenport"])
resumen_c["Wilcoxon-Holm entre todos los pares"] = [p_exacto(WILCOXON_HOLM.loc[f.split(" − ")[0], f.split(" − ")[1]]) for f in resumen_c["Contraste"]]
resumen_c.to_csv(RESULTADOS / "tabla_c.csv", index=False, encoding="utf-8-sig")
tabla(resumen_c.assign(**{c: resumen_c[c].map(p_exacto) for c in ["p (i) Holm", "p Wilcoxon Holm", "p TOST"]}), {"P(ROPE)": "{:.3f}", "d de Cohen pareada": "{:.2f}"})
""")

md("s7-adecuacion", r"""
**Criterio de adecuación de la solución.** Según la sección 7.2 del enunciado, un modelo es adecuado solo si cumple a la vez cuatro condiciones: (i) supera al baseline trivial con p ajustado < 0,05 y una mejora mayor que δ; (ii) su brecha entre entrenamiento y validación es menor que δg con evidencia estadística; (iii) sus probabilidades están calibradas, y (iv) su desempeño en prueba es consistente con el de validación, es decir, el IC por bloques de la diferencia contiene al cero.

Hay que tener en cuenta que el preregistro fija δ, δg y el ECE como métrica secundaria, pero no un umbral numérico para la condición (iii). El umbral que se usa aquí, un ECE menor que 0,05 tras el *temperature scaling* ajustado en validación, se fijó al redactar esta sección, después de la ejecución definitiva. Por tal motivo, y según la política de hallazgos exploratorios de la sección 9.3 del enunciado, la condición (iii) y el veredicto de adecuación se declaran **exploratorios**: no se usan para elegir el modelo final, y la segunda tabla muestra cómo cambia el veredicto con umbrales de 0,03 a 0,07. La pendiente de calibración media se reporta como referencia.
""")

code("s7-adecuacion-calc", r"""
import re

p_trivial = E.holm([stats.ttest_1samp(F1P[m] - F1P["Trivial"], 0).pvalue for m in [x for x in ORDEN if x != "Trivial"]])
ece_t = CALIB.groupby("modelo")["ECE con T"].mean()
UMBRAL_ECE = 0.05  # fijado después de la ejecución definitiva: condición exploratoria
filas_ad, sin_iii = [], {}
for j, m in enumerate([x for x in ORDEN if x != "Trivial"]):
    br = BRECHAS.set_index("Modelo").loc[m]
    c1 = p_trivial[j] < ALFA and F1P[m].mean() - F1P["Trivial"].mean() > DELTA
    c2 = br["p Wilcoxon (g < δg)"] < ALFA
    pend = TABLA_A.set_index("Modelo").loc[m, "Pendiente de calibración (media de clases)"]
    c3 = ece_t[m] < UMBRAL_ECE
    ic_vp = [float(v.replace(",", ".")) for v in br["IC 95 % por bloques"].strip("[]").split("; ")]
    c4 = ic_vp[0] <= 0 <= ic_vp[1]
    sin_iii[m] = c1 and c2 and c4
    filas_ad.append({"Modelo": m, "(i) Supera al trivial": "sí" if c1 else "no", "(ii) Brecha < δg": "sí" if c2 else "no", "(iii) ECE con temperatura < 0,05": "sí" if c3 else "no",
                     "Pendiente media": pend, "(iv) Prueba consistente con validación": "sí" if c4 else "no", "Adecuado": "sí" if all((c1, c2, c3, c4)) else "no"})
ADECUACION = pd.DataFrame(filas_ad)
ADECUACION.to_csv(RESULTADOS / "adecuacion.csv", index=False, encoding="utf-8-sig")
tabla(ADECUACION, {"Pendiente media": "{:.3f}"})
adecuados = ADECUACION.loc[ADECUACION["Adecuado"] == "sí", "Modelo"].tolist()
# Sensibilidad del veredicto al umbral de la condición (iii), que no estaba en el preregistro
filas_sens = []
for u in [0.03, 0.04, 0.05, 0.06, 0.07]:
    ok = [m for m in sin_iii if sin_iii[m] and ece_t[m] < u]
    filas_sens.append({"Umbral del ECE con temperatura": u, "Modelos adecuados": len(ok), "Cuáles": enumerar(ok) if ok else "ninguno"})
SENS_ECE = pd.DataFrame(filas_sens)
SENS_ECE.to_csv(RESULTADOS / "adecuacion_sensibilidad_ece.csv", index=False, encoding="utf-8-sig")
tabla(SENS_ECE, {"Umbral del ECE con temperatura": "{:.2f}"})
n_sens = SENS_ECE.set_index("Umbral del ECE con temperatura")["Modelos adecuados"]
falla = {c: ADECUACION.loc[ADECUACION[c] == "no", "Modelo"].tolist() for c in ["(i) Supera al trivial", "(ii) Brecha < δg", "(iii) ECE con temperatura < 0,05", "(iv) Prueba consistente con validación"]}
peor_c = max(falla, key=lambda c: len(falla[c]))
ece_t = cal_m["ECE con T"]
al_borde = [m for m in falla["(iii) ECE con temperatura < 0,05"] if m in ece_t.index and ece_t[m] < 0.055]
CONDICION_TXT = {"(i) Supera al trivial": "superar al trivial", "(ii) Brecha < δg": "una brecha entre entrenamiento y validación menor que δg",
                 "(iii) ECE con temperatura < 0,05": "un ECE con temperatura menor que 0,05", "(iv) Prueba consistente con validación": "una prueba consistente con la validación"}
razones = {m: [CONDICION_TXT[c] for c in falla if m in falla[c]] for m in dict.fromkeys(["LightGBM", MEJOR_DL]) if m not in adecuados}
texto_razones = " ".join(f"{m} no es adecuado porque no cumple la condición de {enumerar(rz)}." for m, rz in razones.items() if rz)
texto_borde = (f" Hay que tener en cuenta que la condición (iii) separa modelos muy parecidos: {enumerar(al_borde)} {'queda' if len(al_borde) == 1 else 'quedan'} por encima de 0,05 por menos de 0,005, "
               f"frente a, por ejemplo, {adecuados[0]} con {num(ece_t[adecuados[0]], 4)}, de modo que el umbral decide más que la diferencia de calibración." if al_borde and adecuados and adecuados[0] in ece_t.index else "")
n_peor = len(falla[peor_c])
etiqueta_peor = re.sub(r"^[(][ivx]+[)] ", "", peor_c)
interpretar(f'''**Interpretación.** {("Cumple" if len(adecuados) == 1 else "Cumplen") + " las cuatro condiciones de adecuación " + enumerar(adecuados) if adecuados else "Ningún modelo cumple las cuatro condiciones de adecuación"}{f", de {len(ADECUACION)} evaluados" if adecuados else ""}. {f"La condición que más modelos incumplen es la {peor_c.split(' ')[0]}, «{etiqueta_peor}», con {n_peor} {'modelo' if n_peor == 1 else 'modelos'}: {enumerar(falla[peor_c])}." if n_peor else "Ninguna condición se incumple."} En la tabla de sensibilidad, el número de modelos adecuados pasa de {n_sens.iloc[0]} con un umbral de 0,03 a {n_sens.iloc[-1]} con uno de 0,07{(lambda fuera: f", y {enumerar(fuera)} {'queda' if len(fuera) == 1 else 'quedan'} fuera con cualquier umbral, porque {'incumple' if len(fuera) == 1 else 'incumplen'} otra de las condiciones")([m for m in sin_iii if not sin_iii[m]]) if not all(sin_iii.values()) else ""}.

Esto quiere decir que {"hay modelos que el benchmark puede declarar adecuados para este conjunto" + ((". " + (texto_razones + texto_borde).strip().rstrip(".")) if (texto_razones + texto_borde).strip() else "") if adecuados else "ningún modelo satisface a la vez todas las condiciones del enunciado, de modo que, con estos datos y esta partición, ninguno se puede declarar una solución adecuada en sentido estricto"}. {"Hay que tener en cuenta que la condición (iv) depende de la variación entre series y no solo de un defecto del modelo. " if falla["(iv) Prueba consistente con validación"] else ""}{"Todos los modelos superan al trivial por más de δ." if not falla["(i) Supera al trivial"] else ""}

En consecuencia, se decide reportar el veredicto con el umbral de 0,05 como resultado exploratorio, sin usarlo para elegir el modelo final, fijar en el preregistro del Entregable 3 un umbral de calibración antes de entrenar y usar las fallas de este criterio como requisitos explícitos del modelo original.''')
""")
