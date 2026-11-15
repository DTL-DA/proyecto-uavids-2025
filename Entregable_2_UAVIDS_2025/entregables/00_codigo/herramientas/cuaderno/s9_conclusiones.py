from .base import code, md

md("s9", r"""
## 9. Discusión crítica y conclusiones

Esta sección responde las preguntas de la sección 6.4 del enunciado, cada una con la evidencia estadística de las secciones 4 a 8, declara las amenazas a la validez y cierra con las conclusiones y el trabajo futuro. Las cifras salen de las celdas anteriores, de modo que cambian con ellas si se vuelve a ejecutar el cuaderno.

### 9.1 Discusión crítica
""")

code("s9-discusion", r"""
# Insumos de la discusión, todos calculados en las secciones anteriores
f1p_m, f1v_m = F1P.mean(), F1V.mean()
mejor_dl_prueba = max(PROFUNDOS, key=lambda m: f1p_m[m])
comunes_atr = sorted(set.intersection(*[set(f["Tres atributos más importantes"].split(", ")) for _, f in INTERP.iterrows()]))
abl_log = ABLACION[ABLACION["Variante"].str.contains("log")].iloc[0]
abl_eq = [ARTICULO_ABL.get(v_.replace("sin ", "", 1), (v_, v_))[0] for v_ in ABLACION.loc[ABLACION["p TOST"] < ALFA, "Variante"]]
dominio_bw = dominante.index[dominante["Categoría"] == "Confusión Blackhole–Wormhole"].tolist()
bp_max = BRECHAS.set_index("Modelo").loc[PROFUNDOS, "Brecha g"]
ece_min = cal_m["ECE"].idxmin()
ece_min_dl = cal_p["ECE"].idxmin()
n_sobre = int((pend_a < 1).sum())
rob = ROBUSTEZ["Índice medio"]
rank_rob = {m: int((rob > rob[m]).sum()) + 1 for m in rob.index}
rank_f1 = {m: int((f1p_m[[x for x in ORDEN if x != "Trivial"]] > f1p_m[m]).sum()) + 1 for m in ORDEN if m != "Trivial"}
n_modelos = len(ORDEN) - 1
dec_rq1 = RQ1["Decisión"].value_counts()
pct_bloque, pct_modelo = 100 * var_bloque / total_var, 100 * var_modelo / total_var
ic_b2 = r2["IC 95 % por bloques"]
techo_cl = TECHO.iloc[0]
min_lgbm, min_dl = costo.loc["LightGBM", "Minutos"], costo.loc[MEJOR_DL, "Minutos"]

interpretar(f'''**¿Qué modelo funciona mejor y por qué?** Ninguno de forma inequívoca. En prueba, el F1 macro más alto es el de LightGBM ({num(f1p_m["LightGBM"], 4)}), y entre los profundos el de {mejor_dl_prueba} ({num(f1p_m[mejor_dl_prueba], 4)}); por validación, el mejor profundo es {MEJOR_DL} ({num(f1v_m[MEJOR_DL], 4)}). Sin embargo, el *Model Confidence Set* al 90 % conserva {len(MCS)} de los {n_modelos} modelos no triviales, de modo que con {NB_PRU} series de prueba los datos no separan un ganador. En cuanto al porqué, la ablación sobre {MEJOR_DL} muestra que el componente del pipeline con más evidencia es la transformación log(1 + x), cuya ausencia resta {num(abs(abl_log["Diferencia con el completo"]), 4)} de F1 en prueba y {num(abs(abl_log["Diferencia en validación"]), 4)} en validación{f", mientras que {enumerar(abl_eq)} resultan equivalentes a no usarlos dentro de ±δ" if abl_eq else ""}; así mismo, las atribuciones de los tres mejores coinciden en {enumerar(comunes_atr)}, atributos de pérdida y tamaño de paquetes. Es decir, lo que más pesa es la representación de los atributos, que todos los modelos comparten, y no la arquitectura.

**¿Qué modelo generaliza y se calibra mejor?** Ningún modelo sobreajusta el entrenamiento: la brecha entre entrenamiento y validación de los profundos no pasa de {num(bp_max.max(), 4)} ({bp_max.idxmax()}), por debajo de δg = 0,05. Lo que separa a los modelos es el paso a series nuevas: los profundos pierden en mediana {num(caida_dl, 4)} de F1 entre validación y prueba, la regresión logística {num(caida_log, 4)} y LightGBM {num(caida_lgbm, 4)}, de modo que el árbol potenciado es el que mejor generaliza a las series de prueba. En calibración, el menor ECE de prueba es el de {ece_min} ({num(cal_m.loc[ece_min, "ECE"], 4)}), y entre los profundos, el de {ece_min_dl} ({num(cal_p.loc[ece_min_dl, "ECE"], 4)}); {n_sobre} de los {len(pend_a)} modelos tienen pendiente de calibración menor que 1 en prueba, es decir, son sobreconfiados en series nuevas, aunque en validación su temperatura óptima queda cerca de 1.

**¿Qué modelo es más robusto ante perturbaciones y cambio de distribución?** Ante perturbaciones, el más robusto es {mas_rob} (índice {num(rob[mas_rob], 3)}) y el menos, {menos_rob} ({num(rob[menos_rob], 3)}); LightGBM queda en el puesto {rank_rob["LightGBM"]} de {len(rob)}, y la correlación de Spearman entre el F1 limpio y la robustez es de {num(rho_rob.statistic, 2)} ({valor_p(rho_rob.pvalue)}), de modo que robustez y desempeño son ejes distintos. Ante el cambio entre series, en cambio, el más robusto es LightGBM, por su caída menor entre validación y prueba. Esto plantea un compromiso: el modelo que mejor generaliza a series nuevas es de los más frágiles ante ruido y atributos faltantes, y {mas_rob}, el más robusto ante perturbaciones, ocupa el puesto {rank_f1[mas_rob]} de {n_modelos} en F1 de prueba.

**¿Son las diferencias estadísticamente significativas y prácticamente relevantes?** Solo algunas. En RQ1, ningún modelo del Top 10 supera al baseline del Entregable 1 por más de δ: de los 10 contrastes, {enumerar([f"{v} {('es ' if v == 1 else 'son ') if i == 0 else ''}{DECISION_TXT.get(k, (k.lower(), k.lower()))[0 if v == 1 else 1]}" for i, (k, v) in enumerate(dec_rq1.items())])}. En RQ2, LightGBM supera a {MEJOR_DL} por {num(abs(r2["Diferencia media"]), 4)}, una diferencia significativa y relevante según la regla preregistrada, pero que desaparece en la validación cruzada 5 × 2 ({num(d_cv.mean(), 4)}; {valor_p(nb["p"])}) y cuyo IC por bloques, {intervalo(ic_b2)}, contiene al cero: es real en estas series, pero no se puede afirmar que se repita en otras. En RQ3, los modelos que leen los atributos como secuencia y los que no suponen orden {"son equivalentes" if RQ3_DECISION == "Equivalente" else "quedan como «" + RQ3_DECISION.lower() + "»"}, con una diferencia de {num(coef, 4)} en el modelo mixto. La descomposición de la varianza lo resume: el bloque de prueba explica el {num(pct_bloque, 1)} % de la varianza del F1 por bloque, y el modelo, el {num(pct_modelo, 1)} %.

**¿Existe un compromiso relevante entre desempeño y costo?** No en la dirección esperada. Entre los profundos, la correlación entre parámetros y F1 es de {num(rho_p.statistic, 2)} ({valor_p(rho_p.pvalue)}), y LightGBM, que tiene el mayor F1 de prueba, entrena en {num(min_lgbm, 2)} minutos por corrida, frente a {num(min_dl, 2)} de {MEJOR_DL}. Además, en RQ5 {f"{enumerar(equiv5)} {'es equivalente' if len(equiv5) == 1 else 'son equivalentes'} a {MEJOR_DL} dentro de ±δ con menor latencia" if equiv5 else f"ningún modelo más barato es equivalente a {MEJOR_DL} dentro de ±δ"}, de modo que pagar más cómputo no compra F1 en este problema.

**¿Qué tan cerca está el mejor modelo del techo?** Frente al aprendizaje confiado, el techo de exactitud es {num(techo_cl["Exactitud máxima alcanzable"], 4)}, y {mejor_t} llega a {num(acc[mejor_t], 4)}, con γ = {num(g_cl, 3)}; frente a la cota de Cover y Hart, γ = {num(g_ch, 3)}. {"Queda, por tanto, un margen real de mejora" if min(g_cl, g_ch) >= 0.5 else "La mayor parte de la distancia entre el baseline y el techo ya está cubierta"}, que, por lo que muestran los errores (sección 4.5), está concentrado en la frontera entre Blackhole y Wormhole.

**¿Se confirman o se rechazan las hipótesis?** En RQ1 no se rechaza H0 a favor del Top 10: ningún modelo supera al baseline por más de δ. En RQ2 se rechaza H0, en contra del modelo profundo, aunque la ventaja de LightGBM queda acotada a las series de esta prueba. En RQ3 no se rechaza H0, y TOST respalda la equivalencia de los dos grupos. En RQ4, exploratoria, la limitación del paper guía {"es compartida" if compartida else "no es compartida"} por el Top 10: con la partición agrupada, la LSTM confunde Blackhole y Wormhole en el {num(100 * lstm_bench, 2)} % de esos flujos, y los diez modelos, entre el {num(100 * top10_bench.min(), 2)} % y el {num(100 * top10_bench.max(), 2)} %. En RQ5, exploratoria, la respuesta es {"afirmativa" if equiv5 else "negativa"}.''')
""")

md("s9-2", r"""
### 9.2 Amenazas a la validez y limitaciones
""")

code("s9-amenazas", r"""
# Número de caminos examinados (sección 9.3 del enunciado), contados sobre los registros de las secciones anteriores
n_cfg_ajuste, n_arq_f1, n_finalistas, n_corridas = len(ajuste), len(f1), int(FASE2["Finalistas"].sum()), len(registro_exp)
n_confirmatorios = len(RQ1) + len(RQ2) + 1
n_p_ab = len(NA) * (3 * K + 1) + int(NA["p permutación"].notna().sum()) + 2 * len(TABLA_B)
interpretar(f'''**Validez interna.** El mejor modelo profundo y los hiperparámetros se eligen con {NB_VAL} bloques de validación, y la fase 1 de la búsqueda se entrenó con el 25 % de los bloques (desviación 1), lo que hizo que su orden informara poco el de la fase 2 (ρ de {num(FASE2["ρ de Spearman fase 1–fase 2"].min(), 2)} a {num(FASE2["ρ de Spearman fase 1–fase 2"].max(), 2)}); por eso, la conclusión sobre los cuatro modelos con búsqueda se limita a su mejor finalista. La suspensión del equipo durante la fase 2 solo afectó el tiempo de seis corridas, que se corrigió. La réplica del paper guía valida el pipeline con cuatro de sus cinco modelos, pero no reproduce el resultado clave de la LSTM.

**Validez externa.** Todo el benchmark usa una sola versión de un solo conjunto de datos, con {NB_PRU} series de prueba aproximadas por bloques de 1000 flujos, y la serie explica el {num(pct_bloque, 1)} % de la varianza del F1; con otra elección de series, el orden de los modelos puede cambiar, como sugiere la validación cruzada 5 × 2 de RQ2. Además, el atajo del puerto 654 (sección 4.5) indica que parte del desempeño depende de la configuración de los escenarios simulados.

**Validez de constructo.** El F1 macro pesa por igual a las cinco clases, aunque en la práctica no todos los errores cuestan lo mismo: confundir dos ataques de enrutamiento es menos grave que dejar pasar un ataque como tráfico normal. Así mismo, la latencia y la energía se midieron en un solo equipo, en CPU y con 2 hilos por corrida, y el F1 por bloque se promedia sobre las clases presentes en cada bloque (aclaración 2 de la adenda).

**Validez estadística.** Con R = {R} semillas, el tope de cómputo, la diferencia mínima detectable en RQ1 es de {num(ADENDA["piloto"]["diferencia_minima_detectable"], 4)}, mayor que δ, de modo que una mejora real de magnitud δ podría pasar inadvertida; por eso los resultados no significativos se leen con TOST y con la prueba bayesiana. Las pruebas por filas, como McNemar y DeLong, son optimistas porque tratan los flujos como independientes, y las preguntas exploratorias RQ4 y RQ5 no tienen control del error por familia.

**Análisis examinados y tasa de falsos descubrimientos.** Para dimensionar los caminos que se bifurcan (sección 9.3 del enunciado), el benchmark examinó {len(ORDEN)} modelos, {n_cfg_ajuste} configuraciones de hiperparámetros en el ajuste, {num(n_arq_f1, 0)} arquitecturas en la fase 1 y {n_finalistas} finalistas en la fase 2, con {num(n_corridas, 0)} corridas registradas en total y {len(NOMBRES_M)} métricas por corrida. Las afirmaciones confirmatorias se apoyan solo en los {n_confirmatorios} contrastes preregistrados de RQ1 a RQ3, con Holm dentro de cada familia, de modo que las demás pruebas no aumentan su tasa de error. En cambio, solo las tablas A y B suman al menos {n_p_ab} valores p sin corrección por multiplicidad, y si todas sus hipótesis nulas fueran ciertas cabría esperar unos {num(0.05 * n_p_ab, 0)} rechazos por azar; por eso se leen como evidencia de dirección y no como hallazgos. Así mismo, elegir el máximo entre tantas configuraciones infla el puntaje de validación: el F1 interno del ajuste supera al de prueba en {num(np.mean(interno - externo), 4)} en promedio (sección 7), que es el sesgo del ganador acumulado junto con el cambio entre series. Por último, el umbral de ECE de la condición (iii) del criterio de adecuación se fijó después de ver los resultados (sección 7), de modo que ese veredicto es exploratorio y no sustenta ninguna conclusión confirmatoria.''')
""")

md("s9-3", r"""
### 9.3 Conclusiones y trabajo futuro
""")

code("s9-conclusiones", r"""
interpretar(f'''**Conclusiones.**

1. En UAVIDS-2025, con una partición agrupada por series y el mismo protocolo para todos, ningún modelo profundo del Top 10 supera al baseline del Entregable 1, {base}, por más de δ = 0,01 (RQ1): {enumerar(RQ1.loc[RQ1["Decisión"] == "Equivalente", "a"].tolist()) if (RQ1["Decisión"] == "Equivalente").any() else "ninguno"} {"es equivalente" if (RQ1["Decisión"] == "Equivalente").sum() == 1 else "son equivalentes"} a él dentro de ±δ, y ninguno alcanza el umbral de mejora de 0,8775 que fijó el Entregable 1.
2. LightGBM obtiene el mayor F1 macro de prueba, {num(f1p_m["LightGBM"], 4)}, y supera al mejor modelo profundo por validación, {MEJOR_DL}, de forma significativa y relevante (RQ2); sin embargo, la ventaja no se repite en la validación cruzada 5 × 2, de modo que depende de las series de esta prueba y no basta para descartar a los modelos profundos.
3. La serie pesa más que la arquitectura: el bloque de prueba explica el {num(pct_bloque, 1)} % de la varianza del F1 por bloque, frente al {num(pct_modelo, 1)} % del modelo, y los modelos secuenciales y sin orden son equivalentes (RQ3). En consecuencia, la ganancia del Entregable 3 no vendrá de una arquitectura más compleja.
4. La confusión entre Blackhole y Wormhole es la limitación compartida por las diez familias del Top 10 (RQ4) {f"y la categoría que más errores reúne en {enumerar(dominio_bw)}" if dominio_bw else ""}, de modo que es el hueco de investigación que se traslada al Entregable 3; el resultado clave del paper guía, el 94 % de la LSTM, no se reproduce con su propio protocolo.
5. El pipeline es reproducible: en un entorno limpio, las corridas repetidas dan el mismo F1 hasta el último decimal, y la tabla principal se recalcula desde las corridas guardadas.

**Trabajo futuro.** Para el Entregable 3 se propone un modelo original que extienda el paper guía con un mecanismo dirigido a la frontera entre Blackhole y Wormhole, como la pérdida contrastiva supervisada o la mezcla de negativos difíciles, o una etapa jerárquica dedicada a ese par, y que se compare con {base} y con LightGBM con el mismo protocolo. Así mismo, conviene elegir el modelo con validación cruzada por bloques, en lugar de una sola partición de validación, para reducir la dependencia de la serie que mostró RQ2; recalibrar las probabilidades con datos de la serie de destino; exigir robustez ante ruido y atributos faltantes, y, cuando exista, evaluar el modelo en un segundo conjunto de datos de intrusiones en redes de drones.''')
""")

md("s10", r"""
## 10. Bibliografía

Se incluyen solo las referencias citadas en el informe, con su DOI cuando lo tienen; los metadatos de las referencias nuevas de este entregable se consultaron en OpenAlex (`literatura/referencias_e2.json`). La misma lista se exporta en formato BibTeX, con los mismos DOI, en `entregables/11_literatura/referencias.bib`.
""")

code("s10-bibliografia", r"""
import re
import unicodedata

REFS_E2 = json.loads((LIT2 / "referencias_e2.json").read_text(encoding="utf-8"))
# Año y fuente de publicación cuando la versión citada es la de congreso y el DOI es el de arXiv
CONGRESO = {"10.48550/arxiv.1706.03762": (2017, "Advances in Neural Information Processing Systems 30"), "10.48550/arxiv.1612.01474": (2017, "Advances in Neural Information Processing Systems 30"),
            "10.48550/arxiv.1506.02142": (2016, "International Conference on Machine Learning"), "10.48550/arxiv.1703.01365": (2017, "International Conference on Machine Learning"),
            "10.48550/arxiv.1706.04599": (2017, "International Conference on Machine Learning"), "10.48550/arxiv.2004.11362": (2020, "Advances in Neural Information Processing Systems 33"),
            "10.48550/arxiv.2010.01028": (2020, "Advances in Neural Information Processing Systems 33"), "10.48550/arxiv.2106.11959": (2021, "Advances in Neural Information Processing Systems 34")}
def paginas_fmt(paginas):
    # Un artículo de una sola página o con número de artículo se escribe sin rango
    partes = [x.strip() for x in str(paginas).replace("–", "-").split("-") if x.strip()]
    return partes[0] if len(partes) == 2 and partes[0] == partes[1] else "–".join(partes)
def formato_ref(doi, m):
    autores = [partes_nombre(a) for a in m["autores"]]
    lista = [autor_apa(a, n) for a, n in autores]
    texto_aut = lista[0] if len(lista) == 1 else ", ".join(lista[:-1]) + " y " + lista[-1] if len(lista) <= 3 else ", ".join(lista[:3]) + ", et al."
    anio, fuente = CONGRESO.get(doi, (m["anio"], m["fuente"]))
    titulo = m["titulo"].replace("$n$", "n")
    if doi in CONGRESO or m.get("tipo") in ("proceedings-article", "conference-paper") or "Conference" in (fuente or "") or "Proceedings" in (fuente or ""):
        detalle = f"En *{fuente}*."
    elif m.get("tipo") == "book":
        titulo = f"*{titulo}*" + (" (3.ª ed.)" if doi == "10.1002/9781118548387" else "")
        detalle = "Wiley."
    else:
        detalle = f"*{fuente}" + (f", {m['volumen']}*" if m.get("volumen") else "*") + (f"({m['numero']})" if m.get("numero") else "") + (f", {paginas_fmt(m['paginas'])}." if m.get("paginas") else ".")
    return f"- {texto_aut} ({anio}). {titulo}. {detalle} <https://doi.org/{doi}>"
SIN_DOI = [
    "- Benavoli, A., Corani, G., Demšar, J. y Zaffalon, M. (2017). Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis. *Journal of Machine Learning Research, 18*(77), 1–36.",
    "- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2.ª ed.). Lawrence Erlbaum Associates.",
    "- DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics, 44*(3), 837–845.",
    "- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.",
    "- Efron, B. y Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.",
    "- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association, 32*(200), 675–701.",
    "- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M. y Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*(11), 665–673.",
    "- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B. y Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research, 13*, 723–773.",
    "- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.",
    "- Kingma, D. P. y Ba, J. (2015). Adam: A method for stochastic optimization. En *International Conference on Learning Representations*. <https://arxiv.org/abs/1412.6980>",
    "- Lopez-Paz, D. y Oquab, M. (2017). Revisiting classifier two-sample tests. En *International Conference on Learning Representations*.",
    "- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157.",
    "- Nadeau, C. y Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281.",
    "- Page, M. J., McKenzie, J. E., Bossuyt, P. M., et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ, 372*, n71. <https://doi.org/10.1136/bmj.n71>",
    "- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics, 15*(6), 657–680.",
    "- SCImago (2026). *SJR: SCImago Journal & Country Rank* [Conjunto de datos]. <https://www.scimagojr.com>",
    "- Shapiro, S. S. y Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika, 52*(3/4), 591–611.",
    "- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin, 1*(6), 80–83.",
    "- Zeng, Q. (2025). *UAV--IDS2025* [Código fuente]. GitHub. <https://github.com/QingliZeng/UAV--IDS2025>",
    "- Zeng, Q., Bashir, A. y Nait-Abdesselam, F. (2025). *UAVIDS-2025* [Conjunto de datos]. Zenodo. <https://doi.org/10.5281/zenodo.15336998>",
]
FUENTES = {"International journal of intelligent engineering and systems": "International Journal of Intelligent Engineering and Systems",
           "Concurrency and Computation Practice and Experience": "Concurrency and Computation: Practice and Experience", "arXiv (Cornell University)": "arXiv"}
def referencia_estudio(doi):
    m = METADATOS[doi]
    autores = [autor_apa(a, n) for a, n in map(partes_nombre, m["autores"])]
    lista = autores[0] if len(autores) == 1 else ", ".join(autores[:-1]) + " y " + autores[-1] if len(autores) <= 3 else ", ".join(autores[:3]) + ", et al."
    titulo = m["titulo"].capitalize() if m["titulo"].isupper() else m["titulo"]
    fuente = m["fuente"].title() if (m.get("fuente") or "").isupper() else m.get("fuente")
    fuente = FUENTES.get(fuente, fuente)
    paginas = m.get("paginas") or ""
    if m.get("tipo") == "conference-paper":
        detalle = f"En *{fuente}*" + (f" (pp. {paginas_fmt(paginas)})." if paginas else ".")
    else:
        detalle = f"*{fuente}" + (f", {m['volumen']}*" if m.get("volumen") else "*") + (f"({m['numero']})" if m.get("numero") else "") + (f", {paginas_fmt(paginas)}." if paginas else ".")
    enlace_ = m.get("url") if doi.startswith("sin-doi") else f"https://doi.org/{doi}"
    return f"- {lista} ({m['anio']}{SUFIJO.get(doi, '')}). {titulo}. {detalle} <{enlace_}>"
BIBLIOGRAFIA = sorted([referencia_estudio(e["doi"]) for e in EXT] + [formato_ref(d, m) for d, m in REFS_E2.items()] + SIN_DOI, key=clave_orden)
interpretar("\n".join(BIBLIOGRAFIA))

# Misma bibliografía en BibTeX (sección 8 del enunciado), generada de los mismos metadatos
def bib_texto(t):
    return str(t).replace("\\", "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("$n$", "n")
def bib_entrada(tipo, apellido, anio, titulo, campos, usadas):
    base = "".join(c for c in unicodedata.normalize("NFKD", apellido.lower()) if c.isalnum() and not unicodedata.combining(c))
    palabra = next((p for p in re.findall(r"[a-z0-9]+", clave_orden(titulo)) if len(p) > 3), "ref")
    clave = f"{base}{anio}{palabra}"
    while clave in usadas:
        clave += "x"
    usadas.add(clave)
    lineas = [f"  title = {{{{{bib_texto(titulo)}}}}}", f"  year = {{{anio}}}"] + [f"  {k} = {{{bib_texto(v) if k not in ('doi', 'url') else v}}}" for k, v in campos.items() if v]
    return f"@{tipo}{{{clave},\n" + ",\n".join(lineas) + "\n}"
def bib_autores(nombres):
    return " and ".join(f"{a}, {n}" if n.strip() else a for a, n in map(partes_nombre, nombres))
usadas_bib, ENTRADAS_BIB = set(), []
for e in EXT:
    doi = e["doi"]
    m = METADATOS[doi]
    fuente = m["fuente"].title() if (m.get("fuente") or "").isupper() else m.get("fuente")
    fuente = FUENTES.get(fuente, fuente)
    titulo = m["titulo"].capitalize() if m["titulo"].isupper() else m["titulo"]
    enlace_ = {"url": m.get("url")} if doi.startswith("sin-doi") else {"doi": doi}
    if m.get("tipo") == "conference-paper":
        campos = {"author": bib_autores(m["autores"]), "booktitle": fuente, "pages": paginas_fmt(m.get("paginas") or "").replace("–", "--"), **enlace_}
        ENTRADAS_BIB.append(bib_entrada("inproceedings", partes_nombre(m["autores"][0])[0], m["anio"], titulo, campos, usadas_bib))
    else:
        campos = {"author": bib_autores(m["autores"]), "journal": fuente, "volume": m.get("volumen"), "number": m.get("numero"), "pages": paginas_fmt(m.get("paginas") or "").replace("–", "--"), **enlace_}
        ENTRADAS_BIB.append(bib_entrada("article", partes_nombre(m["autores"][0])[0], m["anio"], titulo, campos, usadas_bib))
for doi, m in REFS_E2.items():
    anio, fuente = CONGRESO.get(doi, (m["anio"], m["fuente"]))
    comunes = {"author": bib_autores(m["autores"]), "doi": doi}
    apellido = partes_nombre(m["autores"][0])[0]
    if doi in CONGRESO or m.get("tipo") in ("proceedings-article", "conference-paper") or "Conference" in (fuente or "") or "Proceedings" in (fuente or ""):
        ENTRADAS_BIB.append(bib_entrada("inproceedings", apellido, anio, m["titulo"], {**comunes, "booktitle": fuente, "pages": paginas_fmt(m.get("paginas") or "").replace("–", "--")}, usadas_bib))
    elif m.get("tipo") == "book":
        ENTRADAS_BIB.append(bib_entrada("book", apellido, anio, m["titulo"], {**comunes, "publisher": "Wiley", "edition": "3" if doi == "10.1002/9781118548387" else None}, usadas_bib))
    elif m.get("tipo") == "preprint":
        ENTRADAS_BIB.append(bib_entrada("misc", apellido, anio, m["titulo"], {**comunes, "howpublished": fuente}, usadas_bib))
    else:
        ENTRADAS_BIB.append(bib_entrada("article", apellido, anio, m["titulo"], {**comunes, "journal": fuente, "volume": m.get("volumen"), "number": m.get("numero"),
                                                                              "pages": paginas_fmt(m.get("paginas") or "").replace("–", "--")}, usadas_bib))
# Referencias sin DOI de la lista anterior, transcritas con los mismos datos (solo iniciales donde la lista tiene iniciales)
SIN_DOI_BIB = [
    ("article", "Benavoli", 2017, "Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis", {"author": "Benavoli, A. and Corani, G. and Demšar, J. and Zaffalon, M.", "journal": "Journal of Machine Learning Research", "volume": 18, "number": 77, "pages": "1--36"}),
    ("book", "Cohen", 1988, "Statistical power analysis for the behavioral sciences", {"author": "Cohen, J.", "edition": "2", "publisher": "Lawrence Erlbaum Associates"}),
    ("article", "DeLong", 1988, "Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach", {"author": "DeLong, E. R. and DeLong, D. M. and Clarke-Pearson, D. L.", "journal": "Biometrics", "volume": 44, "number": 3, "pages": "837--845"}),
    ("article", "Demšar", 2006, "Statistical comparisons of classifiers over multiple data sets", {"author": "Demšar, J.", "journal": "Journal of Machine Learning Research", "volume": 7, "pages": "1--30"}),
    ("book", "Efron", 1993, "An introduction to the bootstrap", {"author": "Efron, B. and Tibshirani, R. J.", "publisher": "Chapman & Hall"}),
    ("article", "Friedman", 1937, "The use of ranks to avoid the assumption of normality implicit in the analysis of variance", {"author": "Friedman, M.", "journal": "Journal of the American Statistical Association", "volume": 32, "number": 200, "pages": "675--701"}),
    ("article", "Geirhos", 2020, "Shortcut learning in deep neural networks", {"author": "Geirhos, R. and Jacobsen, J.-H. and Michaelis, C. and Zemel, R. and Brendel, W. and Bethge, M. and Wichmann, F. A.", "journal": "Nature Machine Intelligence", "volume": 2, "number": 11, "pages": "665--673"}),
    ("article", "Gretton", 2012, "A kernel two-sample test", {"author": "Gretton, A. and Borgwardt, K. M. and Rasch, M. J. and Schölkopf, B. and Smola, A.", "journal": "Journal of Machine Learning Research", "volume": 13, "pages": "723--773"}),
    ("article", "Holm", 1979, "A simple sequentially rejective multiple test procedure", {"author": "Holm, S.", "journal": "Scandinavian Journal of Statistics", "volume": 6, "number": 2, "pages": "65--70"}),
    ("inproceedings", "Kingma", 2015, "Adam: A method for stochastic optimization", {"author": "Kingma, D. P. and Ba, J.", "booktitle": "International Conference on Learning Representations", "url": "https://arxiv.org/abs/1412.6980"}),
    ("inproceedings", "Lopez-Paz", 2017, "Revisiting classifier two-sample tests", {"author": "Lopez-Paz, D. and Oquab, M.", "booktitle": "International Conference on Learning Representations"}),
    ("article", "McNemar", 1947, "Note on the sampling error of the difference between correlated proportions or percentages", {"author": "McNemar, Q.", "journal": "Psychometrika", "volume": 12, "number": 2, "pages": "153--157"}),
    ("article", "Nadeau", 2003, "Inference for the generalization error", {"author": "Nadeau, C. and Bengio, Y.", "journal": "Machine Learning", "volume": 52, "number": 3, "pages": "239--281"}),
    ("article", "Page", 2021, "The PRISMA 2020 statement: An updated guideline for reporting systematic reviews", {"author": "Page, M. J. and McKenzie, J. E. and Bossuyt, P. M. and others", "journal": "BMJ", "volume": 372, "pages": "n71", "doi": "10.1136/bmj.n71"}),
    ("article", "Schuirmann", 1987, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability", {"author": "Schuirmann, D. J.", "journal": "Journal of Pharmacokinetics and Biopharmaceutics", "volume": 15, "number": 6, "pages": "657--680"}),
    ("misc", "SCImago", 2026, "SJR: SCImago Journal & Country Rank", {"author": "{SCImago}", "howpublished": "Conjunto de datos", "url": "https://www.scimagojr.com"}),
    ("article", "Shapiro", 1965, "An analysis of variance test for normality (complete samples)", {"author": "Shapiro, S. S. and Wilk, M. B.", "journal": "Biometrika", "volume": 52, "number": "3/4", "pages": "591--611"}),
    ("article", "Wilcoxon", 1945, "Individual comparisons by ranking methods", {"author": "Wilcoxon, F.", "journal": "Biometrics Bulletin", "volume": 1, "number": 6, "pages": "80--83"}),
    ("misc", "Zeng", 2025, "UAV--IDS2025", {"author": "Zeng, Q.", "howpublished": "Código fuente, GitHub", "url": "https://github.com/QingliZeng/UAV--IDS2025"}),
    ("misc", "Zeng", 2025, "UAVIDS-2025", {"author": "Zeng, Q. and Bashir, A. and Nait-Abdesselam, F.", "howpublished": "Conjunto de datos, Zenodo", "doi": "10.5281/zenodo.15336998"}),
]
ENTRADAS_BIB += [bib_entrada(t, a, y_, ti, c, usadas_bib) for t, a, y_, ti, c in SIN_DOI_BIB]
assert len(ENTRADAS_BIB) == len(BIBLIOGRAFIA), f"BibTeX con {len(ENTRADAS_BIB)} entradas y bibliografía con {len(BIBLIOGRAFIA)}"
(LIT2 / "referencias.bib").write_text("% Bibliografía del Entregable 2 (UAVIDS-2025), generada por la sección 10 del cuaderno. Codificación UTF-8.\n\n"
                                      + "\n\n".join(sorted(ENTRADAS_BIB, key=lambda t: clave_orden(t.split("{", 1)[1]))) + "\n", encoding="utf-8")
""")
