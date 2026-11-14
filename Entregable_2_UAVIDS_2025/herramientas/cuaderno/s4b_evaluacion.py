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

Los intervalos por bloques remuestrean con reemplazo los 24 bloques de prueba y recalculan el F1 macro sobre la matriz de confusión agregada de los bloques elegidos. Hay que tener en cuenta que cada bloque de prueba contiene solo una o dos clases (14 bloques con una y 10 con dos), de modo que el F1 macro de un bloque, de un segmento o de un remuestreo se promedia sobre las clases presentes en él: promediar sobre las cinco contaría como cero el F1 de una clase que no aparece, lo que no es un error del modelo. Sobre la prueba completa, con las cinco clases, las dos definiciones coinciden.
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
print(f"Tres mejores modelos profundos por F1 macro medio de validación: {enumerar(TOP3)}")
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

md("s4-5-fichas-md", r"""
**Fichas de modelo y registro de experimentos.** La tabla sigue la plantilla 11.3 del enunciado para los 16 modelos, y el registro consolidado de todas las corridas, de todas las fases, con su configuración, semilla y resultados, queda en `resultados/registro_experimentos.csv`.
""")

code("s4-5-fichas", r"""
fases_registro = ["ajuste", "piloto", "ruido", "fase1", "fase2", "definitiva", "cv5x2", "ablacion", "curvas", "permutacion", "bootstrap"]
registro_exp = pd.concat([RES.tabla_fase(f) for f in fases_registro if len(RES.tabla_fase(f))], ignore_index=True)
columnas_reg = [c for c in ["id", "fase", "modelo", "semilla", "arquitectura", "hp_lr", "hp_weight_decay", "hp_batch", "particion", "max_epocas", "epocas", "mejor_epoca", "divergente",
                            "f1_ent", "f1_val", "f1_eva", "segundos", "memoria_pico_mb", "equipo"] if c in registro_exp.columns]
registro_exp[columnas_reg].to_csv(RESULTADOS / "registro_experimentos.csv", index=False, encoding="utf-8-sig")
DOI_FUENTE = {"DNN": "10.1109/cns66487.2025.11194990", "CNN 1D": "10.1109/cns66487.2025.11194990", "LSTM": "10.1109/cns66487.2025.11194990", "RNN": "10.1038/s41598-026-45063-6",
              "GRU": "10.1038/s41598-026-45063-6", "Transformer": "10.1038/s41598-026-45063-6", "BiLSTM": "ijisae.org/index.php/IJISAE/article/view/8549", "KAN": "10.1016/j.sysarc.2026.104038",
              "TabNet": "10.1609/aaai.v35i8.16826", "ResNet": "10.1109/cvpr.2016.90", "Autoencoder": "10.1126/science.1127647", "DBN": "10.1162/neco.2006.18.7.1527",
              "CNN-LSTM": "10.1109/cvpr.2015.7298878", "LightGBM": "lightgbm 4.7.0", "Logística": "scikit-learn 1.9.0", "Trivial": "Clase mayoritaria"}
commit = subprocess.run(["git", "-C", str(RAIZ_E2), "log", "-1", "--format=%h", "--", "codigo"], capture_output=True, text=True).stdout.strip() or "sin repositorio"
fichas = []
for m in ORDEN:
    sub_m = METRICAS[METRICAS["modelo"] == m]
    arq = ADENDA.get("arquitecturas_ganadoras", {}).get(m)
    fichas.append({"Modelo": m, "Familia": MOD.CATALOGO[m]["familia"] if m in MOD.CATALOGO else "No profundo", "Referencia o versión": DOI_FUENTE[m], "Implementación": f"codigo/, commit {commit}",
                   "Preentrenamiento": "Ninguno", "Parámetros": costo.loc[m, "Parámetros"], "FLOPs por muestra": costo.loc[m, "FLOPs"],
                   "Hiperparámetros finales": ", ".join(f"{k} {cientifico(v)}" if isinstance(v, float) else f"{k} {v}" for k, v in MEJORES_HP.get(m, {}).items()) or "No aplica",
                   "Intentos de ajuste": int((ajuste["modelo"] == m).sum()), "Arquitectura seleccionada": MOD.identificador(arq) if arq else "Definida en la literatura" if m in MOD.CATALOGO else "No aplica",
                   "Entrenamiento": "Adam, ReduceLROnPlateau, parada por F1 de validación, float32" if m in MOD.CATALOGO else "Parada por log-loss de validación" if m == "LightGBM" else "lbfgs" if m == "Logística" else "No aplica",
                   "Épocas efectivas (media)": costo.loc[m, "Épocas"], "Minutos por corrida": costo.loc[m, "Minutos"], "Latencia (ms)": costo.loc[m, "ms_por_muestra"],
                   "Corridas definitivas": len(sub_m), "Incidencias": f"{int(registro_exp.loc[registro_exp['modelo'] == m, 'divergente'].fillna(False).astype(bool).sum())} corridas divergentes"})
FICHAS = pd.DataFrame(fichas)
FICHAS.to_csv(RESULTADOS / "fichas_de_modelo.csv", index=False, encoding="utf-8-sig")
tabla(FICHAS, {"Parámetros": "{:,.0f}", "FLOPs por muestra": "{:,.0f}", "Épocas efectivas (media)": "{:.1f}", "Minutos por corrida": "{:.2f}", "Latencia (ms)": "{:.3f}"})
print(f"Registro de experimentos: {len(registro_exp):,} corridas en {registro_exp['fase'].nunique()} fases".replace(",", " "))
""")

md("s4-5-ajuste", r"""
**Ajuste, sobreajuste y consistencia entre validación y prueba.** Para cada modelo se calcula la brecha entre entrenamiento y validación, $g = s_{ent} - s_{val}$, y se prueba la hipótesis de no inferioridad de la tabla B: $H_0$: la mediana de $g$ es al menos δg = 0,05, frente a $H_1$: es menor, con Wilcoxon unilateral sobre $g - δ_g$ en las R semillas. La diferencia entre validación y prueba se estima con su IC 95 % por *bootstrap* por bloques, remuestreando por separado los bloques de cada conjunto, porque son series distintas.
""")

code("s4-5-brechas", r"""
NB_ENT = len(np.unique(bloques[ID_ENT]))
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
bp = BRECHAS[BRECHAS["Modelo"].isin(PROFUNDOS)]
vp = BRECHAS.set_index("Modelo")["Validación − prueba"]
caida_dl, caida_lgbm, caida_log = vp.loc[PROFUNDOS].median(), vp.get("LightGBM", np.nan), vp.get("Logística", np.nan)
if caida_dl > DELTA and caida_lgbm < caida_dl / 2:
    lectura_caida = (f"la caída entre validación y prueba es común a los modelos profundos (mediana de {num(caida_dl, 4)}) y a la regresión logística ({num(caida_log, 4)}), pero mucho menor en LightGBM "
                     f"({num(caida_lgbm, 4)}). Es decir, depende de qué series cayeron en cada conjunto y también del tipo de modelo: una hipótesis, que la sección 6.3 contrasta con la validación cruzada, "
                     "es que los árboles, que parten cada atributo por umbrales, son menos sensibles a los cambios de escala y de forma de las distribuciones entre series que las redes y el modelo lineal, "
                     "que dependen de combinaciones continuas de los atributos")
elif caida_dl > DELTA:
    lectura_caida = ("la caída entre validación y prueba es común a casi todos los modelos, incluidos los que no se seleccionaron con validación, como la regresión logística, de modo que se explica más "
                     "por qué series cayeron en cada conjunto que por un sobreajuste a la validación")
else:
    lectura_caida = "la caída entre validación y prueba es pequeña, de modo que la prueba es consistente con la validación"
con_ic_cero = [f["Modelo"] for _, f in BRECHAS.iterrows() if float(f["IC 95 % por bloques"].strip("[]").split("; ")[0].replace(",", ".")) <= 0 <= float(f["IC 95 % por bloques"].strip("[]").split("; ")[1].replace(",", "."))]
interpretar(f'''**Interpretación.** En los 13 modelos profundos, la brecha entre entrenamiento y validación va de {num(bp["Brecha g"].min(), 4)} ({bp.loc[bp["Brecha g"].idxmin(), "Modelo"]}) a {num(bp["Brecha g"].max(), 4)} ({bp.loc[bp["Brecha g"].idxmax(), "Modelo"]}), y la prueba de no inferioridad confirma que es menor que δg = 0,05 en {int((bp["p Wilcoxon (g < δg)"] < ALFA).sum())} de ellos. La diferencia entre validación y prueba va de {num(BRECHAS["Validación − prueba"].min(), 4)} a {num(BRECHAS["Validación − prueba"].max(), 4)}, y su IC por bloques contiene al cero en {len(con_ic_cero)} de los {len(BRECHAS)} modelos.

Esto significa que {"ningún modelo sobreajusta el entrenamiento más allá del margen" if (bp["p Wilcoxon (g < δg)"] < ALFA).all() else "algunos modelos no demuestran una brecha menor que el margen"}, y que {lectura_caida}. Hay que tener en cuenta que, con solo {NB_VAL} bloques de validación y {NB_PRU} de prueba, el intervalo por bloques es amplio, porque la unidad independiente es la serie y no el flujo.

Por tal motivo, se reporta el desempeño de prueba con su intervalo por bloques en la tabla principal, y la brecha entre validación y prueba entra en el criterio de adecuación de la sección 7.''')
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
cal_m = CALIB.groupby("modelo").mean(numeric_only=True)
cal_p = cal_m.loc[PROFUNDOS]
interpretar(f'''**Interpretación.** Sin recalibrar, el ECE de los modelos profundos va de {num(cal_p["ECE"].min(), 4)} ({cal_p["ECE"].idxmin()}) a {num(cal_p["ECE"].max(), 4)} ({cal_p["ECE"].idxmax()}), y el de LightGBM es de {num(cal_m.loc["LightGBM", "ECE"], 4)}. La temperatura ajustada en validación es mayor que 1 en {int((cal_p["T"] > 1).sum())} de los 13 modelos profundos, y con ella el ECE medio pasa de {num(cal_p["ECE"].mean(), 4)} a {num(cal_p["ECE con T"].mean(), 4)}; la versión adaptativa, con bins de igual tamaño, cambia de {num(cal_p["ECE adaptativo"].mean(), 4)} a {num(cal_p["ECE adaptativo con T"].mean(), 4)}.

Lo anterior indica que {"la mayoría de los modelos profundos es sobreconfiada en validación, porque la temperatura que la corrige, mayor que 1, suaviza las probabilidades" if (cal_p["T"] > 1).sum() > 6 else "la mayoría de los modelos profundos es subconfiada en validación, porque la temperatura que la corrige, menor que 1, agudiza las probabilidades, lo que es coherente con la pérdida ponderada por clase, que empuja probabilidad hacia las clases minoritarias" if (cal_p["T"] < 1).sum() > 6 else "los modelos profundos se reparten entre sobreconfiados y subconfiados en validación"}, y que una sola temperatura ajustada en validación {"corrige buena parte del error de calibración en prueba" if cal_p["ECE con T"].mean() < cal_p["ECE"].mean() else "no mejora la calibración en prueba, lo que es coherente con que validación y prueba sean series distintas"}. {f"Las temperaturas quedan entre {num(cal_p['T'].min(), 2)} y {num(cal_p['T'].max(), 2)}, muy cerca de 1, de modo que en validación los modelos ya están casi calibrados y el ECE de prueba refleja sobre todo el cambio entre series. " if (cal_p["T"] - 1).abs().max() <= 0.12 else ""}Así mismo, el ECE adaptativo {"confirma la lectura del de bins fijos, de modo que la conclusión no depende de la discretización" if np.sign(cal_p["ECE con T"].mean() - cal_p["ECE"].mean()) == np.sign(cal_p["ECE adaptativo con T"].mean() - cal_p["ECE adaptativo"].mean()) else "cambia en sentido contrario al de bins fijos, de modo que la conclusión sobre la temperatura depende de la discretización"}.

{"Por lo anterior, conviene reportar las probabilidades con la temperatura de validación cuando se usen como puntaje de riesgo, y declarar en el criterio de adecuación cuáles modelos quedan calibrados." if cal_p["ECE con T"].mean() < cal_p["ECE"].mean() else "Por lo anterior, conviene usar las probabilidades sin recalibrar, porque la temperatura de validación no las mejora en series nuevas, y declarar en el criterio de adecuación cuáles modelos quedan calibrados; recalibrar con datos de la serie de destino queda como requisito de una puesta en producción."}''')
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
tax = pd.DataFrame(taxonomia)
dominante = tax.loc[tax.groupby("Modelo")["% de los errores"].idxmax()].set_index("Modelo")
mejor_ind = max(METRICAS.loc[(METRICAS["modelo"] == m) & (METRICAS["semilla"] == SEMILLAS[0]), "prueba F1 macro"].iloc[0] for m in TOP3)
phi = correlacion.loc[TOP3, TOP3].to_numpy()[np.triu_indices(len(TOP3), 1)]
phi_lgbm = correlacion.loc[TOP3, "LightGBM"].to_numpy()
NOMBRE_CAT = {"Confusión Blackhole–Wormhole": "la confusión entre Blackhole y Wormhole", "Otras confusiones entre ataques": "las demás confusiones entre ataques",
              "Ataque predicho como Normal": "los ataques predichos como Normal", "Normal predicho como ataque": "los flujos normales predichos como ataque"}
partes_dom = []
for cat, sub_d in dominante.groupby("Categoría"):
    pct_txt = num(sub_d["% de los errores"].iloc[0], 1) if len(sub_d) == 1 else f"{num(sub_d['% de los errores'].min(), 1)} % a {num(sub_d['% de los errores'].max(), 1)}"
    partes_dom.append(f"{NOMBRE_CAT.get(cat, cat)} en {enumerar(sub_d.index.tolist())} ({pct_txt} % de sus errores)")
concentrados = compartidos >= 0.5
gana_comb = f1_ensamble - mejor_ind > DELTA
if concentrados and not gana_comb:
    lectura_err = "Esto quiere decir que los errores se concentran en los mismos flujos, de modo que, como advierte la sección 12.3 del enunciado, el benchmark mide sobre todo la dificultad de esos casos y no diferencias de arquitectura. En ese sentido, combinar modelos gana menos que δ, porque se equivocan juntos."
elif concentrados:
    lectura_err = "Esto quiere decir que buena parte de los errores se concentra en los mismos flujos, pero el promedio de probabilidades aún gana más que δ, porque corrige los flujos en que solo uno de los tres se equivoca."
elif gana_comb:
    lectura_err = "Esto quiere decir que los errores no se concentran en los mismos flujos: cada modelo falla casos distintos, y por eso promediar sus probabilidades gana más que δ."
else:
    lectura_err = "Esto quiere decir que los errores no se concentran en los mismos flujos, pero promediar las probabilidades gana menos que δ: en los flujos en que discrepan, el promedio no siempre favorece a la clase correcta, como muestra la distancia entre el ensamble y el oráculo."
interpretar(f'''**Interpretación.** En los tres mejores modelos y en LightGBM, la categoría que reúne más errores es {enumerar(partes_dom)}. De los flujos en que se equivoca al menos uno de los tres mejores, el {num(100 * compartidos, 1)} % lo fallan los tres; la correlación de sus errores va de {num(phi.min(), 3)} a {num(phi.max(), 3)}, y con LightGBM, de {num(phi_lgbm.min(), 3)} a {num(phi_lgbm.max(), 3)}, y el promedio de sus probabilidades obtiene un F1 macro de {num(f1_ensamble, 4)}, frente a {num(mejor_ind, 4)} del mejor de ellos con la misma semilla; un oráculo que eligiera siempre al que acierta llegaría a una exactitud de {num(oraculo, 4)}.

{lectura_err}

Por ello, se recomienda que el modelo original del Entregable 3 ataque {"la confusión entre Blackhole y Wormhole, que domina en " + enumerar(dominante.index[dominante["Categoría"] == "Confusión Blackhole–Wormhole"].tolist()) + " y coincide con la limitación del paper guía (sección 5)" if "Confusión Blackhole–Wormhole" in set(dominante["Categoría"]) else "la categoría dominante"}{", en lugar de buscar más capacidad o combinar arquitecturas" if f1_ensamble - mejor_ind <= DELTA else ", y que evalúe la combinación de arquitecturas como línea base"}.''')
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
sg = SEGMENTOS.set_index("Modelo")
clases_f1 = sg[[f"F1 {c}" for c in CLASES]].loc[PROFUNDOS].mean()
dif_puerto = (sg["Con puerto 654"] - sg["Sin puerto 654"]).loc[PROFUNDOS]
dif_casi = (sg["Con casi duplicado"] - sg["Sin casi duplicado"]).loc[PROFUNDOS]
iqr = {m: np.subtract(*np.percentile(v, [75, 25])) for m, v in f1_bloque.items()}
interpretar(f'''**Interpretación.** Promediando los 13 modelos profundos, la clase más difícil es {clases_f1.idxmin().replace("F1 ", "")}, con F1 de {num(clases_f1.min(), 3)}, y la más fácil, {clases_f1.idxmax().replace("F1 ", "")}, con {num(clases_f1.max(), 3)}. La exactitud en los flujos con el puerto 654 difiere de la de los flujos sin él entre {num(dif_puerto.min(), 3)} y {num(dif_puerto.max(), 3)} según el modelo, y los flujos de prueba con un casi duplicado en entrenamiento obtienen un F1 macro {f"mayor que los demás, entre {num(dif_casi.min(), 3)} y {num(dif_casi.max(), 3)}" if (dif_casi > 0).all() else f"menor que los demás, entre {num(-dif_casi.max(), 3)} y {num(-dif_casi.min(), 3)}" if (dif_casi < 0).all() else f"que difiere del de los demás entre {num(dif_casi.min(), 3)} y {num(dif_casi.max(), 3)} según el modelo"}. Entre los {NB_PRU} bloques de prueba, el rango intercuartílico del F1 macro va de {num(min(iqr.values()), 3)} a {num(max(iqr.values()), 3)}.

Es decir, {"el desempeño depende más de la serie y de la clase que del modelo: la dispersión entre bloques es de " + num(np.median(list(iqr.values())) / DELTA, 0) + " veces δ en mediana" if np.median(list(iqr.values())) > 2 * DELTA else "la dispersión entre bloques es pequeña frente a δ"}, y la diferencia por el puerto 654 indica {"que el modelo aprovecha la configuración del escenario que el puerto delata, un atajo indirecto a través de los atributos de tráfico" if dif_puerto.abs().max() > 2 * DELTA else "que el puerto, excluido de las entradas, no deja un atajo apreciable en los atributos de tráfico"}. Así mismo, los casi duplicados son pocos ({num(100 * casi_dup.mean(), 1)} % de la prueba), de modo que su efecto sobre el F1 global es acotado.

Por lo anterior, conviene leer las comparaciones de la sección 6 como pareadas sobre las mismas series y no como estimaciones del desempeño en un enjambre nuevo, y declarar la dependencia de la serie como amenaza a la validez externa.''')
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
peor_pert = ROBUSTEZ[list(AN.NIVELES)].mean().idxmin()
mas_rob = ROBUSTEZ["Índice medio"].idxmax()
menos_rob = ROBUSTEZ["Índice medio"].idxmin()
rho_rob = stats.spearmanr(limpio.loc[ROBUSTEZ.index], ROBUSTEZ["Índice medio"])
interpretar(f'''**Interpretación.** La perturbación que más degrada a los modelos es {peor_pert.lower()}, con un índice medio de {num(ROBUSTEZ[peor_pert].mean(), 3)}. El modelo más robusto en promedio es {mas_rob} ({num(ROBUSTEZ.loc[mas_rob, "Índice medio"], 3)}) y el menos robusto, {menos_rob} ({num(ROBUSTEZ.loc[menos_rob, "Índice medio"], 3)}); LightGBM obtiene {num(ROBUSTEZ.loc["LightGBM", "Índice medio"], 3)}, con {num(ROBUSTEZ.loc["LightGBM", "Atributos faltantes"], 3)} ante atributos faltantes, frente a una mediana de {num(ROBUSTEZ.loc[PROFUNDOS, "Atributos faltantes"].median(), 3)} en los modelos profundos.

Esto significa que la robustez {"no sigue el orden del desempeño sin perturbar" if not (rho_rob.pvalue < ALFA and rho_rob.statistic > 0) else "acompaña al desempeño sin perturbar"} (ρ de Spearman entre el F1 limpio y el índice medio = {num(rho_rob.statistic, 2)}, {valor_p(rho_rob.pvalue)}), de modo que {"es un eje propio de la comparación, como pide la discusión del enunciado" if not (rho_rob.pvalue < ALFA and rho_rob.statistic > 0) else "los mejores modelos son también los más robustos"}. Hay que tener en cuenta que imputar con la media de entrenamiento, que es 0 tras estandarizar, deja el valor en el centro de la distribución, lo que favorece a los modelos que no dependen de un solo atributo.

Por tal motivo, la tabla principal de la sección 6.1 incluye el índice medio de robustez, y la fragilidad ante {peor_pert.lower()} queda como criterio para el modelo original del Entregable 3, junto con la tokenización por atributo de la búsqueda transversal (sección 3.1).''')
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
inestables = INTERP.loc[INTERP["Estabilidad media (ρ)"] <= 0.7, "Modelo"].tolist()
infieles = INTERP.loc[INTERP["Área entre curvas"] <= 0, "Modelo"].tolist()
fiables = [m for m in INTERP["Modelo"] if m not in inestables + infieles]
lectura_interp = ("las explicaciones no son a la vez fieles y estables en todos los modelos"
                  + (f": en {enumerar(inestables)}, la estabilidad media entre semillas no pasa de 0,7, de modo que el orden de los atributos cambia según la semilla" if inestables else "")
                  + (f"{'; y en ' if inestables else ': en '}{enumerar(infieles)}, anular los atributos señalados no reduce la confianza más que un orden aleatorio" if infieles else "")
                  + (f"; en {enumerar(fiables)}, en cambio, son fieles y estables. Por eso, las atribuciones de {enumerar(inestables + infieles)} no deben leerse como evidencia del mecanismo" if fiables
                     else ", de modo que no deben leerse como evidencia del mecanismo"))
interpretar(f'''**Interpretación.** La estabilidad de las atribuciones entre semillas, medida con la ρ de Spearman media, va de {num(INTERP["Estabilidad media (ρ)"].min(), 3)} a {num(INTERP["Estabilidad media (ρ)"].max(), 3)} en los tres mejores modelos, con un mínimo por par de {num(INTERP["Estabilidad mínima (ρ)"].min(), 3)}. En la prueba de borrado, el área entre la curva aleatoria y la de la atribución es {"positiva en los tres modelos" if (INTERP["Área entre curvas"] > 0).all() else "positiva solo en " + enumerar(INTERP.loc[INTERP["Área entre curvas"] > 0, "Modelo"])}, de {num(INTERP["Área entre curvas"].min(), 3)} a {num(INTERP["Área entre curvas"].max(), 3)}. Los tres atributos más importantes son, {"; ".join(f"en {f['Modelo']}, {enumerar(f['Tres atributos más importantes'].split(', '))}" for _, f in INTERP.iterrows())}.

Lo anterior indica que {"las explicaciones son fieles, porque anular primero los atributos que señalan reduce la confianza más rápido que un orden aleatorio, y estables, porque se repiten entre semillas" if (INTERP["Área entre curvas"] > 0).all() and INTERP["Estabilidad media (ρ)"].min() > 0.7 else lectura_interp}. En cuanto al dominio, la coherencia se evalúa frente a lo que distingue a los ataques: Flooding eleva las tasas de envío, Sybil multiplica las identidades y altera el volumen, y Blackhole y Wormhole cambian el número de saltos y la entrega de paquetes; una atribución concentrada en atributos de volumen explicaría bien Flooding, pero no la frontera entre los dos ataques de enrutamiento.

Por ello, se recomienda usar estas atribuciones como diagnóstico y no como explicación causal, y contrastarlas en el Entregable 3 con atribuciones calculadas solo sobre los flujos de Blackhole y Wormhole.''')
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
ARTICULO_ABL = {"dropout": ("el *dropout*", "al *dropout*"), "ponderación de clases": ("la ponderación de clases", "a la ponderación de clases"), "log(1 + x)": ("la transformación log(1 + x)", "a la transformación log(1 + x)")}
val_completo = np.mean([r["f1_val"] for r in DEF if r["modelo"] == MEJOR_DL])
ABLACION.insert(4, "Diferencia en validación", [abl.loc[abl["variante_nombre"] == v_, "f1_val"].mean() - val_completo for v_ in ABLACION["Variante"]])
ABLACION[["p t Holm", "p Wilcoxon Holm"]] = np.column_stack([E.holm(ABLACION["p t"]), E.holm(ABLACION["p Wilcoxon"])])
tabla(ABLACION, {"F1 macro de prueba": "{:.4f}", "Diferencia con el completo": "{:+.4f}", "Diferencia en validación": "{:+.4f}", "p t": "{:.4f}", "p Wilcoxon": "{:.4f}", "p TOST": "{:.4f}", "d de Cohen": "{:.2f}", "p t Holm": "{:.4f}", "p Wilcoxon Holm": "{:.4f}"})
sig_abl = ABLACION[(ABLACION["p t Holm"] < ALFA) & (ABLACION["p Wilcoxon Holm"] < ALFA)]
indicio = ABLACION[(ABLACION["Diferencia con el completo"].abs() > DELTA) & ((ABLACION["p t Holm"] < ALFA) | (ABLACION["p Wilcoxon Holm"] < ALFA)) & ~ABLACION.index.isin(sig_abl.index)]
if (sig_abl["Diferencia con el completo"].abs() > DELTA).any():
    lectura_abl = "el componente con efecto relevante es " + enumerar(sig_abl.loc[sig_abl["Diferencia con el completo"].abs() > DELTA, "Variante"])
    decision_abl = "revisar en el Entregable 3 el componente con efecto relevante"
elif len(indicio):
    f_i = indicio.iloc[0]
    lectura_abl = (f"ningún componente cumple el criterio de las dos pruebas, pero «{f_i['Variante']}» se acerca: su diferencia supera δ, su IC excluye el cero y la "
                   + ("*t* pareada la declara significativa tras Holm (p = " + num(f_i["p t Holm"], 3) + "), aunque Wilcoxon no (p = " + num(f_i["p Wilcoxon Holm"], 3) + ")" if f_i["p t Holm"] < ALFA
                      else "prueba de Wilcoxon la declara significativa tras Holm (p = " + num(f_i["p Wilcoxon Holm"], 3) + "), aunque la *t* no (p = " + num(f_i["p t Holm"], 3) + ")")
                   + f"; en validación, el mismo cambio mueve el F1 en {num(f_i['Diferencia en validación'], 4)}, unas {num(abs(f_i['Diferencia en validación']) / DELTA, 1)} veces δ. Es decir, es el componente del pipeline con más evidencia de aportar")
    otras = [v_ for v_ in ABLACION.loc[ABLACION["p TOST"] < ALFA, "Variante"]]
    decision_abl = ("conservar el pipeline común, que incluye ese componente" + (", y no atribuir el desempeño " + " ni ".join(ARTICULO_ABL.get(v_.replace("sin ", "", 1), (v_, v_))[1] for v_ in otras) + ", que resultan equivalentes a no usarlos dentro de ±δ" if otras else ""))
else:
    lectura_abl = "ningún componente del pipeline común cambia el desempeño en una magnitud mayor que δ con evidencia estadística"
    decision_abl = "no atribuir el desempeño del mejor modelo a la ponderación, la transformación ni el dropout"
interpretar(f'''**Interpretación.** Sobre {MEJOR_DL}, la variante con el mayor cambio es «{ABLACION.loc[ABLACION["Diferencia con el completo"].abs().idxmax(), "Variante"]}», con una diferencia de {num(ABLACION["Diferencia con el completo"].abs().max() * np.sign(ABLACION.loc[ABLACION["Diferencia con el completo"].abs().idxmax(), "Diferencia con el completo"]), 4)} en F1 macro de prueba. {"Las diferencias significativas tras Holm, en ambas pruebas, son las de " + enumerar("«" + sig_abl["Variante"] + "»") if len(sig_abl) else "Ninguna diferencia es significativa tras Holm con las dos pruebas a la vez"}, y {"TOST declara equivalentes al modelo completo, dentro de ±δ, las variantes " + enumerar("«" + ABLACION.loc[ABLACION["p TOST"] < ALFA, "Variante"] + "»") if (ABLACION["p TOST"] < ALFA).any() else "TOST no declara equivalente al modelo completo ninguna variante"}.

Esto significa que {lectura_abl}. Hay que tener en cuenta que la ablación aísla componentes del pipeline y no de la arquitectura, porque el diseño del modelo viene de la literatura.

En consecuencia, se decide {decision_abl}.''')
""")
