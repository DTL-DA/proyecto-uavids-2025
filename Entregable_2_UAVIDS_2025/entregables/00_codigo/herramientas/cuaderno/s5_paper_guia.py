from .base import code, md

md("s5", r"""
## 5. Paper guía y replicación

En esta sección se presenta el paper guía, se replica parcialmente su resultado principal con su propio protocolo para validar el pipeline y se cuantifica una limitación concreta de su modelo, que es el candidato al hueco de investigación del modelo original del Entregable 3.

### 5.1 Referencia, modelo y justificación

**Referencia.** Zeng, Q., Bashir, A. y Naït-Abdesselam, F. (2025). UAVIDS-2025: A Benchmark Dataset for Intrusion Detection in UAV Networks Using Machine Learning Techniques. En *IEEE Conference on Communications and Network Security (CNS)* (pp. 1–9). <https://doi.org/10.1109/cns66487.2025.11194990>. El código de los autores está en <https://github.com/QingliZeng/UAV--IDS2025>.

**Modelo original.** El artículo presenta el conjunto de datos y lo valida con diez modelos, de los cuales los de Deep Learning son una DNN de 16 → 128 → 64 → 32 → 5, una CNN 1D de 32 y 64 filtros y una LSTM de 64 unidades, las tres con *dropout* de 0,2 (sus formulaciones están en las fichas del Entregable 1, sección 3.3). Todos se entrenan con entropía cruzada sin ponderar, Adam con tasa de 10⁻³, lotes de 256, hasta 50 épocas y parada temprana sobre la pérdida de un 10 % del entrenamiento, con paciencia de 5. Los supuestos del protocolo son tres: los 16 atributos numéricos describen el flujo, la estandarización se ajusta con entrenamiento, y una partición aleatoria estratificada 80/20 con semilla 42 basta para evaluar. Su resultado principal entre los modelos profundos es el de la LSTM, con F1 de 94 %, y entre todos, el de XGBoost, con 96 %.

**Justificación.** Es el artículo que define el problema y el conjunto de datos, de modo que su relevancia e idoneidad son máximas; es la referencia de los 12 estudios incluidos que usan UAVIDS-2025, y es el único de los 43 con código público, lo que permite replicarlo bajo sus condiciones exactas. Además, sus tres modelos profundos son los modelos base del proyecto y su XGBoost es el antecedente del baseline clásico fuerte.

### 5.2 Réplica parcial

**Criterio de éxito, declarado en el preregistro.** La réplica es exitosa si la diferencia absoluta entre la media de 10 réplicas (semillas y particiones de 42 a 51, como la opción `--seeds` del código de los autores) y el valor reportado es menor que δ = 0,01, o si el valor reportado cae dentro del IC 95 % de esa media. La métrica principal es el F1 macro y, como sensibilidad, el ponderado, porque el artículo no declara el promedio de su columna de F1.
""")

code("s5-replica", r"""
replica = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in sorted((RESULTADOS / "replica").glob("*.json"))])
filas_r = []
for m in ["LR", "XGBoost", "DNN", "CNN", "LSTM"]:
    sub_r = replica[replica["modelo"] == m]
    for metrica, nombre in (("f1_macro", "F1 macro"), ("f1_ponderado", "F1 ponderado")):
        v = sub_r[metrica].to_numpy()
        margen = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        reportado = sub_r["reportado_f1"].iloc[0]
        exito = abs(v.mean() - reportado) < DELTA or (v.mean() - margen <= reportado <= v.mean() + margen)
        filas_r.append({"Modelo": m, "Métrica": nombre, "Réplicas": len(v), "Media": v.mean(), "DE": v.std(ddof=1), "IC 95 %": intervalo((v.mean() - margen, v.mean() + margen)),
                        "Reportado": reportado, "Diferencia": v.mean() - reportado, "Éxito": "sí" if exito else "no"})
REPLICA = pd.DataFrame(filas_r)
REPLICA.to_csv(RESULTADOS / "replica.csv", index=False, encoding="utf-8-sig")
tabla(REPLICA, {"Media": "{:.4f}", "DE": "{:.4f}", "Reportado": "{:.2f}", "Diferencia": "{:+.4f}"})
lstm_r = REPLICA[(REPLICA["Modelo"] == "LSTM") & (REPLICA["Métrica"] == "F1 macro")].iloc[0]
macro_r = REPLICA[REPLICA["Métrica"] == "F1 macro"].set_index("Modelo")
otros_ok = [m for m in ["LR", "XGBoost", "DNN", "CNN"] if macro_r.loc[m, "Éxito"] == "sí"]
lstm_v = replica.loc[replica["modelo"] == "LSTM", "f1_macro"].to_numpy()
umbral_redondeo = lstm_r["Reportado"] - 0.005
orden_invertido = macro_r.loc["LSTM", "Media"] < min(macro_r.loc["DNN", "Media"], macro_r.loc["CNN", "Media"])
if lstm_r["Éxito"] == "sí":
    lectura_replica = '''Esto significa que el pipeline de este proyecto, con los mismos datos y el mismo protocolo, reproduce las cifras de los autores dentro del margen de relevancia, de modo que la diferencia entre esas cifras y las del benchmark no se debe a un error de implementación sino a la partición y al protocolo. Las discrepancias que quedan son del orden del redondeo del artículo, que reporta porcentajes enteros.

Por tal motivo, se da por validado el pipeline y se usa la réplica como punto de comparación de la limitación de la sección 5.3.'''
else:
    lectura_replica = (f'''Esto significa que la implementación del protocolo queda validada por los modelos que sí se reproducen ({enumerar(otros_ok)}), pero el resultado clave no: con el mismo código y los mismos datos, la LSTM queda {num(abs(lstm_r["Diferencia"]), 4)} por debajo de lo reportado, y ninguna de las {len(lstm_v)} réplicas alcanza {num(umbral_redondeo, 3)}, el menor valor que se redondea a {num(100 * lstm_r["Reportado"], 0)} % (la mejor llega a {num(lstm_v.max(), 4)})'''
                       + (". Además, el orden de la tabla IV se invierte: la LSTM queda por debajo de la DNN y de la CNN, de modo que la ventaja de la recurrencia que sugiere el artículo no se sostiene" if orden_invertido else "")
                       + f'''. Hay que tener en cuenta que el código de los autores también está en PyTorch, de modo que la inicialización y el entrenamiento son los mismos, y que el artículo no reporta cuántas corridas promedia ni su dispersión: el {num(100 * lstm_r["Reportado"], 0)} % podría venir de una sola ejecución favorable o de otra versión del conjunto de datos, y esta réplica no puede distinguir entre esas explicaciones.

Por tal motivo, se da por validado el pipeline con los modelos que sí se reproducen, se declara como no reproducido el resultado clave de la LSTM, y se usa la réplica como punto de comparación de la limitación de la sección 5.3.''')
interpretar(f'''**Interpretación.** En el resultado clave, la LSTM, la réplica obtiene un F1 macro de {num(lstm_r["Media"], 4)} ± {num(lstm_r["DE"], 4)} en 10 réplicas, con IC 95 % de {lstm_r["IC 95 %"]}, frente al 0,94 reportado: la diferencia es de {num(lstm_r["Diferencia"], 4)}, y la réplica {"es exitosa" if lstm_r["Éxito"] == "sí" else "no cumple el criterio"}. De los cinco modelos, {int((REPLICA[REPLICA["Métrica"] == "F1 macro"]["Éxito"] == "sí").sum())} cumplen el criterio con F1 macro y {int((REPLICA[REPLICA["Métrica"] == "F1 ponderado"]["Éxito"] == "sí").sum())} con F1 ponderado.

{lectura_replica}''')
""")

md("s5-3", r"""
### 5.3 Limitación identificada en el paper guía

La limitación declarada en el preregistro es la **confusión entre Blackhole y Wormhole**, que se cuantifica con la tasa de confusión del par, $\frac{n_{B \to W} + n_{W \to B}}{n_B + n_W}$, sobre los flujos de esas dos clases. Las dos son ataques de enrutamiento: en Blackhole, un nodo atrae rutas y descarta los paquetes, y en Wormhole, dos nodos crean un túnel que altera las rutas, de modo que ambos cambian el número de saltos y la entrega de paquetes de forma parecida. Bakiri et al. (2026) la señalan como la frontera más difícil de UAVIDS-2025, y el Entregable 1 midió que la CNN 1D predecía como Wormhole el 17,6 % de los flujos de Blackhole. La limitación se cuantifica en tres escenarios: con el modelo del paper bajo su propio protocolo (la réplica), con el mismo modelo bajo la partición agrupada (la ejecución definitiva) y con tres modelos del Top 10 en el piloto, sobre validación, para saber si es propia del paper o compartida por la familia. Los intervalos se calculan por *bootstrap* por bloques de 1000 flujos, agregando las confusiones de todas las semillas en cada bloque; en el piloto, el intervalo es el de la primera semilla, tal como quedó en la adenda del preregistro.
""")

code("s5-limitacion", r"""
lim_filas = []
for m in ["DNN", "CNN", "LSTM", "XGBoost"]:
    for _, r_r in replica[replica["modelo"] == m].iterrows():
        z = np.load(RESULTADOS / "replica" / f"{m}_{r_r['semilla']}.npz")
        v, p = y[z["indices_prueba"]], z["prediccion"].astype(int)
        lim_filas.append({"Escenario": "Réplica, partición aleatoria 80/20", "Modelo": m, "semilla": r_r["semilla"], "tasa": E.tasa_ruteo(v, p), "indices": z["indices_prueba"], "pred": p})
for m in ["DNN", "CNN 1D", "LSTM"] + CONFIG["modelos"]["top10"] + ["LightGBM"]:
    for s in SEMILLAS:
        r_d = next(x for x in DEF if x["modelo"] == m and x["semilla"] == s)
        p = AN.prob(r_d, "eva").argmax(1)
        lim_filas.append({"Escenario": "Benchmark, partición agrupada (prueba)", "Modelo": m, "semilla": s, "tasa": E.tasa_ruteo(y[ID_PRU], p), "indices": ID_PRU, "pred": p})
LIM = pd.DataFrame(lim_filas)
def ic_limitacion(grupo):
    # Se agregan las semillas: por bloque, se suman las confusiones del par y los flujos del par de todas las semillas,
    # y cada réplica del bootstrap remuestrea los bloques, de modo que el intervalo es el de la tasa media
    ids_b = np.unique(np.concatenate([bloques[ix] for ix in grupo["indices"]]))
    num_b, den_b = np.zeros(len(ids_b)), np.zeros(len(ids_b))
    b_, w_ = E.PAR_RUTEO
    for _, f in grupo.iterrows():
        v, pr, pos = y[f["indices"]], f["pred"], np.searchsorted(ids_b, bloques[f["indices"]])
        num_b += np.bincount(pos, weights=((v == b_) & (pr == w_)) | ((v == w_) & (pr == b_)), minlength=len(ids_b))
        den_b += np.bincount(pos, weights=(v == b_) | (v == w_), minlength=len(ids_b))
    W = AN.pesos_bootstrap(len(ids_b), repeticiones=2000, semilla=42)
    boot = (W @ num_b) / np.maximum(W @ den_b, 1)
    return pd.Series({"Tasa media": grupo["tasa"].mean(), "DE entre semillas": grupo["tasa"].std(), "IC 95 % por bloques": intervalo(100 * np.array(E.ic_percentil(boot)), 1)})
resumen_lim = LIM.groupby(["Escenario", "Modelo"], sort=False).apply(ic_limitacion).reset_index()
resumen_lim["Tasa media"] *= 100
resumen_lim["DE entre semillas"] *= 100
resumen_lim.to_csv(RESULTADOS / "limitacion_paper_guia.csv", index=False, encoding="utf-8-sig")
tabla(resumen_lim.rename(columns={"Tasa media": "Tasa media (%)", "DE entre semillas": "DE entre semillas (pp)"}), {"Tasa media (%)": "{:.2f}", "DE entre semillas (pp)": "{:.2f}"})
piloto_lim = ADENDA["piloto"]["limitacion_paper_guia"]["por_modelo"]
tabla(pd.DataFrame([{"Modelo del piloto (validación)": m, "Tasa media (%)": 100 * v["media_semillas"], "DE (pp)": 100 * v["de_semillas"], "IC 95 % por bloques (primera semilla)": intervalo(100 * np.array(v["ic95_bloques_primera_semilla"]), 1)}
                    for m, v in piloto_lim.items()]), {"Tasa media (%)": "{:.2f}", "DE (pp)": "{:.2f}"})
lstm_rep = LIM[(LIM["Escenario"].str.startswith("Réplica")) & (LIM["Modelo"] == "LSTM")]["tasa"].mean()
lstm_bench = LIM[(LIM["Escenario"].str.startswith("Benchmark")) & (LIM["Modelo"] == "LSTM")]["tasa"].mean()
top10_bench = LIM[(LIM["Escenario"].str.startswith("Benchmark")) & (LIM["Modelo"].isin(CONFIG["modelos"]["top10"]))].groupby("Modelo")["tasa"].mean()
compartida = top10_bench.min() >= 0.5 * lstm_bench
subestima = lstm_rep < lstm_bench
piloto_top = {k: v["media_semillas"] for k, v in piloto_lim.items() if k != "CNN 1D"}
interpretar(f'''**Interpretación.** Con el protocolo de los autores, la LSTM del paper confunde Blackhole y Wormhole en el {num(100 * lstm_rep, 2)} % de los flujos de esas dos clases, y con la partición agrupada la tasa es de {num(100 * lstm_bench, 2)} %. En el benchmark, los diez modelos del Top 10 tienen tasas de {num(100 * top10_bench.min(), 2)} % ({top10_bench.idxmin()}) a {num(100 * top10_bench.max(), 2)} % ({top10_bench.idxmax()}), y en el piloto, los tres modelos del Top 10 evaluados en validación quedan entre {num(100 * min(piloto_top.values()), 2)} % y {num(100 * max(piloto_top.values()), 2)} %.

Esto quiere decir que la limitación {"no es propia de la LSTM del paper, sino compartida por todas las familias evaluadas: ningún modelo del Top 10 baja la tasa a menos de la mitad de la de la LSTM" if compartida else "no afecta por igual a todas las familias: " + top10_bench.idxmin() + " la reduce a menos de la mitad de la de la LSTM"}{", y que la partición aleatoria la subestima, porque los flujos casi gemelos de una misma serie hacen más fácil separar el par" if subestima else ""}. {"Es decir, ninguna arquitectura del Top 10 resuelve la frontera entre los dos ataques de enrutamiento, lo que apunta a una superposición en el espacio de los 16 atributos más que a un problema de capacidad." if compartida else "Es decir, el sesgo inductivo sí cambia cómo se separan los dos ataques, lo que orienta el mecanismo del modelo original."}

Por lo anterior, conviene declarar esta limitación como el hueco de investigación para el Entregable 3, que se resume en la sección 5.4.''')
""")

md("s5-4", r"""
### 5.4 Relación con el benchmark y con el Entregable 3

**Posición del paper guía en el benchmark.** Los tres modelos profundos del paper son los modelos base del proyecto, de modo que su posición frente al Top 10 es la que resuelve RQ1, y su XGBoost es el antecedente del LightGBM de RQ2. Las diferencias con el protocolo del paper son tres: la partición agrupada por serie en lugar de aleatoria, la pérdida ponderada por clase y el ajuste de hiperparámetros con el mismo presupuesto, que el paper no hace.

**Relación con el Entregable 3.** La limitación cuantificada en la sección 5.3 es el candidato natural al hueco de investigación del modelo original. Como no depende de la familia, la mejora no vendrá de cambiar la arquitectura base, sino de un mecanismo que separe explícitamente los dos ataques de enrutamiento. Los candidatos de la búsqueda transversal (sección 3.1) que apuntan a esa limitación son la pérdida contrastiva supervisada (Khosla et al., 2020) y la mezcla de negativos difíciles (Kalantidis et al., 2020), que fuerzan la separación de los pares confundibles en el espacio latente, y una descomposición jerárquica como la de H-MicroKAN (Bakiri et al., 2026), que dedique una etapa al par Blackhole–Wormhole. La recomendación es extender el paper guía con uno de estos mecanismos, sin comprometer aún la elección, que corresponde al Entregable 3.
""")
