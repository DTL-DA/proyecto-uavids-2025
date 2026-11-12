from .base import code, md

md("s1", r"""
## 1. Definición del problema, preguntas de investigación y dataset

En esta sección se formaliza el problema, se declaran las preguntas de investigación con sus hipótesis y se documenta el conjunto de datos con su Data Card y su contrato de datos. Esto es primordial porque fija qué se mide, con qué criterio se decide y sobre qué población valen las conclusiones, antes de ver un solo resultado del benchmark.

### 1.1 Definición formal del problema

Cada observación es un flujo de red de un enjambre de drones, y la tarea es asignarle una de cinco clases:

- **Entradas:** $\mathbf{x} \in \mathcal{X} \subset \mathbb{R}^{16}$, los 16 atributos de tráfico y desempeño del flujo, recortados, transformados con $\log(1 + x)$ y estandarizados con los estadísticos de entrenamiento (Entregable 1, sección 2.2).
- **Salidas:** $y \in \mathcal{Y}$ = {Blackhole, Flooding, Normal, Sybil, Wormhole}.
- **Espacio de hipótesis:** para cada arquitectura $a$, $\mathcal{H}_a = \{ f_\theta : \mathcal{X} \to \Delta^{4} \}$, redes con salida softmax sobre el símplex de 5 clases, con $\theta$ en el espacio de pesos que define la arquitectura; la predicción es $\hat{y} = \arg\max_c f_\theta(\mathbf{x})_c$.
- **Pérdida de entrenamiento:** la entropía cruzada ponderada, $L(f(\mathbf{x}), y) = -w_y \log f(\mathbf{x})_y$, con $w_c = n / (K n_c)$, más el término propio de la arquitectura cuando lo tiene (la dispersión de TabNet o la reconstrucción del autoencoder).
- **Riesgo esperado:** $R(f) = \mathbb{E}_{(\mathbf{x}, y) \sim P}\left[ L(f(\mathbf{x}), y) \right]$, que se estima con el riesgo empírico sobre el conjunto de prueba sellado, $\hat{R}_T(f) = \frac{1}{|T|} \sum_{i \in T} L(f(\mathbf{x}_i), y_i)$. La métrica de decisión es el F1 macro, que no se descompone como un promedio de pérdidas por fila, de modo que se calcula sobre todo $T$ y su incertidumbre se estima remuestreando las unidades independientes.

**Tipo de tarea.** Clasificación con datos tabulares, en su versión multiclase, que corresponde a la familia F1 del enunciado. Ningún modelo usa rezagos ni la posición del flujo como entrada, de modo que la serie solo define la unidad de independencia y no convierte el problema en uno de series de tiempo.

**Relevancia científica y aplicada.** Los enjambres de drones se comunican por enlaces inalámbricos y enrutan el tráfico entre ellos, sin infraestructura fija, de modo que un nodo comprometido puede descartar o desviar paquetes desde adentro, y detectarlo a partir del tráfico no exige modificar el protocolo de los drones (Zeng et al., 2025). La literatura reciente lleva estos detectores a bordo, con restricciones de energía y latencia (Bakiri et al., 2026), y advierte que la forma de partir los datos cambia lo que se mide: Biswas et al. (2026) parten por sesión y por dron para evitar la fuga, y Mohamed et al. (2026) evalúan la detección de ataques no vistos en enjambres. En ese sentido, el aporte de este entregable no es otro número alto, sino una comparación de arquitecturas con una partición que respeta la dependencia entre flujos y con inferencia estadística, que según la revisión de la sección 3 ningún estudio incluido hace sobre este conjunto.

**Métrica, margen y unidad, heredados del Entregable 1.** La métrica primaria es el F1 macro, $F1_{macro} = \frac{1}{K}\sum_{c=1}^{K} \frac{2 P_c R_c}{P_c + R_c}$, porque en seguridad un ataque poco frecuente importa tanto como uno frecuente. El margen de relevancia es δ = 0,01: por la definición del promedio macro, equivale a que el F1 de una sola clase cambie 0,05 con las demás iguales, un cambio que en operación se nota en la tasa de alertas de un ataque. La unidad de independencia es la serie de flujos consecutivos según `FlowID`, aproximada con bloques de 1000 flujos, porque el archivo no identifica la simulación de origen y los flujos vecinos son casi idénticos. Este entregable agrega dos márgenes declarados en el preregistro: δg = 0,05 como brecha máxima aceptable entre entrenamiento y validación, y el nivel de 0,10 del *Model Confidence Set*.
""")

code("s1-apriori", r"""
declaracion = json.loads((D.E1 / "resultados" / "declaracion_a_priori.json").read_text(encoding="utf-8"))
preregistro = json.loads((RESULTADOS / "preregistro_e2.json").read_text(encoding="utf-8"))
tabla(pd.DataFrame({"Elemento": ["Tipo de problema", "Métrica primaria", "Margen δ", "Margen de sobreajuste δg", "Nivel α", "Unidad de independencia", "Partición", "Fijado"],
                    "Valor": [declaracion["tipo_problema"], declaracion["metrica_primaria"], num(DELTA, 2), num(DELTA_G, 2), num(ALFA, 2), declaracion["unidad_independencia"],
                              "64/16/20 %, estratificada por clase y agrupada por bloques, semilla 42", f"Entregable 1, {declaracion['fecha']}; preregistro del Entregable 2, {preregistro['fecha']}"]}))
""")

md("s1-2", r"""
### 1.2 Preguntas de investigación e hipótesis

Las preguntas se fijaron en el preregistro antes de entrenar. Las tres confirmatorias tienen control del error por familia con Holm; las exploratorias se reportan como generadoras de hipótesis y no se usan para elegir el modelo final.

| RQ | Tipo | Pregunta | Hipótesis | Pruebas y criterio |
|---|---|---|---|---|
| RQ1 | Confirmatoria (familia de 10 contrastes) | ¿Supera algún modelo del Top 10 al baseline del Entregable 1 (CNN 1D, reentrenado aquí con el mismo protocolo) en F1 macro de prueba, por más de δ = 0,01? | Para cada modelo k: H0: μk − μCNN = 0; H1: μk − μCNN ≠ 0 | *t* pareada sobre semillas (o *bootstrap* pareado si falla la normalidad), Wilcoxon exacta y Friedman con *post hoc* de Holm; α = 0,05 con Holm. Superior y relevante si ambas pruebas pareadas rechazan, el IC 95 % excluye el cero y la diferencia supera δ |
| RQ2 | Confirmatoria (1 contraste) | ¿Difiere el mejor modelo profundo, elegido por su F1 macro medio de **validación**, del LightGBM ajustado con el mismo presupuesto? | H0: μDL − μLGBM = 0; H1: ≠ 0 | Las mismas tres pruebas, más TOST con ±δ, la prueba bayesiana con ROPE, McNemar, DeLong y la *F* de 5 × 2 cv como sensibilidad |
| RQ3 | Confirmatoria (1 contraste) | ¿Obtienen menor F1 macro los modelos que leen los atributos como una secuencia (CNN 1D, LSTM, RNN, GRU, BiLSTM, CNN-LSTM) que los que no suponen un orden entre ellos (DNN, ResNet, Autoencoder, DBN, KAN, TabNet, Transformer)? | H0: μS − μN = 0, con el promedio de cada grupo por semilla; H1: ≠ 0 | *t* pareada y Wilcoxon sobre semillas, y un modelo de efectos mixtos sobre el F1 por bloque de prueba |
| RQ4 | Exploratoria | ¿Es compartida por los modelos del Top 10 la limitación cuantificada en el paper guía? | Generadora de hipótesis | IC 95 % por *bootstrap* por bloques |
| RQ5 | Exploratoria | ¿Existe un modelo de menor costo equivalente al mejor dentro de ±δ? | Generadora de hipótesis | TOST por pares |

La elección del mejor modelo profundo de RQ2 con validación, y no con prueba, evita la maldición del ganador: si se eligiera el máximo de 13 medias de prueba, la diferencia observada frente a LightGBM quedaría sesgada hacia arriba por la propia selección.
""")

md("s1-3", r"""
### 1.3 Justificación del dataset

**Criterios de selección.** UAVIDS-2025 es, entre los conjuntos públicos revisados, el que reúne flujos de red de un enjambre de drones con ataques de la capa de red (Blackhole, Flooding, Sybil y Wormhole), con 122 171 flujos y cinco clases casi balanceadas (Zeng et al., 2025). Además, tiene uso previo en la literatura: 12 de los 43 estudios incluidos en la revisión lo usan, entre ellos varios de revistas Q1, como Bakiri et al. (2026) y Wang et al. (2026), lo que permite contrastar los resultados con cifras publicadas.

**Licencia, procedencia y ética.** Se publica en Kaggle, Zenodo (licencia CC BY 4.0) e IEEE DataPort (DOI 10.21227/j5p4-zt27). Los flujos provienen de simulaciones en NS-3.24, con movilidad de enjambre BOID extendida, IEEE 802.11ac, enrutamiento AODV, tráfico UDP y canal Nakagami (Zeng et al., 2025). Al ser simulados, no contienen datos personales: las direcciones IP son de la simulación, de modo que no hay riesgos de privacidad, y la licencia permite el uso académico con atribución.

**Limitaciones conocidas.** Los datos provienen de un solo simulador y de una sola configuración de radio y enrutamiento, sin drones reales, lo que limita la representatividad frente a enjambres en campo. El archivo no identifica la simulación de origen, de modo que la dependencia entre flujos se aproxima con bloques; contiene 8219 filas repetidas, y la etiqueta corresponde al escenario y no a cada flujo, lo que puede introducir ruido de etiquetas (sección 2.1.2).

**Partición oficial.** El conjunto no trae una partición oficial. El código de los autores usa una partición aleatoria estratificada 80/20, que se replica en la sección 5 para validar el pipeline, pero no se usa en el benchmark, porque mezcla flujos de una misma serie entre entrenamiento y prueba: en el Entregable 1, esa partición subió el F1 macro de referencia en 0,0689 frente a la agrupada.
""")

md("s1-4", r"""
### 1.4 Descripción técnica (Data Card)

La Data Card resume el tamaño, la estructura, la variable objetivo y su protocolo de etiquetado, la versión y los filtros aplicados. Todo se calcula sobre el archivo verificado.
""")

code("s1-data-card", r"""
repetidas = datos[ATRIBUTOS + ["label"]].duplicated()
conteo = pd.Series(y).value_counts().sort_index()
tipos = datos.dtypes.astype(str).value_counts()
data_card = pd.DataFrame({"Campo": ["Archivo y versión", "SHA-256", "Tamaño", "Estructura", "Atributos de entrada", "Identificadores excluidos", "Variable objetivo",
                                    "Protocolo de etiquetado", "Acuerdo entre anotadores", "Recolección", "Periodo y cobertura", "Filtros aplicados", "Partición"],
                          "Valor": ["UAVIDS-2025.csv, versión 1", D.SHA_CSV, f"{num(len(datos), 0)} filas, {datos.shape[1]} columnas, {num(D.CSV.stat().st_size / 2**20, 1)} MB",
                                    f"Una fila por flujo; tipos: " + ", ".join(f"{v} {k}" for k, v in tipos.items()),
                                    f"{len(ATRIBUTOS)} numéricos de volumen de tráfico y de desempeño", ", ".join(D.IDENTIFICADORES),
                                    "label: " + ", ".join(f"{c} {num(n, 0)}" for c, n in zip(CLASES, conteo)),
                                    "El simulador anota cada flujo con la etiqueta de su escenario; una verificación cruza los tiempos de inyección del ataque con los flujos (Zeng et al., 2025)",
                                    "No aplica: no hay anotadores humanos", "Simulación en NS-3.24, BOID extendido, IEEE 802.11ac, AODV, UDP, canal Nakagami, alcance de 100 m y 20 dBm",
                                    "Escenarios simulados; el archivo no trae marca de tiempo ni identificador de simulación",
                                    f"Se retiran {num(int(repetidas.sum()), 0)} filas repetidas en atributos y etiqueta; se excluyen los 6 identificadores",
                                    f"Heredada del Entregable 1: {num(len(ID_ENT), 0)}, {num(len(ID_VAL), 0)} y {num(len(ID_PRU), 0)} flujos; prueba sellada con SHA-256 {D.SHA_PRUEBA[:16]}…"]})
tabla(data_card)
""")

md("s1-5", r"""
### 1.5 Contrato de datos

El contrato de datos fija el esquema que el archivo debe cumplir y lo verifica automáticamente cada vez que se carga, de modo que una versión distinta del conjunto o un error de lectura se detectan antes de entrenar. Las reglas de rango y de consistencia física se derivan de la definición de cada atributo en el monitor de flujos de NS-3 (Zeng et al., 2025).
""")

code("s1-contrato", r"""
numericas = datos[ATRIBUTOS]
REGLAS = {
    "Sin valores faltantes ni infinitos": bool(datos.notna().all().all() and np.isfinite(numericas.to_numpy()).all()),
    "FlowID único y consecutivo de 1 a N": bool(datos["FlowID"].is_unique and datos["FlowID"].min() == 1 and datos["FlowID"].max() == len(datos)),
    "Conteos enteros y no negativos": bool((datos[["TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes"]] >= 0).all().all()),
    "TxPackets ≥ 1 (todo flujo emite)": bool((datos["TxPackets"] >= 1).all()),
    "RxPackets ≤ TxPackets y RxBytes ≤ TxBytes": bool(((datos["RxPackets"] <= datos["TxPackets"]) & (datos["RxBytes"] <= datos["TxBytes"])).all()),
    "PacketDropRate = LostPackets / TxPackets": bool(np.isclose(datos["PacketDropRate"], datos["LostPackets"] / datos["TxPackets"], atol=1e-4).all()),
    "Throughput/Kbps = 8 · RxByteRate/s / 1000": bool(np.isclose(datos["Throughput/Kbps"], 8 * datos["RxByteRate/s"] / 1000, rtol=1e-3).all()),
    "Duración, tasas, retardo y jitter no negativos": bool((numericas.drop(columns=["PacketDropRate"]) >= 0).all().all()),
    "Protocol = UDP; puertos en {9, 654}": bool((datos["Protocol"] == "UDP").all() and datos["SrcPort"].isin([9, 654]).all() and datos["DstPort"].isin([9, 654]).all()),
    "label en las 5 clases declaradas": bool(datos["label"].str.replace(" Attack", "").str.replace(" Traffic", "").isin(CLASES).all()),
    "PacketDropRate ≤ 1": bool((datos["PacketDropRate"] <= 1).all()),
}
fallas_drop = int((datos["PacketDropRate"] > 1).sum())
perdidas_distintas = float((datos["LostPackets"] != datos["TxPackets"] - datos["RxPackets"]).mean())
tabla(pd.DataFrame({"Regla": list(REGLAS), "Resultado": ["Cumple" if v else "No cumple" for v in REGLAS.values()]}))
ESQUEMA = pd.DataFrame({"Atributo": ATRIBUTOS, "Tipo": [str(datos[a].dtype) for a in ATRIBUTOS], "Mínimo": numericas.min().to_numpy(), "Máximo": numericas.max().to_numpy(),
                        "Unidad": ["s", "paquetes", "paquetes", "paquetes", "bytes", "bytes", "paquetes/s", "paquetes/s", "bytes/s", "bytes/s", "s", "s", "kbit/s", "bytes", "proporción", "saltos"],
                        "Disponible": "Al cierre del flujo", "Anterior a la etiqueta": "Sí: se mide en el flujo, la etiqueta es del escenario"})
tabla(ESQUEMA, {"Mínimo": "{:,.4g}", "Máximo": "{:,.4g}"})
interpretar(f'''**Interpretación.** De las {len(REGLAS)} reglas, {sum(REGLAS.values())} se cumplen en todas las filas. La que falla es `PacketDropRate` ≤ 1, con {fallas_drop} flujos por encima de 1, y la causa está en la propia definición del monitor: la tasa es `LostPackets` / `TxPackets` en todas las filas, pero `LostPackets` no es igual a `TxPackets` − `RxPackets` en el {pct(perdidas_distintas)} de los flujos, porque el monitor cuenta como perdido un paquete que supera un tiempo de espera, aunque después llegue. Así mismo, todos los atributos se miden en el propio flujo y están disponibles al cerrarlo, de modo que ninguno es posterior a la etiqueta del escenario.

Es decir, el esquema es consistente y las relaciones físicas se cumplen, con una excepción conocida y acotada, que no indica corrupción sino una convención del simulador.

Por tal motivo, se mantiene el recorte de `PacketDropRate` en 1 que definió el Entregable 1 dentro de la transformación, y el contrato se ejecuta al cargar los datos: si alguna de las otras reglas fallara, el cuaderno se detendría.''')
assert all(v for k, v in REGLAS.items() if k != "PacketDropRate ≤ 1"), "El archivo no cumple el contrato de datos"
""")

code("s1-poblaciones", r"""
indices_unicos = np.flatnonzero(~repetidas.to_numpy())
poblaciones = pd.DataFrame({"Paso": ["Archivo original", "Sin filas repetidas en atributos y etiqueta", "Entrenamiento", "Validación", "Prueba sellada"],
                            "Flujos": [len(datos), len(indices_unicos), len(ID_ENT), len(ID_VAL), len(ID_PRU)],
                            "Eliminados en el paso": [0, len(datos) - len(indices_unicos), None, None, None],
                            "Bloques de 1000": [len(np.unique(bloques)), len(np.unique(bloques[indices_unicos])), len(np.unique(bloques[ID_ENT])), len(np.unique(bloques[ID_VAL])), len(np.unique(bloques[ID_PRU]))]})
tabla(poblaciones, {"Flujos": "{:,.0f}", "Eliminados en el paso": "{:,.0f}"})
registro_transformaciones = pd.DataFrame({"Orden": [1, 2, 3, 4, 5], "Transformación": ["Verificación de la suma SHA-256 y del contrato de datos", "Eliminación de filas repetidas en atributos y etiqueta",
                                                                                         "Exclusión de FlowID, SrcAddr, SrcPort, DstAddr, DstPort y Protocol", "Recorte de PacketDropRate en 1 y log(1 + x)",
                                                                                         "Estandarización con la media y la desviación de entrenamiento (de cada pliegue, en la validación cruzada)"],
                                          "Ajustada con": ["Ninguno", "Todo el archivo, antes de partir; compara filas y no estima parámetros", "Ninguno", "Ninguno", "Solo entrenamiento"]})
tabla(registro_transformaciones)
""")
