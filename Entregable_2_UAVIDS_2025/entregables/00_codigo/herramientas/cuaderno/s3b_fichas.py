from .base import code, md

code("s3-fichas-funcion", r"""
def evidencia(familia):
    '''Estudios incluidos que usan la familia, con sus datos, valores reportados, dispersión y prueba, y el sesgo de reporte.'''
    lista = [e for e in sorted(EXT, key=lambda e: cita_base(e["doi"]).lower()) if familia in [m.strip() for m in e["modelos"].split(";")]]
    filas = [{"Estudio": citar(e["doi"]), "Dataset": e["dataset"], "Valores reportados": e["valores"], "Dispersión": e["dispersion"], "Prueba": e["prueba"]} for e in lista]
    with pd.option_context("display.max_colwidth", 120):
        tabla(pd.DataFrame(filas))
    aleatoria = sum(not agrupa(e["particion"]) and e["particion"] != "no identificada" for e in lista)
    sin_rep = sum(e["repeticiones"] == "no reportadas" for e in lista)
    publicadas = sum(REPRO[e["doi"]]["baselines_ajustados"] == "no" for e in lista)
    interpretar(f"**Sesgo de reporte.** De los {len(lista)} estudios que usan la familia, {aleatoria} parten sin agrupar filas relacionadas, {sin_rep} no reportan repeticiones, "
                f"{sum(e['dispersion'] == 'sí' for e in lista)} reportan dispersión, {sum(not e['prueba'].startswith('ninguna') for e in lista)} aplican alguna prueba y {publicadas} comparan con cifras publicadas. "
                + ("Ningún estudio de la familia ajusta sus líneas base con un presupuesto equivalente, de modo que las ventajas publicadas no son comparables." if publicadas or sin_rep == len(lista) else ""))
""")

FICHAS = {
    "CNN-LSTM": (r"""
#### Modelo 1: CNN-LSTM

- **Referencias.** Fundacional: Donahue et al. (2015), que combinan convoluciones y LSTM en una red entrenada de extremo a extremo (*long-term recurrent convolutional networks*). En el problema la usan Alharthi (2024), Alruwaili y Tunc (2026), Arulalan et al. (2023), Hamad et al. (2025), Mohamed et al. (2026), en su variante CNN-GRU, y SureshBabu y Geetha (2026), como ConvLSTM.
- **Tipo y familia.** Híbrida convolucional y recurrente.
- **Descripción técnica.** Dos convoluciones 1D extraen patrones locales del vector de 16 atributos, $z^{(l)} = \mathrm{ReLU}(W^{(l)} * z^{(l-1)} + b^{(l)})$, un *max-pooling* reduce la secuencia a 8 posiciones y una LSTM la recorre con las compuertas $i_t, f_t, o_t = \sigma(W x_t + U h_{t-1} + b)$, la memoria $c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)$ y el estado $h_t = o_t \odot \tanh(c_t)$; la salida softmax usa el último estado. El flujo es 1 × 16 → $w_1$ filtros → $w_2$ filtros → 8 pasos → $w_3$ unidades → 5.
- **Innovación.** Separa la extracción de rasgos locales, que hace la convolución con pesos compartidos, del modelado de dependencias a lo largo de la secuencia, que hace la LSTM.
- **Fortalezas y limitaciones.** Combina dos sesgos inductivos y es la familia híbrida más usada en la literatura del problema. Sin embargo, ambos sesgos suponen que el orden de las columnas tiene significado, que en una tabla es arbitrario; además, la recurrencia no se paraleliza a lo largo de la secuencia y suma hiperparámetros.
- **Complejidad.** Parámetros $P = 4 w_1 + (3 w_1 + 1) w_2 + 4[(w_2 + w_3) w_3 + 2 w_3] + 5 (w_3 + 1)$; costo por muestra de orden $\mathcal{O}(L k\, w_1 w_2 + T\, 4 (w_2 + w_3) w_3)$, con $L = 16$, $k = 3$ y $T = 8$. Los estudios no reportan un tiempo de entrenamiento comparable.
- **Definición de la arquitectura.** No está definida: Katib et al. (2025) reportan capas LSTM de 128 y 64 unidades, pero no los filtros de la convolución, y los demás estudios no la describen. Por eso se aplica la búsqueda de la sección 14 del enunciado, con la correspondencia aproximada de bloques convolucionales y recurrente.
""", "CNN-LSTM"),
    "Autoencoder": (r"""
#### Modelo 2: Autoencoder

- **Referencias.** Fundacional: Hinton y Salakhutdinov (2006). En el problema lo usan como referencia Balgoon (2026), Gharami y Moni (2025), como MLP-AE, Mohamed et al. (2026), como DAE, y Wu et al. (2024), como SNDA; Alissa et al. (2022) y Negm et al. (2024) proponen detectores basados en un autoencoder profundo.
- **Tipo y familia.** Codificador-decodificador.
- **Descripción técnica.** Un codificador $h = f_\phi(\mathbf{x})$ comprime los 16 atributos en un código de dimensión $w_3$ y un decodificador $\hat{\mathbf{x}} = g_\psi(h)$ lo reconstruye. Aquí se implementa como autoencoder supervisado: una cabeza softmax clasifica a partir del código y la pérdida suma la entropía cruzada ponderada y la reconstrucción, $\mathcal{L} = \mathcal{L}_{CE} + \lambda \lVert \mathbf{x} - \hat{\mathbf{x}} \rVert^2$, con λ = 1, de modo que el código debe a la vez separar las clases y conservar la información del flujo.
- **Innovación.** La reconstrucción actúa como un regularizador no supervisado que obliga a la representación a describir el tráfico, no solo a separar las etiquetas.
- **Fortalezas y limitaciones.** No supone ningún orden entre atributos y es barato. Sin embargo, la ponderación λ entre ambas pérdidas es un hiperparámetro más, y en la literatura el autoencoder suele usarse para extraer rasgos que luego clasifica otro modelo, de modo que la variante supervisada es una adaptación declarada.
- **Complejidad.** Con anchos $(w_1, w_2, w_3)$, el codificador tiene $\sum_l (w_{l-1} + 1) w_l$ parámetros con $w_0 = 16$, el decodificador es su espejo y la cabeza suma $5 (w_3 + 1)$; la inferencia solo usa el codificador y la cabeza, $\mathcal{O}(\sum_l w_{l-1} w_l)$.
- **Definición de la arquitectura.** No está definida: Alissa et al. (2022) describen la forma $n \to 2h \to h \to 2h \to n$ sin fijar $h$. Se aplica la búsqueda de la sección 14, con el decodificador espejo.
""", "Autoencoder"),
    "RNN": (r"""
#### Modelo 3: RNN

- **Referencias.** Fundacional: Elman (1990). La arquitectura se toma de Biswas et al. (2026), que la usan como línea base en drones; también la usan Abbas et al. (2026), Ashraf et al. (2023), Li et al. (2026b) y Wang et al. (2026).
- **Tipo y familia.** Recurrente simple.
- **Descripción técnica.** En cada paso, $h_t = \mathrm{ReLU}(W_x x_t + W_h h_{t-1} + b)$. Biswas et al. (2026) apilan dos capas, de 128 unidades, que devuelve la secuencia, y de 64, seguidas de normalización por lotes, *dropout* de 0,3 y una densa de 128 con ReLU antes de la salida. El flujo es 16 pasos de un valor → 128 → 64 → 128 → 5.
- **Innovación.** Introduce un estado que se arrastra a lo largo de la secuencia, la idea que después refinan la LSTM y la GRU con compuertas.
- **Fortalezas y limitaciones.** Tiene menos parámetros por unidad que la LSTM y la GRU. Sin embargo, sin compuertas sufre el desvanecimiento o la explosión del gradiente (Bengio et al., 1994), en especial con ReLU, que no acota el estado, y supone un orden entre atributos.
- **Complejidad.** $P = \sum_l (d_{l-1} + h_l + 1) h_l$ más la cabeza; costo de orden $\mathcal{O}(T\, h^2)$ por capa, con $T = 16$, sin paralelismo a lo largo de los pasos.
- **Definición de la arquitectura.** Definida por Biswas et al. (2026); no aplica la búsqueda.
""", "RNN"),
    "Transformer": (r"""
#### Modelo 4: Transformer (TSLT-Net)

- **Referencias.** Fundacional: Vaswani et al. (2017). La arquitectura es TSLT-Net, de Biswas et al. (2026), un Transformer ligero para detectar intrusiones en drones, con su configuración capa a capa en su tabla 1. En el problema usan Transformers Alshamrani y Alghamdi (2026), con FlowTransformer, Li et al. (2026b) y Zhou et al. (2026).
- **Tipo y familia.** Basada en atención.
- **Descripción técnica.** Una densa de 128 con ReLU proyecta el flujo, que se reorganiza en 16 fichas de dimensión 8, se normaliza con LayerNorm y pasa por una atención multicabeza, $\mathrm{Atención}(Q, K, V) = \mathrm{softmax}(Q K^\top / \sqrt{d_k}) V$, con 2 cabezas y $d_k = 4$; luego, un promedio global, una densa de 64 con ReLU, *dropout* de 0,3 y la salida. Biswas et al. (2026) aclaran que las 16 fichas no son pasos de tiempo, sino combinaciones aprendidas de atributos.
- **Innovación.** La atención pondera las interacciones entre fichas según el propio ejemplo, en lugar de fijarlas en los pesos.
- **Fortalezas y limitaciones.** No depende del orden de las columnas, porque la primera densa mezcla todos los atributos, y es muy liviano. Sin embargo, la atención sobre fichas aprendidas es difícil de interpretar, y no tiene conexión residual ni red *feed-forward*, a diferencia del Transformer original.
- **Complejidad.** $\mathcal{O}(n^2 d)$ en la atención, con $n = 16$ fichas y $d = 8$, más $\mathcal{O}(16 \cdot 128)$ en la proyección.
- **Definición de la arquitectura.** Definida por Biswas et al. (2026); no aplica la búsqueda.
""", "Transformer"),
    "BiLSTM": (r"""
#### Modelo 5: BiLSTM

- **Referencias.** Fundacional: Graves y Schmidhuber (2005). La arquitectura se toma de Balgoon (2026), que reporta su BiLSTM como línea base en UAVIDS-2025; también la usan Ashraf et al. (2023) y Wang et al. (2026).
- **Tipo y familia.** Recurrente bidireccional.
- **Descripción técnica.** Dos LSTM recorren la secuencia en sentidos opuestos y su estado se concatena, $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$, con las compuertas de la LSTM. Balgoon (2026) usa dos capas de 128 unidades por dirección con *dropout* de 0,3; la salida concatena el último estado de cada dirección.
- **Innovación.** Cada posición ve el contexto anterior y el posterior, lo que en un vector de atributos equivale a no privilegiar un extremo de la secuencia.
- **Fortalezas y limitaciones.** Es la variante recurrente con más capacidad del Top 10. Sin embargo, es la más costosa, con cerca de 531 000 parámetros, y Balgoon (2026) reporta para ella un F1 macro de solo 77,87 % en UAVIDS-2025.
- **Complejidad.** $\mathcal{O}(2\, T\, 4 (d + h) h)$ por capa, con $T = 16$ y $h = 128$.
- **Definición de la arquitectura.** Definida por Balgoon (2026), donde es un componente de su modelo propuesto evaluado también por separado; no aplica la búsqueda.
""", "BiLSTM"),
    "ResNet": (r"""
#### Modelo 6: ResNet

- **Referencias.** Fundacional: He et al. (2016). En el problema la usan Hakeem et al. (2025) y Katib et al. (2025), en ambos casos como red convolucional para imágenes adaptada a otros datos; para tablas, Gorishniy et al. (2021) proponen la variante residual densa que aquí se implementa.
- **Tipo y familia.** Residual.
- **Descripción técnica.** Cada bloque aprende un residuo sobre su entrada, $\mathbf{z}_{l} = \mathrm{ReLU}(F(\mathbf{z}_{l-1}) + S \mathbf{z}_{l-1})$, con $F$ = densa, normalización, ReLU, densa y normalización, y $S$ la identidad o una proyección lineal si cambia el ancho; 3 bloques de anchos $(w_1, w_2, w_3)$ y la salida.
- **Innovación.** El atajo de identidad permite entrenar redes profundas sin degradación, porque el gradiente fluye por él aunque $F$ se sature.
- **Fortalezas y limitaciones.** No supone orden entre atributos y es estable de entrenar. Sin embargo, en tablas de 16 columnas la profundidad aporta poco, y en datos tabulares su ventaja frente a un perceptrón depende del conjunto (Gorishniy et al., 2021).
- **Complejidad.** $\mathcal{O}(\sum_l w_{l-1} w_l + w_l^2)$ por muestra.
- **Definición de la arquitectura.** No está definida en la literatura del problema; se aplica la búsqueda de la sección 14, con el ancho de cada bloque residual.
""", "ResNet"),
    "GRU": (r"""
#### Modelo 7: GRU

- **Referencias.** Fundacional: Cho et al. (2014). La arquitectura se toma de Biswas et al. (2026); también la usan Aburasain (2024), Ashraf et al. (2023), Attaullah et al. (2026b) y Hamad et al. (2025).
- **Tipo y familia.** Recurrente con compuertas.
- **Descripción técnica.** La compuerta de actualización $z_t = \sigma(W_z x_t + U_z h_{t-1})$ y la de reinicio $r_t = \sigma(W_r x_t + U_r h_{t-1})$ combinan el estado anterior con el candidato $\tilde{h}_t = \tanh(W x_t + U (r_t \odot h_{t-1}))$ en $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. Biswas et al. (2026) apilan GRU de 128 y de 64, cada una con normalización por lotes y *dropout* de 0,3, y una densa de 128 antes de la salida.
- **Innovación.** Logra un control del flujo del gradiente parecido al de la LSTM con dos compuertas en lugar de tres y sin celda de memoria aparte.
- **Fortalezas y limitaciones.** Es más estable que la RNN simple. Sin embargo, supone un orden entre atributos y, con 128 unidades, es la segunda más costosa del Top 10.
- **Complejidad.** $\mathcal{O}(T\, 3 (d + h) h)$ por capa.
- **Definición de la arquitectura.** Definida por Biswas et al. (2026); no aplica la búsqueda.
""", "GRU"),
    "KAN": (r"""
#### Modelo 8: KAN

- **Referencias.** Fundacional: Liu et al. (2024). La arquitectura es el clasificador de ataques de H-MicroKAN, de Bakiri et al. (2026), el único estudio del problema que la usa, sobre UAVIDS-2025 y con partición por nodo.
- **Tipo y familia.** Red de Kolmogorov-Arnold.
- **Descripción técnica.** Por el teorema de Kolmogorov-Arnold, una función continua de varias variables se escribe como composición de funciones de una variable. Una capa KAN pone una función aprendible en cada arista, $y_j = \sum_i \varphi_{ij}(x_i)$, y aquí cada $\varphi_{ij}(x) = \sum_{k=1}^{K} w_{ijk}\, \exp(-((x - \mu_k)/h)^2)$ usa $K = 5$ bases gaussianas fijas en [−5, 5], la formulación FastKAN que adoptan Bakiri et al. (2026). La red aplica una compuerta $\mathbf{x} \odot \sigma(W_2 \mathrm{ReLU}(W_1 \mathbf{x}))$, dos capas KAN de 28 y 16 neuronas con normalización, GELU y *dropout* de 0,25 y 0,20, y la salida.
- **Innovación.** Traslada la no linealidad de los nodos a las aristas, lo que da fronteras flexibles con pocos parámetros y funciones de cada atributo que se pueden graficar.
- **Fortalezas y limitaciones.** Es el modelo más liviano del Top 10 después del Transformer y se diseñó para microcontroladores. Sin embargo, el recorte en [−5, 5] trunca los valores extremos, y Bakiri et al. (2026) lo usan como la segunda etapa de un clasificador jerárquico, de modo que aplicarlo directamente a las 5 clases es una adaptación declarada.
- **Complejidad.** $n_{in} n_{out} K + n_{out}$ parámetros por capa, la fórmula que publican sus autores; costo $\mathcal{O}(n_{in} n_{out} K)$.
- **Definición de la arquitectura.** Definida por Bakiri et al. (2026), en su tabla 2; no aplica la búsqueda.
""", "KAN"),
    "TabNet": (r"""
#### Modelo 9: TabNet

- **Referencias.** Fundacional: Arik y Pfister (2021). En el problema lo usan Hadi et al. (2026), como segunda etapa de un detector en tres niveles.
- **Tipo y familia.** Atención secuencial para datos tabulares.
- **Descripción técnica.** En cada paso de decisión, un transformador atento elige qué atributos mirar con una máscara $M[i] = \mathrm{sparsemax}(P[i-1] \cdot h_i(a[i-1]))$, donde $P[i-1]$ penaliza reutilizar atributos ya elegidos con el factor γ, y un transformador de atributos con bloques GLU procesa $M[i] \odot \mathbf{x}$; la salida suma las decisiones de los pasos. La pérdida agrega una regularización de dispersión de las máscaras.
- **Innovación.** La selección de atributos es por ejemplo e interpretable, porque las máscaras indican qué atributos usó cada decisión.
- **Fortalezas y limitaciones.** Fue diseñado para tablas y no supone orden entre columnas. Sin embargo, es sensible a sus hiperparámetros, la normalización por lotes fantasma depende del tamaño del lote, y en la comparación de Gorishniy et al. (2021) no supera de forma consistente a otros modelos ni a los árboles potenciados.
- **Complejidad.** $\mathcal{O}(N_{steps}\, (d\, n_a + (n_d + n_a)^2))$ por muestra.
- **Definición de la arquitectura.** El artículo define los bloques y su número, y se usa la configuración por defecto de la implementación de referencia, pytorch-tabnet: $n_d = n_a = 8$, 3 pasos, γ = 1,3 y λ de dispersión de 10⁻³. No aplica la búsqueda.
""", "TabNet"),
    "DBN": (r"""
#### Modelo 10: DBN

- **Referencias.** Fundacional: Hinton et al. (2006). En el problema la usan como referencia Alotaibi et al. (2024), Babu et al. (2024) y Negm et al. (2024), como IDBN y DPC-DBN, y Tan et al. (2019) proponen una DBN optimizada con PSO.
- **Tipo y familia.** Generativa preentrenada capa a capa.
- **Descripción técnica.** Una pila de máquinas de Boltzmann restringidas, cada una con energía $E(\mathbf{v}, \mathbf{h}) = -\mathbf{b}^\top \mathbf{v} - \mathbf{c}^\top \mathbf{h} - \mathbf{h}^\top W \mathbf{v}$ (la primera con unidades visibles gaussianas), se preentrena sin etiquetas con divergencia contrastiva de un paso, $\Delta W \propto \langle \mathbf{h} \mathbf{v}^\top \rangle_{datos} - \langle \mathbf{h} \mathbf{v}^\top \rangle_{1}$, y luego se ajusta con la entropía cruzada como un perceptrón con sigmoides.
- **Innovación.** Fue el primer método que permitió entrenar redes profundas, al inicializar los pesos con una representación generativa en lugar de valores aleatorios.
- **Fortalezas y limitaciones.** El preentrenamiento puede ayudar con pocos datos etiquetados. Sin embargo, con optimizadores modernos y decenas de miles de ejemplos su ventaja suele desaparecer, y ningún estudio de 2025 o 2026 la usa, lo que muestra que la familia está en retirada.
- **Complejidad.** $\mathcal{O}(\sum_l w_{l-1} w_l)$ por muestra en inferencia, más el preentrenamiento, de orden $\mathcal{O}(E\, n \sum_l w_{l-1} w_l)$ con $E$ épocas por RBM.
- **Definición de la arquitectura.** No está definida: Tan et al. (2019) la optimizan con PSO para KDD Cup 99, sin una configuración aplicable a estos datos en los pasajes revisados, y los demás estudios la toman de otros trabajos. Se aplica la búsqueda de la sección 14, con las unidades de cada RBM.
""", "DBN"),
}

for i, (modelo, (texto, familia)) in enumerate(FICHAS.items(), 1):
    md(f"s3-ficha-{i}", texto)
    code(f"s3-evidencia-{i}", f'evidencia("{familia}")')

md("s3-4", r"""
### 3.4 Comparación crítica y *gap* de investigación

**Matriz comparativa.** La matriz reúne, para los diez modelos, la familia, los parámetros y FLOPs de la implementación de este entregable, los datos que requiere, el mejor valor que reporta la literatura del problema y la fortaleza y la limitación principales. Los valores de la literatura no se comparan entre sí, porque provienen de conjuntos de datos y protocolos distintos.
""")

code("s3-matriz", r"""
MATRIZ_CUALITATIVA = {
    "CNN-LSTM": ("Secuencia de atributos", "Rasgos locales y dependencias en un solo modelo", "Supone un orden de columnas arbitrario"),
    "Autoencoder": ("Vector, sin orden", "La reconstrucción regulariza la representación", "Un hiperparámetro más (λ)"),
    "RNN": ("Secuencia de atributos", "Pocos parámetros por unidad", "Gradiente inestable sin compuertas"),
    "Transformer": ("Vector, sin orden", "Muy liviano; atención por ejemplo", "Fichas aprendidas difíciles de interpretar"),
    "BiLSTM": ("Secuencia de atributos", "Contexto en ambos sentidos", "El más costoso del Top 10"),
    "ResNet": ("Vector, sin orden", "Entrenamiento estable", "Profundidad poco útil con 16 columnas"),
    "GRU": ("Secuencia de atributos", "Control del gradiente con dos compuertas", "Costosa y dependiente del orden"),
    "KAN": ("Vector acotado en [−5, 5]", "Liviana; funciones por atributo graficables", "Trunca valores extremos"),
    "TabNet": ("Vector, sin orden", "Selección de atributos interpretable", "Sensible a hiperparámetros y al lote"),
    "DBN": ("Vector, sin orden", "Preentrenamiento sin etiquetas", "Familia en retirada; costo de preentrenar"),
}
filas_m = []
for _, f in complejidad_modelos[complejidad_modelos["Modelo"].isin(MATRIZ_CUALITATIVA)].iterrows():
    entrada, fuerte, debil = MATRIZ_CUALITATIVA[f["Modelo"]]
    filas_m.append({"Modelo": f["Modelo"], "Familia": f["Familia"], "Parámetros": f["Parámetros"], "FLOPs por muestra": f["FLOPs por muestra"], "Entrada que requiere": entrada,
                    "Estudios": int(puntaje.loc[puntaje["modelo"] == f["Modelo"], "estudios"].item()), "Con UAVIDS-2025": int(puntaje.loc[puntaje["modelo"] == f["Modelo"], "estudios_uavids"].item()),
                    "Fortaleza": fuerte, "Limitación": debil})
tabla(pd.DataFrame(filas_m), {"Parámetros": "{:,.0f}", "FLOPs por muestra": "{:,.0f}"})
anios = pd.DataFrame([{"anio": int(e["anio"]), "familia": m.strip()} for e in EXT for m in e["modelos"].split(";") if m.strip()])
tendencia = anios.assign(periodo=np.where(anios["anio"] >= 2025, "2025–2026", "2019–2024")).groupby(["familia", "periodo"]).size().unstack(fill_value=0)
estudios_periodo = pd.Series([int(e["anio"]) >= 2025 for e in EXT]).map({True: "2025–2026", False: "2019–2024"}).value_counts()
tendencia_pct = (100 * tendencia / estudios_periodo).round(1)
tabla(tendencia_pct.loc[["DNN/MLP", "CNN", "LSTM", "CNN-LSTM", "GRU", "Transformer", "GNN", "KAN", "TabNet", "DBN", "Autoencoder"]].reset_index(names="Familia"),
      {c: "{:.1f}" for c in tendencia_pct.columns})
interpretar(f'''**Interpretación.** En los {estudios_periodo["2019–2024"]} estudios de 2019 a 2024 y los {estudios_periodo["2025–2026"]} de 2025 a 2026, la proporción que usa Transformers pasa de {num(tendencia_pct.loc["Transformer", "2019–2024"], 1)} % a {num(tendencia_pct.loc["Transformer", "2025–2026"], 1)} %, la de GNN de {num(tendencia_pct.loc["GNN", "2019–2024"], 1)} % a {num(tendencia_pct.loc["GNN", "2025–2026"], 1)} %, y KAN y TabNet aparecen solo en el periodo reciente, mientras que la DBN pasa de {num(tendencia_pct.loc["DBN", "2019–2024"], 1)} % a {num(tendencia_pct.loc["DBN", "2025–2026"], 1)} %. En costo, la matriz va de unos 3400 parámetros en el Transformer a más de medio millón en la BiLSTM.

Así mismo, la tendencia es clara hacia la atención, los grafos y los modelos livianos para despliegue a bordo, con modelos híbridos y propuestos que superan a líneas base sin ajustar. En reproducibilidad, el panorama es el de la sección 3.1: un solo estudio con código público, ninguno con líneas base ajustadas con el mismo presupuesto y solo dos que agrupan las filas relacionadas al partir.

En consecuencia, se formula el *gap* sobre esas carencias, que se pueden refutar con la hoja de extracción.''')
""")

code("s3-gap", r"""
SECUENCIALES = {"CNN", "LSTM", "CNN-LSTM", "RNN", "GRU", "BiLSTM"}
n_secuenciales = sum(bool(SECUENCIALES & {m.strip() for m in e["modelos"].split(";")}) for e in EXT)
uavids_ext = [e for e in EXT if "UAVIDS-2025" in e["dataset"]]
interpretar(f'''**Gap en la literatura.** Ningún estudio incluido compara diez o más arquitecturas de Deep Learning de familias distintas sobre UAVIDS-2025 con una partición que agrupe los flujos de una misma serie, con el mismo presupuesto de ajuste para todos los modelos y con inferencia estadística sobre repeticiones. La hoja de extracción permite refutarlo: de los {len(uavids_ext)} estudios que usan UAVIDS-2025, {sum(agrupa(e["particion"]) for e in uavids_ext)} agrupa las filas al partir, y lo hace por nodo y no por serie (Bakiri et al., 2026); ninguno ajusta sus líneas base con un presupuesto equivalente, {sum(not e["prueba"].startswith("ninguna") for e in uavids_ext)} aplican alguna prueba estadística y ninguno corrige por comparaciones múltiples.

**Conexión con las preguntas de investigación.** RQ1 y RQ2 responden directamente al *gap*, porque comparan el Top 10 con el baseline del Entregable 1 y con un árbol potenciado ajustado con el mismo presupuesto, bajo la partición agrupada. RQ3 pone a prueba un supuesto que la literatura da por hecho sin contrastarlo: que leer los atributos como una secuencia aporta, como hacen la CNN, la LSTM y sus variantes en {n_secuenciales} de los {len(EXT)} estudios. Si RQ3 rechaza su hipótesis nula con una diferencia negativa, buena parte de esas arquitecturas estaría imponiendo un sesgo inductivo que perjudica en datos tabulares.''')
""")
