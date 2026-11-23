"""Sección 3: Modelos (líderes, predicción en vivo con explicación e incertidumbre, diagnósticos, comparación estadística y ficha)."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, dash_table, dcc, html

from src import inference as I
from src import plots
from src.data import contrastes, diagnosticos, ficha_original, lideres, resumen, uavids

dash.register_page(__name__, path="/modelos", name="3. Modelos", title="Modelos · UAVIDS-2025", order=3)

PSEUDOCODIGO = """Entrada: flujo x ∈ R^16 (log(1 + x) estandarizado), parámetros θ, temperatura mínima τ_min
1. s_{j,t} ← σ((x_j − θ_{j,t}) / τ_j),  τ_j = τ_min + softplus(ρ_j),  t = 1..T     (escalones suaves)
2. p_u ← softmax(MLP_u(s))                                                       (experto de umbrales)
3. p_s ← softmax(MLP_s(x))                                                       (experto suave)
4. g ← softmax(MLP_g(x)) ∈ Δ²                                                    (compuerta por flujo)
5. p(y | x) ← g_u p_u(y) + g_s p_s(y)                                             (mezcla de probabilidades)
Entrenamiento cooperativo (por lote B, pesos de clase w):
   L = CE_w(log mezcla con p_u, p_s detenidos) + CE_w(p_u) + CE_w(p_s) + γ KL(ḡ ‖ U)
   Adam, ReduceLROnPlateau, recorte del gradiente en 5, parada temprana con el F1 macro de validación (paciencia 6)
Salida: p(y | x) y la cota K tal que ‖p(x) − p(x′)‖₂ ≤ K ‖x − x′‖₂"""


def _card(*hijos, cls="seccion mb-4"):
    return dbc.Card(dbc.CardBody(list(hijos)), className=cls)


def _tabla_lideres():
    t = lideres().copy()
    t["F1 macro (± DE)"] = [f"{f:.4f} ± {d:.4f}" for f, d in zip(t["F1 macro"], t["DE"])]
    t["IC 95 %"] = [f"[{a:.4f}; {b:.4f}]" for a, b in zip(t["IC95 inf"], t["IC95 sup"])]
    columnas = ["Modelo", "Grupo", "Familia", "Semillas", "F1 macro (± DE)", "IC 95 %", "ECE", "ECE calibrado", "Parámetros", "FLOPs",
                "Latencia mediana (ms)", "Latencia p95 (ms)", "Entrenamiento (s)"]
    columnas = [c for c in columnas if c in t.columns]
    formato = {c: dash_table.FormatTemplate.Format(precision=3, scheme=dash_table.Format.Scheme.fixed) for c in ("ECE", "ECE calibrado", "Latencia mediana (ms)",
                                                                                                               "Latencia p95 (ms)")}
    return dash_table.DataTable(
        data=t[columnas].to_dict("records"),
        columns=[{"name": c, "id": c, "type": "numeric" if c in formato or c in ("Parámetros", "FLOPs", "Semillas", "Entrenamiento (s)") else "text",
                  **({"format": formato[c]} if c in formato else {})} for c in columnas],
        sort_action="native", filter_action="native", page_size=20, style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Lato, 'Segoe UI', Arial, sans-serif", "fontSize": 13, "padding": "6px"},
        style_header={"fontWeight": 700, "backgroundColor": "#f4f6f9"},
        style_data_conditional=[{"if": {"filter_query": '{Modelo} = "Original"'}, "backgroundColor": "#fdf0e9", "fontWeight": 700}])


def layout():
    modelos = I.modelos_disponibles()
    r = resumen()
    lider = _card(html.H3("Tabla de líderes", className="h5"),
                  html.P("Top 10, baselines y modelo original, con todas sus semillas en la prueba sellada. Se puede ordenar y filtrar "
                         "(por ejemplo, Grupo = Top 10).", className="nota"), _tabla_lideres())
    intervalos = _card(dcc.Graph(figure=plots.intervalos_lideres(lideres()), config=plots.CONFIG),
                       html.P("Los intervalos son de la media sobre semillas; si dos intervalos no se solapan, la diferencia es estable frente a la semilla. "
                              "La significancia frente a cada rival se decide con las tres pruebas de la tabla de contrastes, no con este gráfico.",
                              className="insight mt-2 mb-0"))
    prediccion = _card(
        html.H3("Predicción en vivo", className="h5"),
        dbc.Row([
            dbc.Col([dbc.Label("Modelo"), dcc.Dropdown(modelos, modelos[0] if modelos else None, id="m-modelo", clearable=False)], md=4),
            dbc.Col([dbc.Label("Clase del flujo de prueba al azar"), dcc.Dropdown(["Cualquiera"] + I.CLASES, "Cualquiera", id="m-clase-azar", clearable=False)], md=4),
            dbc.Col([dbc.Label(" "), html.Div([dbc.Button("Flujo de prueba al azar", id="m-azar", color="secondary", className="me-2"),
                                                    dbc.Button("Predecir", id="m-predecir", color="primary")])], md=4),
        ], className="mb-3"),
        dbc.Row([dbc.Col([dbc.Label(a, html_for=f"m-x{j}", className="small mb-0"),
                          dbc.Input(id=f"m-x{j}", type="number", value=0, size="sm", debounce=True)], md=3, xs=6, className="mb-2")
                 for j, a in enumerate(I.ATRIBUTOS)]),
        html.Div(id="m-verdad", className="nota mb-2"),
        dbc.Alert(id="m-ood", is_open=False, color="warning", className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(id="m-prob", config=plots.CONFIG), lg=6), dbc.Col(dcc.Graph(id="m-expl", config=plots.CONFIG), lg=6)]),
        html.P(id="m-texto", className="insight mb-0"))
    diagnostico = _card(html.H3("Diagnóstico del modelo seleccionado", className="h5"),
                        dbc.Row([dbc.Col(dcc.Graph(id="m-confusion", config=plots.CONFIG), lg=6), dbc.Col(dcc.Graph(id="m-confiabilidad", config=plots.CONFIG), lg=6)]),
                        dcc.Graph(id="m-f1clase", config=plots.CONFIG), html.P(id="m-diag-texto", className="insight mb-0"))
    tabla_c = contrastes()
    if tabla_c is not None:
        cols = [c for c in ["rival", "familia", "n_semillas", "diferencia", "p T1 t Holm", "p T2 Wilcoxon Holm", "p T3 bootstrap bloques Holm", "delta Cliff",
                            "supera (R3)", "peor con significancia"] if c in tabla_c]
        contenido_c = dash_table.DataTable(data=tabla_c[cols].round(4).astype(str).to_dict("records"), columns=[{"name": c, "id": c} for c in cols],
                                           sort_action="native", style_table={"overflowX": "auto"}, style_cell={"fontSize": 12, "padding": "5px"},
                                           style_header={"fontWeight": 700, "backgroundColor": "#f4f6f9"})
    else:
        contenido_c = html.P("Las pruebas confirmatorias se publican al terminar las corridas definitivas.", className="nota")
    comparacion = _card(html.H3("Comparación estadística confirmatoria (m = 12, Holm)", className="h5"),
                        html.P("Original frente a cada rival: t pareado y Wilcoxon sobre semillas, y bootstrap por bloques de 1000 flujos de prueba. "
                               "La decisión usa el mayor de los tres p ajustados.", className="nota"), contenido_c,
                        dcc.Graph(figure=plots.rangos_friedman(r["rangos medios (1 = mejor)"], r.get("cd_nemenyi")) if r.get("rangos medios (1 = mejor)")
                                  else plots.vacia("Diagrama de diferencia crítica: pendiente de las corridas definitivas", 200), config=plots.CONFIG))
    f = ficha_original()
    ficha = _card(html.H3("Ficha del modelo original", className="h5"), dbc.Row([
        dbc.Col([html.P([html.B("Mecanismo: "), "compuerta por flujo entre un experto de escalones suaves (umbrales y temperaturas aprendibles, sesgo de árbol) "
                         "y un experto neuronal suave, con entrenamiento cooperativo y regularizador de balance."]),
                 html.P([html.B("Procedencia: "), "propio, con piezas de Jacobs et al. (1991), Gorishniy et al. (2022) y Shazeer et al. (2017), reimplementadas desde "
                         "su formulación (experiments/procedencia.md)."]),
                 html.P([html.B("Propiedad demostrada: "), "cota de Lipschitz de las probabilidades respecto de la entrada, verificada numéricamente en las "
                         "pruebas unitarias."]),
                 html.P([html.B("Pruebas unitarias: "), f.get("pruebas", "14 pruebas con pytest en integración continua (sobreajuste de un lote, reproducibilidad, "
                                                              "formas, gradientes, Lipschitz, Jensen, guardado y ONNX).")]),
                 html.P([html.B("Uso previsto y límites: "), "apoyo a un analista de seguridad de un enjambre simulado; no generaliza a escenarios de simulación "
                         "no vistos (ver la página de contexto y MODEL_CARD.md)."], className="mb-0")], lg=6),
        dbc.Col(html.Pre(PSEUDOCODIGO, className="pseudocodigo"), lg=6)]))
    return html.Div([html.H2("Modelos", className="mb-3"), lider, intervalos, prediccion, diagnostico, comparacion, ficha])


@callback([Output(f"m-x{j}", "value") for j in range(len(I.ATRIBUTOS))] + [Output("m-verdad", "children")],
          Input("m-azar", "n_clicks"), State("m-clase-azar", "value"))
def flujo_al_azar(_, clase):
    df = uavids()
    prueba = df[df["particion"] == "prueba"]
    if clase and clase != "Cualquiera":
        prueba = prueba[prueba["clase"] == clase]
    fila = prueba.sample(1, random_state=None if _ else 7).iloc[0]
    return [float(fila[a]) for a in I.ATRIBUTOS] + [f"Flujo de prueba FlowID {int(fila['flow_id']):,} · clase real: {fila['clase']}".replace(",", " ")]


@callback(Output("m-prob", "figure"), Output("m-expl", "figure"), Output("m-ood", "children"), Output("m-ood", "is_open"), Output("m-texto", "children"),
          Input("m-predecir", "n_clicks"), Input("m-modelo", "value"), [Input(f"m-x{j}", "value") for j in range(len(I.ATRIBUTOS))])
def predecir(_, modelo, *valores):
    if modelo is None or any(v is None for v in valores):
        v = plots.vacia("Completa los 16 atributos")
        return v, v, "", False, ""
    x = np.array(valores, dtype=float).reshape(1, -1)
    p = I.probabilidades(x, modelo)[0]
    conjunto = I.conjunto_conforme(p[None, :], modelo)[0]
    referencia = np.array(I.calibracion().get("_dominio", {}).get("mediana_entrenamiento", [0.0] * len(I.ATRIBUTOS)))
    aportes = I.explicacion_local(x, modelo, referencia)
    dominio = I.fuera_de_dominio(x, modelo)
    aviso = ""
    if dominio["fuera"]:
        partes = []
        if dominio["atributos_fuera_de_rango"]:
            partes.append("atributos fuera del rango del entrenamiento: " + ", ".join(dominio["atributos_fuera_de_rango"]))
        if dominio["distancia"] is not None and dominio["umbral"] and dominio["distancia"] > dominio["umbral"]:
            partes.append(f"distancia de Mahalanobis {dominio['distancia']:.1f} > umbral del percentil 99 ({dominio['umbral']:.1f})")
        aviso = "Fuera de dominio: " + "; ".join(partes) + ". La predicción puede no ser confiable."
    c = I.CLASES[int(p.argmax())]
    texto = (f"{modelo} predice {c} con probabilidad calibrada de {plots.num(100 * p.max(), 1)} %. El conjunto conforme al 90 % es {{{', '.join(conjunto)}}}"
             f"{': el modelo duda entre varias clases' if len(conjunto) > 1 else ''}. El atributo que más sostiene la predicción es "
             f"{aportes[0]['atributo']}.")
    return plots.barras_probabilidad(p, conjunto), plots.barras_explicacion(aportes), aviso, bool(aviso), texto


@callback(Output("m-confusion", "figure"), Output("m-confiabilidad", "figure"), Output("m-f1clase", "figure"), Output("m-diag-texto", "children"),
          Input("m-modelo", "value"))
def diagnostico(modelo):
    d = diagnosticos().get(modelo)
    if d is None:
        v = plots.vacia("Sin diagnóstico para este modelo")
        return v, v, v, ""
    ref = diagnosticos().get("LightGBM") if modelo != "LightGBM" else None
    conf = np.array(d["confusion"], float)
    bw = (conf[0, 4] + conf[4, 0]) / max(conf[0].sum() + conf[4].sum(), 1)
    cal = I.calibracion().get(modelo, {})
    texto = (f"Confusión Blackhole–Wormhole de {modelo}: {plots.num(100 * bw, 1)} % de esos flujos se asignan a la otra clase. ECE en prueba: {cal.get('ece_antes', float('nan')):.3f} "
             f"antes y {cal.get('ece_despues', float('nan')):.3f} después del temperature scaling (T = {cal.get('temperatura', float('nan')):.2f}). Cobertura "
             f"del conjunto conforme en prueba: {plots.num(100 * cal.get('cobertura_prueba', float('nan')), 1)} % (nominal 90 %), con {cal.get('tamano_medio_conjunto', float('nan')):.2f} "
             "clases en promedio.")
    return (plots.matriz_confusion(d["confusion"]), plots.confiabilidad(d["confiabilidad"]),
            plots.f1_por_clase(d["f1_por_clase"], ref["f1_por_clase"] if ref else None, modelo, "LightGBM"), texto)
