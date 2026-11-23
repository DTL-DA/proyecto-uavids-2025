"""Sección 2: EDA interactivo (filtros que actualizan cada gráfico con callbacks)."""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from src import plots
from src.data import uavids
from src.inference import ATRIBUTOS, CLASES

dash.register_page(__name__, path="/eda", name="2. EDA", title="EDA · UAVIDS-2025", order=2)

PARTICIONES = ["entrenamiento", "validación", "prueba"]


def _tarjeta(titulo_id, figura_id, texto):
    return dbc.Card(dbc.CardBody([dcc.Loading(dcc.Graph(id=figura_id, config=plots.CONFIG)), html.P(texto, id=titulo_id, className="insight mt-2 mb-0")]),
                    className="seccion mb-4")


def layout():
    df = uavids()
    duplicados = int((df["particion"] == "excluida (duplicada)").sum())
    filtros = dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([dbc.Label("Atributo"), dcc.Dropdown(ATRIBUTOS, "PacketDropRate", id="eda-atributo", clearable=False)], md=4),
        dbc.Col([dbc.Label("Clases"), dcc.Dropdown(CLASES, CLASES, id="eda-clases", multi=True)], md=4),
        dbc.Col([dbc.Label("Particiones"), dcc.Checklist(PARTICIONES, PARTICIONES, id="eda-particiones", inline=True, inputClassName="me-1", labelClassName="me-3")], md=2),
        dbc.Col([dbc.Label("Escala"), dcc.RadioItems([{"label": "log(1 + x)", "value": "log"}, {"label": "original", "value": "lin"}], "log", id="eda-escala",
                                                     inline=True, inputClassName="me-1", labelClassName="me-3")], md=2),
    ])), className="seccion mb-4")
    calidad = dbc.Card(dbc.CardBody([html.H3("Calidad de los datos", className="h5"), dbc.Row([
        dbc.Col(plots_stat(f"{duplicados:,}".replace(",", " "), "duplicados exactos excluidos antes de partir"), md=3, xs=6),
        dbc.Col(plots_stat("0", "valores faltantes en los 16 atributos"), md=3, xs=6),
        dbc.Col(plots_stat("0", "bloques compartidos entre particiones"), md=3, xs=6),
        dbc.Col(plots_stat("70,5 %", "filas de prueba en un escenario con filas en entrenamiento"), md=3, xs=6),
    ])]), className="seccion mb-4")
    return html.Div([
        html.H2("Análisis exploratorio", className="mb-1"),
        html.P("Los filtros actualizan todas las figuras. Los atributos son estadísticos por flujo con colas largas; la escala log(1 + x) es la "
               "transformación que reciben los modelos.", className="nota"),
        filtros, calidad,
        dbc.Row([dbc.Col(_tarjeta("eda-t1", "eda-balance", ""), lg=6), dbc.Col(_tarjeta("eda-t2", "eda-distribucion", ""), lg=6)]),
        _tarjeta("eda-t3", "eda-escenarios",
                 "Panel del tipo de problema: cada clase aparece en tramos largos y contiguos, que son los escenarios de simulación. La partición por "
                 "bloques de 1000 flujos evita mezclar vecinos inmediatos, pero un mismo escenario puede tener bloques en entrenamiento y en prueba. "
                 "Los dos escenarios de Sybil sombreados quedaron aislados por azar y funcionan como prueba de generalización a un escenario nuevo."),
        dbc.Row([dbc.Col(_tarjeta("eda-t4", "eda-ecdf", ""), lg=6), dbc.Col(_tarjeta("eda-t5", "eda-correlacion", ""), lg=6)]),
    ])


def plots_stat(cifra, texto):
    return html.Div([html.Div(cifra, className="cifra-media"), html.Div(texto, className="nota")], className="stat stat-claro mb-2")


@callback(Output("eda-balance", "figure"), Output("eda-t1", "children"), Output("eda-distribucion", "figure"), Output("eda-t2", "children"),
          Output("eda-escenarios", "figure"), Output("eda-ecdf", "figure"), Output("eda-t4", "children"), Output("eda-correlacion", "figure"),
          Output("eda-t5", "children"),
          Input("eda-atributo", "value"), Input("eda-clases", "value"), Input("eda-particiones", "value"), Input("eda-escala", "value"))
def actualizar(atributo, clases, particiones, escala):
    df = uavids()
    sel = df[df["clase"].isin(clases or []) & df["particion"].isin(particiones or [])]
    if sel.empty:
        v = plots.vacia("Sin flujos con estos filtros")
        return v, "", v, "", v, v, "", v, ""
    conteo = sel["clase"].value_counts(normalize=True)
    t1 = (f"Con estos filtros hay {len(sel):,} flujos. ".replace(",", " ") + f"La clase más frecuente es {conteo.index[0]} ({plots.num(100 * conteo.iloc[0], 0)} %) y la menos "
          f"frecuente {conteo.index[-1]} ({plots.num(100 * conteo.iloc[-1], 0)} %); por ese desbalance moderado la métrica primaria es el F1 macro y la pérdida se pondera por clase.")
    medianas = sel.groupby("clase")[atributo].median().sort_values()
    t2 = (f"La mediana de {atributo} va de {plots.num(medianas.iloc[0], 3)} ({medianas.index[0]}) a {plots.num(medianas.iloc[-1], 3)} ({medianas.index[-1]}). "
          "Donde las cajas de dos clases se solapan, como Blackhole y Wormhole en la mayoría de los atributos, ningún umbral sobre ese atributo las separa solo.")
    t4 = ("Si las curvas de entrenamiento, validación y prueba se separan, ese atributo cambia de distribución entre particiones, algo esperable porque los "
          "bloques provienen de escenarios distintos. Es la razón por la que el desempeño de prueba es menor que el de validación.")
    t5 = ("Los pares con |ρ| cercano a 1 (por ejemplo, paquetes y bytes transmitidos) aportan información redundante; los modelos los reciben todos, pero "
          "la explicación local se reparte entre ellos.")
    return (plots.balance(sel), t1, plots.distribucion(sel, atributo, escala == "log"), t2, plots.estructura_escenarios(df[df["clase"].isin(clases or [])]),
            plots.ecdf_particiones(df[df["clase"].isin(clases or [])], atributo), t4, plots.correlacion(sel.sample(min(len(sel), 20000), random_state=0), ATRIBUTOS), t5)
