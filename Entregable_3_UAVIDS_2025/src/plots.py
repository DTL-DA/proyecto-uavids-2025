"""Figuras del dashboard con una plantilla común de Plotly (misma paleta en todas las páginas)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

TINTA, TINTA_2, GRIS, REJILLA = "#2C3E50", "#4A5568", "#8A94A6", "#E6E8EC"
AZUL, NARANJA = "#2A78D6", "#EB6834"
CLASES = ["Blackhole", "Flooding", "Normal", "Sybil", "Wormhole"]
COLOR_CLASE = dict(zip(CLASES, ["#2A78D6", "#EDA100", "#8A94A6", "#1BAF7A", "#EB6834"]))
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"]}

pio.templates["uav"] = go.layout.Template(layout=dict(
    font=dict(family="Lato, 'Segoe UI', Arial, sans-serif", size=13, color=TINTA),
    title=dict(font=dict(size=15, color=TINTA), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=list(COLOR_CLASE.values()),
    xaxis=dict(gridcolor=REJILLA, linecolor="#C9CED6", zerolinecolor=REJILLA, ticks="", automargin=True),
    yaxis=dict(gridcolor=REJILLA, linecolor="#C9CED6", zerolinecolor=REJILLA, ticks="", automargin=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    margin=dict(l=10, r=10, t=56, b=10), separators=", ",
))
pio.templates.default = "uav"


def num(x, d=3):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def vacia(mensaje: str, alto: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=TINTA_2), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.update_layout(height=alto)


# ------------------------------------------------------------------------------------------------ EDA
def balance(df: pd.DataFrame) -> go.Figure:
    t = df.groupby(["particion", "clase"]).size().unstack(fill_value=0).reindex(columns=CLASES, fill_value=0)
    fig = go.Figure([go.Bar(name=c, x=t.index, y=t[c], marker_color=COLOR_CLASE[c], hovertemplate=f"{c}: %{{y:,}} flujos<extra>%{{x}}</extra>") for c in CLASES])
    fig.update_yaxes(title_text="Flujos", tickformat=",d")
    fig.update_xaxes(title_text="Partición")
    return fig.update_layout(title="Flujos por clase y partición", barmode="group", height=360)


def distribucion(df: pd.DataFrame, atributo: str, log: bool) -> go.Figure:
    fig = go.Figure()
    for c in CLASES:
        v = df.loc[df["clase"] == c, atributo]
        if v.empty:
            continue
        fig.add_trace(go.Violin(y=np.log1p(np.maximum(v, 0)) if log else v, name=c, line_color=COLOR_CLASE[c], box_visible=True, meanline_visible=True,
                                points=False, spanmode="hard"))
    fig.update_yaxes(title_text=f"log(1 + {atributo})" if log else atributo)
    return fig.update_layout(title=f"Distribución de {atributo} por clase", height=380, showlegend=False)


def correlacion(df: pd.DataFrame, atributos: list[str]) -> go.Figure:
    c = df[atributos].rank().corr().round(2)
    fig = go.Figure(go.Heatmap(z=c.values, x=atributos, y=atributos, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                               colorbar=dict(title="ρ de Spearman"), hovertemplate="%{y} · %{x}: ρ = %{z}<extra></extra>"))
    return fig.update_layout(title="Correlación de Spearman entre atributos", height=520, xaxis=dict(tickangle=45))


def estructura_escenarios(df: pd.DataFrame) -> go.Figure:
    """Panel del tipo de problema: los escenarios de simulación son tramos largos de una clase a lo largo de FlowID."""
    muestra = df.iloc[::25]
    fig = go.Figure()
    for c in CLASES:
        m = muestra[muestra["clase"] == c]
        fig.add_trace(go.Scattergl(x=m["flow_id"], y=m["clase"], mode="markers", name=c, marker=dict(color=COLOR_CLASE[c], size=4, opacity=0.6),
                                   customdata=m["particion"], hovertemplate="FlowID %{x:,}<br>%{y} · %{customdata}<extra></extra>"))
    for a, b in ((94285, 94979), (108071, 109045)):
        fig.add_vrect(x0=a, x1=b, fillcolor=NARANJA, opacity=0.15, line_width=0)
    fig.add_annotation(x=101500, y=1.08, yref="paper", text="Escenarios de Sybil aislados", showarrow=False, font=dict(color=NARANJA, size=12))
    fig.update_xaxes(title_text="FlowID (orden de la simulación)", tickformat=",d")
    fig.update_yaxes(title_text="Clase", categoryorder="array", categoryarray=CLASES)
    return fig.update_layout(title="Estructura por escenario de simulación", height=360, showlegend=False)


def ecdf_particiones(df: pd.DataFrame, atributo: str) -> go.Figure:
    fig = go.Figure()
    for p, color in (("entrenamiento", AZUL), ("validación", "#1BAF7A"), ("prueba", NARANJA)):
        v = np.sort(np.log1p(np.maximum(df.loc[df["particion"] == p, atributo].to_numpy(), 0)))
        if len(v):
            fig.add_trace(go.Scatter(x=v, y=np.arange(1, len(v) + 1) / len(v), mode="lines", name=p, line=dict(color=color, width=2)))
    fig.update_xaxes(title_text=f"log(1 + {atributo})")
    fig.update_yaxes(title_text="Proporción acumulada", range=[0, 1])
    return fig.update_layout(title=f"Cambio de distribución entre particiones: {atributo}", height=360)


# ------------------------------------------------------------------------------------------------ Modelos
def barras_probabilidad(p: np.ndarray, conjunto: list[str]) -> go.Figure:
    colores = [COLOR_CLASE[c] if c in conjunto else "#D5DAE1" for c in CLASES]
    fig = go.Figure(go.Bar(x=p * 100, y=CLASES, orientation="h", marker_color=colores, text=[f"{num(v * 100, 1)} %" for v in p], textposition="outside",
                           cliponaxis=False, hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Probabilidad calibrada (%)", range=[0, 115])
    fig.update_yaxes(categoryorder="array", categoryarray=CLASES[::-1])
    return fig.update_layout(title="Probabilidad por clase (en color: conjunto conforme al 90 %)", height=300, showlegend=False)


def barras_explicacion(aportes: list[dict]) -> go.Figure:
    a = aportes[:8][::-1]
    fig = go.Figure(go.Bar(x=[d["aporte"] * 100 for d in a], y=[d["atributo"] for d in a], orientation="h",
                           marker_color=[AZUL if d["aporte"] >= 0 else NARANJA for d in a], hovertemplate="%{y}: %{x:+.1f} pp<extra></extra>"))
    fig.update_xaxes(title_text="Caída de la probabilidad de la clase predicha al neutralizar el atributo (pp)", zeroline=True, zerolinecolor=TINTA_2)
    return fig.update_layout(title="Explicación local por oclusión (8 atributos más influyentes)", height=330, showlegend=False)


def matriz_confusion(conf: list[list[int]]) -> go.Figure:
    m = np.array(conf, dtype=float)
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=norm, x=CLASES, y=CLASES, colorscale="Blues", zmin=0, zmax=1, text=[[f"{v:.2f}" for v in fila] for fila in norm],
                               texttemplate="%{text}", customdata=m.astype(int), colorbar=dict(title="Proporción"),
                               hovertemplate="Real %{y} → predicho %{x}: %{customdata:,} flujos (%{z:.1%})<extra></extra>"))
    fig.update_xaxes(title_text="Clase predicha")
    fig.update_yaxes(title_text="Clase real", autorange="reversed")
    return fig.update_layout(title="Matriz de confusión en prueba (normalizada por fila)", height=400)


def confiabilidad(curva: list[dict]) -> go.Figure:
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS, dash="dash"), name="Calibración perfecta"),
                     go.Scatter(x=[c["confianza"] for c in curva], y=[c["exactitud"] for c in curva], mode="lines+markers", name="Modelo",
                                line=dict(color=AZUL, width=2), marker=dict(size=[max(5, min(16, np.sqrt(c["n"]) / 4)) for c in curva]),
                                customdata=[c["n"] for c in curva], hovertemplate="Confianza %{x:.2f} → exactitud %{y:.2f} (%{customdata:,} flujos)<extra></extra>")])
    fig.update_xaxes(title_text="Confianza de la clase predicha", range=[0, 1])
    fig.update_yaxes(title_text="Exactitud observada", range=[0, 1.02])
    return fig.update_layout(title="Curva de confiabilidad (tras temperature scaling)", height=360)


def f1_por_clase(f1: list[float], f1_ref: list[float] | None, nombre: str, nombre_ref: str) -> go.Figure:
    fig = go.Figure(go.Bar(name=nombre, x=CLASES, y=f1, marker_color=AZUL, hovertemplate="%{x}: %{y:.3f}<extra>" + nombre + "</extra>"))
    if f1_ref is not None:
        fig.add_trace(go.Bar(name=nombre_ref, x=CLASES, y=f1_ref, marker_color=GRIS, hovertemplate="%{x}: %{y:.3f}<extra>" + nombre_ref + "</extra>"))
    fig.update_yaxes(title_text="F1 en prueba", range=[0, 1])
    return fig.update_layout(title="F1 por clase (corrida exportada)", barmode="group", height=340)


def intervalos_lideres(t: pd.DataFrame) -> go.Figure:
    t = t.sort_values("F1 macro")
    colores = [NARANJA if m == "Original" else (AZUL if g in ("Baseline fuerte", "Baseline del E1") else TINTA_2) for m, g in zip(t["Modelo"], t["Grupo"])]
    fig = go.Figure(go.Scatter(x=t["F1 macro"], y=t["Modelo"], mode="markers", marker=dict(color=colores, size=10),
                               error_x=dict(type="data", symmetric=False, array=t["IC95 sup"] - t["F1 macro"], arrayminus=t["F1 macro"] - t["IC95 inf"], color=GRIS),
                               customdata=np.stack([t["Semillas"], t["DE"]], 1),
                               hovertemplate="%{y}: %{x:.4f} (DE %{customdata[1]:.4f}; %{customdata[0]} semillas)<extra></extra>"))
    fig.update_xaxes(title_text="F1 macro medio en prueba (IC 95 % bootstrap sobre semillas)")
    return fig.update_layout(title="F1 macro por modelo", height=max(360, 26 * len(t) + 80), showlegend=False)


def rangos_friedman(rangos: dict, cd: float | None) -> go.Figure:
    orden = sorted(rangos.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Scatter(x=[v for _, v in orden], y=[k for k, _ in orden], mode="markers",
                               marker=dict(size=11, color=[NARANJA if k == "Original" else AZUL for k, _ in orden]),
                               hovertemplate="%{y}: rango medio %{x:.2f}<extra></extra>"))
    if cd:
        mejor = orden[0][1]
        fig.add_vrect(x0=mejor, x1=mejor + cd, fillcolor=AZUL, opacity=0.08, line_width=0)
        fig.add_annotation(x=mejor + cd / 2, y=1.04, yref="paper", text=f"Diferencia crítica de Nemenyi = {num(cd, 2)}", showarrow=False, font=dict(size=12))
    fig.update_xaxes(title_text="Rango medio en las semillas (1 = mejor)")
    fig.update_yaxes(autorange="reversed")
    return fig.update_layout(title="Rangos medios de Friedman (diagrama de diferencia crítica)", height=max(360, 26 * len(orden) + 80), showlegend=False)
