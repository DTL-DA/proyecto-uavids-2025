"""Punto de entrada del dashboard predictivo de UAVIDS-2025 (Proyecto Final de Deep Learning, Entregable 3).

Local:   python app.py        ->  http://127.0.0.1:8050
Render:  gunicorn app:server  (ver render.yaml)

La aplicación solo carga artefactos ya entrenados (ONNX) y hace inferencia en CPU: nunca entrena ni importa PyTorch.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import html  # noqa: E402

from src import plots  # noqa: E402,F401  registra la plantilla común de Plotly
from src.data import JBOOK, VERSION, lideres, uavids  # noqa: E402

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
                title="UAVIDS-2025 · Detección de intrusiones en drones", suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
                           {"name": "description", "content": "Dashboard predictivo de detección de intrusiones en enjambres de UAV"}])
server = app.server

# Datos y tabla de líderes se cargan una sola vez; los modelos ONNX se cargan bajo demanda (src/inference.py)
uavids()
lideres()

navbar = dbc.NavbarSimple(
    children=[dbc.NavItem(dbc.NavLink(p["name"], href=p["relative_path"], active="exact")) for p in dash.page_registry.values()],
    brand="UAVIDS-2025 · Proyecto final de Deep Learning", brand_href="/", color="primary", dark=True, sticky="top")

app.layout = html.Div([
    navbar,
    dbc.Container(dash.page_container, fluid="xl", className="py-4"),
    html.Footer(dbc.Container(html.Small([
        f"Donny Torres · Maestría en Analítica de Datos, Universidad del Norte · Versión {VERSION} · "
        "Datos: UAVIDS-2025 (Zeng, 2025; Zenodo, CC BY 4.0) · ",
        html.A("Jupyter Book del proyecto", href=JBOOK, target="_blank", rel="noopener noreferrer", className="enlace-pie")]), fluid="xl"),
        className="pie"),
])

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))
