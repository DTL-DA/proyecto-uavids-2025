"""Sección 1: Contexto (narrativa estática y resumen visual)."""
import dash
import dash_bootstrap_components as dbc
from dash import html

from src.data import lideres, resumen, uavids
from src.plots import num

dash.register_page(__name__, path="/", name="1. Contexto", title="Contexto · UAVIDS-2025", order=1)


def _stat(cifra, texto, claro=False):
    return html.Div([html.Div(cifra, className="cifra"), html.Div(texto, className="cifra-texto")], className="stat" + (" stat-claro" if claro else ""))


def _seccion(titulo, *hijos):
    return dbc.Card(dbc.CardBody([html.H3(titulo, className="h5 mb-3"), *hijos]), className="seccion mb-4")


def _paso(icono, titulo, texto, pendiente=False):
    return html.Div([html.I(className=f"bi {icono} paso-icono"), html.Div(titulo, className="paso-titulo"), html.Div(texto, className="paso-texto")],
                    className="paso" + (" paso-pendiente" if pendiente else ""))


def layout():
    df = uavids()
    t = lideres().set_index("Modelo")
    r = resumen()
    mejor_rival = t.drop(index="Original", errors="ignore")["F1 macro"].idxmax()
    original = t["F1 macro"].get("Original")
    hero = html.Div([
        html.Div("Proyecto final de Deep Learning · Entregable 3", className="eyebrow"),
        html.H1("Detección de intrusiones en enjambres de drones", className="mb-3"),
        html.P("Un enjambre de UAV se comunica por una red ad hoc sin infraestructura fija. Un nodo comprometido puede descartar paquetes "
               "(Blackhole), tunelizarlos (Wormhole), suplantar identidades (Sybil) o saturar la red (Flooding). Este tablero compara 15 modelos "
               "y un modelo original, MoE-ES, sobre 122 171 flujos simulados de UAVIDS-2025, con un protocolo preregistrado.", className="lead"),
        dbc.Row([
            dbc.Col(_stat(f"{len(df):,}".replace(",", " "), "flujos de red (5 clases, 16 atributos)"), md=3, xs=6, className="mb-2"),
            dbc.Col(_stat(num(t.loc[mejor_rival, "F1 macro"], 3), f"F1 macro en prueba del mejor rival ({mejor_rival})"), md=3, xs=6, className="mb-2"),
            dbc.Col(_stat(num(original, 3) if original is not None else "—", "F1 macro en prueba del modelo original"), md=3, xs=6, className="mb-2"),
            dbc.Col(_stat("12", "contrastes confirmatorios con Holm y 3 pruebas cada uno"), md=3, xs=6, className="mb-2"),
        ], className="mt-3"),
    ], className="hero mb-4")
    por_que = _seccion("Por qué importa",
        html.P("Los drones se usan en vigilancia, logística y atención de desastres. Sus redes no tienen un perímetro que proteger: cada nodo "
               "enruta el tráfico de los demás, así que un solo nodo malicioso puede aislar una parte del enjambre. Un detector útil debe "
               "reconocer el tipo de ataque, no solo su presencia, porque la respuesta es distinta en cada caso."),
        html.P("La literatura reporta exactitudes de 97 a 100 % sobre UAVIDS-2025 con particiones aleatorias. Este proyecto mide el desempeño "
               "con una partición por bloques que evita mezclar flujos vecinos entre entrenamiento y prueba, y por eso las cifras son más bajas "
               "y más honestas.", className="insight"))
    preguntas = _seccion("Preguntas de investigación", html.Ol([
        html.Li("¿Supera el modelo original, con significancia estadística, a cada uno de los 10 modelos del Top 10 y a los dos baselines fuertes "
                "(LightGBM y CNN 1D)?"),
        html.Li("¿Reduce el experto de escalones suaves la confusión Blackhole–Wormhole, la limitación del paper guía?"),
        html.Li("¿Por qué el escenario de simulación pesa más que la arquitectura en el desempeño, y qué tan bien generalizan los modelos a un "
                "escenario no visto?"),
    ]))
    ficha = _seccion("Ficha de datos", dbc.Table([html.Tbody([
        html.Tr([html.Th("Fuente"), html.Td("UAVIDS-2025 (Zeng, 2025), Zenodo DOI 10.5281/zenodo.15336998, licencia CC BY 4.0")]),
        html.Tr([html.Th("Generación"), html.Td("Simulación de enjambres de UAV en NS-3.24 con movilidad BOID; un registro por flujo")]),
        html.Tr([html.Th("Observaciones"), html.Td("122 171 flujos; 8 219 duplicados exactos se excluyen antes de partir")]),
        html.Tr([html.Th("Variables"), html.Td("16 estadísticos por flujo (duración, paquetes, bytes, tasas, retardo, jitter, pérdida, rendimiento, saltos). "
                                               "IP, puertos, protocolo y FlowID se excluyen por ser identificadores")]),
        html.Tr([html.Th("Objetivo y tarea"), html.Td("Clase del flujo: Blackhole, Flooding, Normal, Sybil o Wormhole. Clasificación multiclase; métrica "
                                                      "primaria F1 macro con margen de relevancia δ = 0,01")]),
        html.Tr([html.Th("Partición"), html.Td("Bloques de 1000 flujos contiguos: 73 267 de entrenamiento, 17 802 de validación y 22 883 de prueba "
                                               "(sellada en el Entregable 1, verificada con SHA-256)")]),
    ])], bordered=False, size="sm", className="tabla-texto mb-0"))
    pipeline = _seccion("Flujo metodológico", html.Div([
        _paso("bi-database-check", "Datos", "Deduplicación, log(1 + x) y partición por bloques"),
        _paso("bi-shield-check", "Auditoría", "Fuga, escenarios y determinismo"),
        _paso("bi-diagram-3", "Benchmark", "15 modelos y 20 a 48 semillas, mismo presupuesto"),
        _paso("bi-cpu", "Modelo original (MoE-ES)", "Compuerta entre escalones suaves y un experto suave"),
        _paso("bi-clipboard-data", "Protocolo", "Preregistrado con etiqueta de Git"),
        _paso("bi-graph-up", "Pruebas", "3 pruebas por contraste, Holm y potencia"),
        _paso("bi-globe", "Ampliada", "3 conjuntos adicionales y robustez"),
    ], className="pipeline"))
    limites = _seccion("Limitaciones, ética y autoría",
        html.Ul([
            html.Li("Los datos son simulados. Los escenarios de ataque son tramos largos de una sola clase, y el 70 % de las filas de prueba pertenecen "
                    "a un escenario con filas en entrenamiento. Ningún modelo reconoce bien un escenario nuevo."),
            html.Li("Un detector con falsos negativos deja pasar ataques; uno con falsos positivos puede aislar drones legítimos. Las predicciones "
                    "de este tablero son una ayuda para un analista, no una decisión automática."),
            html.Li("Los conjuntos de predicción conforme suponen intercambiabilidad, que la dependencia entre bloques no garantiza; su cobertura "
                    "se verifica en la prueba."),
        ]),
        html.P(["Autor: Donny Torres (Maestría en Analítica de Datos, Universidad del Norte). Uso de IA declarado en el último capítulo del libro. "
                "Paper guía: Zeng (2025). ", f"Resultado de R3: {'cumplido' if r.get('R3 (supera a los 12)') else 'ver la página de modelos'}."],
               className="nota mb-0"))
    return html.Div([hero, dbc.Row([dbc.Col(por_que, lg=6), dbc.Col(preguntas, lg=6)]), ficha, pipeline, limites])
