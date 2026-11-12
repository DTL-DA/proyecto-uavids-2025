"""Auxiliares de presentación del cuaderno: formato de números en español, tablas y estilo de las gráficas."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from pandas.io.formats.style import Styler

NEGRO, GRIS_OSCURO, GRIS_MEDIO, GRIS_CLARO = "#000000", "#4a4a4a", "#7a7a7a", "#d9d9d9"
LINEAS = ["-", "--", ":", "-."]
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*"]


def configurar():
    """Coma decimal en tablas y gráficas en escala de grises."""
    pd.set_option("styler.format.decimal", ",")
    pd.set_option("styler.format.precision", 4)
    pd.set_option("display.float_format", lambda v: f"{v:.5f}".rstrip("0").rstrip(".").replace(".", ","))
    if not getattr(Styler, "_con_coma", False):
        original = Styler.format

        def formato_con_coma(self, *a, **k):
            k.setdefault("decimal", ",")
            k.setdefault("thousands", " ")
            k.setdefault("na_rep", "")
            return original(self, *a, **k)

        Styler.format = formato_con_coma
        Styler._con_coma = True
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "image.cmap": "Greys", "axes.grid": True, "grid.color": GRIS_CLARO,
                         "axes.edgecolor": GRIS_MEDIO, "axes.prop_cycle": mpl.cycler(color=[NEGRO, GRIS_OSCURO, GRIS_MEDIO]),
                         "axes.titlesize": 10.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.formatter.use_locale": False})


def num(valor, decimales=3):
    """Número para el texto: coma decimal y espacio de miles."""
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "no disponible"
    return f"{valor:,.{decimales}f}".replace(",", " ").replace(".", ",")


def pct(valor, decimales=1):
    return num(100 * valor, decimales) + " %"


def enumerar(elementos):
    elementos = [str(e) for e in elementos]
    return ", ".join(elementos[:-1]) + " y " + elementos[-1] if len(elementos) > 1 else "".join(elementos)


def valor_p(p):
    return "p < 0,001" if p < 0.001 else f"p = {num(p, 3)}"


def p_exacto(p):
    """p exacto para tablas: notación científica cuando es muy pequeño."""
    if p is None or not np.isfinite(p):
        return "no disponible"
    return f"{p:.2e}".replace(".", ",") if p < 0.0001 else num(p, 4)


def intervalo(par, decimales=4):
    return f"[{num(par[0], decimales)}; {num(par[1], decimales)}]"


def interpretar(texto):
    display(Markdown(texto))


def tabla(df, formatos=None, indice=False):
    estilo = df.style if indice else df.style.hide(axis="index")
    display(estilo.format(formatos or {}, precision=4))
