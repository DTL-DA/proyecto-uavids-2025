"""Exploración de los escenarios de simulación con SOLO las filas de entrenamiento del E2.
Uso: python explorar_escenarios.py <raiz_repo_E3>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

raiz = Path(sys.argv[1])
sys.path.insert(0, str(raiz / "Entregable_2_UAVIDS_2025"))
from codigo import datos as D

datos, bloques, _ = D.crudo()
y = D.etiquetas()
ids = D.heredados()["ids"]
ent = np.sort(ids["entrenamiento"])
t = datos.iloc[ent].copy()
t["clase"] = [D.CLASES[i] for i in y[ent]]
t["bloque"] = bloques[ent]
t["red_src"] = t["SrcAddr"].astype(str).str.rsplit(".", n=1).str[0]
t["red_dst"] = t["DstAddr"].astype(str).str.rsplit(".", n=1).str[0]
print(t[["FlowID", "SrcAddr", "SrcPort", "DstAddr", "DstPort", "Protocol"]].head(3).to_string())
# Tramos de una clase sobre las filas de entrenamiento ordenadas
c = t["clase"].to_numpy()
tramo = np.concatenate([[0], np.cumsum(c[1:] != c[:-1])])
t["tramo"] = tramo
r = t.groupby("tramo").agg(clase=("clase", "first"), ini=("FlowID", "min"), fin=("FlowID", "max"), n=("clase", "size"),
                           bloques=("bloque", "nunique"), redes_src=("red_src", lambda s: ",".join(sorted(s.unique())[:4])),
                           redes_dst=("red_dst", lambda s: ",".join(sorted(s.unique())[:4])), dur_med=("FlowDuration/s", "median"))
grandes = r[r["n"] >= 50]
pd.set_option("display.width", 250)
print(grandes.to_string())
print("tramos totales", len(r), "| filas en tramos >=50:", grandes["n"].sum(), "de", len(t))
# Región intercalada Blackhole/Normal: tramos cortos
cortos = r[r["n"] < 50]
print(cortos.groupby("clase").agg(tramos=("n", "size"), filas=("n", "sum"), ini=("ini", "min"), fin=("fin", "max")))
# Redes por clase
print(t.groupby(["clase", "red_src"]).size().unstack(fill_value=0).T.to_string())
