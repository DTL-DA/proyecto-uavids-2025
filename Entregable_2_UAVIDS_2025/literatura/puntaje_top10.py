"""Puntaje explícito y reproducible para seleccionar el Top 10 (sección 3.2 del enunciado).

Candidatos: las familias de Deep Learning de la tabla de frecuencia de la literatura, sin las tres
de los modelos base del Entregable 1 (DNN/MLP, CNN y LSTM). Cada criterio se lleva a [0, 1]
dividiendo por el máximo entre candidatos, y el puntaje es la suma ponderada:

  P = 0,30·frecuencia + 0,15·citación normalizada + 0,15·recencia + 0,15·evidencia en UAVIDS-2025
      + 0,10·implementación disponible + 0,15·representatividad de la familia

- Frecuencia: estudios incluidos que usan la familia.
- Citación normalizada por antigüedad: log10(1 + citas por año) del artículo fundacional, según OpenAlex.
- Recencia: proporción de los estudios de la familia publicados en 2025 o 2026.
- Evidencia en datos comparables: estudios que la usan sobre UAVIDS-2025.
- Implementación: 1 si hay una implementación de referencia o de librería estándar; 0,5 si hay que programarla desde el artículo.
- Representatividad: 1 si aporta una familia o un sesgo inductivo que no tienen los modelos base; 0,5 si es una variante de una familia base.

Un candidato solo entra al Top 10 si es aplicable a flujos individuales con los 16 atributos de tráfico,
sin los identificadores que el Entregable 1 identificó como atajos. Las citas se consultan una vez y quedan en cache_citas.json.
"""
import csv
import json
import math
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

CARPETA = Path(__file__).parent
E1 = CARPETA.parents[1] / "Entregable_1_UAVIDS_2025" / "literatura"
PESOS = {"frecuencia": 0.30, "citacion": 0.15, "recencia": 0.15, "uavids": 0.15, "implementacion": 0.10, "representatividad": 0.15}
ANIO_CONSULTA = 2026
FUNDACIONALES = {
    "CNN-LSTM": ("Donahue et al. (2015)", "10.1109/cvpr.2015.7298878"),
    "RNN": ("Elman (1990)", "10.1207/s15516709cog1402_1"),
    "GRU": ("Cho et al. (2014)", "10.3115/v1/d14-1179"),
    "Autoencoder": ("Hinton y Salakhutdinov (2006)", "10.1126/science.1127647"),
    "BiLSTM": ("Graves y Schmidhuber (2005)", "10.1016/j.neunet.2005.06.042"),
    "DBN": ("Hinton et al. (2006)", "10.1162/neco.2006.18.7.1527"),
    "Transformer": ("Vaswani et al. (2017)", "10.48550/arxiv.1706.03762"),
    "GNN": ("Scarselli et al. (2009)", "10.1109/tnn.2008.2005605"),
    "ResNet": ("He et al. (2016)", "10.1109/cvpr.2016.90"),
    "GAN": ("Goodfellow et al. (2020)", "10.1145/3422622"),
    "TabNet": ("Arik y Pfister (2021)", "10.1609/aaai.v35i8.16826"),
    "KAN": ("Liu et al. (2024)", "10.48550/arxiv.2404.19756"),
}
IMPLEMENTACION = {"DBN": 0.5}
VARIANTE_BASE = {"RNN", "GRU", "BiLSTM"}
NO_APLICABLE = {
    "GNN": "Los tres estudios arman el grafo con direcciones IP, nodos o paquetes, información que este conjunto no trae o que funciona como atajo",
    "GAN": "Es un modelo generativo: los estudios lo usan para aumentar datos o generar ataques, no como clasificador",
}


def citas(doi, cache):
    if doi not in cache:
        url = f"https://api.openalex.org/works/doi:{doi}?select=doi,title,publication_year,cited_by_count"
        w = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
        cache[doi] = {"titulo": w["title"], "anio": w["publication_year"], "citas": w["cited_by_count"], "fecha_consulta": time.strftime("%Y-%m-%d")}
        time.sleep(1)
    return cache[doi]


def main():
    estudios = list(csv.DictReader((E1 / "extraccion_estudios.csv").open(encoding="utf-8-sig")))
    estudios += list(csv.DictReader((CARPETA / "extraccion_estudios_e2.csv").open(encoding="utf-8-sig")))
    por_familia = defaultdict(list)
    for e in estudios:
        for familia in {m.strip() for m in e["modelos"].split(";") if m.strip()}:
            por_familia[familia].append(e)
    ruta_cache = CARPETA / "cache_citas.json"
    cache = json.loads(ruta_cache.read_text(encoding="utf-8")) if ruta_cache.exists() else {}
    filas = []
    for modelo, (ref, doi) in FUNDACIONALES.items():
        lista = por_familia[modelo]
        c = citas(doi, cache)
        filas.append({"modelo": modelo, "estudios": len(lista), "estudios_uavids": sum("UAVIDS-2025" in e["dataset"] for e in lista),
                      "recientes": sum(int(e["anio"]) >= 2025 for e in lista), "articulo_fundacional": ref, "doi_fundacional": doi, "anio_fundacional": c["anio"],
                      "citas": c["citas"], "citas_por_anio": c["citas"] / (ANIO_CONSULTA - c["anio"] + 1),
                      "implementacion": IMPLEMENTACION.get(modelo, 1.0), "representatividad": 0.5 if modelo in VARIANTE_BASE else 1.0})
    ruta_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    brutos = {"frecuencia": lambda f: f["estudios"], "citacion": lambda f: math.log10(1 + f["citas_por_anio"]),
              "recencia": lambda f: f["recientes"] / f["estudios"] if f["estudios"] else 0.0, "uavids": lambda f: f["estudios_uavids"],
              "implementacion": lambda f: f["implementacion"], "representatividad": lambda f: f["representatividad"]}
    maximos = {k: max(fn(f) for f in filas) for k, fn in brutos.items()}
    for f in filas:
        f["puntaje"] = 0.0
        for k, fn in brutos.items():
            f[f"c_{k}"] = fn(f) / maximos[k] if maximos[k] else 0.0
            f["puntaje"] += PESOS[k] * f[f"c_{k}"]
        f["aplicable"] = "no" if f["modelo"] in NO_APLICABLE else "sí"
        f["motivo_no_aplicable"] = NO_APLICABLE.get(f["modelo"], "")
    filas.sort(key=lambda f: -f["puntaje"])
    seleccionados = 0
    for i, f in enumerate(filas, 1):
        f["posicion"] = i
        f["seleccionado"] = "sí" if f["aplicable"] == "sí" and seleccionados < 10 else "no"
        seleccionados += f["seleccionado"] == "sí"
    with (CARPETA / "puntaje_top10.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        for f in filas:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in f.items()})
    for f in filas:
        print(f"{f['posicion']:>2} {f['modelo']:12s} {f['puntaje']:.3f} estudios={f['estudios']} uavids={f['estudios_uavids']} recientes={f['recientes']} "
              f"citas/año={f['citas_por_anio']:.0f} aplicable={f['aplicable']} seleccionado={f['seleccionado']}")


if __name__ == "__main__":
    main()
