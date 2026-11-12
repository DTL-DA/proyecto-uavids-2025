"""Búsqueda del Entregable 2 en las fuentes abiertas (OpenAlex y arXiv).

Extiende la revisión del Entregable 1 con tres grupos de cadenas:
- C1 a C3: las mismas del Entregable 1, ejecutadas de nuevo para captar lo publicado después del 3 de noviembre de 2026.
- C4: cadena ampliada por familias de modelos candidatas al Top 10, sobre el mismo problema.
- T1 a T4: búsqueda transversal de mecanismos en otras modalidades, con "transfer" o "adaptation" (sección 3.1 del enunciado).

WoS, Scopus e IEEE Xplore se consultan desde su sitio web con las cadenas de consultas_e2_wos_scopus_ieee.md,
y sus exportaciones se dejan en exportaciones/.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

SALIDA = Path(__file__).parent / "exportaciones"
SALIDA.mkdir(exist_ok=True)
HOY = date.today().isoformat()
UA = {"User-Agent": "Mozilla/5.0"}
DRON = '(drone OR UAV OR "unmanned aerial" OR FANET)'
DL = '("deep learning" OR "neural network" OR CNN OR LSTM OR MLP)'
CADENAS = {
    "C1": f'{DRON} AND ("intrusion detection" OR "attack detection") AND {DL}',
    "C2": f'{DRON} AND (swarm OR "ad hoc" OR "network traffic") AND (blackhole OR wormhole OR sybil OR flooding OR cyberattack) AND {DL}',
    "C3": 'UAVIDS OR "UAVIDS-2025"',
    "C4": '(drone OR UAV OR "unmanned aerial" OR FANET OR "Internet of Drones") AND ("intrusion detection" OR "attack detection" OR "anomaly detection") '
          'AND (GRU OR "gated recurrent" OR BiLSTM OR "bidirectional LSTM" OR transformer OR TabNet OR "Kolmogorov-Arnold" OR autoencoder OR "deep belief" '
          'OR ResNet OR residual OR "CNN-LSTM" OR "graph neural" OR "temporal convolutional" OR mamba OR "state space")',
    "T1": '("supervised contrastive" OR "contrastive learning") AND (image OR "computer vision" OR visual) AND (transfer OR adaptation)',
    "T2": '("mixture of experts" OR "gated experts") AND ("natural language" OR "language model" OR vision) AND (transfer OR adaptation)',
    "T3": '("domain generalization" OR "invariant risk minimization" OR "distributionally robust") AND (image OR vision OR speech) AND (transfer OR adaptation)',
    "T4": '("Kolmogorov-Arnold" OR "feature tokenizer" OR "tabular transformer") AND ("time series" OR vision OR image OR language) AND (transfer OR adaptation)',
}
# Las cadenas transversales devuelven miles de registros: se exportan los más citados, que se revisan por título y resumen
MAXIMOS = {"C1": 1000, "C2": 1000, "C3": 1000, "C4": 1000, "T1": 100, "T2": 100, "T3": 100, "T4": 100}
DESDE = {"C1": "2019-01-01", "C2": "2019-01-01", "C3": "2019-01-01", "C4": "2019-01-01", "T1": "2018-01-01", "T2": "2018-01-01", "T3": "2018-01-01", "T4": "2018-01-01"}


def pedir(url, intentos=6):
    for i in range(intentos):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (401, 429, 500, 502, 503) and i < intentos - 1:
                time.sleep(10 * (i + 1))
                continue
            raise


def abstract_openalex(inv):
    if not inv:
        return ""
    return " ".join(w for _, w in sorted((p, w) for w, ps in inv.items() for p in ps))


def openalex(clave, q):
    filtro = f"title_and_abstract.search:{q},from_publication_date:{DESDE[clave]},to_publication_date:{HOY}"
    regs, total, pag = [], None, 1
    while total is None or (len(regs) < min(total, MAXIMOS[clave])):
        url = ("https://api.openalex.org/works?per-page=100&sort=cited_by_count:desc&page=%d&filter=" % pag + urllib.parse.quote(filtro))
        d = json.loads(pedir(url))
        total = d["meta"]["count"]
        if not d["results"]:
            break
        for w in d["results"]:
            oa = w.get("open_access") or {}
            regs.append({"titulo": w.get("title"), "anio": w.get("publication_year"), "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                         "fuente": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"), "tipo": w.get("type"),
                         "citas": w.get("cited_by_count"), "oa_url": oa.get("oa_url"), "resumen": abstract_openalex(w.get("abstract_inverted_index"))})
        pag += 1
        time.sleep(3)
    return total, regs[:MAXIMOS[clave]]


def arxiv(clave, q):
    campo = re.sub(r'(\(|\s)(?!AND|OR)("[^"]+"|\b\w[\w-]*\b)', lambda m: f"{m.group(1)}abs:{m.group(2)}", " " + q).strip()
    regs, total, ini = [], None, 0
    anio_min = int(DESDE[clave][:4])
    while ini == 0 or (ini < (total or 0) and len(regs) < MAXIMOS[clave]):
        x = pedir("http://export.arxiv.org/api/query?max_results=100&start=%d&search_query=%s" % (ini, urllib.parse.quote(campo)))
        total = int(re.search(r"totalResults[^>]*>(\d+)<", x).group(1))
        for e in re.findall(r"<entry>(.*?)</entry>", x, re.S):
            t = re.search(r"<title>(.*?)</title>", e, re.S).group(1)
            s = re.search(r"<summary>(.*?)</summary>", e, re.S).group(1)
            u = re.search(r"<id>(.*?)</id>", e).group(1)
            y = re.search(r"<published>(\d{4})", e).group(1)
            d = re.search(r"<arxiv:doi[^>]*>(.*?)</arxiv:doi>", e)
            if int(y) >= anio_min:
                regs.append({"titulo": " ".join(t.split()), "anio": int(y), "doi": d.group(1) if d else "", "fuente": "arXiv", "tipo": "preprint",
                             "citas": None, "oa_url": u, "resumen": " ".join(s.split())})
        ini += 100
        if ini >= 300 and clave.startswith("T"):
            break
        time.sleep(4)
    return total, regs[:MAXIMOS[clave]]


if __name__ == "__main__":
    resumen = []
    for nombre_fuente, fn in (("OpenAlex", openalex), ("arXiv", arxiv)):
        for c, q in CADENAS.items():
            total, regs = fn(c, q)
            (SALIDA / f"{HOY}_{nombre_fuente}_{c}.json").write_text(
                json.dumps({"fuente": nombre_fuente, "cadena": q, "fecha": HOY, "total": total, "registros": regs}, ensure_ascii=False, indent=1), encoding="utf-8")
            resumen.append({"fuente": nombre_fuente, "cadena": c, "total": total, "exportados": len(regs)})
            print(nombre_fuente, c, "total", total, "exportados", len(regs), flush=True)
            time.sleep(3)
    (SALIDA / f"{HOY}_resumen_e2.json").write_text(json.dumps({"cadenas": CADENAS, "maximos": MAXIMOS, "desde": DESDE, "resultados": resumen}, indent=1), encoding="utf-8")
