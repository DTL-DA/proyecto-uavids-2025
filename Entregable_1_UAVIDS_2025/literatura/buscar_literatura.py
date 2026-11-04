"""Ejecuta las cadenas de búsqueda en las fuentes abiertas (OpenAlex y arXiv) y guarda las exportaciones crudas.

WoS, Scopus e IEEE Xplore se consultan desde su sitio web y sus exportaciones se dejan en esta misma carpeta.
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
HOY = date.today().isoformat()
UA = {"User-Agent": "Mozilla/5.0"}
DRON = '(drone OR UAV OR "unmanned aerial" OR FANET)'
DL = '("deep learning" OR "neural network" OR CNN OR LSTM OR MLP)'
CADENAS = {
    "C1": f'{DRON} AND ("intrusion detection" OR "attack detection") AND {DL}',
    "C2": f'{DRON} AND (swarm OR "ad hoc" OR "network traffic") AND (blackhole OR wormhole OR sybil OR flooding OR cyberattack) AND {DL}',
    "C3": 'UAVIDS OR "UAVIDS-2025"',
}
MAX_REG = 1000


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


def openalex(q):
    filtro = f"title_and_abstract.search:{q},from_publication_date:2019-01-01,to_publication_date:{HOY}"
    regs, total, pag = [], None, 1
    while total is None or (len(regs) < min(total, MAX_REG)):
        url = ("https://api.openalex.org/works?per-page=100&sort=cited_by_count:desc&page=%d&filter=" % pag
               + urllib.parse.quote(filtro))
        d = json.loads(pedir(url))
        total = d["meta"]["count"]
        if not d["results"]:
            break
        for w in d["results"]:
            oa = w.get("open_access") or {}
            regs.append({"titulo": w.get("title"), "anio": w.get("publication_year"),
                         "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                         "fuente": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                         "tipo": w.get("type"), "citas": w.get("cited_by_count"),
                         "oa_url": oa.get("oa_url"), "resumen": abstract_openalex(w.get("abstract_inverted_index"))})
        pag += 1
        time.sleep(3)
    return total, regs


def arxiv(q):
    campo = re.sub(r'(\(|\s)(?!AND|OR)("[^"]+"|\b\w[\w-]*\b)', lambda m: f"{m.group(1)}abs:{m.group(2)}", " " + q).strip()
    regs, total, ini = [], None, 0
    while ini == 0 or (ini < (total or 0) and len(regs) < MAX_REG):
        x = pedir("http://export.arxiv.org/api/query?max_results=100&start=%d&search_query=%s" % (ini, urllib.parse.quote(campo)))
        total = int(re.search(r"totalResults[^>]*>(\d+)<", x).group(1))
        for e in re.findall(r"<entry>(.*?)</entry>", x, re.S):
            t = re.search(r"<title>(.*?)</title>", e, re.S).group(1)
            s = re.search(r"<summary>(.*?)</summary>", e, re.S).group(1)
            u = re.search(r"<id>(.*?)</id>", e).group(1)
            y = re.search(r"<published>(\d{4})", e).group(1)
            d = re.search(r"<arxiv:doi[^>]*>(.*?)</arxiv:doi>", e)
            if int(y) >= 2019:
                regs.append({"titulo": " ".join(t.split()), "anio": int(y), "doi": d.group(1) if d else "", "fuente": "arXiv",
                             "tipo": "preprint", "citas": None, "oa_url": u, "resumen": " ".join(s.split())})
        ini += 100
        time.sleep(4)
    return total, regs


resumen = []
for nombre_fuente, fn in (("OpenAlex", openalex), ("arXiv", arxiv)):
    for c, q in CADENAS.items():
        total, regs = fn(q)
        (SALIDA / f"{HOY}_{nombre_fuente}_{c}.json").write_text(
            json.dumps({"fuente": nombre_fuente, "cadena": q, "fecha": HOY, "total": total, "registros": regs},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        resumen.append({"fuente": nombre_fuente, "cadena": c, "total": total, "exportados": len(regs)})
        print(nombre_fuente, c, "total", total, "exportados", len(regs), flush=True)
        time.sleep(3)
(SALIDA / f"{HOY}_resumen.json").write_text(json.dumps({"cadenas": CADENAS, "resultados": resumen}, indent=1), encoding="utf-8")
