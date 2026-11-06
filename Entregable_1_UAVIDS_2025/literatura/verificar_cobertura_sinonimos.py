"""Verificación de cobertura de sinónimos, ejecutada en OpenAlex el 5 de noviembre de 2026.

Amplía C1 y C2 con sinónimos de plataforma, tarea y método que no estaban en las cadenas originales y
guarda los registros nuevos que superan el mismo filtro automático del proyecto. El cribado manual de esos
registros está en cribado_cobertura_sinonimos.csv; no se debe volver a ejecutar para no sobrescribirlo.
"""
import collections
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
L = Path(__file__).resolve().parent
salida = L
UA = {"User-Agent": "Mozilla/5.0 (academic literature review)"}
PLATAFORMA = '(drone OR UAV OR UAS OR "unmanned aerial" OR "unmanned aircraft" OR FANET OR "flying ad hoc" OR "Internet of Drones" OR IoD OR quadcopter)'
METODO = '("deep learning" OR "neural network" OR CNN OR LSTM OR MLP OR GRU OR RNN OR transformer OR autoencoder)'
CADENAS = {
    "C1 ampliada": f'{PLATAFORMA} AND ("intrusion detection" OR "attack detection" OR "anomaly detection" OR IDS OR "cyberattack detection" OR "threat detection") AND {METODO}',
    "C2 ampliada": f'{PLATAFORMA} AND (swarm OR "ad hoc" OR "network traffic") AND (blackhole OR "black hole" OR wormhole OR sybil OR flooding OR DDoS OR cyberattack) AND {METODO}',
}


def pedir(url):
    for i in range(6):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
        except Exception:
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def abstract(inv):
    return " ".join(w for _, w in sorted((p, w) for w, ps in (inv or {}).items() for p in ps))


def clave(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


# Mismo filtro automático del proyecto, leído de su script
fuente = (L / "depurar_registros.py").read_text(encoding="utf-8")
bloque = fuente[fuente.index("DRON = re.compile"): fuente.index('    return ""\n', fuente.index("def motivo_exclusion")) + len('    return ""\n')]
espacio = {"re": re}
exec(bloque, espacio)
motivo_exclusion = espacio["motivo_exclusion"]

registros = list(csv.DictReader((L / "registros_depurados.csv").open(encoding="utf-8-sig")))
dois = {r["doi"].lower() for r in registros if r["doi"]}
titulos = {clave(r["titulo"]) for r in registros}
nuevos = {}
for nombre, q in CADENAS.items():
    filtro = f"title_and_abstract.search:{q},from_publication_date:2019-01-01,to_publication_date:2026-11-04"
    total, pag, n = None, 1, 0
    while total is None or (n < min(total, 3000)):
        d = pedir("https://api.openalex.org/works?per-page=100&page=%d&filter=" % pag + urllib.parse.quote(filtro))
        total = d["meta"]["count"]
        if not d["results"]:
            break
        for w in d["results"]:
            n += 1
            doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            if (doi and doi in dois) or clave(w.get("title")) in titulos:
                continue
            k = doi or clave(w.get("title"))
            nuevos.setdefault(k, {"titulo": w.get("title"), "anio": w.get("publication_year"), "doi": doi, "fuente": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"), "tipo": w.get("type"),
                                  "resumen": abstract(w.get("abstract_inverted_index")), "cadenas": set()})["cadenas"].add(nombre)
        pag += 1
        time.sleep(1)
    print(nombre, "| total en OpenAlex:", total, "| revisados:", n)
for r in nuevos.values():
    r["motivo"] = motivo_exclusion(r)
pasan = [r for r in nuevos.values() if not r["motivo"] or r["motivo"].startswith("sin resumen")]
print("registros nuevos frente a los 1109:", len(nuevos), "| superan el filtro automático:", len(pasan))
print("motivos del filtro:", collections.Counter(r["motivo"] or "pasa" for r in nuevos.values()).most_common(8))
with (salida / "registros_cobertura_sinonimos.csv").open("w", encoding="utf-8-sig", newline="") as s:
    w = csv.writer(s)
    w.writerow(["n", "anio", "fuente", "titulo", "doi", "cadenas"])
    for i, r in enumerate(sorted(pasan, key=lambda x: (x["titulo"] or "").lower())):
        w.writerow([i, r["anio"], r["fuente"], r["titulo"], r["doi"], "; ".join(sorted(r["cadenas"]))])
