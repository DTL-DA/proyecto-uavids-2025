"""Asigna el cuartil SJR por título y, si no coincide, por ISSN de OpenAlex, y recalcula la frecuencia de modelos por cuartil.

Uso: python asignar_cuartiles.py <archivo de scimagojr.com> <carpeta literatura> <carpeta de salida>
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
csv.field_size_limit(10_000_000)
sjr_ruta, literatura, salida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
salida.mkdir(parents=True, exist_ok=True)


def clave(t):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^(the|proceedings of( the)?)\s+", "", (t or "").lower().strip()))


por_titulo, por_issn = {}, {}
for f in csv.DictReader(sjr_ruta.read_bytes().decode("utf-8-sig", "replace").splitlines(), delimiter=";"):
    q = f.get("SJR Best Quartile") or ""
    dato = (q if q in ("Q1", "Q2", "Q3", "Q4") else "sin cuartil", f["Title"], f.get("Type", ""))
    por_titulo[clave(f["Title"])] = dato
    for i in re.findall(r"[0-9Xx]{8}", f.get("Issn", "")):
        por_issn[i.upper()] = dato


def leer(nombre):
    return list(csv.DictReader((literatura / nombre).open(encoding="utf-8-sig")))


registros = leer("registros_depurados.csv")
elegibles = [r for r in registros if r["cribado"] == "elegible"]
incluidos = leer("extraccion_estudios.csv")
ruta_cache = salida / "cache_issn.json"
cache = json.loads(ruta_cache.read_text(encoding="utf-8")) if ruta_cache.exists() else {}


def issn_de(doi):
    if not doi or not doi.startswith("10."):
        return []
    if doi not in cache:
        try:
            url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="/.()")
            obra = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (academic literature review)"}), timeout=40).read())
            cache[doi] = ((obra.get("primary_location") or {}).get("source") or {}).get("issn") or []
        except Exception:
            cache[doi] = []
        time.sleep(0.15)
    return cache[doi]


def asignar(fuente, doi):
    d = por_titulo.get(clave(fuente))
    if d:
        return d[0], d[1], "título"
    for i in issn_de(doi):
        d = por_issn.get(i.replace("-", "").upper())
        if d:
            return d[0], d[1], "ISSN"
    return "no está en SJR", "", ""


fuentes = {}
for r in elegibles:
    r["_q"] = asignar(r["fuente_publicacion"], r["doi"])
    fuentes.setdefault(r["fuente_publicacion"], r["_q"])
for e in incluidos:
    e["_q"] = asignar(e["fuente"], e["doi"])
    fuentes.setdefault(e["fuente"], e["_q"])
ruta_cache.write_text(json.dumps(cache), encoding="utf-8")

with (salida / "cuartiles_fuentes.csv").open("w", encoding="utf-8-sig", newline="") as s:
    w = csv.writer(s)
    w.writerow(["fuente", "cuartil_sjr_2025", "titulo_en_sjr", "coincidencia"])
    for f, q in sorted(fuentes.items()):
        if f:
            w.writerow([f, *q])

print("=== 24 incluidos")
for e in sorted(incluidos, key=lambda x: (x["_q"][0], x["fuente"])):
    print(f"  {e['_q'][0]:15} {e['anio']} {e['autores'].split(';')[0][:22]:22} {e['fuente'][:48]:48} {e['_q'][2]} {e['_q'][1][:34] if e['_q'][2]=='ISSN' else ''}")
print(" ", collections.Counter(e["_q"][0] for e in incluidos).most_common())

print("\n=== frecuencia por cuartil")
grupos = {"Todos": incluidos, "Q1": [e for e in incluidos if e["_q"][0] == "Q1"], "Q1 o Q2": [e for e in incluidos if e["_q"][0] in ("Q1", "Q2")]}
familias = sorted({f for e in incluidos for f in e["modelos"].split("; ")})
filas = []
for f in familias:
    filas.append((f, *[sum(f in e["modelos"].split("; ") for e in g) for g in grupos.values()]))
print(f"  {'familia':14}", *[f"{k} [n={len(v)}]" for k, v in grupos.items()], sep=" | ")
for fila in sorted(filas, key=lambda x: (-x[2], -x[1])):
    print(f"  {fila[0]:14}", *fila[1:], sep=" | ")

print("\n=== elegibles")
tabla = collections.defaultdict(collections.Counter)
for r in elegibles:
    tabla[r["texto_completo"]][r["_q"][0]] += 1
for k, v in tabla.items():
    print(" ", k, dict(v))

pendientes = [r for r in elegibles if r["texto_completo"] == "no recuperado"]
orden = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "sin cuartil": 4, "no está en SJR": 5}
filas_salida = [{"cuartil_sjr_2025": r["_q"][0], "anio": r["anio"], "titulo": r["titulo"], "fuente": r["fuente_publicacion"], "doi": r["doi"], "enlace_doi": f"https://doi.org/{r['doi']}" if r["doi"] else "",
                 "enlace_abierto": r["acceso_abierto"], "cadenas": r["cadenas"]} for r in sorted(pendientes, key=lambda x: (orden[x["_q"][0]], -int(x["anio"] or 0), x["titulo"]))]
with (salida / "priorizacion_texto_completo.csv").open("w", encoding="utf-8-sig", newline="") as s:
    w = csv.DictWriter(s, fieldnames=list(filas_salida[0]))
    w.writeheader()
    w.writerows(filas_salida)
q1 = [f for f in filas_salida if f["cuartil_sjr_2025"] == "Q1"]
print("\nno recuperados Q1:", len(q1), "| con enlace abierto:", sum(bool(f["enlace_abierto"]) for f in q1), "| IEEE:", sum(f["doi"].startswith("10.1109") for f in q1), "| Elsevier:", sum(f["doi"].startswith("10.1016") for f in q1), "| Springer:", sum(f["doi"].startswith("10.1007") for f in q1), "| MDPI:", sum(f["doi"].startswith("10.3390") for f in q1))
print("sin coincidencia entre elegibles:", collections.Counter(r["fuente_publicacion"] for r in elegibles if r["_q"][0] == "no está en SJR").most_common(16))
