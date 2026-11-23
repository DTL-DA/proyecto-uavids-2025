"""Búsqueda por citación (snowballing) y en repositorios de código (sección 6.3 del enunciado).

Trabajos semilla: el paper guía, los tres trabajos más cercanos de la matriz y las dos fuentes del mecanismo.
  - Hacia adelante: obras de OpenAlex que citan cada semilla y que, en título o resumen, hablan de intrusiones (para
    las fuentes del mecanismo, con miles de citas, además de UAV, drones o FANET).
  - Hacia atrás: referencias (referenced_works) de los trabajos cercanos con mezcla de expertos, compuerta, fusión,
    árboles o codificación de atributos en el título.
  - Repositorios: GitHub (API pública de búsqueda de repositorios) y Hugging Face Hub (modelos y conjuntos de datos).
    Papers with Code cerró en julio de 2025 y redirige a Hugging Face, por eso no se consulta aparte.
Escribe exportaciones/<fecha>_citacion_repositorios_resumen_novedad.json; la clasificación de cada registro es manual
y queda en matriz_novedad.md.

Uso: python experiments/novedad/citacion_y_repositorios.py
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "exportaciones" / f"{date.today().isoformat()}_citacion_repositorios_resumen_novedad.json"
AGENTE = {"User-Agent": "busqueda-novedad-e3"}
SEMILLAS = {
    "Zeng et al. (2025), paper guía UAVIDS-2025": ("doi:10.1109/cns66487.2025.11194990", "intrusion"),
    "Zarkadis et al. (2026), XAI y stacking en UAVIDS-2025": ("doi:10.1109/cits70307.2026.11637173", "intrusion"),
    "Wang et al. (2025), Switch IDS con mezcla de expertos": ("doi:10.1109/jiot.2025.3559902", "intrusion"),
    "CNN + MoE para intrusiones en 5G (2024)": ("search:Convolutional Neural Networks and Mixture of Experts for Intrusion Detection in 5G Networks", "intrusion"),
    "Jacobs et al. (1991), mezcla de expertos": ("doi:10.1162/neco.1991.3.1.79", "intrusion AND (UAV OR drone OR FANET OR unmanned)"),
    "Gorishniy et al. (2022), codificación de atributos numéricos": ("doi:10.48550/arXiv.2203.05556", "intrusion"),
}
PATRON_ATRAS = re.compile(r"expert|gat(e|ing)|fusion|tree|boost|embedding|piecewise|hybrid|ensemble|lipschitz", re.I)
GITHUB = ["mixture of experts intrusion detection", "gating network intrusion detection", "UAVIDS-2025", "UAV intrusion detection mixture",
          "soft decision tree intrusion detection", "drone intrusion detection gated"]
HF = ["intrusion detection mixture of experts", "UAVIDS", "UAV intrusion detection", "drone intrusion"]


def pedir(url):
    for intento in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=40) as r:
                return json.loads(r.read())
        except Exception:  # reintenta ante límites de tasa o errores transitorios
            time.sleep(6 * (intento + 1))
    return None


def openalex_id(clave):
    if clave.startswith("doi:"):
        d = pedir("https://api.openalex.org/works/" + urllib.parse.quote(clave, safe=":/"))
        return (d or {}).get("id", "").rsplit("/", 1)[-1], (d or {}).get("title"), (d or {}).get("referenced_works", [])
    d = pedir("https://api.openalex.org/works?per-page=1&search=" + urllib.parse.quote(clave[7:]))
    w = (d or {}).get("results", [{}])[0]
    return w.get("id", "").rsplit("/", 1)[-1], w.get("title"), w.get("referenced_works", [])


def adelante(wid, filtro):
    url = ("https://api.openalex.org/works?per-page=100&select=id,title,publication_year,doi&filter="
           + urllib.parse.quote(f"cites:{wid},title_and_abstract.search:{filtro}"))
    d = pedir(url) or {}
    return d.get("meta", {}).get("count"), [{"titulo": w["title"], "anio": w["publication_year"], "doi": w.get("doi")} for w in d.get("results", [])]


def atras(referencias):
    if not referencias:
        return []
    ids = "|".join(r.rsplit("/", 1)[-1] for r in referencias[:100])
    d = pedir("https://api.openalex.org/works?per-page=100&select=id,title,publication_year,doi&filter=" + urllib.parse.quote(f"openalex:{ids}")) or {}
    return [{"titulo": w["title"], "anio": w["publication_year"], "doi": w.get("doi")} for w in d.get("results", []) if w.get("title") and PATRON_ATRAS.search(w["title"])]


def github(q):
    d = pedir("https://api.github.com/search/repositories?per_page=10&q=" + urllib.parse.quote(q)) or {}
    time.sleep(7)  # 10 búsquedas por minuto sin autenticación
    return d.get("total_count"), [{"repo": r["full_name"], "descripcion": r.get("description"), "estrellas": r.get("stargazers_count")} for r in d.get("items", [])]


def huggingface(q):
    salida = {}
    for tipo in ("models", "datasets"):
        d = pedir(f"https://huggingface.co/api/{tipo}?limit=20&search=" + urllib.parse.quote(q))
        salida[tipo] = [x.get("id") for x in (d or [])]
    return salida


def main():
    r = {"fecha": date.today().isoformat(), "semillas": {}, "github": {}, "huggingface": {}}
    for nombre, (clave, filtro) in SEMILLAS.items():
        wid, titulo, refs = openalex_id(clave)
        total, citantes = adelante(wid, filtro) if wid else (None, [])
        r["semillas"][nombre] = {"openalex": wid, "titulo": titulo, "filtro hacia adelante": filtro, "citantes filtrados": total, "citantes": citantes,
                                 "referencias": len(refs), "referencias relacionadas con el mecanismo": atras(refs)}
        print(f"{nombre}: {wid}, {total} citantes filtrados, {len(refs)} referencias", flush=True)
    for q in GITHUB:
        total, repos = github(q)
        r["github"][q] = {"total": total, "repositorios": repos}
        print(f"GitHub «{q}»: {total}", flush=True)
    for q in HF:
        r["huggingface"][q] = huggingface(q)
        print(f"Hugging Face «{q}»: {sum(len(v) for v in r['huggingface'][q].values())}", flush=True)
    SALIDA.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
