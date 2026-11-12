"""Intenta recuperar el texto completo de los registros elegibles en fuentes abiertas: OpenAlex, Europe PMC y arXiv.

Uso: python recuperar_texto.py <carpeta_destino>. Cada documento queda en su propia carpeta y el resultado de cada
intento se registra en recuperacion_texto.csv, de modo que lo no recuperado también queda documentado.
"""
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
CARPETA = Path(__file__).parent
DESTINO = Path(sys.argv[1])
DESTINO.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (literature review; academic use)"}
PDFTOTEXT = shutil.which("pdftotext")


def traer(url, limite=25_000_000, espera=40):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=espera) as r:
        return r.read(limite), r.headers.get("Content-Type", "")


def desde_pdf(url, carpeta):
    datos, _ = traer(url)
    if not datos.startswith(b"%PDF"):
        return ""
    pdf = carpeta / "doc.pdf"
    pdf.write_bytes(datos)
    subprocess.run([PDFTOTEXT, "-layout", str(pdf), str(carpeta / "doc.txt")], check=True, timeout=120)
    return (carpeta / "doc.txt").read_text(encoding="utf-8", errors="replace")


def europe_pmc(doi, carpeta):
    consulta = urllib.parse.quote(f'DOI:"{doi}"')
    respuesta = json.loads(traer(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={consulta}&format=json")[0])
    for r in respuesta.get("resultList", {}).get("result", []):
        if r.get("pmcid"):
            xml, _ = traer(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{r['pmcid']}/fullTextXML")
            texto = " ".join(ET.fromstring(xml).itertext())
            (carpeta / "doc.txt").write_text(texto, encoding="utf-8")
            return texto
    return ""


def openalex(doi, carpeta):
    obra = json.loads(traer("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="/.")) [0])
    urls = [l.get("pdf_url") for l in obra.get("locations", []) if l.get("pdf_url")]
    for url in dict.fromkeys(urls):
        try:
            texto = desde_pdf(url, carpeta)
            if texto:
                return texto, url
        except Exception:
            continue
    return "", ""


def arxiv(titulo, carpeta):
    limpio = re.sub(r"[^A-Za-z0-9 ]", " ", titulo)
    consulta = urllib.parse.quote(f'ti:"{" ".join(limpio.split()[:12])}"')
    # arXiv responde 429 si las consultas van muy seguidas
    for espera in (4, 30, 60, 120):
        time.sleep(espera)
        try:
            xml, _ = traer(f"http://export.arxiv.org/api/query?search_query={consulta}&max_results=3")
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or espera == 120:
                raise
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in ET.fromstring(xml).findall("a:entry", ns):
        hallado = re.sub(r"[^a-z0-9]", "", e.find("a:title", ns).text.lower())
        if hallado[:50] == re.sub(r"[^a-z0-9]", "", titulo.lower())[:50]:
            pid = e.find("a:id", ns).text.split("/abs/")[-1]
            return desde_pdf(f"https://arxiv.org/pdf/{pid}", carpeta), f"arXiv {pid}"
    return "", ""


ruta_registro = CARPETA / "recuperacion_texto.csv"
hechos = {r["clave"]: r for r in csv.DictReader(ruta_registro.open(encoding="utf-8-sig"))} if ruta_registro.exists() else {}
elegibles = [r for r in csv.DictReader((CARPETA / "cribado_titulo_resumen.csv").open(encoding="utf-8-sig")) if r["decision"] == "elegible"]
for n, r in enumerate(elegibles):
    clave = r["doi"].lower() or re.sub(r"[^a-z0-9]", "", r["titulo"].lower())[:60]
    if clave in hechos and hechos[clave]["caracteres"] not in ("0", ""):
        continue
    carpeta = DESTINO / f"{n:03d}"
    carpeta.mkdir(exist_ok=True)
    texto, via = "", ""
    intentos = [("Europe PMC", lambda: (europe_pmc(r["doi"], carpeta), "Europe PMC")) if r["doi"] else None,
                ("OpenAlex", lambda: openalex(r["doi"], carpeta)) if r["doi"] else None,
                ("Acceso abierto", lambda: (desde_pdf(r["acceso_abierto"], carpeta), r["acceso_abierto"])) if r["acceso_abierto"] else None,
                ("arXiv", lambda: arxiv(r["titulo"], carpeta))]
    errores = []
    for intento in filter(None, intentos):
        try:
            texto, via = intento[1]()
            if len(texto) > 3000:
                break
            texto = ""
        except Exception as error:
            errores.append(f"{intento[0]}: {repr(error)[:50]}")
        time.sleep(0.4)
    hechos[clave] = {"clave": clave, "carpeta": carpeta.name, "titulo": r["titulo"], "anio": r["anio"], "doi": r["doi"], "caracteres": len(texto), "via": via, "errores": " | ".join(errores)}
    print(n, len(texto), (via or "-")[:30], r["titulo"][:60], flush=True)
    with ruta_registro.open("w", encoding="utf-8-sig", newline="") as s:
        w = csv.DictWriter(s, fieldnames=["clave", "carpeta", "titulo", "anio", "doi", "caracteres", "via", "errores"])
        w.writeheader()
        w.writerows(hechos.values())
print("recuperados:", sum(1 for h in hechos.values() if int(h["caracteres"]) > 0), "de", len(hechos))
