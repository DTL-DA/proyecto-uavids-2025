"""Disponibilidad de código y de datos en los estudios incluidos (sección 3.4 del enunciado).

Para cada estudio incluido busca el texto completo en acceso abierto (OpenAlex) o en una carpeta local
de textos ya extraídos, y registra los pasajes que declaran el código o los datos. El texto de los
artículos no se guarda en el repositorio: solo la decisión y la URL del código, si existe.

    python reproducibilidad_estudios.py <carpeta_de_textos>
"""
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

CARPETA = Path(__file__).parent
E1 = CARPETA.parents[1] / "Entregable_1_UAVIDS_2025" / "literatura"
TEXTOS = Path(sys.argv[1])
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DATOS_PUBLICOS = re.compile(r"UAVIDS|UAV-NIDD|CICIDS|CIC-IDS|NSL-KDD|KDD|UNSW|ToN|TON_IoT|Edge-IIoT|ISOT|CICIoT|CIC IoT|ISCXVPN|WSN-DS|InSDN|5G-NIDD|NF-UQ|UAV IDS|UAV-IDS|GCS-NIDD|UNR-IDD|CICIoMT|UAV Attack|IEEE DataPort", re.I)
CODIGO = re.compile(r"(https?://(?:www\.)?(?:github\.com|gitlab\.com|zenodo\.org|figshare\.com|codeocean\.com)/\S+|code (?:is|will be) (?:publicly )?available|source code (?:is|will be) available|"
                    r"code availability|available upon (?:reasonable )?request)", re.I)


# Revisión manual de los enlaces detectados: dos son de datos o de herramientas, y el código de Zeng existe aunque el artículo no lo cite
CORRECCIONES = {
    "10.3389/fenrg.2024.1491332": ("no", "", "El enlace a GitHub es la fuente de un conjunto de datos, no el código del estudio"),
    "10.1109/ojcoms.2026.3683883": ("no", "", "El enlace a GitHub es la herramienta CICFlowMeter, citada como referencia"),
    "10.1109/cns66487.2025.11194990": ("sí", "https://github.com/QingliZeng/UAV--IDS2025", "Repositorio de los autores con el CSV y el código; el artículo no lo cita"),
}


def texto_local(doi):
    for p in TEXTOS.rglob("*.txt"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        if doi.lower() in t[:20000].lower():
            return t
    return None


def descargar(doi):
    try:
        w = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://api.openalex.org/works/doi:{doi}?select=best_oa_location,open_access", headers=UA), timeout=60).read())
        url = (w.get("best_oa_location") or {}).get("pdf_url") or (w.get("open_access") or {}).get("oa_url")
        if not url:
            return None
        datos = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        if not datos.startswith(b"%PDF"):
            return None
        import fitz

        return "\n".join(pagina.get_text() for pagina in fitz.open(stream=datos, filetype="pdf"))
    except Exception:
        return None


def main():
    estudios = list(csv.DictReader((E1 / "extraccion_estudios.csv").open(encoding="utf-8-sig")))
    estudios += list(csv.DictReader((CARPETA / "extraccion_estudios_e2.csv").open(encoding="utf-8-sig")))
    filas = []
    for e in estudios:
        doi = e["doi"]
        texto = texto_local(doi) if doi else None
        if texto is None and doi:
            texto = descargar(doi)
            time.sleep(1)
        if texto:
            hallazgos = sorted({m.group(0).rstrip(".,;)") for m in CODIGO.finditer(texto)})
            enlaces = [h for h in hallazgos if h.lower().startswith("http")]
            if enlaces:
                codigo = "sí"
            elif any("request" in h.lower() for h in hallazgos):
                codigo = "a solicitud"
            elif hallazgos:
                codigo = "declarado sin enlace"
            else:
                codigo = "no"
        else:
            enlaces, codigo = [], "no verificable"
        nota = e.get("nota", "")
        ajustados = "no" if "cifras publicadas" in nota or "tomados de trabajos publicados" in e.get("detalle", "") else "no reportado"
        nota_codigo = ""
        if doi in CORRECCIONES:
            codigo, enlace, nota_codigo = CORRECCIONES[doi]
            enlaces = [enlace] if enlace else []
        filas.append({"doi": doi, "anio": e["anio"], "texto_revisado": "sí" if texto else "no", "codigo_publico": codigo, "enlace_codigo": "; ".join(enlaces[:3]), "nota_codigo": nota_codigo,
                      "datos_publicos": "sí" if DATOS_PUBLICOS.search(e["dataset"]) else "no identificado", "baselines_ajustados": ajustados})
        print(doi, filas[-1]["texto_revisado"], codigo, filas[-1]["datos_publicos"], flush=True)
    with (CARPETA / "reproducibilidad_estudios.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    main()
