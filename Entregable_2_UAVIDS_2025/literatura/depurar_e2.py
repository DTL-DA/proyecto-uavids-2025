"""Une las exportaciones de las cadenas C1 a C4 del Entregable 2, retira lo que ya revisó el
Entregable 1 y aplica a lo nuevo el mismo filtro automático por título y resumen.

Salida: registros_nuevos_e2.csv. El cribado por título y resumen se decide a mano en las columnas
`cribado`, `motivo_cribado` y `texto_completo`, que se conservan al regenerar.
"""
import csv
import json
import re
from pathlib import Path

CARPETA = Path(__file__).parent
E1 = CARPETA.parents[1] / "Entregable_1_UAVIDS_2025" / "literatura"
SALIDA = CARPETA / "registros_nuevos_e2.csv"
FECHA = "2026-11-12"


def clave_titulo(titulo):
    return re.sub(r"[^a-z0-9]", "", (titulo or "").lower())


conocidos_doi, conocidos_titulo = set(), set()
for nombre in ("registros_depurados.csv", "registros_cobertura_sinonimos.csv"):
    for f in csv.DictReader((E1 / nombre).open(encoding="utf-8-sig")):
        if f.get("doi"):
            conocidos_doi.add(f["doi"].lower().strip())
        conocidos_titulo.add(clave_titulo(f.get("titulo")))

registros = {}
for ruta in sorted((CARPETA / "exportaciones").glob(f"{FECHA}_*_C*.json")):
    d = json.loads(ruta.read_text(encoding="utf-8"))
    cadena = ruta.stem.split("_")[-1]
    for r in d["registros"]:
        doi = (r.get("doi") or "").lower().strip()
        clave = clave_titulo(r.get("titulo"))
        if not clave:
            continue
        previo = registros.get(clave)
        if previo is None:
            registros[clave] = {**r, "doi": doi, "fuentes": {d["fuente"]}, "cadenas": {cadena}}
        else:
            previo["fuentes"].add(d["fuente"])
            previo["cadenas"].add(cadena)
            previo["doi"] = previo["doi"] or doi
            previo["resumen"] = previo.get("resumen") or r.get("resumen")

DRON = re.compile(r"\b(drones?|uavs?|uas|unmanned aerial|unmanned aircraft|quadcopters?|multirotors?|fanets?|flying ad[- ]?hoc|internet of drones|iod|uavids\S*)\b", re.I)
SEGURIDAD = re.compile(r"\b(intrusions?|attacks?|cyber\w*|anomal\w*|blackhole|black[- ]hole|wormhole|sybil|flooding|ddos|dos|malicious|spoof\w*|misbehav\w*|ids|nids)\b", re.I)
DL = re.compile(r"\b(deep|cnn\w*|convolutional|neural networks?|mlps?|multi-?layer perceptrons?|lstm\w*|bilstm\w*|gru\w*|rnn\w*|recurrent|autoencoders?|transformers?|resnet\w*|dnn\w*|tabnet\w*|"
                r"attention|kolmogorov|kans?|deep belief|dbn|graph neural|gnn\w*|gat\w*|temporal convolutional|tcn|mamba|state[- ]space)\b", re.I)
RECONOCER = re.compile(r"\b(detect\w*|classif\w*|identif\w*|recogni\w*)\b", re.I)
OTRA_TAREA = re.compile(
    r"\b(uav[- ]aided|uav[- ]assisted|uav[- ]enabled|uav[- ]mounted|trajectory (?:design|optimization)|path planning|resource allocation|beamforming|"
    r"channel estimation|power allocation|mobile edge computing|remote sensing|aerial images?|object (?:tracking|detection)|yolo\w*|crop|wildfire|road|"
    r"energy harvesting|face|vehicle detection)\b", re.I)


def motivo_exclusion(r):
    texto = f"{r.get('titulo') or ''} {r.get('resumen') or ''}"
    if r.get("tipo") in {"paratext", "editorial", "erratum", "retraction", "letter", "peer-review", "dataset"}:
        return "tipo de documento"
    if re.search(r"\b(survey|review|tutorial|state of the art)\b", r.get("titulo") or "", re.I):
        return "revisión, no estudio primario"
    if not r.get("resumen"):
        return "sin resumen disponible: revisar manualmente"
    if not DRON.search(texto):
        return "no trata de drones"
    if not SEGURIDAD.search(texto):
        return "no aborda intrusiones ni ataques"
    if not DL.search(texto):
        return "no usa Deep Learning"
    if not RECONOCER.search(texto):
        return "no aborda detección o clasificación"
    if OTRA_TAREA.search(r.get("titulo") or ""):
        return "el dron es plataforma o la tarea es otra"
    return ""


previas = {clave_titulo(f["titulo"]): f for f in csv.DictReader(SALIDA.open(encoding="utf-8-sig"))} if SALIDA.exists() else {}
filas = []
for clave, r in registros.items():
    if (r["doi"] and r["doi"] in conocidos_doi) or clave in conocidos_titulo:
        continue
    motivo = motivo_exclusion(r)
    previa = previas.get(clave, {})
    filas.append({"titulo": r.get("titulo"), "anio": r.get("anio"), "doi": r["doi"], "fuente_publicacion": r.get("fuente"), "tipo": r.get("tipo"),
                  "citas": r.get("citas"), "bases": "; ".join(sorted(r["fuentes"])), "cadenas": "; ".join(sorted(r["cadenas"])),
                  "acceso_abierto": r.get("oa_url") or "", "exclusion_filtro": motivo,
                  "cribado": previa.get("cribado", "" if motivo and not motivo.startswith("sin resumen") else "sin cribar"),
                  "motivo_cribado": previa.get("motivo_cribado", ""), "texto_completo": previa.get("texto_completo", "")})
filas.sort(key=lambda f: (f["exclusion_filtro"] != "", -(int(f["anio"] or 0)), f["titulo"] or ""))
with SALIDA.open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)
print(f"Únicos en C1 a C4: {len(registros)} | ya revisados en el Entregable 1: {len(registros) - len(filas)} | nuevos: {len(filas)} | "
      f"retirados por el filtro: {sum(bool(f['exclusion_filtro']) and not f['exclusion_filtro'].startswith('sin resumen') for f in filas)} | por cribar: {sum(f['cribado'] == 'sin cribar' for f in filas)}")
