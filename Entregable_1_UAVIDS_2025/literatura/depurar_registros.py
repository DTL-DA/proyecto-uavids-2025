"""Une las exportaciones, elimina duplicados y aplica el primer filtro por título y resumen.

Salida: registros_depurados.csv, con el motivo de exclusión de cada registro descartado por el filtro automático.
El filtro solo retira lo que claramente no corresponde; la decisión final de inclusión se toma leyendo cada estudio.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

CARPETA = Path(__file__).parent
EXPORTACIONES = CARPETA / "exportaciones"
SALIDA = CARPETA / "registros_depurados.csv"


def clave_titulo(titulo):
    return re.sub(r"[^a-z0-9]", "", (titulo or "").lower())


registros = {}
identificados = Counter()
for ruta in sorted(EXPORTACIONES.glob("*.json")):
    if ruta.name.endswith("_resumen.json"):
        continue
    d = json.loads(ruta.read_text(encoding="utf-8"))
    for r in d["registros"]:
        identificados[d["fuente"]] += 1
        doi = (r.get("doi") or "").lower().strip()
        clave = doi or clave_titulo(r.get("titulo"))
        if not clave:
            continue
        previo = registros.get(clave)
        if previo is None:
            registros[clave] = {**r, "doi": doi, "fuentes": {d["fuente"]}, "cadenas": {ruta.stem.split("_")[-1]}}
        else:
            previo["fuentes"].add(d["fuente"])
            previo["cadenas"].add(ruta.stem.split("_")[-1])
            previo["resumen"] = previo.get("resumen") or r.get("resumen")
            previo["oa_url"] = previo.get("oa_url") or r.get("oa_url")

# Segunda pasada: el mismo trabajo puede aparecer con DOI en una fuente y sin DOI en otra
por_titulo = {}
for r in registros.values():
    k = clave_titulo(r.get("titulo"))
    previo = por_titulo.get(k)
    if previo is None or (not previo["doi"] and r["doi"]):
        if previo is not None:
            r["fuentes"] |= previo["fuentes"]
            r["cadenas"] |= previo["cadenas"]
            r["resumen"] = r.get("resumen") or previo.get("resumen")
            r["oa_url"] = r.get("oa_url") or previo.get("oa_url")
        por_titulo[k] = r
    else:
        previo["fuentes"] |= r["fuentes"]
        previo["cadenas"] |= r["cadenas"]
        previo["resumen"] = previo.get("resumen") or r.get("resumen")
        previo["oa_url"] = previo.get("oa_url") or r.get("oa_url")
unicos = list(por_titulo.values())

DRON = re.compile(r"\b(drones?|uavs?|uas|unmanned aerial|unmanned aircraft|quadcopters?|multirotors?|fanets?|flying ad[- ]?hoc|internet of drones|iod|uavids\S*)\b", re.I)
SEGURIDAD = re.compile(r"\b(intrusions?|attacks?|cyber\w*|anomal\w*|blackhole|black[- ]hole|wormhole|sybil|flooding|ddos|dos|malicious|spoof\w*|misbehav\w*|ids|nids)\b", re.I)
DL = re.compile(r"\b(deep|cnn\w*|convolutional|neural networks?|mlps?|multi-?layer perceptrons?|lstm\w*|bilstm\w*|gru\w*|rnn\w*|recurrent|autoencoders?|transformers?|resnet\w*|dnn\w*|tabnet\w*)\b", re.I)
RECONOCER = re.compile(r"\b(detect\w*|classif\w*|identif\w*|recogni\w*)\b", re.I)
OTRA_TAREA = re.compile(
    r"\b(uav[- ]aided|uav[- ]assisted|uav[- ]enabled|uav[- ]mounted|trajectory (?:design|optimization)|path planning|resource allocation|beamforming|"
    r"channel estimation|power allocation|mobile edge computing|remote sensing|aerial images?|object (?:tracking|detection)|yolo\w*|crop|wildfire|road|"
    r"energy harvesting|face|vehicle detection)\b", re.I)


def motivo_exclusion(r):
    texto = f"{r.get('titulo') or ''} {r.get('resumen') or ''}"
    if r.get("tipo") in {"paratext", "editorial", "erratum", "retraction", "letter", "peer-review"}:
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
    titulo = r.get("titulo") or ""
    if OTRA_TAREA.search(titulo):
        return "el dron es plataforma o la tarea es otra"
    return ""


# El cribado por título y resumen se decide a mano y queda en estas columnas: al regenerar se conserva, y lo nuevo queda sin cribar
previas = {clave_titulo(f["titulo"]): f for f in csv.DictReader(SALIDA.open(encoding="utf-8-sig"))} if SALIDA.exists() else {}
# Los resúmenes son texto de terceros: el filtro los usa, pero no se copian a la salida
filas = []
for r in unicos:
    motivo = motivo_exclusion(r)
    previa = previas.get(clave_titulo(r.get("titulo")), {})
    pendiente = "" if motivo and not motivo.startswith("sin resumen") else "sin cribar"
    filas.append({"titulo": r.get("titulo"), "anio": r.get("anio"), "doi": r.get("doi"), "fuente_publicacion": r.get("fuente"),
                  "tipo": r.get("tipo"), "citas": r.get("citas"), "bases": "; ".join(sorted(r["fuentes"])),
                  "cadenas": "; ".join(sorted(r["cadenas"])), "acceso_abierto": r.get("oa_url") or "",
                  "exclusion_filtro": motivo, "cribado": previa.get("cribado") or pendiente, "motivo_cribado": previa.get("motivo_cribado", ""),
                  "texto_completo": previa.get("texto_completo", "")})
filas.sort(key=lambda f: (f["exclusion_filtro"] != "", -(f["citas"] or 0)))
with SALIDA.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)

print("identificados por base:", dict(identificados), "| total", sum(identificados.values()))
print("únicos tras depurar duplicados:", len(unicos))
print("motivos del filtro:", Counter(f["exclusion_filtro"] or "pasa a lectura" for f in filas).most_common())
print("pasan con acceso abierto:", sum(1 for f in filas if not f["exclusion_filtro"] and f["acceso_abierto"]))
print("años de los que pasan:", sorted(Counter(f["anio"] for f in filas if not f["exclusion_filtro"]).items()))
