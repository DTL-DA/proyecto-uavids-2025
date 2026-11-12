"""Integra el cribado de las bases nuevas y los textos leídos con el acceso de la biblioteca.

Agrega a cribado_titulo_resumen.csv las decisiones de cribado_bases_nuevas.csv, porque cribado.py identifica las
decisiones anteriores por su posición en el archivo y no debe volver a ejecutarse. También marca en
recuperacion_texto.csv los estudios de estudios_biblioteca.json y da de alta los elegibles nuevos.

Uso: python integrar_bases_nuevas.py <recuperacion de la segunda búsqueda> <carpeta con los textos convertidos>
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
CARPETA = Path(__file__).parent
BUSQUEDA, TEXTOS = Path(sys.argv[1]), Path(sys.argv[2])
VIA = "PDF descargado por el autor con el acceso de la biblioteca de la Universidad del Norte"
# Se leyó en la página de IEEE Xplore con el mismo acceso, y su texto no se conservó
LEIDO_EN_LINEA = {"10.1109/jiot.2024.3360231": 53846}


def leer(nombre):
    with (CARPETA / nombre).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clave(titulo):
    return re.sub(r"[^a-z0-9]", "", (titulo or "").lower())[:70]


def escribir(nombre, filas):
    with (CARPETA / nombre).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


registros = {clave(r["titulo"]): r for r in leer("registros_depurados.csv")}
cribado = leer("cribado_titulo_resumen.csv")
ya = {clave(f["titulo"]) for f in cribado}
nuevas = leer("cribado_bases_nuevas.csv")
for f in nuevas:
    if clave(f["titulo"]) not in ya:
        r = registros[clave(f["titulo"])]
        cribado.append({"titulo": f["titulo"], "anio": f["anio"], "doi": f["doi"], "fuente_publicacion": f["fuente_publicacion"], "citas": r["citas"], "bases": r["bases"], "cadenas": r["cadenas"],
                        "acceso_abierto": r["acceso_abierto"], "decision": f["decision"], "motivo": f["motivo"]})
escribir("cribado_titulo_resumen.csv", cribado)

with BUSQUEDA.open(encoding="utf-8-sig") as f:
    recuperacion = {clave(r["titulo"]): r for r in csv.DictReader(f)}
for f in nuevas:
    if f["decision"] == "elegible" and clave(f["titulo"]) not in recuperacion:
        recuperacion[clave(f["titulo"])] = {"clave": f["doi"] or clave(f["titulo"]), "carpeta": "", "titulo": f["titulo"], "anio": f["anio"], "doi": f["doi"], "caracteres": 0, "via": "", "errores": ""}
por_doi = {r["doi"].lower(): r for r in recuperacion.values() if r["doi"]}
biblioteca = json.loads((CARPETA / "estudios_biblioteca.json").read_text(encoding="utf-8"))
leidos = 0
for e in biblioteca["incluidos"] + biblioteca["excluidos"]:
    doi = e["doi"]
    ruta = TEXTOS / re.sub(r"[^a-z0-9]+", "_", doi) / "doc.txt"
    if doi in LEIDO_EN_LINEA:
        por_doi[doi].update(caracteres=LEIDO_EN_LINEA[doi], via="Página de IEEE Xplore, con el acceso de la biblioteca de la Universidad del Norte", errores="")
        leidos += 1
    elif ruta.exists():
        por_doi[doi].update(caracteres=len(ruta.read_text(encoding="utf-8", errors="replace")), via=VIA, errores="")
        leidos += 1
    else:
        assert int(por_doi[doi]["caracteres"]) > 3000, doi
escribir("recuperacion_texto.csv", list(recuperacion.values()))
print("cribado:", len(cribado), "| recuperación:", len(recuperacion), "| con texto:", sum(int(r["caracteres"] or 0) > 3000 for r in recuperacion.values()), "| leídos con la biblioteca:", leidos)
