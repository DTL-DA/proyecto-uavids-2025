"""Convierte las exportaciones de Scopus, IEEE Xplore y Web of Science al formato de buscar_literatura.py.

Lee de exportaciones/ los archivos AAAA-MM-DD_Base_Cadena (.csv de Scopus e IEEE, .xls de Web of Science) y escribe un
JSON por archivo, de modo que depurar_registros.py los una con OpenAlex y arXiv con el mismo procedimiento. Las
exportaciones son texto de terceros y no se versionan. Los archivos .xls de Web of Science requieren xlrd.

Uso: python importar_exportaciones.py [carpeta de exportaciones]
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
csv.field_size_limit(10_000_000)
CARPETA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "exportaciones"
COLUMNAS = {
    "Scopus": {"titulo": "Title", "anio": "Year", "doi": "DOI", "fuente": "Source title", "tipo": "Document Type", "citas": "Cited by", "resumen": "Abstract"},
    "IEEE": {"titulo": "Document Title", "anio": "Publication Year", "doi": "DOI", "fuente": "Publication Title", "tipo": "Document Identifier", "citas": "Article Citation Count", "resumen": "Abstract"},
    "WoS": {"titulo": "Article Title", "anio": "Publication Year", "doi": "DOI", "fuente": "Source Title", "tipo": "Document Type", "citas": "Times Cited, All Databases", "resumen": "Abstract"},
}
# Tipos que depurar_registros.py descarta, con el nombre que les da cada base
TIPOS = {"editorial": "editorial", "editorial material": "editorial", "erratum": "erratum", "correction": "erratum", "retracted": "retraction", "retraction": "retraction", "letter": "letter", "note": "paratext"}


def leer(ruta):
    if ruta.suffix.lower() == ".csv":
        return list(csv.DictReader(ruta.read_bytes().decode("utf-8-sig", "replace").splitlines()))
    import xlrd
    hoja = xlrd.open_workbook(ruta).sheet_by_index(0)
    encabezado = hoja.row_values(0)
    return [dict(zip(encabezado, hoja.row_values(i))) for i in range(1, hoja.nrows)]


def entero(valor):
    try:
        return int(float(valor))
    except (TypeError, ValueError):
        return None


def texto(valor):
    return str(valor).strip() if valor not in (None, "") else ""


conteo = {}
for ruta in sorted(p for p in CARPETA.iterdir() if re.fullmatch(r"\d{4}-\d{2}-\d{2}_(Scopus|IEEE|WoS)_C\d\.(csv|xls)", p.name)):
    fecha, base, cadena = ruta.stem.split("_")
    c = COLUMNAS[base]
    filas = leer(ruta)
    faltantes = [v for v in c.values() if filas and v not in filas[0]]
    assert not faltantes, (ruta.name, faltantes)
    registros = []
    for f in filas:
        tipo = texto(f[c["tipo"]]).lower()
        registros.append({"titulo": texto(f[c["titulo"]]), "anio": entero(f[c["anio"]]), "doi": texto(f[c["doi"]]).lower(), "fuente": texto(f[c["fuente"]]), "tipo": TIPOS.get(tipo.split(";")[0].strip(), tipo),
                          "citas": entero(f[c["citas"]]), "oa_url": None, "resumen": texto(f[c["resumen"]])})
    (CARPETA / f"{fecha}_{base}_{cadena}.json").write_text(json.dumps({"fuente": base, "fecha": fecha, "total": len(registros), "registros": registros}, ensure_ascii=False, indent=1), encoding="utf-8")
    conteo[(base, cadena)] = len(registros)
    print(base, cadena, len(registros), "| con resumen:", sum(bool(r["resumen"]) for r in registros), "| con DOI:", sum(bool(r["doi"]) for r in registros))
print("total importado:", sum(conteo.values()))
