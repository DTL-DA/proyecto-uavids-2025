"""Arma el cuaderno «Entregable 3 - UAVIDS-2025.ipynb» con los capítulos del libro y lo ejecuta de principio a fin.

    python Entregable_3_UAVIDS_2025/book/crear_cuaderno.py [--sin-ejecutar]

El libro es la fuente única: el cuaderno toma, en el orden de _toc.yml, el texto y las celdas de código de cada
capítulo (MyST), convierte las citas {cite:p} y {cite:t} a autor y año con references.bib, y queda en la raíz del
repositorio con todas las salidas, como los cuadernos de los Entregables 1 y 2. Las celdas se ejecutan desde la
carpeta del libro, igual que al compilarlo, así que leen los mismos resultados versionados.
"""
import argparse
import ast
import codecs
import re
import sys
import time
from pathlib import Path

import latexcodec  # noqa: F401  (registra el códec "ulatex" para los acentos de LaTeX del .bib)
import nbformat
from nbclient import NotebookClient
from pybtex.database import parse_file

LIBRO = Path(__file__).resolve().parent
RAIZ_REPO = LIBRO.parents[1]
DESTINO = RAIZ_REPO / "Entregable 3 - UAVIDS-2025.ipynb"
BIB = parse_file(str(LIBRO / "references.bib"))

INICIO_CELDA = re.compile(r"^```\{code-cell\}")
CITA = re.compile(r"\{cite:([pt])\}`([^`]+)`")
ETIQUETA_ECUACION = re.compile(r"^\$\$\s*\(.+\)\s*$")

PORTADA = """# Entregable 3 · Modelo original y validación rigurosa para detectar intrusiones en redes de drones

**Compuerta por flujo entre un experto de escalones suaves y un experto neuronal, protocolo preregistrado, evaluación ampliada en tres conjuntos adicionales y despliegue en un dashboard**

| | |
|---|---|
| **Autor** | Donny Torres |
| **Programa** | Maestría en Analítica de Datos, Universidad del Norte (Barranquilla) |
| **Asignatura** | Deep Learning |
| **Fecha** | Noviembre de 2026 |
| **Conjunto de datos** | *UAVIDS-2025*, versión 1; Zenodo, licencia CC BY 4.0, DOI <https://doi.org/10.5281/zenodo.15336998> |
| **Preregistro** | `Entregable_3_UAVIDS_2025/experiments/protocolo.md`, etiqueta `e3-protocolo` del repositorio, congelado el 22/11/2026 |
| **Jupyter Book** | <https://dtl-da.github.io/proyecto-uavids-2025/entregable-3/> |
| **Dashboard** | <https://uavids-2025-dashboard.onrender.com> (plan gratuito de Render: la primera visita tras 15 minutos sin uso tarda cerca de un minuto) |
| **Repositorio** | <https://github.com/DTL-DA/proyecto-uavids-2025>, carpeta `Entregable_3_UAVIDS_2025/`, etiqueta `entregable-3` |
| **Archivo en Zenodo** | DOI <https://doi.org/10.5281/zenodo.24486475> (versión 1.0.0) |

Este cuaderno reúne, en el orden del libro, el texto y todo el código de la introducción y de los once capítulos del Jupyter Book, y se entrega ejecutado con todas las salidas: sus celdas de código corrieron en orden, sin errores, como lo muestran sus números de ejecución consecutivos, y las de preparación de cada capítulo terminan con una línea que imprime lo que dejaron definido. Se genera con `Entregable_3_UAVIDS_2025/book/crear_cuaderno.py` a partir de las mismas fuentes del libro, así que su contenido coincide con la versión publicada.

Las corridas costosas, que son las semillas de cada modelo, los conjuntos adicionales, la robustez y la exportación a ONNX, se precalculan con los scripts de `training/` y `experiments/`. Aquí, igual que al compilar el libro, se leen sus resultados versionados y se regeneran todas las tablas, pruebas y figuras. Para ejecutarlo de nuevo desde la raíz del repositorio basta con `python Entregable_3_UAVIDS_2025/book/crear_cuaderno.py`, o con *Run All* en Jupyter, en el entorno de `Entregable_3_UAVIDS_2025/requirements-train.txt`. Las citas aparecen como autor y año, y la lista de referencias está al final, en el capítulo 11."""

PREPARACION = """# Los capítulos leen los resultados con rutas relativas a la carpeta del libro, como al compilarlo con Jupyter Book
import os
from datetime import datetime
from pathlib import Path

if Path("Entregable_3_UAVIDS_2025", "book").is_dir():
    os.chdir(Path("Entregable_3_UAVIDS_2025", "book"))
assert Path("_toc.yml").exists(), "Abre el cuaderno desde la raíz del repositorio"
print("Cuaderno ejecutado el", datetime.now().strftime("%d/%m/%Y a las %H:%M"))"""


def limpiar(texto: str) -> str:
    return codecs.decode(texto, "ulatex").replace("{", "").replace("}", "").strip()


def apellido(persona) -> str:
    return limpiar(" ".join(persona.prelast_names + persona.last_names))


def autores_cortos(clave: str) -> str:
    entrada = BIB.entries[clave]
    personas = entrada.persons.get("author", [])
    if not personas:
        return limpiar(entrada.fields.get("organization", entrada.fields.get("title", clave)))
    nombres = [apellido(p) for p in personas]
    if len(nombres) == 1:
        return nombres[0]
    if len(nombres) == 2:
        return f"{nombres[0]} y {nombres[1]}"
    return f"{nombres[0]} et al."


def anio(clave: str) -> str:
    return BIB.entries[clave].fields.get("year", "s. f.")


def referencia(clave: str) -> str:
    entrada = BIB.entries[clave]
    campos = entrada.fields
    personas = entrada.persons.get("author", [])
    nombres = []
    for p in personas:
        iniciales = " ".join(limpiar(n)[0] + "." for n in p.first_names + p.middle_names if limpiar(n))
        nombres.append(f"{apellido(p)}, {iniciales}" if iniciales else apellido(p))
    autores = ", ".join(nombres[:-1]) + " y " + nombres[-1] if len(nombres) > 1 else (nombres[0] if nombres else autores_cortos(clave))
    partes = [f"{autores} ({anio(clave)}). {limpiar(campos.get('title', ''))}."]
    medio = campos.get("journal") or campos.get("booktitle") or campos.get("publisher") or campos.get("howpublished")
    if medio:
        partes.append(f"*{limpiar(medio)}*.")
    if "doi" in campos:
        partes.append(f"<https://doi.org/{campos['doi']}>")
    elif "url" in campos:
        partes.append(f"<{campos['url']}>")
    return " ".join(partes)


def convertir_citas(texto: str, citadas: set) -> str:
    def reemplazo(m):
        claves = [c.strip() for c in m.group(2).split(",")]
        citadas.update(claves)
        if m.group(1) == "p":
            return "(" + "; ".join(f"{autores_cortos(c)}, {anio(c)}" for c in claves) + ")"
        return " y ".join(f"{autores_cortos(c)} ({anio(c)})" for c in claves)
    return CITA.sub(reemplazo, texto)


def adaptar_texto(lineas: list[str], titulo: str, citadas: set) -> str:
    """Baja un nivel los títulos (el cuaderno tiene su propia portada), numera el capítulo y quita la sintaxis propia de MyST."""
    salida, en_bloque, primer_titulo, i = [], False, True, 0
    while i < len(lineas):
        linea = lineas[i]
        if linea.startswith("```{bibliography}"):
            while lineas[i] != "```":
                i += 1
            salida.extend(f"- {referencia(c)}" for c in sorted(citadas, key=lambda c: (autores_cortos(c).lower(), anio(c))))
        elif linea.startswith("```"):
            en_bloque = not en_bloque
            salida.append(linea)
        elif not en_bloque and re.match(r"^#{1,5} ", linea):
            if primer_titulo and linea.startswith("# "):
                linea, primer_titulo = f"# {titulo}", False
            salida.append("#" + linea)
        elif not en_bloque and ETIQUETA_ECUACION.match(linea):
            salida.append("$$")
        else:
            salida.append(convertir_citas(linea, citadas) if not en_bloque else linea)
        i += 1
    return "\n".join(salida).strip("\n")


def leer_capitulo(ruta: Path, titulo: str, citadas: set) -> list[tuple[str, str]]:
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    if lineas and lineas[0] == "---":
        lineas = lineas[lineas.index("---", 1) + 1:]
    celdas, texto, i = [], [], 0

    def cerrar_texto():
        bloque = adaptar_texto(texto, titulo, citadas)
        if bloque:
            celdas.append(("markdown", bloque))
        texto.clear()

    while i < len(lineas):
        if INICIO_CELDA.match(lineas[i]):
            cerrar_texto()
            i += 1
            while lineas[i].startswith(":"):  # opciones de la celda, como :tags: [hide-input]
                i += 1
            codigo = []
            while lineas[i] != "```":
                codigo.append(lineas[i])
                i += 1
            celdas.append(("code", "\n".join(codigo).strip("\n")))
        else:
            texto.append(lineas[i])
        i += 1
    cerrar_texto()
    return celdas


def confirmar_preparacion(codigo: str, capitulo: int) -> str:
    """Agrega a una celda que no mostró salida (las de preparación: importan y definen) una línea que imprime lo que
    dejó definido, para que en el cuaderno toda celda ejecutada muestre una salida."""
    arbol = ast.parse(codigo)
    nombres = []
    for nodo in arbol.body:
        if isinstance(nodo, (ast.Import, ast.ImportFrom)):
            nombres += [a.asname or a.name.split(".")[0] for a in nodo.names]
        elif isinstance(nodo, ast.Assign):
            nombres += [t.id for t in nodo.targets if isinstance(t, ast.Name)]
        elif isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
            nombres.append(nodo.name)
    lista = ", ".join(dict.fromkeys(nombres))
    return codigo + f'\nprint("Preparación del capítulo {capitulo} ejecutada. Quedan definidos: {lista}")'


def capitulos() -> list[str]:
    toc = (LIBRO / "_toc.yml").read_text(encoding="utf-8")
    return [re.search(r"^root:\s*(\S+)", toc, re.M).group(1)] + re.findall(r"^\s*- file:\s*(\S+)", toc, re.M)


def armar() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbformat.v4.new_markdown_cell(PORTADA, id="portada"),
                nbformat.v4.new_code_cell(PREPARACION, id="preparacion")]
    citadas: set = set()
    for n, nombre in enumerate(capitulos()):
        ruta = LIBRO / f"{nombre}.md"
        titulo_libro = next(l[2:] for l in ruta.read_text(encoding="utf-8").splitlines() if l.startswith("# "))
        titulo = titulo_libro if n == 0 else f"Capítulo {n}. {titulo_libro}"
        for k, (tipo, fuente) in enumerate(leer_capitulo(ruta, titulo, citadas)):
            ident = f"c{n:02d}-{k:02d}"
            celda = nbformat.v4.new_code_cell(fuente, id=ident) if tipo == "code" else nbformat.v4.new_markdown_cell(fuente, id=ident)
            nb.cells.append(celda)
    return nb


def ejecutar(nb: nbformat.NotebookNode) -> None:
    cliente = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(RAIZ_REPO)}})
    inicio = time.time()
    with cliente.setup_kernel():
        for i, celda in enumerate(nb.cells):
            if celda.cell_type != "code":
                continue
            t = time.time()
            cliente.execute_cell(celda, i)
            print(f"{celda.id:>12} {time.time() - t:6.1f} s", flush=True)
    print(f"Ejecutado en {(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sin-ejecutar", action="store_true", help="solo arma el cuaderno, sin salidas")
    args = parser.parse_args()
    cuaderno = armar()
    if not args.sin_ejecutar:
        ejecutar(cuaderno)
        vacias = [c for c in cuaderno.cells if c.cell_type == "code" and not c.outputs]
        if vacias:
            # Segunda ejecución completa, en un núcleo nuevo, para que los números de ejecución sigan consecutivos
            for c in vacias:
                c.source = confirmar_preparacion(c.source, int(c.id[1:3]))
            print(f"{len(vacias)} celdas de preparación sin salida: se les agrega la confirmación y se ejecuta de nuevo")
            ejecutar(cuaderno)
    nbformat.validate(cuaderno)
    nbformat.write(cuaderno, DESTINO)
    codigo = [c for c in cuaderno.cells if c.cell_type == "code"]
    print(f"{DESTINO.name}: {len(cuaderno.cells)} celdas, {len(codigo)} de código, "
          f"{sum(1 for c in codigo if c.get('outputs'))} con salidas, {DESTINO.stat().st_size / 1e6:.1f} MB")
