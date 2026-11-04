"""Ensambla las fuentes del Jupyter Book del Entregable 1 a partir del cuaderno ya ejecutado en Colab.

Cada sección ``## N.`` del cuaderno es un capítulo del libro. Además, el script:

- sube un nivel los encabezados de cada capítulo y les agrega una etiqueta MyST (``sec-2-2-5`` para la sección 2.2.5);
- convierte en referencias cruzadas las menciones a secciones de este cuaderno («sección 2.2.5»);
- oculta el código de cada celda detrás de un botón (etiqueta ``hide-input``), con sus salidas visibles;
- quita las barras de avance y las advertencias (salidas por stderr), que no aportan al informe.

El libro no vuelve a ejecutar el cuaderno (``execute_notebooks: off``): usa las salidas de la corrida en Colab.

Uso:
    python construir_libro.py --libro <carpeta de destino> [--cuaderno <cuaderno ejecutado>]
    jupyter-book build <carpeta de destino>
"""

import argparse
import copy
import re
import shutil
from pathlib import Path

import nbformat as nbf

CUADERNO = Path(__file__).resolve().parents[2] / "Entregable 1 - UAVIDS-2025.ipynb"
CAPITULOS = {0: "00_portada", 1: "01_problema_y_dataset", 2: "02_analisis_exploratorio", 3: "03_estado_del_arte", 4: "04_modelos_base",
             5: "05_benchmark", 6: "06_guia_de_procedimiento", 7: "07_limitaciones", 8: "08_conclusiones", 9: "09_bibliografia"}

CONFIG = '''title: "Entregable 1 · Detección de intrusiones en redes de drones con UAVIDS-2025"
author: Donny Torres · Maestría en Analítica de Datos, Universidad del Norte
copyright: "2026"
only_build_toc_files: true

execute:
  execute_notebooks: "off"

parse:
  myst_enable_extensions:
    - dollarmath
    - amsmath
    - colon_fence
    - html_image

sphinx:
  config:
    language: es
    nb_merge_streams: true
    nb_render_markdown_format: myst
    nb_code_prompt_show: "Mostrar el código de la celda"
    nb_code_prompt_hide: "Ocultar el código de la celda"
    suppress_warnings: ["myst.header"]
    html_theme_options:
      repository_url: https://github.com/DTL-DA/proyecto-final-dl-uavids-2025
      repository_branch: main
      use_repository_button: true
      show_toc_level: 2
      use_download_button: true
      use_fullscreen_button: true
      home_page_in_toc: true
'''


def subir_encabezados(texto):
    """Sube un nivel cada encabezado (## -> #), sin tocar los bloques de código."""
    salida, en_codigo = [], False
    for linea in texto.split("\n"):
        if linea.startswith("```"):
            en_codigo = not en_codigo
        if not en_codigo and re.match(r"^#{2,6} ", linea):
            linea = linea[1:]
        salida.append(linea)
    return "\n".join(salida)


ENCABEZADO = re.compile(r"^(#{1,5}) (\d+(?:\.\d+){0,2})\.? ", re.M)


def etiquetar(texto, etiquetas):
    """Antepone una etiqueta MyST a cada encabezado numerado y la registra."""
    def poner(m):
        etiqueta = "sec-" + m.group(2).replace(".", "-")
        etiquetas.add(etiqueta)
        return f"({etiqueta})=\n{m.group(0)}"
    return ENCABEZADO.sub(poner, texto)


REF = re.compile(r"([Ss]ecci(?:ón|ones)\s+)(\d+(?:\.\d+){1,2}(?:(?:,\s*|\s+(?:y|a|e|o)\s+)\d+(?:\.\d+){1,2})*)")
AJENA = re.compile(r"[Ee]nunciado|[Ee]ntregable")


def enlazar_secciones(texto, etiquetas):
    """Convierte «sección 2.2.5» en una referencia cruzada si la sección es de este libro."""
    def cambiar(m):
        antes = texto[:m.start()]
        corte = max(max(antes.rfind(c) for c in "|;(\n"), antes.rfind(". "))
        if AJENA.search(antes[corte + 1:]) or re.match(r"\s*(del|de la|de su|de ese|de dicho)\b", texto[m.end():]):
            return m.group(0)

        def ref(n):
            etiqueta = "sec-" + n.group(0).replace(".", "-")
            return f"{{ref}}`{n.group(0)} <{etiqueta}>`" if etiqueta in etiquetas else n.group(0)
        return m.group(1) + re.sub(r"\d+(?:\.\d+){1,2}", ref, m.group(2))
    return REF.sub(cambiar, texto)


def fuera_de_codigo(funcion):
    """Aplica ``funcion`` solo al texto fuera de los bloques de código (```...```), que se dejan intactos."""
    def aplicar(texto):
        partes = re.split(r"(```.*?```)", texto, flags=re.S)
        return "".join(p if p.startswith("```") else funcion(p) for p in partes)
    return aplicar


def inicio_seccion(c):
    return re.search(r"^## (\d+)\. ", c.source, re.M) if c.cell_type == "markdown" else None


def transformar(c, funcion):
    if c.cell_type == "markdown":
        c.source = funcion(c.source)
    elif c.cell_type == "code":
        for o in c.get("outputs", []):
            if "data" in o and "text/markdown" in o["data"]:
                o["data"]["text/markdown"] = funcion(o["data"]["text/markdown"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libro", required=True, help="Carpeta de destino de las fuentes del libro")
    ap.add_argument("--cuaderno", default=CUADERNO, help="Cuaderno ya ejecutado")
    argumentos = ap.parse_args()
    libro = Path(argumentos.libro).resolve()
    if libro.exists():
        shutil.rmtree(libro)
    libro.mkdir(parents=True)
    cuaderno = nbf.read(argumentos.cuaderno, as_version=4)

    inicios = [i for i, c in enumerate(cuaderno.cells) if inicio_seccion(c)]
    limites = [0] + inicios + [len(cuaderno.cells)]
    capitulos = []
    for k in range(len(limites) - 1):
        celdas = [copy.deepcopy(c) for c in cuaderno.cells[limites[k]:limites[k + 1]]]
        capitulos.append((0 if k == 0 else int(inicio_seccion(celdas[0]).group(1)), celdas))

    etiquetas = set()
    for numero, celdas in capitulos:
        for c in celdas:
            if numero:
                transformar(c, subir_encabezados)
            transformar(c, lambda t: etiquetar(t, etiquetas))
            if c.cell_type == "code":
                c.metadata = {"tags": ["hide-input"]}
                c.outputs = [o for o in c.outputs if o.get("name") != "stderr"]
            else:
                c.metadata = {}
    metadatos = {k: v for k, v in cuaderno.metadata.items() if k in ("kernelspec", "language_info")}
    con_error = [c.get("id") for _, celdas in capitulos for c in celdas if c.cell_type == "code" and any(o.output_type == "error" for o in c.outputs)]
    sin_correr = [c.get("id") for _, celdas in capitulos for c in celdas if c.cell_type == "code" and not c.execution_count]
    for numero, celdas in capitulos:
        for c in celdas:
            transformar(c, fuera_de_codigo(lambda t: enlazar_secciones(t, etiquetas)))
        capitulo = nbf.v4.new_notebook(cells=celdas, metadata=copy.deepcopy(metadatos))
        nbf.write(capitulo, libro / f"{CAPITULOS[numero]}.ipynb")
        print(f"  {CAPITULOS[numero]}.ipynb ({len(celdas)} celdas)")

    (libro / "_config.yml").write_text(CONFIG, encoding="utf-8")
    (libro / "_toc.yml").write_text("format: jb-book\nroot: 00_portada\nchapters:\n" + "".join(f"  - file: {CAPITULOS[n]}\n" for n, _ in capitulos if n), encoding="utf-8")
    # Los repositorios para jurados no llevan rutas personales
    rutas = [r.name for r in libro.glob("*.ipynb") if re.search(r"C:\\\\Users|/Users/|/home/", r.read_text(encoding="utf-8"))]
    print(f"Libro en {libro.name}: {len(capitulos)} capítulos, {len(etiquetas)} etiquetas de sección")
    print(f"Celdas con error: {con_error or 'ninguna'} | sin ejecutar: {len(sin_correr)} | con rutas personales: {rutas or 'ninguna'}")


if __name__ == "__main__":
    main()
