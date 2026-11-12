"""Reúne en entregables/ los productos que pide el enunciado del Entregable 1 y arma los ZIP para enviarlos.

No mueve nada: el cuaderno y el Entregable 2 siguen leyendo de resultados/ y literatura/. Todo lo que hay en
entregables/ se genera aquí, de modo que, si cambia el cuaderno, basta con volver a ejecutar este script.

Salida en la raíz del repositorio, sin versionar:
- Entregable_1_UAVIDS-2025.zip: repositorio y entregables, para adjuntar al correo. No lleva archivos .js,
  que los servicios de correo pueden bloquear incluso dentro de un ZIP.
- Jupyter_Book_sitio_web.zip: el libro compilado en HTML, para publicarlo o compartirlo por enlace.

Uso:
    python organizar_entregables.py
"""

import base64
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path

HERRAMIENTAS = Path(__file__).resolve().parent
PROYECTO = HERRAMIENTAS.parent
RAIZ = PROYECTO.parent
RESULTADOS = PROYECTO / "resultados"
LITERATURA = PROYECTO / "literatura"
ENTREGABLES = PROYECTO / "entregables"
CUADERNO = RAIZ / "Entregable 1 - UAVIDS-2025.ipynb"
ZIP_CORREO = RAIZ / "Entregable_1_UAVIDS-2025.zip"
ZIP_SITIO = RAIZ / "Jupyter_Book_sitio_web.zip"
LIMITE_CORREO_MB = 25
ENTORNO = {**os.environ, "PYTHONUTF8": "1"}

celdas = json.loads(CUADERNO.read_text(encoding="utf-8"))["cells"]


class TablaHTML(HTMLParser):
    """Lee las filas de una tabla HTML, ignorando los estilos que agrega pandas."""

    def __init__(self):
        super().__init__()
        self.filas, self.celda = [], None

    def handle_starttag(self, etiqueta, atributos):
        if etiqueta == "tr":
            self.filas.append([])
        elif etiqueta in ("td", "th"):
            self.celda = ""

    def handle_endtag(self, etiqueta):
        if etiqueta in ("td", "th") and self.celda is not None:
            self.filas[-1].append(self.celda.strip())
            self.celda = None

    def handle_data(self, datos):
        if self.celda is not None:
            self.celda += datos


def tabla_html(columna):
    """Devuelve la primera tabla de las salidas del cuaderno cuyo encabezado tenga esa columna."""
    for celda in celdas:
        for salida in celda.get("outputs", []):
            html = "".join(salida.get("data", {}).get("text/html", []))
            if "<table" in html and columna in html:
                lector = TablaHTML()
                lector.feed(html)
                if columna in lector.filas[0]:
                    return lector.filas
    raise LookupError(f"No se encontró la tabla con la columna {columna!r}")


def tabla_markdown(inicio):
    """Devuelve la tabla de una celda de texto cuya primera línea empieza con ``inicio``."""
    for celda in celdas:
        if celda["cell_type"] != "markdown":
            continue
        lineas = "".join(celda["source"]).split("\n")
        for i, linea in enumerate(lineas):
            if linea.startswith(inicio):
                filas = []
                for fila in lineas[i:]:
                    if not fila.startswith("|"):
                        break
                    if not re.fullmatch(r"\|[-| :]+\|", fila.strip()):
                        filas.append([x.strip() for x in fila.strip().strip("|").split("|")])
                return filas
    raise LookupError(f"No se encontró la tabla que empieza con {inicio!r}")


def figura(marca):
    """Devuelve los bytes de la primera imagen de la celda de código que contiene ``marca``."""
    for celda in celdas:
        if celda["cell_type"] == "code" and marca in "".join(celda["source"]):
            for salida in celda["outputs"]:
                if "image/png" in salida.get("data", {}):
                    return base64.b64decode(salida["data"]["image/png"])
    raise LookupError(f"No se encontró la figura de {marca!r}")


def escribir_csv(ruta, filas):
    with open(ruta, "w", encoding="utf-8-sig", newline="") as archivo:
        csv.writer(archivo).writerows(filas)


def carpeta(nombre):
    ruta = ENTREGABLES / nombre
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def corridas():
    """Una fila por corrida: fase, modelo, semilla, hiperparámetros, costo y equipo."""
    filas = [["fase", "modelo", "semilla", "lr", "weight_decay", "batch", "epocas", "segundos", "parametros",
              "f1_entrenamiento", "f1_validacion", "ms_por_muestra", "dispositivo", "archivo"]]
    for ruta in sorted((RESULTADOS / "corridas").glob("*.json")):
        d = json.loads(ruta.read_text(encoding="utf-8"))
        filas.append([d["fase"], d["modelo"], d["semilla"], d["hp"]["lr"], d["hp"]["weight_decay"], d["hp"]["batch"],
                      d["epocas"], d["segundos"], d["parametros"], d.get("f1_ent", ""), d.get("f1_val", ""),
                      d.get("ms_por_muestra", ""), d["dispositivo"], ruta.name])
    return filas


def compilar_libro(libro, constructor):
    codigo = "from jupyter_book.cli.main import main; main()"
    subprocess.run([sys.executable, "-c", codigo, "build", ".", "--builder", constructor], cwd=libro, env=ENTORNO,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def imprimir_pdf(html, pdf):
    """Imprime el libro en una sola página HTML a PDF, con Edge o Chrome sin interfaz."""
    candidatos = [shutil.which("msedge"), shutil.which("chrome"),
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    navegador = next((c for c in candidatos if c and Path(c).exists()), None)
    if navegador is None:
        print("  Sin Edge ni Chrome: no se generó el PDF")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as perfil:
        subprocess.run([navegador, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={perfil}", "--virtual-time-budget=60000", f"--print-to-pdf={pdf}",
                        html.as_uri()], check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # El navegador puede devolver el control antes de terminar de escribir: se espera a que el tamaño se estabilice
        tamano, estable, limite = -1, 0, time.monotonic() + 180
        while estable < 3:
            actual = pdf.stat().st_size if pdf.exists() else 0
            estable = estable + 1 if actual and actual == tamano else 0
            tamano = actual
            if time.monotonic() > limite:
                raise TimeoutError(f"No se generó {pdf.name}")
            time.sleep(2)


def comprimir(destino, archivos, base, prefijo):
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for archivo in archivos:
            z.write(archivo, Path(prefijo) / archivo.relative_to(base))
    return destino.stat().st_size / 1024 ** 2


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if ENTREGABLES.exists():
        shutil.rmtree(ENTREGABLES)

    # 3. Dataset
    shutil.copy2(HERRAMIENTAS / "obtener_datos.py", carpeta("3_dataset"))

    # 4. Literatura
    literatura = carpeta("4_literatura")
    escribir_csv(literatura / "tabla_sintesis.csv", tabla_markdown("| Referencia | Problema"))
    escribir_csv(literatura / "modelos_base_extraidos.csv", tabla_markdown("| Modelo | Arquitectura tomada de"))
    shutil.copy2(LITERATURA / "frecuencia_modelos.csv", literatura)
    shutil.copy2(LITERATURA / "arquitecturas_referencia.csv", literatura)
    shutil.copy2(LITERATURA / "cuartiles_fuentes.csv", literatura)
    shutil.copy2(LITERATURA / "cribado_cobertura_sinonimos.csv", literatura)
    shutil.copy2(LITERATURA / "estudios_cobertura.json", literatura)
    shutil.copy2(LITERATURA / "consultas_wos_scopus_ieee.md", literatura)
    (literatura / "diagrama_prisma.png").write_bytes(figura("CONTEOS_PRISMA"))

    # 5. Comparación, baseline e inferencia
    benchmark = carpeta("5_benchmark")
    for nombre in ("tabla_benchmark.csv", "baseline.json", "comparaciones_pareadas.csv",
                   "comparaciones_validacion_repetida.csv", "metricas_definitiva.csv", "sensibilidad_dropout.csv"):
        shutil.copy2(RESULTADOS / nombre, benchmark)
    for nombre, columna in (("anova_y_friedman", "Estadístico"), ("nemenyi", "p Nemenyi"),
                            ("mcnemar", "p de McNemar (mediana)"), ("delong", "p DeLong (mediana)"),
                            ("bootstrap_diferencias", "IC 95 % por bloques"),
                            ("descomposicion_varianza", "Suma de cuadrados")):
        escribir_csv(benchmark / f"{nombre}.csv", tabla_html(columna))

    # 6. Dependencias, semillas y configuración
    reproducibilidad = carpeta("6_reproducibilidad")
    shutil.copy2(RAIZ / "requirements.txt", reproducibilidad)
    shutil.copy2(RESULTADOS / "preregistro.json", reproducibilidad)
    shutil.copy2(RESULTADOS / "mejores_hiperparametros.json", reproducibilidad)
    escribir_csv(reproducibilidad / "semillas_y_configuracion.csv", corridas())
    escribir_csv(reproducibilidad / "configuracion_comun.csv", tabla_html("Definición"))
    escribir_csv(reproducibilidad / "entorno.csv", tabla_html("Componente"))

    # 7. Declaración a priori
    declaracion = carpeta("7_declaracion_a_priori")
    shutil.copy2(RESULTADOS / "declaracion_a_priori.json", declaracion)
    shutil.copy2(RESULTADOS / "prueba.sha256", declaracion)

    # 1. Informe en Jupyter Book
    libro = ENTREGABLES / "1_informe_jupyter_book"
    subprocess.run([sys.executable, str(HERRAMIENTAS / "construir_libro.py"), "--libro", str(libro)], check=True, env=ENTORNO)
    # Se compila en una carpeta temporal, porque en Windows la ruta del proyecto más los archivos del libro
    # pueden superar el límite de 260 caracteres
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporal:
        copia = Path(temporal) / "libro"
        shutil.copytree(libro, copia)
        compilar_libro(copia, "html")
        compilar_libro(copia, "singlehtml")
        imprimir_pdf(copia / "_build" / "singlehtml" / "index.html", libro / "Informe_Entregable_1.pdf")
        sitio = copia / "_build" / "html"
        megas_sitio = comprimir(ZIP_SITIO, [a for a in sitio.rglob("*") if a.is_file()], sitio, "jupyter_book")

    # ZIP para el correo: lo que Git versionaría, con los entregables recién generados
    salida = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=RAIZ,
                            capture_output=True, check=True).stdout.decode("utf-8")
    archivos = [RAIZ / r for r in salida.split("\0") if r and not r.startswith(".git") and (RAIZ / r).exists()]
    assert any(a.suffix == ".pdf" for a in archivos), "El ZIP del correo debe llevar el informe en PDF"
    bloqueados = [a.name for a in archivos if a.suffix.lower() == ".js"]
    assert not bloqueados, f"El ZIP del correo no puede llevar archivos .js: {bloqueados}"
    megas = comprimir(ZIP_CORREO, archivos, RAIZ, "Entregable_1_UAVIDS-2025")
    assert megas < LIMITE_CORREO_MB, f"El ZIP pesa {megas:.1f} MB y supera el límite de {LIMITE_CORREO_MB} MB del correo"

    print(f"{ZIP_CORREO.name}: {megas:.1f} MB, {len(archivos)} archivos")
    print(f"{ZIP_SITIO.name}: {megas_sitio:.1f} MB")


if __name__ == "__main__":
    main()
