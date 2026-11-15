"""Reúne en entregables/ cada producto que pide el enunciado del Entregable 2, compila el Jupyter Book y arma
los ZIP para enviarlos. Todo lo que hay en entregables/ se genera aquí desde el cuaderno ejecutado, resultados/ y
literatura/, de modo que, si cambia el cuaderno, basta con volver a ejecutar este script.

Salida en la raíz del repositorio, sin versionar:
- Entregable_2_UAVIDS-2025.zip: entregables, cuaderno y requisitos, para adjuntar al correo. No lleva archivos .js,
  que los servicios de correo pueden bloquear incluso dentro de un ZIP.
- Jupyter_Book_E2_sitio_web.zip: el libro compilado en HTML, para publicarlo o compartirlo por enlace.
- Corridas_E2_probabilidades_y_pesos.zip: probabilidades y pesos de cada corrida, que no caben en el correo y
  permiten verificar las tablas sin reentrenar.

Uso:
    python Entregable_2_UAVIDS_2025/herramientas/organizar_entregables.py [--sin-libro] [--sin-corridas]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERRAMIENTAS = Path(__file__).resolve().parent
E2 = HERRAMIENTAS.parent
RAIZ = E2.parent
RES, LIT = E2 / "resultados", E2 / "literatura"
DESTINO = E2 / "entregables"
CUADERNO = RAIZ / "Entregable 2 - UAVIDS-2025.ipynb"
ZIP_CORREO = RAIZ / "Entregable_2_UAVIDS-2025.zip"
ZIP_SITIO = RAIZ / "Jupyter_Book_E2_sitio_web.zip"
ZIP_CORRIDAS = RAIZ / "Corridas_E2_probabilidades_y_pesos.zip"
LIMITE_CORREO_MB = 25
ENTORNO = {**os.environ, "PYTHONUTF8": "1"}

# Carpeta: (lo que pide el enunciado, archivos o carpetas que la llenan)
CONTENIDO = {
    "00_codigo": ("Código completo y documentado: EDA, 10 modelos, baselines, ajuste y sobreajuste, análisis estadístico, robustez, incertidumbre e interpretabilidad",
                  [E2 / "codigo", E2 / "configuracion", E2 / "herramientas"]),
    "01_informe_jupyter_book": ("Informe en Jupyter Book con ejecución reproducible de celdas (fuentes del libro e informe en PDF)", []),
    "02_dataset_y_contrato": ("Dataset o instrucciones verificables para obtenerlo, con la Data Card y el contrato de datos (secciones 1.4 y 1.5 del informe)",
                              [RAIZ / "Entregable_1_UAVIDS_2025" / "herramientas" / "obtener_datos.py"]),
    "03_preregistro": ("Preregistro fechado y versionado, con su adenda",
                       [E2 / "preregistro_e2.md", E2 / "preregistro_e2_adenda.md", RES / "preregistro_e2.json", RES / "preregistro_e2_adenda.json", E2 / "configuracion" / "experimento.json"]),
    "04_auditoria_de_fuga": ("Reporte de auditoría de fuga", [RES / "auditoria_fuga.csv"]),
    "05_ficha_de_ruta": ("Ficha de ruta metodológica: tablas A, B y C, criterio de adecuación y su sensibilidad al umbral del ECE",
                         [RES / "tabla_a.csv", RES / "tabla_b.csv", RES / "tabla_c.csv", RES / "adecuacion.csv", RES / "adecuacion_sensibilidad_ece.csv"]),
    "06_tabla_comparativa": ("Tabla comparativa con métricas, costos e inferencia estadística, y KS y beneficio neto de la decisión de alarma",
                             [RES / "tabla_benchmark.csv", RES / "contrastes_rq1.csv", RES / "contrastes_rq2.csv", RES / "ks_y_beneficio_neto.csv"]),
    "07_registro_y_fichas": ("Registro de experimentos, fichas de modelo e importancia de los hiperparámetros",
                             [RES / "registro_experimentos.csv", RES / "fichas_de_modelo.csv", RES / "importancia_hiperparametros.csv"]),
    "08_techo_e_incertidumbre": ("Reporte de techo de desempeño, incertidumbre y calibración por clase", [RES / "techo.csv", RES / "conforme.csv", RES / "calibracion_por_clase.csv"]),
    "09_busqueda_de_arquitectura": ("Reporte de la búsqueda de arquitectura: las 1000 arquitecturas, las dos fases, los finalistas y el análisis de efectos",
                                    [RES / "busqueda_fase1.csv", RES / "busqueda_fase2.csv", RES / "busqueda_efectos.csv", RES / "arquitecturas_espacio.csv"]),
    "10_paper_guia": ("Paper guía: réplica parcial y limitación identificada", [RES / "replica.csv", RES / "limitacion_paper_guia.csv"]),
    "11_literatura": ("Revisión sistemática: hoja de extracción, puntaje del Top 10, mecanismos transferibles, consultas y bibliografía en BibTeX",
                      [LIT / "hoja_extraccion_e2.csv", LIT / "puntaje_top10.csv", LIT / "mecanismos_transferibles.csv", LIT / "reproducibilidad_estudios.csv",
                       LIT / "registros_nuevos_e2.csv", LIT / "consultas_e2_wos_scopus_ieee.md", LIT / "referencias.bib"]),
    "12_reproducibilidad": ("Lista de verificación de reproducibilidad (apéndice B), entorno y prueba de entorno limpio",
                            [RAIZ / "requirements.txt", E2 / "Dockerfile", RES / "reproduccion_entorno_limpio.json", RES / "lista_verificacion_b.csv"]),
}
# El material de preparación de la presentación oral no se entrega
SIN_ENTREGAR = ("__pycache__", "*.pyc", "*.log", "crear_guion_defensa.py")

README = """# Entregable 2 · Benchmark de Deep Learning sobre UAVIDS-2025

Esta carpeta reúne, en el orden del enunciado, los productos del Entregable 2. Todo se genera con
`herramientas/organizar_entregables.py` a partir del cuaderno ejecutado y de `resultados/`, de modo que no se edita a mano.

| Carpeta | Entregable del enunciado |
|---|---|
{filas}| (cuaderno en la raíz) | `Entregable 2 - UAVIDS-2025.ipynb`, ejecutado de principio a fin |

## Cómo reproducir

1. Instalar las dependencias: `pip install -r requirements.txt` (Python 3.13, CPU).
2. Obtener los datos con `02_dataset_y_contrato/obtener_datos.py`, que descarga `UAVIDS-2025.csv` y verifica su suma SHA-256.
3. Reentrenar, si se quiere, con `python Entregable_2_UAVIDS_2025/herramientas/orquestar.py`, que encadena las fases del preregistro:
   ajuste, piloto, búsqueda de arquitectura, ejecución definitiva, experimentos complementarios y análisis. Cada corrida queda en
   `resultados/corridas/` y el ejecutor salta las que ya existen.
4. Ejecutar el cuaderno con `python Entregable_2_UAVIDS_2025/herramientas/ejecutar_cuaderno.py "Entregable 2 - UAVIDS-2025.ipynb"`.
   El cuaderno no reentrena: lee las corridas guardadas y recalcula todas las métricas, pruebas, tablas y gráficas.
5. Para verificar sin reentrenar, descomprimir `Corridas_E2_probabilidades_y_pesos.zip` en la raíz del repositorio, que agrega a
   `resultados/corridas/` las probabilidades y los pesos de cada corrida; por su tamaño, se entrega aparte del ZIP del correo.

La prueba de entorno limpio (`herramientas/entorno_limpio.py`) instala un entorno nuevo desde `requirements.txt`, reentrena la
primera semilla de cada modelo que tarda menos de 15 minutos y compara el F1; su resultado está en
`12_reproducibilidad/reproduccion_entorno_limpio.json` y en la sección 8 del informe.
"""


def historial_preregistro(destino):
    """Etiquetas del preregistro y commits del repositorio, con fecha, para verificar el orden sin acceso al repositorio."""
    git = lambda *a: subprocess.run(["git", "-C", str(RAIZ), *a], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    etiquetas = git("for-each-ref", "refs/tags/preregistro-e2*", "--sort=taggerdate",
                    "--format=%(refname:short) | etiqueta %(taggerdate:iso) | commit %(*objectname:short) | %(contents:subject)")
    commits = git("log", "--reverse", "--date=iso", "--format=%h | %ad | %s")
    if not etiquetas:
        print("  Sin repositorio git: no se generó el historial del preregistro")
        return
    destino.write_text("Historial del preregistro del Entregable 2, generado con git por herramientas/organizar_entregables.py\n\n"
                       "Etiquetas (nombre | fecha de la etiqueta | commit | mensaje):\n" + etiquetas + "\n\n"
                       "Commits del repositorio en orden cronológico (commit | fecha | mensaje):\n" + commits + "\n", encoding="utf-8")


def compilar_libro(libro, constructor):
    codigo = "from jupyter_book.cli.main import main; main()"
    r = subprocess.run([sys.executable, "-c", codigo, "build", ".", "--builder", constructor], cwd=libro, env=ENTORNO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"Falló la compilación {constructor} del libro")
    return sum("WARNING" in linea for linea in (r.stdout + r.stderr).splitlines())


def imprimir_pdf(html, pdf):
    """Imprime el libro en una sola página HTML a PDF, con Edge o Chrome sin interfaz."""
    candidatos = [shutil.which("msedge"), shutil.which("chrome"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    navegador = next((c for c in candidatos if c and Path(c).exists()), None)
    if navegador is None:
        print("  Sin Edge ni Chrome: no se generó el PDF")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as perfil:
        subprocess.run([navegador, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={perfil}", "--virtual-time-budget=60000",
                        f"--print-to-pdf={pdf}", html.as_uri()], check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # El navegador puede devolver el control antes de terminar de escribir: se espera a que el tamaño se estabilice
        tamano, estable, limite = -1, 0, time.monotonic() + 180
        while estable < 3:
            actual = pdf.stat().st_size if pdf.exists() else 0
            estable = estable + 1 if actual and actual == tamano else 0
            tamano = actual
            if time.monotonic() > limite:
                raise TimeoutError(f"No se generó {pdf.name}")
            time.sleep(2)


def comprimir(destino, archivos, base, prefijo=""):
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for archivo in archivos:
            z.write(archivo, Path(prefijo) / archivo.relative_to(base))
    return destino.stat().st_size / 1024 ** 2


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-libro", action="store_true", help="No compila el Jupyter Book ni el PDF")
    ap.add_argument("--sin-corridas", action="store_true", help="No arma el ZIP de probabilidades y pesos")
    a = ap.parse_args()
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    faltan = []
    for carpeta, (_, archivos) in CONTENIDO.items():
        (DESTINO / carpeta).mkdir(parents=True)
        for f in archivos:
            if f.is_dir():
                shutil.copytree(f, DESTINO / carpeta / f.name, ignore=shutil.ignore_patterns(*SIN_ENTREGAR))
            elif f.exists():
                shutil.copy2(f, DESTINO / carpeta / f.name)
            else:
                faltan.append(str(f.relative_to(RAIZ)))
    historial_preregistro(DESTINO / "03_preregistro" / "historial_git_preregistro.txt")
    filas = "".join(f"| `{c}/` | {d} |\n" for c, (d, _) in CONTENIDO.items())
    (DESTINO / "README.md").write_text(README.format(filas=filas), encoding="utf-8")

    if not a.sin_libro:
        libro = DESTINO / "01_informe_jupyter_book"
        subprocess.run([sys.executable, str(HERRAMIENTAS / "construir_libro.py"), "--libro", str(libro), "--cuaderno", str(CUADERNO)], check=True, env=ENTORNO)
        # Se compila en una carpeta temporal, porque la ruta del proyecto más los archivos del libro supera el límite de 260 caracteres de Windows
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporal:
            copia = Path(temporal) / "libro"
            shutil.copytree(libro, copia)
            avisos = compilar_libro(copia, "html")
            compilar_libro(copia, "singlehtml")
            imprimir_pdf(copia / "_build" / "singlehtml" / "index.html", libro / "Informe_Entregable_2.pdf")
            sitio = copia / "_build" / "html"
            megas_sitio = comprimir(ZIP_SITIO, [x for x in sitio.rglob("*") if x.is_file()], sitio, "jupyter_book")
        print(f"Libro: {avisos} advertencias; {ZIP_SITIO.name}: {megas_sitio:.1f} MB")

    archivos = [x for x in DESTINO.rglob("*") if x.is_file()] + [CUADERNO, RAIZ / "requirements.txt"]
    bloqueados = [x.name for x in archivos if x.suffix.lower() == ".js"]
    assert not bloqueados, f"El ZIP del correo no puede llevar archivos .js: {bloqueados}"
    megas = comprimir(ZIP_CORREO, archivos, RAIZ, "Entregable_2_UAVIDS-2025")
    print(f"{ZIP_CORREO.name}: {megas:.1f} MB, {len(archivos)} archivos" + (f" (supera los {LIMITE_CORREO_MB} MB del correo)" if megas > LIMITE_CORREO_MB else ""))

    if not a.sin_corridas:
        pesados = [x for x in (RES / "corridas").rglob("*") if x.suffix in (".npy", ".pt", ".pkl")]
        megas_c = comprimir(ZIP_CORRIDAS, pesados, RAIZ)
        print(f"{ZIP_CORRIDAS.name}: {megas_c:.1f} MB, {len(pesados)} archivos")
    print(f"Entregables en {DESTINO.relative_to(RAIZ)}; faltan: {faltan or 'ninguno'}")


if __name__ == "__main__":
    main()
