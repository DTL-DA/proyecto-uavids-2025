"""Reúne en entregables/ cada producto que pide la sección 12 del enunciado del Entregable 3, imprime el libro en PDF y
arma los ZIP de la entrega, igual que organizar_entregables.py del Entregable 2.

    python Entregable_3_UAVIDS_2025/organizar_entregables.py [--sin-pdf]

Salida, sin versionar:
- Entregable_3_UAVIDS_2025/entregables/: una carpeta por producto de la sección 12 (más las correcciones del E2, la
  literatura y los análisis) y 00_codigo con todo el código. Todo archivo versionado del Entregable 3 queda en alguna
  carpeta, de modo que no falta ningún archivo del proceso.
- Entregable_3_UAVIDS-2025.zip, en la raíz: las carpetas anteriores, el cuaderno ejecutado y el índice con los enlaces,
  para la plataforma o el correo. No lleva archivos .js, que los servicios de correo bloquean incluso dentro de un ZIP.
- Jupyter_Book_E3_sitio_web.zip, en la raíz: el libro compilado en HTML.
El paquete de reproducción (paquete_reproduccion_e3.zip) lo arma verificacion_externa/empaquetar.py.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

E3 = Path(__file__).resolve().parent
RAIZ = E3.parent
DESTINO = E3 / "entregables"
CUADERNO = RAIZ / "Entregable 3 - UAVIDS-2025.ipynb"
ZIP_ENTREGA = RAIZ / "Entregable_3_UAVIDS-2025.zip"
ZIP_SITIO = RAIZ / "Jupyter_Book_E3_sitio_web.zip"
LIMITE_CORREO_MB = 25
ENTORNO = {**os.environ, "PYTHONUTF8": "1"}

REPO = "https://github.com/DTL-DA/proyecto-uavids-2025"
ENLACES = {
    "Jupyter Book publicado": "https://dtl-da.github.io/proyecto-uavids-2025/entregable-3/",
    "Cuaderno ejecutado (.ipynb)": f"{REPO}/blob/entregable-3/Entregable%203%20-%20UAVIDS-2025.ipynb",
    "Dashboard en Render": "https://uavids-2025-dashboard.onrender.com",
    "Repositorio (rama entregable-3)": f"{REPO}/tree/entregable-3",
    "Integración continua (pruebas)": f"{REPO}/actions/workflows/entregable3-pruebas.yml",
    "Release entregable-3 (con el paquete de reproducción)": f"{REPO}/releases/tag/entregable-3",
}
CI = RAIZ / ".github" / "workflows" / "entregable3-pruebas.yml"
EXTENSIONES_CODIGO = {".py", ".css", ".ini", ".yml", ".yaml"}

# Carpeta: (entregable del enunciado, lo que la llena). Cada elemento es una ruta relativa al Entregable 3 (admite
# comodines), una ruta absoluta o una pareja (ruta, nombre en la carpeta). De las carpetas solo se copian los archivos versionados.
PRODUCTOS = {
    "01_dashboard": ("1. URL pública del dashboard desplegado en Render, con su código, el despliegue y la prueba de humo",
                     ["app.py", "pages", "assets", "src/inference.py", "src/data.py", "src/plots.py", "render.yaml", "requirements.txt",
                      "data/uavids.parquet", "artifacts/dashboard", "experiments/resultados/humo_dashboard.json", "verificacion_externa/humo_dashboard.py",
                      "book/09_dashboard.md"]),
    "02_repositorio": ("2. Repositorio de GitHub: README, requisitos con versiones exactas, Dockerfile, licencia, CITATION.cff e integración continua",
                       ["README.md", (RAIZ / "README.md", "README_raiz_del_repositorio.md"), "requirements.txt", "requirements-train.txt",
                        "requirements-gpu.txt", "Dockerfile", "LICENSE", "CITATION.cff", "pytest.ini", CI]),
    "03_jupyter_book": ("3. Jupyter Book: fuentes, informe en PDF y enlace a la versión publicada", ["book"]),
    "04_modelo_original": ("4. Código del modelo original con docstrings y sus pruebas unitarias, que corren en integración continua",
                           ["src/original_model.py", "tests", CI]),
    "05_estandar_protocolo_bitacora": ("5. Declaración del estándar, protocolo preregistrado con su etiqueta de Git y fecha, y bitácora de diseño",
                                       ["experiments/estandar.md", "experiments/protocolo.md", "experiments/design_log.md", "experiments/diseno"]),
    "06_procedencia": ("6. Documento de procedencia del mecanismo", ["experiments/procedencia.md"]),
    "07_demostracion": ("7. Demostración formal de la cota de Lipschitz, con sus supuestos y su verificación numérica (capítulo 5)",
                        ["book/05_modelo_diseno.md", (E3 / "book" / "_build" / "jupyter_execute" / "05_modelo_diseno.ipynb", "05_modelo_diseno_ejecutado.ipynb")]),
    "08_tabla_maestra": ("8. Tabla maestra de experimentos (una fila por corrida, con métricas, tiempos, costos y semillas) y el registro de cada corrida",
                         ["experiments/resultados/tabla_maestra.csv", "experiments/tabla_maestra.py", "experiments/resultados/corridas"]),
    "09_artefactos": ("9. Artefactos de los modelos en ONNX, con su metadata.json y el reporte de la prueba de ida y vuelta",
                      ["artifacts", "training/exportar_onnx.py"]),
    "10_ficha_modelo_y_datos": ("10. Ficha del modelo y declaración de los datos",
                                ["MODEL_CARD.md", "DATA_STATEMENT.md", "data/procedencia_conjuntos_adicionales.md", "data/*_resumen.json"]),
    "11_archivo_doi": ("11. Archivo versionado del repositorio con identificador persistente (Zenodo)", ["CITATION.cff", RAIZ / ".zenodo.json"]),
    "12_autoria": ("12. Declaración de autoría, contribuciones y uso de herramientas, firmada por el autor", []),
    "13_reproduccion_y_humo": ("13. Reproducción (scripts, paquete y corrida del autor) y prueba de humo del despliegue",
                               ["verificacion_externa/reproducir.py", "verificacion_externa/empaquetar.py", "verificacion_externa/humo_dashboard.py",
                                "experiments/resultados/humo_dashboard.json", (E3 / "verificacion_externa" / "salida", "reproduccion_del_autor")]),
    "14_correcciones_e2": ("Bloque A: tabla de trazabilidad de correcciones del Entregable 2, auditoría de fuga y prueba de D3",
                           ["experiments/bloque_a", "book/01_correcciones.md"]),
    "15_literatura_y_novedad": ("Revisión de literatura: cadenas de búsqueda, matriz de novedad, conteos por base y bibliografía en BibTeX",
                                ["experiments/novedad", "book/references.bib"]),
    "16_resultados_y_analisis": ("Análisis confirmatorio, pruebas del cuadro 4, ablación, conjuntos adicionales, robustez y adecuación, con sus scripts",
                                 ["experiments/resultados/confirmatorio", "experiments/resultados/ablacion", "experiments/resultados/adicionales",
                                  "experiments/resultados/robustez", "experiments/resultados/adecuacion", "experiments/resultados/potencia.json",
                                  "experiments/analisis_*.py", "experiments/potencia.py"]),
}


def largo(p: Path) -> str:
    """Ruta extendida de Windows: la carpeta del proyecto en OneDrive deja algunas rutas por encima de 260 caracteres."""
    s = str(p)
    return "\\\\?\\" + s if os.name == "nt" and not s.startswith("\\\\?\\") else s


def copiar(origen: Path, destino: Path) -> None:
    os.makedirs(largo(destino.parent), exist_ok=True)
    shutil.copy2(largo(origen), largo(destino))


def escribir(destino: Path, texto: str) -> None:
    os.makedirs(largo(destino.parent), exist_ok=True)
    with open(largo(destino), "w", encoding="utf-8") as f:
        f.write(texto)


def listar(carpeta: Path) -> list[Path]:
    """Archivos de una carpeta, recorridos con la ruta extendida y devueltos con la ruta normal."""
    base = largo(carpeta)
    return [carpeta / os.path.relpath(os.path.join(d, n), base) for d, _, nombres in os.walk(base) for n in nombres]


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(RAIZ), *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()


VERSIONADOS = {RAIZ / p for p in git("ls-files", "-z", "--", "Entregable_3_UAVIDS_2025").split("\0") if p}


def expandir(elemento) -> list[tuple[Path, str]]:
    """Devuelve (origen, nombre en la carpeta) para cada archivo de un elemento de PRODUCTOS."""
    ruta, nombre = elemento if isinstance(elemento, tuple) else (elemento, None)
    rutas = [Path(ruta)] if isinstance(ruta, Path) else sorted(E3.glob(ruta)) or [E3 / ruta]
    salida = []
    for r in rutas:
        if r.is_dir():
            todos = [f for f in r.rglob("*") if f.is_file() and not {"__pycache__", "_build"} & set(f.relative_to(r).parts)]
            versionados = [f for f in todos if f in VERSIONADOS]
            salida += [(f, f"{nombre or r.name}/{f.relative_to(r).as_posix()}") for f in (versionados or todos)]
        elif r.exists():
            salida.append((r, nombre or r.name))
        else:
            print(f"  Falta: {r}")
    return salida


def doi() -> str | None:
    m = re.search(r'^doi:\s*"?([^"\n]+)', (E3 / "CITATION.cff").read_text(encoding="utf-8"), re.M)
    return m.group(1).strip() if m else None


def declaracion_autoria() -> str:
    texto = (E3 / "book" / "11_reproducibilidad.md").read_text(encoding="utf-8")
    seccion = re.search(r"^## Declaración de autoría.*?(?=^## )", texto, re.M | re.S).group(0)
    return "# Entregable 3 · " + seccion[3:].strip() + "\n\nTomada del capítulo 11 del Jupyter Book (`book/11_reproducibilidad.md`).\n"


def historial_git() -> str:
    etiquetas = git("for-each-ref", "refs/tags/e3-*", "refs/tags/entregable-3", "--sort=taggerdate",
                    "--format=%(refname:short) | etiqueta %(taggerdate:iso) | commit %(*objectname:short)%(objectname:short) | %(contents:subject)")
    commits = git("log", "--reverse", "--date=iso", "--format=%h | %ad | %s", "--", "Entregable_3_UAVIDS_2025")
    return ("Historial del Entregable 3, generado con git por organizar_entregables.py\n\n"
            "Etiquetas (nombre | fecha de la etiqueta | commit | mensaje):\n" + etiquetas + "\n\n"
            "Commits del Entregable 3 en orden cronológico (commit | fecha | mensaje):\n" + commits + "\n")


def indice(identificador: str | None) -> str:
    enlaces = dict(ENLACES)
    enlaces["DOI de Zenodo"] = f"https://doi.org/{identificador}" if identificador else "se asigna al publicar la release `entregable-3`"
    filas_enlaces = "".join(f"| {k} | {v} |\n" for k, v in enlaces.items())
    filas = "".join(f"| `{c}/` | {d} |\n" for c, (d, _) in PRODUCTOS.items())
    return f"""# Entregable 3 · Modelo original y validación rigurosa para detectar intrusiones en redes de drones (UAVIDS-2025)

**Autor:** Donny Torres · Maestría en Analítica de Datos, Universidad del Norte · Deep Learning · Noviembre de 2026

Esta carpeta reúne, en el orden de la sección 12 del enunciado, los productos del Entregable 3. La genera
`organizar_entregables.py` desde el repositorio, así que no se edita a mano. El informe completo está en el Jupyter Book
publicado y en el cuaderno `Entregable 3 - UAVIDS-2025.ipynb`, que se entrega ejecutado con todas las salidas.

## Enlaces

| Producto | Enlace |
|---|---|
{filas_enlaces}
## Carpetas

| Carpeta | Entregable del enunciado |
|---|---|
| `00_codigo/` | Todo el código del Entregable 3 con la estructura del repositorio (sección 9), más cualquier archivo versionado que no esté en otra carpeta |
{filas}| (en la raíz del ZIP) | `Entregable 3 - UAVIDS-2025.ipynb`, ejecutado de principio a fin |

## Cómo reproducir

Los pasos completos están en la sección "Cómo reproducirlo" de `02_repositorio/README.md`. En resumen: descomprimir
`paquete_reproduccion_e3.zip` (publicado con la *release*), construir el contenedor con
`docker build -t uavids-e3 -f Entregable_3_UAVIDS_2025/Dockerfile .` y ejecutar
`docker run --rm uavids-e3 python verificacion_externa/reproducir.py`, que regenera la tabla principal y reporta la
discrepancia máxima. La corrida del autor está en `13_reproduccion_y_humo/reproduccion_del_autor/`.
"""


def compilar_libro(constructor: str, salida: Path | None = None) -> None:
    """Compila el libro; con `salida`, deja _build en esa carpeta (de ruta corta, por el límite de Windows)."""
    codigo = "from jupyter_book.cli.main import main; main()"
    extra = ["--path-output", str(salida)] if salida else []
    r = subprocess.run([sys.executable, "-c", codigo, "build", "book", "--builder", constructor, *extra], cwd=E3, env=ENTORNO,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"Falló la compilación {constructor} del libro")


def imprimir_pdf(html: Path, pdf: Path) -> None:
    """Imprime el libro en una sola página HTML a PDF, con Edge o Chrome sin interfaz."""
    candidatos = [shutil.which("msedge"), shutil.which("chrome"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    navegador = next((c for c in candidatos if c and Path(c).exists()), None)
    if navegador is None:
        print("  Sin Edge ni Chrome: no se generó el PDF")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as perfil:
        subprocess.run([navegador, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={perfil}",
                        "--virtual-time-budget=60000", f"--print-to-pdf={pdf}", html.as_uri()],
                       check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tamano, estable, limite = -1, 0, time.monotonic() + 180
        while estable < 3:
            actual = pdf.stat().st_size if pdf.exists() else 0
            estable = estable + 1 if actual and actual == tamano else 0
            tamano = actual
            if time.monotonic() > limite:
                raise TimeoutError(f"No se generó {pdf.name}")
            time.sleep(2)


def comprimir(destino: Path, archivos: list[tuple[Path, str]]) -> float:
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for origen, nombre in archivos:
            z.write(largo(origen), nombre)
    return destino.stat().st_size / 1024 ** 2


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sin-pdf", action="store_true", help="no compila el libro en una sola página ni lo imprime en PDF")
    a = ap.parse_args()

    if DESTINO.exists():
        shutil.rmtree(largo(DESTINO))
    usados: set[Path] = set()
    for carpeta, (_, elementos) in PRODUCTOS.items():
        for origen, nombre in [par for e in elementos for par in expandir(e)]:
            copiar(origen, DESTINO / carpeta / nombre)
            usados.add(origen)
        os.makedirs(largo(DESTINO / carpeta), exist_ok=True)

    # 00_codigo: todo el código con la estructura del repositorio, más los archivos versionados que no quedaron en otra carpeta
    resto = [f for f in VERSIONADOS if f.is_file() and (f.suffix in EXTENSIONES_CODIGO or f.name == "Dockerfile" or f.name.startswith("requirements") or f not in usados)]
    for f in sorted(resto):
        copiar(f, DESTINO / "00_codigo" / f.relative_to(E3))
        usados.add(f)
    copiar(CI, DESTINO / "00_codigo" / ".github" / "workflows" / CI.name)

    escribir(DESTINO / "05_estandar_protocolo_bitacora" / "historial_git.txt", historial_git())
    escribir(DESTINO / "12_autoria" / "declaracion_autoria.md", declaracion_autoria())
    identificador = doi()
    escribir(DESTINO / "11_archivo_doi" / "DOI.md",
             f"# Identificador persistente\n\nDOI: https://doi.org/{identificador}\n" if identificador else
             "# Identificador persistente\n\nEl DOI de Zenodo se asigna al publicar la release `entregable-3`; se agrega a CITATION.cff y se vuelve a generar esta carpeta.\n")
    escribir(DESTINO / "README.md", indice(identificador))
    sin_ubicar = sorted(str(f.relative_to(E3)) for f in VERSIONADOS if f.is_file() and f not in usados)
    assert not sin_ubicar, f"Archivos versionados sin carpeta: {sin_ubicar}"

    html = E3 / "book" / "_build" / "html"
    if not (html / "intro.html").exists():
        compilar_libro("html")
    if not a.sin_pdf:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporal:
            compilar_libro("singlehtml", Path(temporal))
            imprimir_pdf(Path(temporal) / "_build" / "singlehtml" / "index.html", DESTINO / "03_jupyter_book" / "Informe_Entregable_3.pdf")
    megas_sitio = comprimir(ZIP_SITIO, [(f, f"jupyter_book_e3/{f.relative_to(html).as_posix()}") for f in html.rglob("*") if f.is_file()])
    print(f"{ZIP_SITIO.name}: {megas_sitio:.1f} MB")

    prefijo = "Entregable_3_UAVIDS-2025"
    archivos = [(f, f"{prefijo}/{f.relative_to(DESTINO).as_posix()}") for f in listar(DESTINO)]
    archivos.append((CUADERNO, f"{prefijo}/{CUADERNO.name}"))
    bloqueados = [n for _, n in archivos if n.lower().endswith(".js")]
    assert not bloqueados, f"El ZIP de la entrega no puede llevar archivos .js: {bloqueados}"
    megas = comprimir(ZIP_ENTREGA, archivos)
    print(f"{ZIP_ENTREGA.name}: {megas:.1f} MB, {len(archivos)} archivos"
          + (f" (supera los {LIMITE_CORREO_MB} MB del correo)" if megas > LIMITE_CORREO_MB else ""))
    print(f"Entregables en {DESTINO.relative_to(RAIZ)}: {len(PRODUCTOS) + 1} carpetas; {len(VERSIONADOS)} archivos versionados, todos ubicados")


if __name__ == "__main__":
    main()
