"""Publica el libro compilado en GitHub Pages, en la carpeta /entregable-3/ de la rama gh-pages.

    jupyter-book build book/            # primero compilar (ejecuta todas las celdas)
    python book/publicar.py              # copia book/_build/html a gh-pages:/entregable-3/ y hace el commit
    git push origin gh-pages             # publicar (después activar Pages: Settings → Pages → rama gh-pages, carpeta /)

La URL queda en https://dtl-da.github.io/proyecto-uavids-2025/entregable-3/, la misma que enlaza el dashboard
(src/data.py, JBOOK). La rama gh-pages solo contiene HTML estático; no lleva código ni datos.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

LIBRO = Path(__file__).resolve().parent
RAIZ = LIBRO.parents[1]
HTML = LIBRO / "_build" / "html"
DESTINO = "entregable-3"


def git(*args, cwd=RAIZ):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main():
    if not (HTML / "intro.html").exists():
        raise SystemExit("Primero compile el libro: jupyter-book build book/")
    commit = git("rev-parse", "--short", "HEAD")
    existe = bool(subprocess.run(["git", "show-ref", "--verify", "--quiet", "refs/heads/gh-pages"], cwd=RAIZ).returncode == 0)
    with tempfile.TemporaryDirectory() as tmp:
        arbol = Path(tmp) / "gh-pages"
        if existe:
            git("worktree", "add", str(arbol), "gh-pages")
        else:
            git("worktree", "add", "--detach", str(arbol))
            git("checkout", "--orphan", "gh-pages", cwd=arbol)
            git("rm", "-rf", "--quiet", ".", cwd=arbol)
        try:
            (arbol / ".nojekyll").write_text("", encoding="utf-8")
            shutil.rmtree(arbol / DESTINO, ignore_errors=True)
            shutil.copytree(HTML, arbol / DESTINO)
            git("add", "-A", cwd=arbol)
            mensaje = f"Libro del Entregable 3 compilado desde {commit}" + "\n\n" + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
            git("commit", "-m", mensaje, cwd=arbol)
            print(f"Listo: rama gh-pages con /{DESTINO}/ (libro de {commit}). Falta: git push origin gh-pages")
        finally:
            git("worktree", "remove", "--force", str(arbol))


if __name__ == "__main__":
    main()
