"""Celdas del cuaderno y escritura con conservación de salidas."""
import json
from pathlib import Path

CELDAS = []


def md(identificador, texto):
    CELDAS.append({"cell_type": "markdown", "id": identificador, "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)})


def code(identificador, texto):
    CELDAS.append({"cell_type": "code", "id": identificador, "metadata": {}, "execution_count": None, "outputs": [],
                   "source": texto.strip("\n").splitlines(keepends=True)})


def escribir(destino):
    destino = Path(destino)
    ids = [c["id"] for c in CELDAS]
    repetidos = {i for i in ids if ids.count(i) > 1}
    assert not repetidos, f"Identificadores repetidos: {repetidos}"
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "cells": CELDAS}
    if destino.exists():
        previas = json.loads(destino.read_text(encoding="utf-8"))["cells"]
        por_codigo = {"".join(c["source"]): c for c in previas if c["cell_type"] == "code" and c.get("outputs")}
        for c in CELDAS:
            if c["cell_type"] == "code" and "".join(c["source"]) in por_codigo:
                anterior = por_codigo["".join(c["source"])]
                c["outputs"], c["execution_count"] = anterior["outputs"], anterior.get("execution_count")
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("escrito", destino.name, "| celdas:", len(CELDAS), "| con salidas:", sum(1 for c in CELDAS if c.get("outputs")))
