"""Pruebas del dashboard: páginas, inferencia ONNX, conformal, fuera de dominio e ida y vuelta de los artefactos."""
import numpy as np
import pandas as pd
import pytest

from src import inference as I
from src.data import uavids

MODELOS = I.modelos_disponibles()


def test_paginas_responden():
    import app

    cliente = app.app.server.test_client()
    for ruta in ("/", "/eda", "/modelos"):
        assert cliente.get(ruta).status_code == 200


@pytest.mark.parametrize("modelo", MODELOS)
def test_probabilidades_validas(modelo):
    prueba = uavids().query("particion == 'prueba'").head(50)
    p = I.probabilidades(prueba[I.ATRIBUTOS].to_numpy(), modelo)
    assert p.shape == (50, 5)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    assert len(I.conjunto_conforme(p, modelo)) == 50


def test_exactitud_coherente_con_metadatos():
    """El F1 de la exportación en memoria coincide con el registrado en metadata.json (prueba completa, sin calibrar)."""
    from sklearn.metrics import f1_score

    prueba = uavids().query("particion == 'prueba'")
    for modelo in ("LightGBM", "Transformer"):
        if modelo not in MODELOS:
            continue
        p = I.probabilidades(prueba[I.ATRIBUTOS].to_numpy(), modelo, calibrar=False)
        f1 = f1_score([I.CLASES.index(c) for c in prueba["clase"]], p.argmax(1), average="macro")
        assert abs(f1 - I.metadatos(modelo)["metricas"]["f1_prueba_onnx"]) < 1e-6


def test_fuera_de_dominio_detecta_valores_absurdos():
    x = np.full(16, 1e12)
    assert I.fuera_de_dominio(x, MODELOS[0])["fuera"]
    normal = uavids().query("particion == 'entrenamiento'")[I.ATRIBUTOS].median().to_numpy()
    assert not I.fuera_de_dominio(normal, MODELOS[0])["atributos_fuera_de_rango"]


def test_reporte_ida_y_vuelta():
    t = pd.read_csv(I.ARTEFACTOS / "reporte_ida_y_vuelta.csv")
    redes = t[~t["modelo"].isin(["LightGBM"])]
    assert (redes["error_max_prob"] < 1e-4).all()
    assert (t["misma_clase_%"] == 100).all()
    assert (t["dif_f1"].abs() < 1e-9).all()


def test_lideres_consistentes_con_tabla_maestra():
    """El dashboard se alimenta de la tabla maestra: la tabla de líderes publicada debe coincidir con ella (sección 9 del enunciado)."""
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    lideres = pd.read_csv(raiz / "artifacts" / "dashboard" / "lideres.csv").set_index("Modelo")
    maestra = pd.read_csv(raiz / "experiments" / "resultados" / "tabla_maestra.csv")
    defin = maestra[(maestra["fase"] == "definitiva") & (maestra["conjunto"] == "UAVIDS-2025") & maestra["f1_eva"].notna()]
    resumen = defin.groupby("modelo")["f1_eva"].agg(["mean", "std", "count"])
    for modelo, fila in lideres.iterrows():
        assert int(fila["Semillas"]) == int(resumen.loc[modelo, "count"]), modelo
        assert abs(fila["F1 macro"] - resumen.loc[modelo, "mean"]) < 1e-9, modelo
        assert abs(fila["DE"] - resumen.loc[modelo, "std"]) < 1e-9, modelo


def test_adicionales_consistentes_con_analisis():
    """Los F1 de los conjuntos adicionales que muestra el dashboard son los del análisis del capítulo 8."""
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1]
    ruta = raiz / "artifacts" / "dashboard" / "adicionales.csv"
    if not ruta.exists():
        pytest.skip("conjuntos adicionales aún sin publicar en el dashboard")
    tablero = pd.read_csv(ruta).set_index(["conjunto", "modelo"])["F1 macro"]
    for c in ("uav_nidd", "unsw_nb15", "unr_idd"):
        t = pd.read_csv(raiz / "experiments" / "resultados" / "adicionales" / f"tabla_{c}.csv").set_index(["conjunto", "modelo"])["F1 macro"]
        assert ((tablero.loc[t.index] - t).abs() < 1e-12).all()
