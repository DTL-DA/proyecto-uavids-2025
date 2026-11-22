"""Pruebas unitarias del modelo original (sección 6.5 del enunciado, puntos i a vi)."""
import io
import math

import pytest
import torch

from src.original_model import EscalonesSuaves, ModeloOriginal


def lote_sintetico(n=64, d=16, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    x = torch.randn(n, d, generator=g)
    y = (x[:, 0] > 0).long() + 2 * (x[:, 1] > 0.5).long()  # 4 clases definidas por umbrales en dos ejes
    return x, y


def entrenar(modelo, x, y, pasos=300, lr=1e-2):
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    modelo.train()
    for _ in range(pasos):
        opt.zero_grad()
        perdida = torch.nn.functional.cross_entropy(modelo(x), y)
        if modelo.perdida_extra is not None:
            perdida = perdida + modelo.perdida_extra
        perdida.backward()
        opt.step()
    return perdida.item()


# (i) puede sobreajustar un lote diminuto
def test_sobreajusta_lote_pequeno():
    x, y = lote_sintetico()
    modelo = ModeloOriginal(semilla=0)
    modelo.preentrenar(x)
    assert entrenar(modelo, x, y) < 0.1


# (ii) reproducible con la misma semilla
def test_reproducible_con_misma_semilla():
    x, _ = lote_sintetico()
    torch.testing.assert_close(ModeloOriginal(semilla=1).predict_proba(x), ModeloOriginal(semilla=1).predict_proba(x))


# (iii) formas de salida, probabilidades válidas y entradas inválidas
def test_formas_y_probabilidades():
    x, _ = lote_sintetico(n=10)
    p = ModeloOriginal(semilla=0).predict_proba(x)
    assert p.shape == (10, 5)
    torch.testing.assert_close(p.sum(1), torch.ones(10))
    assert (p >= 0).all()


def test_entrada_invalida():
    with pytest.raises(RuntimeError):
        ModeloOriginal(d=16)(torch.randn(4, 7))
    with pytest.raises(ValueError):
        ModeloOriginal(compuerta="otra")


# (iv) todos los parámetros reciben gradiente finito
def test_todos_los_parametros_reciben_gradiente():
    x, y = lote_sintetico()
    modelo = ModeloOriginal(semilla=0)
    modelo.preentrenar(x)
    modelo.train()
    perdida = torch.nn.functional.cross_entropy(modelo(x), y) + modelo.perdida_extra
    perdida.backward()
    for nombre, p in modelo.named_parameters():
        assert p.grad is not None, nombre
        assert torch.isfinite(p.grad).all(), nombre
        assert p.grad.abs().sum() > 0, nombre


# (v) verificación numérica de la propiedad demostrada: cota de Lipschitz y desigualdad de Jensen
@pytest.mark.parametrize("modo,entrada", [("aprendida", "x+expertos"), ("aprendida", "x"), ("promedio", "x"), ("umbral", "x"), ("suave", "x")])
def test_cota_lipschitz_no_se_excede(modo, entrada):
    torch.manual_seed(0)
    x, y = lote_sintetico(n=256)
    modelo = ModeloOriginal(semilla=0, compuerta=modo, entrada_compuerta=entrada)
    modelo.preentrenar(x)
    entrenar(modelo, x, y, pasos=50)
    modelo.eval()
    K = modelo.cota_lipschitz()
    with torch.no_grad():
        for escala in (1e-3, 1e-1, 1.0):
            dx = escala * torch.randn_like(x)
            num = (modelo(x).exp() - modelo(x + dx).exp()).norm(dim=1)
            assert (num / dx.norm(dim=1)).max().item() <= K * (1 + 1e-6)


def test_codificador_lipschitz():
    cod = EscalonesSuaves(d=3, T=5, tau_min=0.1)
    x = torch.randn(200, 3)
    dx = 1e-3 * torch.randn(200, 3)
    cociente = (cod(x + dx) - cod(x)).norm(dim=1) / dx.norm(dim=1)
    assert cociente.max().item() <= math.sqrt(5) / (4 * cod.tau().min().item()) + 1e-6


def test_desigualdad_de_jensen_de_la_mezcla():
    """-log sum_k g_k p_k(y) <= sum_k g_k (-log p_k(y)) para cada flujo."""
    x, y = lote_sintetico()
    modelo = ModeloOriginal(semilla=3)
    modelo.preentrenar(x)
    modelo.eval()
    with torch.no_grad():
        nll_mezcla = -modelo(x).gather(1, y[:, None]).squeeze(1)
        log_u, log_s = modelo.expertos(x)
        g = modelo.compuerta(x)
        cota = -(g[:, 0] * log_u.gather(1, y[:, None]).squeeze(1) + g[:, 1] * log_s.gather(1, y[:, None]).squeeze(1))
    assert (nll_mezcla <= cota + 1e-6).all()


# (vi) guardar, cargar y exportar conserva las salidas
def test_guardar_y_cargar_conserva_salidas():
    x, _ = lote_sintetico()
    m1 = ModeloOriginal(semilla=2)
    m1.preentrenar(x)
    buffer = io.BytesIO()
    torch.save(m1.state_dict(), buffer)
    buffer.seek(0)
    m2 = ModeloOriginal(semilla=99)
    m2.load_state_dict(torch.load(buffer))
    torch.testing.assert_close(m1.predict_proba(x), m2.predict_proba(x))


def test_exportacion_onnx_ida_y_vuelta(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    x, _ = lote_sintetico()
    modelo = ModeloOriginal(semilla=4)
    modelo.preentrenar(x)
    modelo.eval()
    ruta = tmp_path / "original.onnx"
    torch.onnx.export(modelo, x[:1], str(ruta), input_names=["x"], output_names=["log_p"], dynamic_axes={"x": {0: "n"}, "log_p": {0: "n"}},
                      opset_version=17)
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    p_onnx = torch.from_numpy(sesion.run(None, {"x": x.numpy()})[0]).exp()
    assert (p_onnx - modelo.predict_proba(x)).abs().max().item() < 1e-4
