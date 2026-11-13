"""Catálogo de modelos del benchmark del Entregable 2.

Cada modelo se construye con ``construir(nombre, arquitectura=None)``. Los modelos con
arquitectura definida en la literatura ignoran ``arquitectura``; los que no la tienen
reciben ``arquitectura = (w1, w2, w3, b1, b2, b3)``, el punto del espacio de la sección 14
del enunciado (anchos en {16, 32, 64, 128, 256} y dropout de 0,2 tras la capa i si bi = 1).

Todos reciben el mismo vector de 16 atributos, transformados con log(1 + x) y
estandarizados con los estadísticos de entrenamiento. Los modelos secuenciales leen el
vector como una secuencia de 16 pasos de un valor, en el orden de las columnas del
archivo, como el código de los autores del conjunto de datos (Zeng, 2025).
"""
from __future__ import annotations

import itertools
import math

import torch
from torch import nn

N_ATRIBUTOS = 16
N_CLASES = 5
ANCHOS = (16, 32, 64, 128, 256)
TASA_DROPOUT_BUSQUEDA = 0.2
REFERENCIA = (64, 64, 64, 0, 1, 0)


def espacio_arquitecturas():
    """Las 1000 arquitecturas de la sección 14.1: producto cartesiano de W^3 y {0, 1}^3."""
    return [w + b for w in itertools.product(ANCHOS, repeat=3) for b in itertools.product((0, 1), repeat=3)]


def identificador(arquitectura):
    w1, w2, w3, b1, b2, b3 = arquitectura
    return f"w{w1}-{w2}-{w3}_d{b1}{b2}{b3}"


class NormalizacionSecuencia(nn.Module):
    """BatchNorm sobre los canales de una secuencia (lote, pasos, canales), como en Keras."""

    def __init__(self, canales):
        super().__init__()
        self.bn = nn.BatchNorm1d(canales)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


# --------------------------------------------------------------------------------------
# Modelos base del Entregable 1, con el dropout del código de los autores (Zeng, 2025)
# --------------------------------------------------------------------------------------
class DNN(nn.Module):
    """16 → 128 → 64 → 32 → 5, ReLU, dropout de 0,2 tras las capas de 128 y 64."""

    def __init__(self, p=0.2):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(N_ATRIBUTOS, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, N_CLASES))

    def forward(self, x):
        return self.red(x)


class CNN1D(nn.Module):
    """Convoluciones de 32 y 64 filtros de tamaño 3, max-pooling, dropout de 0,2 y densa de 64."""

    def __init__(self, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.cabeza = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(64 * (N_ATRIBUTOS // 2), 64), nn.ReLU(), nn.Linear(64, N_CLASES))

    def forward(self, x):
        return self.cabeza(self.conv(x.unsqueeze(1)))


class LSTM(nn.Module):
    """Una capa de 64 unidades sobre la secuencia de 16 atributos, dropout de 0,2 antes de la salida."""

    def __init__(self, p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.cabeza = nn.Sequential(nn.Dropout(p), nn.Linear(64, N_CLASES))

    def forward(self, x):
        return self.cabeza(self.lstm(x.unsqueeze(-1))[1][0][-1])


# --------------------------------------------------------------------------------------
# Top 10 con arquitectura definida en la literatura
# --------------------------------------------------------------------------------------
class RNNBiswas(nn.Module):
    """Biswas et al. (2026): SimpleRNN de 128 (secuencia) y de 64, con ReLU, normalización por lotes,
    dropout de 0,3 y densa de 128 con ReLU antes de la salida."""

    def __init__(self, p=0.3):
        super().__init__()
        self.rnn1 = nn.RNN(1, 128, nonlinearity="relu", batch_first=True)
        self.rnn2 = nn.RNN(128, 64, nonlinearity="relu", batch_first=True)
        self.cabeza = nn.Sequential(nn.BatchNorm1d(64), nn.Dropout(p), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, N_CLASES))

    def forward(self, x):
        h = self.rnn1(x.unsqueeze(-1))[0]
        return self.cabeza(self.rnn2(h)[1][-1])


class GRUBiswas(nn.Module):
    """Biswas et al. (2026): GRU de 128 (secuencia) → BN → dropout 0,3 → GRU de 64 → BN → dropout 0,3
    → densa de 128 con ReLU → dropout 0,3 → salida."""

    def __init__(self, p=0.3):
        super().__init__()
        self.gru1 = nn.GRU(1, 128, batch_first=True)
        self.norma1 = NormalizacionSecuencia(128)
        self.drop1 = nn.Dropout(p)
        self.gru2 = nn.GRU(128, 64, batch_first=True)
        self.cabeza = nn.Sequential(nn.BatchNorm1d(64), nn.Dropout(p), nn.Linear(64, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, N_CLASES))

    def forward(self, x):
        h = self.drop1(self.norma1(self.gru1(x.unsqueeze(-1))[0]))
        return self.cabeza(self.gru2(h)[1][-1])


class BiLSTMBalgoon(nn.Module):
    """Balgoon (2026): dos capas BiLSTM de 128 unidades por dirección, con dropout de 0,3;
    la salida concatena el último estado de ambas direcciones."""

    def __init__(self, p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, num_layers=2, bidirectional=True, dropout=p, batch_first=True)
        self.cabeza = nn.Sequential(nn.Dropout(p), nn.Linear(256, N_CLASES))

    def forward(self, x):
        h = self.lstm(x.unsqueeze(-1))[1][0]
        return self.cabeza(torch.cat([h[-2], h[-1]], dim=1))


class TSLTNet(nn.Module):
    """Biswas et al. (2026), tabla 1: densa de 128 con ReLU → 16 fichas de 8 → LayerNorm →
    atención multicabeza (2 cabezas, dimensión de clave 4) → promedio global → densa de 64 con ReLU →
    dropout 0,3 → salida."""

    def __init__(self, p=0.3):
        super().__init__()
        self.entrada = nn.Sequential(nn.Linear(N_ATRIBUTOS, 128), nn.ReLU())
        self.norma = nn.LayerNorm(8)
        self.atencion = nn.MultiheadAttention(8, num_heads=2, batch_first=True)
        self.cabeza = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, N_CLASES))
        self.pesos_atencion = None

    def forward(self, x):
        fichas = self.norma(self.entrada(x).view(-1, 16, 8))
        salida, pesos = self.atencion(fichas, fichas, fichas, need_weights=not self.training)
        if not self.training:
            self.pesos_atencion = pesos
        return self.cabeza(salida.mean(1))


class TabNetModelo(nn.Module):
    """Arik y Pfister (2021), con la red de la implementación de referencia pytorch-tabnet y su
    configuración por defecto (n_d = n_a = 8, 3 pasos, γ = 1,3, máscara sparsemax).
    La pérdida agrega −λ·M_loss, con λ = 10⁻³, como en esa implementación."""

    def __init__(self, lambda_dispersion=1e-3):
        super().__init__()
        from pytorch_tabnet.tab_network import TabNet
        from pytorch_tabnet.utils import create_group_matrix

        # Sin grupos de atributos: la matriz identidad, como hace TabNetClassifier por defecto
        self.red = TabNet(N_ATRIBUTOS, N_CLASES, group_attention_matrix=create_group_matrix([], N_ATRIBUTOS))
        self.lambda_dispersion = lambda_dispersion
        self.perdida_extra = None

    def forward(self, x):
        salida, m_loss = self.red(x)
        self.perdida_extra = -self.lambda_dispersion * m_loss
        return salida


class CapaKAN(nn.Module):
    """Capa KAN con bases gaussianas fijas (FastKAN): y_j = Σ_i Σ_k w_ijk φ_k(x_i) + b_j,
    con φ_k(x) = exp(-((x - μ_k) / h)²) y centros μ_k uniformes en [-5, 5]."""

    def __init__(self, n_entrada, n_salida, K, rango=(-5.0, 5.0)):
        super().__init__()
        self.register_buffer("centros", torch.linspace(rango[0], rango[1], K))
        self.h = (rango[1] - rango[0]) / (K - 1)
        self.rango = rango
        self.pesos = nn.Parameter(torch.empty(n_entrada * K, n_salida))
        nn.init.trunc_normal_(self.pesos, std=0.1)
        self.sesgo = nn.Parameter(torch.zeros(n_salida))
        self.n_entrada, self.n_salida, self.K = n_entrada, n_salida, K

    def forward(self, x):
        x = x.clamp(*self.rango)
        base = torch.exp(-(((x.unsqueeze(-1) - self.centros) / self.h) ** 2))
        return base.flatten(1) @ self.pesos + self.sesgo


class KANBakiri(nn.Module):
    """Clasificador de ataques de H-MicroKAN (Bakiri et al., 2026), aplicado a las 5 clases:
    compuerta d → r → d con sigmoide, KAN d → 28 (K = 5) → BN → GELU → dropout 0,25 →
    KAN 28 → 16 (K = 5) → BN → GELU → dropout 0,20 → lineal."""

    def __init__(self, r=4):
        super().__init__()
        self.compuerta = nn.Sequential(nn.Linear(N_ATRIBUTOS, r), nn.ReLU(), nn.Linear(r, N_ATRIBUTOS), nn.Sigmoid())
        self.cuerpo = nn.Sequential(CapaKAN(N_ATRIBUTOS, 28, 5), nn.BatchNorm1d(28), nn.GELU(), nn.Dropout(0.25),
                                    CapaKAN(28, 16, 5), nn.BatchNorm1d(16), nn.GELU(), nn.Dropout(0.20), nn.Linear(16, N_CLASES))

    def forward(self, x):
        return self.cuerpo(x * self.compuerta(x))


class FTTransformerModelo(nn.Module):
    """FT-Transformer de Gorishniy et al. (2021), con la implementación oficial de los autores
    (rtdl_revisiting_models) y su configuración por defecto."""

    def __init__(self):
        super().__init__()
        from rtdl_revisiting_models import FTTransformer

        self.red = FTTransformer(n_cont_features=N_ATRIBUTOS, cat_cardinalities=[], d_out=N_CLASES, **FTTransformer.get_default_kwargs())

    def forward(self, x):
        return self.red(x, None)


# --------------------------------------------------------------------------------------
# Modelos sin arquitectura definida: espacio de la sección 14 (3 bloques, anchos y dropout)
# --------------------------------------------------------------------------------------
def _desempacar(arquitectura):
    w1, w2, w3, b1, b2, b3 = arquitectura
    return (w1, w2, w3), (b1, b2, b3)


class ResNetBusqueda(nn.Module):
    """Red residual densa (He et al., 2016) de 3 bloques: Linear → BN → ReLU → Linear → BN, más un
    atajo, identidad o proyección lineal si cambia el ancho, y ReLU; dropout tras el bloque i si bi = 1."""

    def __init__(self, arquitectura):
        super().__init__()
        anchos, patron = _desempacar(arquitectura)
        self.bloques, self.atajos, self.drops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        entrada = N_ATRIBUTOS
        for ancho, b in zip(anchos, patron):
            self.bloques.append(nn.Sequential(nn.Linear(entrada, ancho), nn.BatchNorm1d(ancho), nn.ReLU(), nn.Linear(ancho, ancho), nn.BatchNorm1d(ancho)))
            self.atajos.append(nn.Identity() if entrada == ancho else nn.Linear(entrada, ancho, bias=False))
            self.drops.append(nn.Dropout(TASA_DROPOUT_BUSQUEDA) if b else nn.Identity())
            entrada = ancho
        self.salida = nn.Linear(entrada, N_CLASES)

    def forward(self, x):
        for bloque, atajo, drop in zip(self.bloques, self.atajos, self.drops):
            x = drop(torch.relu(bloque(x) + atajo(x)))
        return self.salida(x)


class AutoencoderBusqueda(nn.Module):
    """Autoencoder supervisado: codificador 16 → w1 → w2 → w3, decodificador espejo w3 → w2 → w1 → 16
    y cabeza softmax sobre el código. Pérdida: entropía cruzada + λ·MSE de reconstrucción, con λ = 1.
    El dropout del bloque i se aplica tras la capa i del codificador y tras su espejo en el decodificador."""

    def __init__(self, arquitectura, lambda_reconstruccion=1.0):
        super().__init__()
        anchos, patron = _desempacar(arquitectura)
        cod, entrada = [], N_ATRIBUTOS
        for ancho, b in zip(anchos, patron):
            cod += [nn.Linear(entrada, ancho), nn.ReLU()] + ([nn.Dropout(TASA_DROPOUT_BUSQUEDA)] if b else [])
            entrada = ancho
        dec = []
        destinos = list(anchos[:-1][::-1]) + [N_ATRIBUTOS]
        espejo = list(patron[:-1][::-1])
        for i, ancho in enumerate(destinos):
            dec.append(nn.Linear(entrada, ancho))
            if i < len(destinos) - 1:
                dec.append(nn.ReLU())
                if espejo[i]:
                    dec.append(nn.Dropout(TASA_DROPOUT_BUSQUEDA))
            entrada = ancho
        self.codificador, self.decodificador = nn.Sequential(*cod), nn.Sequential(*dec)
        self.salida = nn.Linear(anchos[-1], N_CLASES)
        self.lambda_reconstruccion = lambda_reconstruccion
        self.perdida_extra = None

    def forward(self, x):
        codigo = self.codificador(x)
        if self.training:
            self.perdida_extra = self.lambda_reconstruccion * torch.mean((self.decodificador(codigo) - x) ** 2)
        return self.salida(codigo)


class DBNBusqueda(nn.Module):
    """Red de creencia profunda (Hinton et al., 2006): 3 RBM preentrenadas capa a capa con divergencia
    contrastiva (CD-1), la primera gaussiana-bernoulli, y ajuste fino supervisado con sigmoides;
    dropout tras la capa i si bi = 1."""

    EPOCAS_RBM = 3
    TASA_RBM = (0.002, 0.01, 0.01)
    LOTE_RBM = 256

    def __init__(self, arquitectura):
        super().__init__()
        anchos, patron = _desempacar(arquitectura)
        entradas = (N_ATRIBUTOS,) + anchos[:-1]
        self.capas = nn.ModuleList([nn.Linear(a, b) for a, b in zip(entradas, anchos)])
        for capa in self.capas:
            nn.init.normal_(capa.weight, std=0.01)
            nn.init.zeros_(capa.bias)
        self.drops = nn.ModuleList([nn.Dropout(TASA_DROPOUT_BUSQUEDA) if b else nn.Identity() for b in patron])
        self.salida = nn.Linear(anchos[-1], N_CLASES)

    def forward(self, x):
        for capa, drop in zip(self.capas, self.drops):
            x = drop(torch.sigmoid(capa(x)))
        return self.salida(x)

    @torch.no_grad()
    def preentrenar(self, X, generador):
        visible = X
        for i, capa in enumerate(self.capas):
            W, bh = capa.weight, capa.bias
            bv = torch.zeros(W.shape[1])
            tasa = self.TASA_RBM[i]
            for _ in range(self.EPOCAS_RBM):
                for lote in torch.randperm(len(visible), generator=generador).split(self.LOTE_RBM):
                    v0 = visible[lote]
                    ph0 = torch.sigmoid(v0 @ W.T + bh)
                    h0 = torch.bernoulli(ph0, generator=generador)
                    media_v = h0 @ W + bv
                    v1 = media_v if i == 0 else torch.sigmoid(media_v)
                    ph1 = torch.sigmoid(v1 @ W.T + bh)
                    n = len(lote)
                    W += tasa * (ph0.T @ v0 - ph1.T @ v1) / n
                    bh += tasa * (ph0 - ph1).mean(0)
                    bv += tasa * (v0 - v1).mean(0)
            visible = torch.sigmoid(visible @ W.T + bh)


class CNNLSTMBusqueda(nn.Module):
    """CNN-LSTM: convolución 1D de w1 filtros (tamaño 3) → convolución de w2 filtros → max-pooling de 2 →
    LSTM de w3 unidades sobre los 8 pasos resultantes → salida. Correspondencia aproximada de la sección 14.2:
    bloques 1 y 2 convolucionales y bloque 3 recurrente, con dropout tras el bloque i si bi = 1."""

    def __init__(self, arquitectura):
        super().__init__()
        (w1, w2, w3), (b1, b2, b3) = _desempacar(arquitectura)
        drop = lambda b: nn.Dropout(TASA_DROPOUT_BUSQUEDA) if b else nn.Identity()
        self.conv = nn.Sequential(nn.Conv1d(1, w1, 3, padding=1), nn.ReLU(), drop(b1), nn.Conv1d(w1, w2, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), drop(b2))
        self.lstm = nn.LSTM(w2, w3, batch_first=True)
        self.cabeza = nn.Sequential(drop(b3), nn.Linear(w3, N_CLASES))

    def forward(self, x):
        z = self.conv(x.unsqueeze(1)).transpose(1, 2)
        return self.cabeza(self.lstm(z)[1][0][-1])


# --------------------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------------------
CATALOGO = {
    "DNN": {"clase": DNN, "familia": "Perceptrón multicapa", "grupo": "Base E1", "busqueda": False},
    "CNN 1D": {"clase": CNN1D, "familia": "Convolucional", "grupo": "Base E1", "busqueda": False},
    "LSTM": {"clase": LSTM, "familia": "Recurrente", "grupo": "Base E1", "busqueda": False},
    "RNN": {"clase": RNNBiswas, "familia": "Recurrente", "grupo": "Top 10", "busqueda": False},
    "GRU": {"clase": GRUBiswas, "familia": "Recurrente", "grupo": "Top 10", "busqueda": False},
    "BiLSTM": {"clase": BiLSTMBalgoon, "familia": "Recurrente", "grupo": "Top 10", "busqueda": False},
    "Transformer": {"clase": TSLTNet, "familia": "Atención", "grupo": "Top 10", "busqueda": False},
    "TabNet": {"clase": TabNetModelo, "familia": "Atención", "grupo": "Top 10", "busqueda": False},
    "KAN": {"clase": KANBakiri, "familia": "Kolmogorov-Arnold", "grupo": "Top 10", "busqueda": False},
    "FT-Transformer": {"clase": FTTransformerModelo, "familia": "Atención", "grupo": "Candidato", "busqueda": False},
    "ResNet": {"clase": ResNetBusqueda, "familia": "Residual", "grupo": "Top 10", "busqueda": True},
    "Autoencoder": {"clase": AutoencoderBusqueda, "familia": "Codificador-decodificador", "grupo": "Top 10", "busqueda": True},
    "DBN": {"clase": DBNBusqueda, "familia": "Generativa preentrenada", "grupo": "Top 10", "busqueda": True},
    "CNN-LSTM": {"clase": CNNLSTMBusqueda, "familia": "Híbrida", "grupo": "Top 10", "busqueda": True},
}


def construir(nombre, arquitectura=None):
    ficha = CATALOGO[nombre]
    if ficha["busqueda"]:
        return ficha["clase"](tuple(arquitectura) if arquitectura is not None else REFERENCIA)
    return ficha["clase"]()


def parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Conteo de multiplicaciones-acumulaciones (MAC) por muestra; FLOPs ≈ 2 × MAC
# --------------------------------------------------------------------------------------
def _mac_recurrente(m, pasos):
    puertas = {"RNN": 1, "GRU": 3, "LSTM": 4}[type(m).__name__]
    direcciones = 2 if m.bidirectional else 1
    total, entrada = 0, m.input_size
    for _ in range(m.num_layers):
        total += direcciones * pasos * puertas * (entrada + m.hidden_size) * m.hidden_size
        entrada = m.hidden_size * direcciones
    return total


def contar_mac(modelo):
    """MAC por muestra en una pasada hacia adelante, con ganchos sobre las capas con pesos."""
    modelo = modelo.eval()
    total = [0]
    ganchos = []

    def gancho(m, entrada, salida):
        x = entrada[0]
        if isinstance(m, nn.Linear):
            filas = x.numel() // x.shape[-1] // x.shape[0]
            total[0] += filas * m.in_features * m.out_features
        elif isinstance(m, nn.Conv1d):
            total[0] += salida.shape[-1] * m.out_channels * m.in_channels * m.kernel_size[0] // m.groups
        elif isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
            total[0] += _mac_recurrente(m, x.shape[1])
        elif isinstance(m, nn.MultiheadAttention):
            pasos, d = x.shape[1], m.embed_dim
            total[0] += 4 * pasos * d * d + 2 * pasos * pasos * d
        elif isinstance(m, CapaKAN):
            total[0] += m.n_entrada * m.K * m.n_salida
        elif type(m).__name__ == "MultiheadAttention" and hasattr(m, "W_q"):
            pasos, d = x.shape[1], x.shape[2]
            total[0] += 2 * pasos * pasos * d

    for m in modelo.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.RNN, nn.GRU, nn.LSTM, nn.MultiheadAttention, CapaKAN)) or (type(m).__name__ == "MultiheadAttention" and hasattr(m, "W_q")):
            ganchos.append(m.register_forward_hook(gancho))
    with torch.no_grad():
        modelo(torch.zeros(2, N_ATRIBUTOS))
    for g in ganchos:
        g.remove()
    return total[0]
