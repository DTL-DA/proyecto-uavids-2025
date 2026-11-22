"""Modelo original del Entregable 3: compuerta por flujo entre un experto de umbrales y un experto suave.

Motivación (limitación del paper guía, registrada en el E2): en UAVIDS-2025 los errores del sesgo de árbol
(umbrales por eje, LightGBM) y del sesgo neuronal (funciones suaves, MLP y redes del Top 10) son
complementarios. LightGBM confunde menos Blackhole con Wormhole y las redes reconocen mejor Sybil. Una regla de
fusión fija (promedio simple, pesos por error global, *stacking*) usa el mismo peso para todos los flujos. Aquí
una red de compuerta decide, flujo por flujo, cuánto pesa cada experto.

Formulación. Para un flujo x en R^d (d = 16 atributos transformados y estandarizados) y K = 5 clases:

    Experto de umbrales (E_u):  s_{j,t}(x) = sigma((x_j - theta_{j,t}) / tau_j),  t = 1..T
                                z_u(x) = MLP_u(s(x))                 (escalones suaves con umbrales aprendibles)
    Experto suave (E_s):        z_s(x) = MLP_s(x)
    Compuerta (G):              g(x) = softmax(MLP_g(x)) en el símplex de 2 expertos
    Mezcla:                     p(y | x) = g_u(x) softmax(z_u(x))_y + g_s(x) softmax(z_s(x))_y

El módulo devuelve log p(y | x). Como sum_y p(y | x) = 1, la entropía cruzada de PyTorch sobre esa salida es
exactamente la log-verosimilitud negativa de la mezcla, y el modelo entra sin cambios al motor de entrenamiento
del E2 (misma pérdida ponderada, optimizador, parada temprana y recorte del gradiente).

Componentes propios (sujetos a ablación):
  1. Codificador de escalones suaves con umbrales y temperaturas aprendibles, con tau_j >= tau_min, lo que acota
     la sensibilidad del experto de umbrales (ver ``cota_lipschitz``).
  2. Compuerta por flujo entre dos expertos de sesgo distinto (frente al promedio simple o pesos globales).
  3. Regularizador de balance de la compuerta: KL(media de g en el lote || uniforme), que evita que la
     compuerta colapse en un solo experto.

Fuentes de las piezas (reimplementadas desde su formulación, sin código ajeno):
  - Mezcla de expertos con compuerta: Jacobs, Jordan, Nowlan y Hinton (1991), Neural Computation 3(1):79-87.
  - Codificación por tramos de atributos numéricos: Gorishniy, Rubachev y Babenko (2022), NeurIPS.
  - Regularización de balance de la compuerta: Shazeer et al. (2017), ICLR.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

N_ATRIBUTOS = 16
N_CLASES = 5


def _mlp(entrada: int, ocultas: tuple[int, ...], salida: int) -> nn.Sequential:
    capas, previa = [], entrada
    for ancho in ocultas:
        capas += [nn.Linear(previa, ancho), nn.ReLU()]
        previa = ancho
    capas.append(nn.Linear(previa, salida))
    return nn.Sequential(*capas)


class EscalonesSuaves(nn.Module):
    """Codificador de escalones suaves por atributo: s_{j,t} = sigma((x_j - theta_{j,t}) / tau_j).

    Los umbrales theta (d x T) se inicializan en cuantiles del entrenamiento con ``inicializar`` y se aprenden.
    La temperatura es tau_j = tau_min + softplus(rho_j) >= tau_min, de modo que cada escalón tiene pendiente
    máxima 1 / (4 tau_j) y el codificador es Lipschitz con constante sqrt(T) / (4 tau_min) en norma 2.
    """

    def __init__(self, d: int, T: int, tau_min: float = 0.05, tau_inicial: float = 0.25):
        super().__init__()
        if tau_inicial <= tau_min:
            raise ValueError("tau_inicial debe ser mayor que tau_min")
        self.d, self.T, self.tau_min = d, T, tau_min
        self.theta = nn.Parameter(torch.linspace(-2.0, 2.0, T).repeat(d, 1))
        rho0 = math.log(math.expm1(tau_inicial - tau_min))  # softplus^{-1}(tau_inicial - tau_min)
        self.rho = nn.Parameter(torch.full((d, 1), rho0))

    def tau(self) -> torch.Tensor:
        return self.tau_min + F.softplus(self.rho)

    @torch.no_grad()
    def inicializar(self, X: torch.Tensor) -> None:
        """Umbrales en los cuantiles t / (T + 1) de cada atributo; si hay empates (masas en un valor), los
        umbrales repetidos se reparten de forma uniforme entre el mínimo y el máximo del atributo."""
        niveles = torch.arange(1, self.T + 1, dtype=X.dtype) / (self.T + 1)
        q = torch.quantile(X, niveles, dim=0).T  # d x T
        for j in range(self.d):
            unicos = torch.unique(q[j])
            if len(unicos) < self.T:
                relleno = torch.linspace(float(X[:, j].min()), float(X[:, j].max()), self.T + 2)[1:-1]
                unicos = torch.unique(torch.cat([unicos, relleno]))
                indices = torch.linspace(0, len(unicos) - 1, self.T).round().long()
                unicos = unicos[indices]
            self.theta[j] = unicos.sort().values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.d:
            raise RuntimeError(f"Se esperaba una matriz (n, {self.d}); llegó {tuple(x.shape)}")
        z = (x.unsqueeze(-1) - self.theta) / self.tau()  # n x d x T
        return torch.sigmoid(z).flatten(1)


class ModeloOriginal(nn.Module):
    """Compuerta por flujo entre un experto de umbrales y un experto suave (salida: log-probabilidades).

    Parámetros
    ----------
    d, n_clases : dimensión de entrada y número de clases.
    T : número de umbrales por atributo del experto de umbrales.
    ocultas_umbral, ocultas_suave, ocultas_compuerta : anchos de las capas ocultas de cada MLP.
    tau_min : temperatura mínima de los escalones (controla la cota de Lipschitz).
    gamma : peso del regularizador de balance de la compuerta (0 lo desactiva).
    compuerta : "aprendida" (modelo completo), "promedio" (pesos fijos 1/2, ablación) o "umbral"/"suave"
        (un solo experto, ablación).
    entrada_compuerta : "x" (la compuerta ve solo el flujo) o "x+expertos" (ve además las distribuciones que
        predice cada experto, lo que le permite aprender en qué clase confiar en cada experto).
    codificador : "escalones" (modelo completo) o "ninguno" (ablación A2: el experto de umbrales se reemplaza por
        un MLP sobre x con capas ocultas (128, 64), de capacidad comparable).
    semilla : si se da, fija la inicialización de los pesos.
    """

    def __init__(self, d: int = N_ATRIBUTOS, n_clases: int = N_CLASES, T: int = 8, ocultas_umbral=(64,), ocultas_suave=(64, 64),
                 ocultas_compuerta=(32,), tau_min: float = 0.05, gamma: float = 0.01, compuerta: str = "aprendida",
                 entrada_compuerta: str = "x", codificador: str = "escalones", semilla: int | None = None):
        super().__init__()
        if compuerta not in ("aprendida", "promedio", "umbral", "suave"):
            raise ValueError(f"compuerta desconocida: {compuerta}")
        if entrada_compuerta not in ("x", "x+expertos"):
            raise ValueError(f"entrada de la compuerta desconocida: {entrada_compuerta}")
        if codificador not in ("escalones", "ninguno"):
            raise ValueError(f"codificador desconocido: {codificador}")
        if semilla is not None:
            torch.manual_seed(semilla)
        self.d, self.n_clases, self.gamma, self.modo, self.entrada_compuerta = d, n_clases, gamma, compuerta, entrada_compuerta
        if codificador == "escalones":
            self.codificador = EscalonesSuaves(d, T, tau_min)
            self.experto_umbral = _mlp(d * T, tuple(ocultas_umbral), n_clases)
        else:
            self.codificador = None
            self.experto_umbral = _mlp(d, (128, 64), n_clases)
        self.experto_suave = _mlp(d, tuple(ocultas_suave), n_clases)
        ancho_entrada = d + (2 * n_clases if entrada_compuerta == "x+expertos" else 0)
        self.red_compuerta = _mlp(ancho_entrada, tuple(ocultas_compuerta), 2)
        self.perdida_extra = None
        self._ultima_compuerta = None

    # -- gancho del motor del E2: se llama una vez con las filas de entrenamiento antes de optimizar --
    @torch.no_grad()
    def preentrenar(self, X: torch.Tensor, generador=None) -> None:
        if self.codificador is not None:
            self.codificador.inicializar(X)

    def compuerta(self, x: torch.Tensor, log_u: torch.Tensor | None = None, log_s: torch.Tensor | None = None) -> torch.Tensor:
        """Pesos (g_u, g_s) de cada flujo, en el símplex."""
        if self.modo == "aprendida":
            if self.entrada_compuerta == "x+expertos":
                if log_u is None:
                    log_u, log_s = self.expertos(x)
                x = torch.cat([x, log_u.exp(), log_s.exp()], dim=1)
            return torch.softmax(self.red_compuerta(x), dim=-1)
        fijo = {"promedio": (0.5, 0.5), "umbral": (1.0, 0.0), "suave": (0.0, 1.0)}[self.modo]
        return torch.tensor(fijo, dtype=x.dtype).expand(x.shape[0], 2)

    def expertos(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probabilidades de cada experto, (n, K) cada una."""
        entrada_u = self.codificador(x) if self.codificador is not None else x
        return (F.log_softmax(self.experto_umbral(entrada_u), dim=-1),
                F.log_softmax(self.experto_suave(x), dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.d:
            raise RuntimeError(f"Se esperaba una matriz (n, {self.d}); llegó {tuple(x.shape)}")
        log_u, log_s = self.expertos(x)
        g = self.compuerta(x, log_u, log_s)
        self._ultima_compuerta = g.detach()
        log_g = torch.log(g.clamp_min(1e-12))
        salida = torch.logsumexp(torch.stack([log_g[:, :1] + log_u, log_g[:, 1:] + log_s]), dim=0)  # log p(y|x)
        if self.training and self.modo == "aprendida" and self.gamma > 0:
            media = g.mean(0)
            self.perdida_extra = self.gamma * torch.sum(media * torch.log(media.clamp_min(1e-12) * 2.0))  # KL(media || uniforme)
        else:
            self.perdida_extra = None
        return salida

    def perdida_cooperativa(self, x: torch.Tensor, y: torch.Tensor, criterio, lambda_expertos: float = 1.0) -> torch.Tensor:
        """Esquema de entrenamiento cooperativo (componente propio).

        L = criterio(log p_mezcla con expertos detenidos, y)              -> solo entrena la compuerta
          + lambda_expertos * [criterio(log p_u, y) + criterio(log p_s, y)]  -> cada experto sigue siendo un
                                                                               clasificador completo
          + gamma * KL(media de g || uniforme)                             -> balance de la compuerta
        El entrenamiento conjunto clásico (Jacobs et al., 1991) usa solo la log-verosimilitud de la mezcla y hace que
        cada experto se especialice en las clases que la compuerta le asigna; en el diseño (d02) eso empeoró el F1.
        """
        log_u, log_s = self.expertos(x)
        if self.modo != "aprendida":
            return criterio(self.forward(x), y)
        g = self.compuerta(x, log_u.detach(), log_s.detach())
        log_g = torch.log(g.clamp_min(1e-12))
        log_mezcla = torch.logsumexp(torch.stack([log_g[:, :1] + log_u.detach(), log_g[:, 1:] + log_s.detach()]), dim=0)
        total = criterio(log_mezcla, y) + lambda_expertos * (criterio(log_u, y) + criterio(log_s, y))
        if self.gamma > 0:
            media = g.mean(0)
            total = total + self.gamma * torch.sum(media * torch.log(media.clamp_min(1e-12) * 2.0))
        return total

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x).exp()

    # -- propiedad demostrada: cota de Lipschitz de las probabilidades respecto de la entrada (norma 2) --
    @torch.no_grad()
    def cota_lipschitz(self) -> float:
        """K tal que ||p(x) - p(x')||_2 <= K ||x - x'||_2 para todo x, x' (demostración en el libro, cap. 5).

        K = sqrt(2) * L_g + max(L_u, L_s), con
            L_u = (1/2) * L(MLP_u) * sqrt(T) / (4 tau_min_efectiva),   L_s = (1/2) * L(MLP_s),
            L_g = (1/2) * L(MLP_g) * (1 + L_u + L_s) si la compuerta ve las distribuciones de los expertos,
                  (1/2) * L(MLP_g) si ve solo x,
        donde L(MLP) es el producto de las normas espectrales de sus matrices (ReLU es 1-Lipschitz) y 1/2 es la
        constante de Lipschitz del softmax en norma 2. Con compuerta fija, el término de la compuerta es 0.
        """
        def lip(red):
            return math.prod(torch.linalg.matrix_norm(m.weight, ord=2).item() for m in red if isinstance(m, nn.Linear))

        if self.codificador is not None:
            l_cod = math.sqrt(self.codificador.T) / (4 * float(self.codificador.tau().min()))
        else:
            l_cod = 1.0
        l_u = 0.5 * lip(self.experto_umbral) * l_cod
        l_s = 0.5 * lip(self.experto_suave)
        l_g = 0.0
        if self.modo == "aprendida":
            l_g = 0.5 * lip(self.red_compuerta) * ((1 + l_u + l_s) if self.entrada_compuerta == "x+expertos" else 1.0)
        if self.modo == "umbral":
            return l_u
        if self.modo == "suave":
            return l_s
        return math.sqrt(2) * l_g + max(l_u, l_s)
