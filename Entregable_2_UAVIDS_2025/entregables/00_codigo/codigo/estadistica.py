"""Métricas y pruebas estadísticas del benchmark (tablas A, B y C del enunciado).

Las funciones de potencia, TOST, DeLong y Nemenyi provienen del Entregable 1; aquí se agregan
Friedman con Iman y Davenport, Wilcoxon-Holm, la prueba bayesiana de rangos con signo con ROPE,
la t corregida de Nadeau y Bengio, la F de 5 × 2 cv, la esfericidad de Mauchly, el Model Confidence
Set de Hansen, el bootstrap por bloques y las pruebas de calibración.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import rankdata, studentized_range
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from statsmodels.stats.multitest import multipletests

ALFA = 0.05
DELTA = 0.01
POTENCIA_OBJETIVO = 0.80
K = 5
PAR_RUTEO = (0, 4)  # Blackhole y Wormhole


# ------------------------------------------------------------------ métricas por corrida
def f1_macro_rapido(verdad, prediccion, k=K):
    """F1 macro sobre las clases presentes en `verdad`: en un bloque o segmento sin alguna clase, esa clase no entra al promedio."""
    confusion = np.bincount(k * verdad + prediccion, minlength=k * k).reshape(k, k)
    presentes = confusion.sum(1) > 0
    return float(np.mean((2 * np.diag(confusion) / np.maximum(confusion.sum(0) + confusion.sum(1), 1))[presentes]))


def confusion(verdad, prediccion, k=K):
    return np.bincount(k * verdad + prediccion, minlength=k * k).reshape(k, k)


def ece(prob, verdad, bins=15, adaptativo=False):
    """Error de calibración esperado sobre la confianza de la clase predicha; con `adaptativo`, bins de igual tamaño."""
    confianza, acierto = prob.max(1), (prob.argmax(1) == verdad).astype(float)
    if adaptativo:
        orden = np.argsort(confianza)
        grupos = np.array_split(orden, bins)
    else:
        cortes = np.linspace(0, 1, bins + 1)
        indice = np.clip(np.digitize(confianza, cortes[1:-1]), 0, bins - 1)
        grupos = [np.flatnonzero(indice == b) for b in range(bins)]
    return float(sum(len(g) / len(verdad) * abs(acierto[g].mean() - confianza[g].mean()) for g in grupos if len(g)))


def brier(prob, verdad):
    return float(np.mean(((prob - np.eye(prob.shape[1])[verdad]) ** 2).sum(1)))


def tasa_ruteo(verdad, prediccion):
    """Confusión del par Blackhole–Wormhole: (n_B→W + n_W→B) / (n_B + n_W)."""
    b, w = PAR_RUTEO
    m = (verdad == b) | (verdad == w)
    return float((((verdad == b) & (prediccion == w)) | ((verdad == w) & (prediccion == b))).sum() / max(m.sum(), 1))


def metricas(prob, verdad):
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(verdad, pred, average=None, labels=range(K), zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(verdad, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(verdad, pred, average="weighted", zero_division=0)
    uno = np.eye(K)[verdad]
    prob_c = np.clip(prob, 1e-7, 1)
    return {"F1 macro": fm, "Accuracy": accuracy_score(verdad, pred), "Balanced accuracy": balanced_accuracy_score(verdad, pred), "Precision macro": pm, "Recall macro": rm,
            "F1 ponderado": fw, "Precision ponderada": pw, "Recall ponderado": rw, "AUC-ROC macro": roc_auc_score(verdad, prob, multi_class="ovr", average="macro"),
            "AUC-PR macro": float(np.mean([average_precision_score(uno[:, c], prob[:, c]) for c in range(K)])), "MCC": matthews_corrcoef(verdad, pred),
            "Log-loss": log_loss(verdad, prob_c / prob_c.sum(1, keepdims=True), labels=range(K)), "Brier": brier(prob, verdad), "ECE": ece(prob, verdad), "ECE adaptativo": ece(prob, verdad, adaptativo=True),
            "Confusión Blackhole–Wormhole": tasa_ruteo(verdad, pred), **{f"F1 {c}": f[i] for i, c in enumerate(("Blackhole", "Flooding", "Normal", "Sybil", "Wormhole"))}}


def f1_por_bloque(verdad, prediccion, bloques):
    return {int(b): f1_macro_rapido(verdad[bloques == b], prediccion[bloques == b]) for b in np.unique(bloques)}


# ------------------------------------------------------------------ bootstrap por bloques
def bootstrap_bloques(estadistico, bloques, repeticiones=2000, semilla=42):
    """Remuestrea los bloques (unidades independientes) con reemplazo; `estadistico` recibe los índices de filas."""
    unicos = np.unique(bloques)
    filas = [np.flatnonzero(bloques == b) for b in unicos]
    rng = np.random.default_rng(semilla)
    valores = np.empty(repeticiones)
    for i in range(repeticiones):
        elegidos = rng.integers(0, len(unicos), len(unicos))
        valores[i] = estadistico(np.concatenate([filas[j] for j in elegidos]))
    return valores


def ic_percentil(valores, nivel=0.95):
    a = (1 - nivel) / 2
    return tuple(np.nanpercentile(valores, [100 * a, 100 * (1 - a)]))


# ------------------------------------------------------------------ potencia (Entregable 1)
Z_ALFA, Z_POTENCIA = stats.norm.ppf(1 - ALFA / 2), stats.norm.ppf(POTENCIA_OBJETIVO)


def potencia_pareada(desviacion, n, efecto, alfa=ALFA):
    if desviacion <= 0:
        return 1.0
    gl = n - 1
    nc = efecto / (desviacion / np.sqrt(n))
    critico = stats.t.ppf(1 - alfa / 2, gl)
    valor = stats.nct.sf(critico, gl, nc) + stats.nct.cdf(-critico, gl, nc)
    return 1.0 if not np.isfinite(valor) else float(min(valor, 1.0))


def minima_detectable(desviacion, n, alfa=ALFA):
    if desviacion <= 0:
        return 0.0
    return brentq(lambda e: potencia_pareada(desviacion, n, e, alfa) - POTENCIA_OBJETIVO, 1e-9, 10 * desviacion / np.sqrt(n))


def ejecuciones_necesarias(sigma_d, delta=DELTA, comparaciones=10):
    z = stats.norm.ppf(1 - ALFA / comparaciones / 2) + Z_POTENCIA
    return int(np.ceil((z * sigma_d / delta) ** 2))


# ------------------------------------------------------------------ comparación pareada sobre semillas
def bayes_rangos_signo(d, rope=DELTA, muestras=20000, semilla=42):
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2017), con proceso de Dirichlet y una pseudoobservación en 0."""
    x = np.concatenate(([0.0], np.asarray(d, float)))
    rng = np.random.default_rng(semilla)
    w = rng.dirichlet(np.ones(len(x)), muestras)
    suma = x[:, None] + x[None, :]
    izquierda, derecha = (suma < -2 * rope).astype(float), (suma > 2 * rope).astype(float)
    pi = np.einsum("ki,ij,kj->k", w, izquierda, w)
    pd_ = np.einsum("ki,ij,kj->k", w, derecha, w)
    pr = 1 - pi - pd_
    ganador = np.argmax(np.stack([pi, pr, pd_]), 0)
    return {"P(A < B)": float((ganador == 0).mean()), "P(ROPE)": float((ganador == 1).mean()), "P(A > B)": float((ganador == 2).mean())}


def comparar_par(a, b, delta=DELTA, repeticiones=10000, semilla=42):
    """Diferencia a − b sobre semillas pareadas: t, bootstrap, Wilcoxon, TOST, bayesiana y tamaños de efecto."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n, media, sd = len(d), d.mean(), d.std(ddof=1)
    error = sd / np.sqrt(n) if sd > 0 else 0.0
    margen = stats.t.ppf(0.975, n - 1) * error
    p_shapiro = stats.shapiro(d).pvalue if sd > 0 and n >= 3 else 1.0
    p_t = stats.ttest_rel(a, b).pvalue if sd > 0 else (1.0 if media == 0 else 0.0)
    p_w = stats.wilcoxon(d, alternative="two-sided", method="exact" if n <= 25 else "auto").pvalue if np.any(d != 0) else 1.0
    rng = np.random.default_rng(semilla)
    medias = d[rng.integers(0, n, (repeticiones, n))].mean(1)
    centradas = (d - media)[rng.integers(0, n, (repeticiones, n))].mean(1)
    p_boot = float((1 + (np.abs(centradas) >= abs(media)).sum()) / (repeticiones + 1))
    rangos = rankdata(np.abs(d[d != 0]))
    r_biserial = (rangos[d[d != 0] > 0].sum() - rangos[d[d != 0] < 0].sum()) / rangos.sum() if rangos.size else 0.0
    p_tost = max(stats.t.sf((media + delta) / error, n - 1), stats.t.cdf((media - delta) / error, n - 1)) if sd > 0 else float(abs(media) >= delta)
    d_cohen = media / sd if sd > 0 else 0.0
    ic_d = (d_cohen - stats.t.ppf(0.975, n - 1) * np.sqrt(1 / n + d_cohen ** 2 / (2 * n)), d_cohen + stats.t.ppf(0.975, n - 1) * np.sqrt(1 / n + d_cohen ** 2 / (2 * n)))
    return {"n": n, "Diferencia media": media, "DE de la diferencia": sd, "IC 95 % t": (media - margen, media + margen), "IC 95 % bootstrap": tuple(np.percentile(medias, [2.5, 97.5])),
            "Shapiro-Wilk p": p_shapiro, "p t pareada": p_t, "p bootstrap": p_boot, "p Wilcoxon": p_w, "p TOST": p_tost, "d de Cohen pareada": d_cohen, "IC 95 % de d": ic_d,
            "r biserial": r_biserial, "δ de Cliff": float(np.sign(np.subtract.outer(a, b)).mean()), "P. de superioridad": float((d > 0).mean() + 0.5 * (d == 0).mean()),
            "Potencia para δ": potencia_pareada(sd, n, delta), "Mínima detectable": minima_detectable(sd, n), **bayes_rangos_signo(d, delta)}


def decision(r, p_param_ajustado, p_wilcoxon_ajustado, delta=DELTA):
    """Regla del preregistro: significativa y relevante, significativa sin relevancia, equivalente o no concluyente."""
    ic = r["IC 95 % t"] if r["Shapiro-Wilk p"] >= ALFA else r["IC 95 % bootstrap"]
    significativa = p_param_ajustado < ALFA and p_wilcoxon_ajustado < ALFA and (ic[0] > 0 or ic[1] < 0)
    if significativa and abs(r["Diferencia media"]) > delta:
        return "Significativa y relevante"
    if significativa:
        return "Significativa sin relevancia"
    if r["p TOST"] < ALFA:
        return "Equivalente"
    return "No concluyente"


def holm(p):
    return multipletests(np.asarray(p, float), alpha=ALFA, method="holm")[1]


# ------------------------------------------------------------------ comparaciones múltiples
def friedman(M):
    """M: semillas × modelos. Friedman, corrección de Iman y Davenport y W de Kendall."""
    n, k = M.shape
    chi2, p = stats.friedmanchisquare(*M.T)
    f_id = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) - chi2 > 0 else np.inf
    p_id = stats.f.sf(f_id, k - 1, (k - 1) * (n - 1))
    return {"chi2": chi2, "p": p, "F Iman-Davenport": f_id, "p Iman-Davenport": p_id, "W de Kendall": chi2 / (n * (k - 1))}


def rangos_medios(M):
    return np.apply_along_axis(lambda f: rankdata(-f), 1, M).mean(0)


def nemenyi(M):
    n, k = M.shape
    rangos = rangos_medios(M)
    escala = np.sqrt(k * (k + 1) / (6 * n))
    p = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            p[i, j] = p[j, i] = studentized_range.sf(abs(rangos[i] - rangos[j]) / escala * np.sqrt(2), k, np.inf)
    return rangos, p, studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * escala


def anova_repetidas(M):
    n, k = M.shape
    gran = M.mean()
    ss_modelo = n * ((M.mean(0) - gran) ** 2).sum()
    ss_semilla = k * ((M.mean(1) - gran) ** 2).sum()
    ss_error = ((M - gran) ** 2).sum() - ss_modelo - ss_semilla
    gl_modelo, gl_error = k - 1, (k - 1) * (n - 1)
    f = (ss_modelo / gl_modelo) / (ss_error / gl_error)
    centro = np.eye(k) - 1 / k
    doble = centro @ np.cov(M, rowvar=False) @ centro
    epsilon = np.trace(doble) ** 2 / ((k - 1) * (doble ** 2).sum())
    return {"F": f, "gl modelo": gl_modelo, "gl error": gl_error, "épsilon GG": epsilon, "p sin corregir": stats.f.sf(f, gl_modelo, gl_error),
            "p GG": stats.f.sf(f, gl_modelo * epsilon, gl_error * epsilon), "η² parcial": ss_modelo / (ss_modelo + ss_error),
            "ss": {"Modelo": ss_modelo, "Semilla": ss_semilla, "Residuo": ss_error}}


def mauchly(M):
    """Prueba de esfericidad de Mauchly con contrastes ortonormales."""
    n, k = M.shape
    C = np.linalg.qr(np.vstack([np.ones(k), np.eye(k)[:-1]]).T)[0][:, 1:]
    S = C.T @ np.cov(M, rowvar=False) @ C
    p_ = k - 1
    det = np.linalg.det(S)
    if det <= 0 or n - 1 <= p_:
        return {"W": np.nan, "chi2": np.nan, "gl": np.nan, "p": np.nan}
    W = det / (np.trace(S) / p_) ** p_
    factor = (n - 1) - (2 * p_ ** 2 + p_ + 2) / (6 * p_)
    chi2 = -factor * np.log(W)
    gl = p_ * (p_ + 1) / 2 - 1
    return {"W": W, "chi2": chi2, "gl": gl, "p": stats.chi2.sf(chi2, gl)}


def model_confidence_set(perdidas, nivel=0.10, repeticiones=2000, semilla=42):
    """MCS de Hansen, Lunde y Nason (2011) con el estadístico T_max y bootstrap sobre las unidades (filas de `perdidas`)."""
    L = np.asarray(perdidas, float)
    n, m = L.shape
    rng = np.random.default_rng(semilla)
    indices = rng.integers(0, n, (repeticiones, n))
    vivos, eliminados = list(range(m)), []
    while len(vivos) > 1:
        sub = L[:, vivos]
        d_i = sub.mean(1, keepdims=True)
        d = (sub - d_i).mean(0)
        boot = np.stack([(sub[ix] - sub[ix].mean(1, keepdims=True)).mean(0) for ix in indices])
        var = ((boot - d) ** 2).mean(0)
        t = d / np.sqrt(np.maximum(var, 1e-18))
        t_boot = ((boot - d) / np.sqrt(np.maximum(var, 1e-18))).max(1)
        p = float((t_boot >= t.max()).mean())
        if p >= nivel:
            break
        peor = vivos[int(np.argmax(t))]
        eliminados.append((peor, p))
        vivos.remove(peor)
    return {"conjunto": vivos, "eliminados": eliminados}


# ------------------------------------------------------------------ validación cruzada 5 × 2
def nadeau_bengio(diferencias, n_prueba, n_entrenamiento):
    d = np.asarray(diferencias, float)
    J = len(d)
    var = d.var(ddof=1) * (1 / J + n_prueba / n_entrenamiento)
    t = d.mean() / np.sqrt(var) if var > 0 else 0.0
    return {"t": t, "gl": J - 1, "p": 2 * stats.t.sf(abs(t), J - 1), "error corregido": np.sqrt(var)}


def f_5x2cv(diferencias):
    """F combinada de Alpaydin (1999): diferencias en una matriz 5 × 2 (repetición × pliegue)."""
    p = np.asarray(diferencias, float).reshape(5, 2)
    s2 = ((p - p.mean(1, keepdims=True)) ** 2).sum(1)
    f = (p ** 2).sum() / (2 * s2.sum()) if s2.sum() > 0 else np.inf
    return {"F": f, "gl": (10, 5), "p": stats.f.sf(f, 10, 5)}


# ------------------------------------------------------------------ pruebas sobre filas de prueba
def delong(verdad, puntaje_a, puntaje_b):
    """DeLong et al. (1988) para dos AUC binarios correlacionados."""
    verdad = np.asarray(verdad, bool)
    m, n = int(verdad.sum()), int((~verdad).sum())
    comp = []
    for puntaje in (puntaje_a, puntaje_b):
        pos, neg = puntaje[verdad], puntaje[~verdad]
        tx, ty, tz = rankdata(pos), rankdata(neg), rankdata(np.concatenate([pos, neg]))
        comp.append(((tz[:m].sum() - m * (m + 1) / 2) / (m * n), (tz[:m] - tx) / n, 1 - (tz[m:] - ty) / m))
    (auc_a, v10_a, v01_a), (auc_b, v10_b, v01_b) = comp
    c10, c01 = np.cov(v10_a, v10_b), np.cov(v01_a, v01_b)
    var = (c10[0, 0] + c10[1, 1] - 2 * c10[0, 1]) / m + (c01[0, 0] + c01[1, 1] - 2 * c01[0, 1]) / n
    z = (auc_a - auc_b) / np.sqrt(var) if var > 0 else 0.0
    return {"diferencia": auc_a - auc_b, "p": 2 * stats.norm.sf(abs(z)), "error": np.sqrt(max(var, 0))}


def mcnemar_exacta(acierto_a, acierto_b):
    b = int((acierto_a & ~acierto_b).sum())
    c = int((~acierto_a & acierto_b).sum())
    if b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    else:
        p = stats.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1)
    return {"b": b, "c": c, "p": p}


# ------------------------------------------------------------------ bondad de ajuste y calibración (tabla A)
def binomial_frente_mayoritaria(aciertos, n, proporcion):
    return stats.binomtest(int(aciertos), int(n), proporcion, alternative="greater").pvalue


def hosmer_lemeshow(prob, y, g=10):
    orden = np.argsort(prob)
    grupos = np.array_split(orden, g)
    chi2 = 0.0
    for grupo in grupos:
        o, e, nn = y[grupo].sum(), prob[grupo].sum(), len(grupo)
        if 0 < e < nn:
            chi2 += (o - e) ** 2 / (e * (1 - e / nn))
    return {"chi2": chi2, "gl": g - 2, "p": stats.chi2.sf(chi2, g - 2)}


def spiegelhalter(prob, y):
    num = ((y - prob) * (1 - 2 * prob)).sum()
    den = np.sqrt((((1 - 2 * prob) ** 2) * prob * (1 - prob)).sum())
    z = num / den if den > 0 else 0.0
    return {"Z": z, "p": 2 * stats.norm.sf(abs(z))}


def pendiente_intercepto(prob, y):
    """Regresión logística de y sobre el logit predicho; razón de verosimilitud para pendiente = 1 e intercepto = 0."""
    import statsmodels.api as sm

    z = logit(np.clip(prob, 1e-6, 1 - 1e-6))
    ajuste = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    ll_libre = ajuste.llf
    p0 = expit(z)
    ll_nulo = float((y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum())
    rv = 2 * (ll_libre - ll_nulo)
    return {"intercepto": ajuste.params[0], "pendiente": ajuste.params[1], "IC pendiente": tuple(ajuste.conf_int()[1]), "RV": rv, "p": stats.chi2.sf(rv, 2)}


def escalar_temperatura(logits_val, y_val):
    """Temperatura T > 0 que minimiza la log-verosimilitud negativa en validación."""
    from scipy.optimize import minimize_scalar

    def nll(t):
        z = logits_val / t
        z = z - z.max(1, keepdims=True)
        return -(z[np.arange(len(y_val)), y_val] - np.log(np.exp(z).sum(1))).mean()

    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def logits_de(prob):
    return np.log(np.clip(prob, 1e-12, 1))


def suavizar(prob, temperatura):
    z = logits_de(prob) / temperatura
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ predicción conforme
def conforme(prob_cal, y_cal, prob_pru, alfa=0.10, metodo="LAC"):
    """Conjuntos de predicción conforme: LAC (1 − p de la clase verdadera) o APS (masa acumulada)."""
    n = len(y_cal)
    nivel = np.ceil((n + 1) * (1 - alfa)) / n
    if metodo == "LAC":
        puntajes = 1 - prob_cal[np.arange(n), y_cal]
        q = np.quantile(puntajes, min(nivel, 1), method="higher")
        return prob_pru >= 1 - q
    orden = np.argsort(-prob_cal, 1)
    acumulada = np.take_along_axis(prob_cal, orden, 1).cumsum(1)
    posicion = np.argmax(orden == y_cal[:, None], 1)
    puntajes = acumulada[np.arange(n), posicion]
    q = np.quantile(puntajes, min(nivel, 1), method="higher")
    orden_p = np.argsort(-prob_pru, 1)
    acum_p = np.take_along_axis(prob_pru, orden_p, 1).cumsum(1)
    incluir = (acum_p - np.take_along_axis(prob_pru, orden_p, 1)) < q
    conjunto = np.zeros_like(prob_pru, dtype=bool)
    np.put_along_axis(conjunto, orden_p, incluir, 1)
    return conjunto


def cobertura(conjuntos, y):
    return float(conjuntos[np.arange(len(y)), y].mean()), float(conjuntos.sum(1).mean())


def auc_ponderado(positivo, puntaje, pesos):
    """AUC con pesos por fila, en O(n log n): la probabilidad ponderada de que un positivo supere a un negativo."""
    orden = np.argsort(puntaje, kind="mergesort")
    p, s, w = positivo[orden], puntaje[orden], pesos[orden]
    _, inicio, conteo = np.unique(s, return_index=True, return_counts=True)
    grupo = np.repeat(np.arange(len(inicio)), conteo)
    w_pos = np.bincount(grupo, weights=w * p)
    w_neg = np.bincount(grupo, weights=w * ~p)
    menores = np.cumsum(w_neg) - w_neg
    total = w_pos.sum() * w_neg.sum()
    return float((w_pos * (menores + w_neg / 2)).sum() / total) if total > 0 else np.nan


def auc_macro_bootstrap(verdad, prob, bloques, repeticiones=2000, semilla=42):
    """IC del AUC macro uno contra todos remuestreando bloques: cada réplica pondera las filas por las veces que se eligió su bloque."""
    unicos, inversa = np.unique(bloques, return_inverse=True)
    # El orden de los puntajes no cambia entre réplicas: se ordena una vez por clase y en cada réplica solo cambian los pesos
    previo = []
    for c in range(prob.shape[1]):
        orden = np.argsort(prob[:, c], kind="mergesort")
        _, inicio, conteo = np.unique(prob[orden, c], return_index=True, return_counts=True)
        previo.append((orden, np.repeat(np.arange(len(inicio)), conteo), (verdad == c)[orden], inversa[orden]))
    rng = np.random.default_rng(semilla)
    valores = np.empty(repeticiones)
    for i in range(repeticiones):
        veces = np.bincount(rng.integers(0, len(unicos), len(unicos)), minlength=len(unicos)).astype(float)
        aucs = []
        for orden, grupo, positivo, bloque_ord in previo:
            w = veces[bloque_ord]
            w_pos = np.bincount(grupo, weights=w * positivo)
            w_neg = np.bincount(grupo, weights=w * ~positivo)
            total = w_pos.sum() * w_neg.sum()
            aucs.append((w_pos * (np.cumsum(w_neg) - w_neg / 2)).sum() / total if total > 0 else np.nan)
        valores[i] = np.nanmean(aucs)
    return valores


def mcc_desde_conteos(C):
    C = C.reshape(*C.shape[:-1], K, K)
    t, p = C.sum(-1), C.sum(-2)
    c, s = np.trace(C, axis1=-2, axis2=-1), C.sum((-2, -1))
    num_ = c * s - (t * p).sum(-1)
    den = np.sqrt((s ** 2 - (p ** 2).sum(-1)) * (s ** 2 - (t ** 2).sum(-1)))
    return np.where(den > 0, num_ / np.maximum(den, 1e-12), 0.0)
