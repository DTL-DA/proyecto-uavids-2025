"""Réplica parcial del paper guía (Zeng et al., 2025) con el protocolo de su código público.

Reproduce, sin ejecutar el código descargado, los pasos de github.com/QingliZeng/UAV--IDS2025:
16 atributos (se excluyen FlowID, SrcAddr, SrcPort, DstAddr, DstPort y Protocol), partición aleatoria
estratificada 80/20 con `random_state` igual a la semilla, StandardScaler ajustado en entrenamiento y
los modelos de la tabla IV: regresión logística, XGBoost, DNN, CNN y LSTM, con sus hiperparámetros
(`hparams.py`) y su entrenamiento (`deep.py`: Adam, lr 1e-3, lote de 256, hasta 50 épocas, 10 % del
entrenamiento para la parada temprana sobre la pérdida de validación con paciencia de 5, entropía
cruzada sin pesos).

    python Entregable_2_UAVIDS_2025/codigo/replica.py --semillas 42 43 ... --hilos 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

RAIZ_E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_E2))
from codigo import datos  # noqa: E402
from codigo.modelos import CNN1D, DNN, LSTM  # noqa: E402

SALIDA = RAIZ_E2 / "resultados" / "replica"
MODELOS = ["LR", "XGBoost", "DNN", "CNN", "LSTM"]
REPORTADO = {"LR": 0.82, "XGBoost": 0.96, "DNN": 0.93, "CNN": 0.93, "LSTM": 0.94}
HP_PROFUNDOS = {"epochs": 50, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "val_fraction": 0.1, "patience": 5, "dropout": 0.2}


class _DNNAutores(nn.Module):
    """La DNN de deep.py de los autores: dropout tras las capas de 128 y 64."""

    def __init__(self, p):
        super().__init__()
        self.net = DNN(p).red

    def forward(self, x):
        return self.net(x)


def particion(semilla):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    tabla = datos.crudo()[0]
    X = tabla[datos.ATRIBUTOS].to_numpy(dtype=np.float64)
    y = datos.etiquetas()
    indices = np.arange(len(tabla))
    i_ent, i_pru = train_test_split(indices, test_size=0.2, stratify=y, random_state=semilla)
    escala = StandardScaler().fit(X[i_ent])
    return escala.transform(X[i_ent]).astype(np.float32), escala.transform(X[i_pru]).astype(np.float32), y[i_ent], y[i_pru], i_pru


def ajustar_profundo(nombre, X, y, semilla):
    """Traducción línea a línea de TorchClassifier.fit del código de los autores."""
    hp = HP_PROFUNDOS
    torch.manual_seed(semilla)
    rng = np.random.RandomState(semilla)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * hp["val_fraction"])
    va, tr = idx[:n_val], idx[n_val:]
    Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr]).long()
    Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y[va]).long()
    modelo = {"DNN": lambda: _DNNAutores(hp["dropout"]), "CNN": lambda: CNN1D(hp["dropout"]), "LSTM": lambda: LSTM(hp["dropout"])}[nombre]()
    opt = torch.optim.Adam(modelo.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    perdida = nn.CrossEntropyLoss()
    mejor, mejor_estado, malas, epocas = float("inf"), None, 0, 0
    for _ in range(hp["epochs"]):
        epocas += 1
        modelo.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), hp["batch_size"]):
            b = perm[i:i + hp["batch_size"]]
            opt.zero_grad()
            perdida(modelo(Xt[b]), yt[b]).backward()
            opt.step()
        modelo.eval()
        with torch.no_grad():
            vl = perdida(modelo(Xv), yv).item()
        if vl < mejor - 1e-4:
            mejor, malas = vl, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            malas += 1
            if malas >= hp["patience"]:
                break
    modelo.load_state_dict(mejor_estado)
    modelo.eval()

    def probabilidades(Z):
        with torch.no_grad():
            return np.vstack([torch.softmax(modelo(torch.from_numpy(Z[i:i + 4096])), 1).numpy() for i in range(0, len(Z), 4096)])

    return probabilidades, epocas


def ajustar(nombre, X, y, semilla, hilos):
    if nombre == "LR":
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(random_state=semilla, max_iter=1000).fit(X, y)
        return m.predict_proba, None
    if nombre == "XGBoost":
        from xgboost import XGBClassifier

        m = XGBClassifier(random_state=semilla, n_jobs=hilos, eval_metric="mlogloss", n_estimators=100, max_depth=6, learning_rate=0.3).fit(X, y)
        return m.predict_proba, None
    return ajustar_profundo(nombre, X, y, semilla)


def correr(nombre, semilla, hilos):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

    ruta = SALIDA / f"{nombre}_{semilla}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    X_ent, X_pru, y_ent, y_pru, i_pru = particion(semilla)
    inicio = time.time()
    prob_fn, epocas = ajustar(nombre, X_ent, y_ent, semilla, hilos)
    segundos = time.time() - inicio
    prob = prob_fn(X_pru)
    pred = prob.argmax(1)
    macro = precision_recall_fscore_support(y_pru, pred, average="macro", zero_division=0)
    ponderado = precision_recall_fscore_support(y_pru, pred, average="weighted", zero_division=0)
    r = {"modelo": nombre, "semilla": semilla, "accuracy": accuracy_score(y_pru, pred), "precision_macro": macro[0], "recall_macro": macro[1], "f1_macro": macro[2],
         "precision_ponderada": ponderado[0], "recall_ponderado": ponderado[1], "f1_ponderado": ponderado[2],
         "auc_macro": roc_auc_score(y_pru, prob, multi_class="ovr", average="macro"), "segundos": round(segundos, 1), "epocas": epocas,
         "matriz_confusion": confusion_matrix(y_pru, pred).tolist(), "reportado_f1": REPORTADO[nombre]}
    np.savez_compressed(ruta.with_suffix(".npz"), indices_prueba=i_pru.astype(np.int32), prediccion=pred.astype(np.int8))
    ruta.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    print(f"{nombre} semilla {semilla}: F1 macro {r['f1_macro']:.4f} | ponderado {r['f1_ponderado']:.4f} | {segundos:.0f} s", flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--modelos", nargs="+", default=MODELOS)
    ap.add_argument("--hilos", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    SALIDA.mkdir(parents=True, exist_ok=True)
    for s in a.semillas:
        for m in a.modelos:
            correr(m, s, a.hilos)
