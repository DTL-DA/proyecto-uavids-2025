"""BiLSTM y GRU de los conjuntos adicionales en GPU (sección 5 de protocolo.md: opción preregistrada y declarada).

El protocolo permite pasar a GPU solo las BiLSTM y GRU de los conjuntos adicionales, con determinismo de CUDA. Este script:
  - fija CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.use_deterministic_algorithms(True), cudnn.deterministic=True y
    cudnn.benchmark=False antes de tocar la GPU;
  - reemplaza motor.entrenar y motor.predecir del E2 por copias fieles que solo envían los tensores a la GPU: el mismo
    orden de lotes (generador en CPU), la misma inicialización (el modelo se construye en CPU con la semilla y después
    se mueve), la misma pérdida ponderada, el mismo recorte del gradiente, el mismo planificador y la misma parada temprana;
  - corre ``correr_adicional`` de ejecutar_definitivas.py sin cambios y agrega al registro el dispositivo, la GPU, las
    versiones de CUDA y cuDNN y el modo de determinismo.
En CPU, ``entrenar_gpu`` reproduce bit a bit a ``motor.entrenar`` (comprobado con ``--verificar``).

Uso, desde Entregable_3_UAVIDS_2025:
    python training/ejecutar_gpu.py [procesos]        # corre las BiLSTM y GRU pendientes (por defecto 4 procesos)
    python training/ejecutar_gpu.py --verificar       # compara entrenar_gpu con motor.entrenar en CPU (2 épocas)
    python training/ejecutar_gpu.py --listar          # solo muestra las corridas pendientes
Al terminar deja resultados_gpu.zip (JSON y probabilidades de prueba) en Entregable_3_UAVIDS_2025/.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch import nn  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI), str(AQUI.parent), str(AQUI.parents[1] / "Entregable_2_UAVIDS_2025")]
import ejecutar_definitivas as ED  # noqa: E402
from codigo import modelos, motor  # noqa: E402

E3 = AQUI.parent
CARPETA = ED.CORRIDAS / "adicionales"
MODELOS_GPU = ["BiLSTM", "GRU"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODO = {"determinismo": "estricto"}


def fijar_determinismo(warn_only=False):
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predecir_gpu(modelo, X, indices, lote=4096):
    """motor.predecir con el modelo en su dispositivo; devuelve los logits en CPU, como el original."""
    modelo.eval()
    dev = next(modelo.parameters()).device
    with torch.no_grad():
        return torch.cat([modelo(X[b.to(X.device)].to(dev)).cpu() for b in indices.split(lote)])


def entrenar_gpu(nombre, arquitectura, hp, semilla, D, ids_ent, ids_val, max_epocas=motor.MAX_EPOCAS, paciencia=motor.PACIENCIA, etiquetas=None,
                 variante=None, dispositivo=None):
    """Copia fiel de motor.entrenar del E2; las únicas diferencias son los .to(dispositivo)."""
    dev = dispositivo or DEV
    variante = variante or {}
    motor.fijar_semilla(semilla)
    X, pesos = D["X"], D["pesos"]
    if not variante.get("ponderar", True):
        pesos = torch.ones_like(pesos)
    Y = D["Y"] if etiquetas is None else etiquetas
    modelo = modelos.construir(nombre, arquitectura)
    if not variante.get("dropout", True):
        for m in modelo.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        for m in modelo.modules():
            if isinstance(m, nn.LSTM) and m.num_layers > 1:
                m.dropout = 0.0
    generador = torch.Generator().manual_seed(semilla)
    inicio_pre = time.time()
    if hasattr(modelo, "preentrenar"):
        modelo.preentrenar(X[ids_ent], generador)
    segundos_pre = time.time() - inicio_pre
    modelo.to(dev)
    Xd, Yd = X.to(dev), Y.to(dev)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, factor=0.1, patience=2)
    perdida = nn.CrossEntropyLoss(weight=pesos.to(dev))
    mejor, mejor_estado, sin_mejora, curvas, divergente = -1.0, None, 0, [], False
    for epoca in range(1, max_epocas + 1):
        inicio = time.time()
        modelo.train()
        acumulado, n = 0.0, 0
        for lote in ids_ent[torch.randperm(len(ids_ent), generator=generador)].split(hp["batch"]):
            if len(lote) < 2:
                continue
            lote_d = lote.to(dev)
            optimizador.zero_grad(set_to_none=True)
            valor = perdida(modelo(Xd[lote_d]), Yd[lote_d])
            extra = getattr(modelo, "perdida_extra", None)
            if extra is not None:
                valor = valor + extra
            if not torch.isfinite(valor):
                divergente = True
                break
            valor.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), motor.NORMA_GRADIENTE)
            optimizador.step()
            acumulado += valor.item() * len(lote)
            n += len(lote)
        if divergente:
            break
        logits = predecir_gpu(modelo, Xd, ids_val)
        perdida_val = perdida(logits.to(dev), Yd[ids_val.to(dev)]).item()
        f1_val = f1_score(Y[ids_val].numpy(), logits.argmax(1).numpy(), average="macro", zero_division=0)
        planificador.step(perdida_val)
        curvas.append({"epoca": epoca, "perdida_ent": acumulado / max(n, 1), "perdida_val": perdida_val, "f1_val": f1_val,
                       "lr": optimizador.param_groups[0]["lr"], "segundos": round(time.time() - inicio, 2)})
        if f1_val > mejor:
            mejor, sin_mejora = f1_val, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo, curvas, divergente, segundos_pre


def preparar():
    """Inicializador de cada proceso: determinismo, hilos y sustitución de motor.entrenar y motor.predecir."""
    torch.set_num_threads(2)
    fijar_determinismo()
    motor.entrenar = entrenar_gpu
    motor.predecir = predecir_gpu


def es_pendiente(ruta):
    return not ruta.exists() or json.loads(ruta.read_text(encoding="utf-8")).get("pendiente_gpu", False)


def correr(trabajo, carpeta):
    ruta = Path(carpeta) / f"{trabajo['id']}.json"
    if ruta.exists() and es_pendiente(ruta):
        ruta.unlink()  # marcador de "pendiente de GPU"
    try:
        r = ED.correr_adicional(trabajo, carpeta)
    except RuntimeError as e:
        if "determinis" not in str(e).lower():
            raise
        fijar_determinismo(warn_only=True)  # una operación sin versión determinista: se registra y se declara
        MODO["determinismo"] = "warn_only: " + str(e).splitlines()[0][:200]
        r = ED.correr_adicional(trabajo, carpeta)
    if "dispositivo" not in r:
        r.update({"dispositivo": str(DEV), "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None, "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(), "determinismo": MODO["determinismo"], "torch": torch.__version__})
        tmp = ruta.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        tmp.replace(ruta)
    return r


def verificar():
    """En CPU, entrenar_gpu debe dar exactamente lo mismo que motor.entrenar (2 épocas de GRU en UNR-IDD, semilla 6001)."""
    import numpy as np

    from conjuntos_adicionales import cargar

    d = cargar("unr_idd")
    modelos.N_ATRIBUTOS, modelos.N_CLASES = d["X"].shape[1], int(d["y"].max()) + 1
    D = {"X": torch.from_numpy(d["X"]), "pesos": torch.from_numpy(d["pesos"]), "Y": torch.from_numpy(d["y"])}
    hp, arq = ED.hp_e2("GRU")
    ent, val = torch.from_numpy(d["ent"]), torch.from_numpy(d["val"])
    a = motor.entrenar("GRU", arq, hp, 6001, D, ent, val, max_epocas=2)
    b = entrenar_gpu("GRU", arq, hp, 6001, D, ent, val, max_epocas=2, dispositivo=torch.device("cpu"))
    dif = max(float((a[0].state_dict()[k] - b[0].state_dict()[k]).abs().max()) for k in a[0].state_dict())
    pa = motor.predecir(a[0], D["X"], torch.from_numpy(d["eva"])).numpy()
    pb = predecir_gpu(b[0], D["X"], torch.from_numpy(d["eva"])).numpy()
    sin_tiempo = lambda curvas: [{k: v for k, v in c.items() if k != "segundos"} for c in curvas]  # noqa: E731
    print(json.dumps({"diferencia máxima de pesos": dif, "diferencia máxima de logits": float(np.abs(pa - pb).max()),
                      "curvas iguales": sin_tiempo(a[1]) == sin_tiempo(b[1]), "épocas": len(b[1])}, ensure_ascii=False))


def empaquetar_resultados():
    destino = E3 / "resultados_gpu.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for m in MODELOS_GPU:
            for p in sorted(CARPETA.glob(f"*_{m}_*")):
                if p.suffix in (".json", ".npy") and not es_pendiente(p.with_name(p.name.replace("_eva.npy", ".json"))):
                    z.write(p, p.relative_to(E3).as_posix())
    return destino


def main():
    if "--verificar" in sys.argv:
        verificar()
        return
    numeros = [a for a in sys.argv[1:] if a.isdigit()]
    procesos = int(numeros[0]) if numeros else 4
    print(f"Dispositivo: {DEV} {torch.cuda.get_device_name(0) if DEV.type == 'cuda' else ''} | torch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)
    CARPETA.mkdir(parents=True, exist_ok=True)
    trabajos = [t for t in ED.plan_adicionales(MODELOS_GPU) if es_pendiente(CARPETA / f"{t['id']}.json")]
    print(f"{len(trabajos)} corridas pendientes", flush=True)
    if "--listar" in sys.argv:
        print([t["id"] for t in trabajos])
        return
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=procesos, mp_context=mp.get_context("spawn"), initializer=preparar) as ex:
        futuros = {ex.submit(correr, t, str(CARPETA)): t for t in trabajos}
        for i, f in enumerate(as_completed(futuros), 1):
            try:
                r = f.result()
                print(f"[{i}/{len(trabajos)}] {r['id']}: f1_val={r.get('f1_val', float('nan')):.4f} f1_eva={r.get('f1_eva', float('nan')):.4f} "
                      f"({r.get('segundos')} s; {(time.time() - inicio) / 60:.1f} min; {r.get('determinismo')})", flush=True)
            except Exception as e:  # una corrida fallida no detiene las demás
                print(f"[{i}/{len(trabajos)}] ERROR en {futuros[f]['id']}: {e!r}", flush=True)
    print(f"Listo: {empaquetar_resultados()}", flush=True)


if __name__ == "__main__":
    main()
