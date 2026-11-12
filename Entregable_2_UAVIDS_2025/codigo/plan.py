"""Genera las listas de trabajos de cada fase a partir de configuracion/experimento.json y de los
resultados de las fases anteriores.

    python Entregable_2_UAVIDS_2025/codigo/plan.py <fase>

Fases: ajuste, piloto, ruido, fase1, fase2, definitiva, cv5x2.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

RAIZ_E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_E2))
from codigo import modelos  # noqa: E402

CONFIG = json.loads((RAIZ_E2 / "configuracion" / "experimento.json").read_text(encoding="utf-8"))
CORRIDAS = RAIZ_E2 / "resultados" / "corridas"
TRABAJOS = RAIZ_E2 / "resultados" / "trabajos"
# Segundos por época medidos con lote de 256 en este equipo; solo ordenan la cola de trabajos
COSTO = {"DNN": 1, "CNN 1D": 2.5, "LSTM": 4.4, "RNN": 8, "GRU": 24, "BiLSTM": 50, "Transformer": 2.8, "TabNet": 14, "KAN": 2.3, "ResNet": 4.5,
         "Autoencoder": 2.5, "DBN": 1.5, "CNN-LSTM": 12, "LightGBM": 1, "Logística": 0.5, "Trivial": 0.01}


def profundos():
    return CONFIG["modelos"]["base_e1"] + CONFIG["modelos"]["top10"]


def nombre_archivo(modelo):
    return modelo.replace(" ", "_").replace("í", "i")


def configuraciones(espacio, n, semilla):
    claves = list(espacio)
    todas = [dict(zip(claves, v)) for v in itertools.product(*(espacio[c] for c in claves))]
    return random.Random(semilla).sample(todas, min(n, len(todas)))


def configuraciones_ajuste():
    a = CONFIG["ajuste"]
    return {"profundos": configuraciones(a["espacio_profundos"], a["intentos_por_modelo"], a["semilla_muestreo"]),
            "LightGBM": configuraciones(a["espacio_lightgbm"], a["intentos_por_modelo"], a["semilla_muestreo"]),
            "Logística": [{"C": c} for c in a["espacio_logistica"]["C"]]}


def leer(fase):
    carpeta = CORRIDAS / fase
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(carpeta.glob("*.json"))] if carpeta.exists() else []


def mejores_hp():
    """Configuración elegida por modelo: el mayor F1 macro de validación del ajuste."""
    mejor = {}
    for r in leer("ajuste"):
        if r.get("f1_val") is None:
            continue
        actual = mejor.get(r["modelo"])
        if actual is None or r["f1_val"] > actual["f1_val"] or (r["f1_val"] == actual["f1_val"] and r["intento"] < actual["intento"]):
            mejor[r["modelo"]] = r
    return {m: r["hp"] for m, r in mejor.items()}


def trabajo(fase, modelo, hp, semilla, sufijo="", **extra):
    epocas = extra.get("max_epocas", CONFIG["entrenamiento"]["max_epocas"])
    t = {"id": f"{fase}_{nombre_archivo(modelo)}_{semilla}{sufijo}", "fase": fase, "modelo": modelo, "hp": hp, "semilla": semilla,
         "costo": COSTO.get(modelo, 1) * (epocas or 1) * (256 / hp.get("batch", 256)) ** 0.3}
    t.update(extra)
    return t


def plan_ajuste():
    a = CONFIG["ajuste"]
    c = configuraciones_ajuste()
    t = []
    for m in profundos():
        for i, hp in enumerate(c["profundos"]):
            extra = {"arquitectura": list(modelos.REFERENCIA)} if m in CONFIG["modelos"]["con_busqueda"] else {}
            t.append(trabajo("ajuste", m, hp, a["semilla_entrenamiento"], f"_i{i:02d}", intento=i, max_epocas=a["max_epocas"], **extra))
    for m in ("LightGBM", "Logística"):
        for i, hp in enumerate(c[m]):
            t.append(trabajo("ajuste", m, hp, a["semilla_entrenamiento"], f"_i{i:02d}", intento=i))
    return t


def plan_piloto():
    hp = mejores_hp()
    p = CONFIG["piloto"]
    return [trabajo("piloto", m, hp[m], s, guardar_prob_val=True) for m in p["modelos"] for s in p["semillas"]]


def muestra_arquitecturas():
    r = CONFIG["ruido_semilla_busqueda"]
    return random.Random(r["semilla_muestreo"]).sample(modelos.espacio_arquitecturas(), r["arquitecturas_por_modelo"])


SUBCONJUNTO_FASE1 = "fase1_25_bloques"


def subconjunto_fase1(fraccion_bloques=0.25):
    """Desviación del 12/11/2026: el 25 % de los bloques de entrenamiento, estratificado por clase, para la fase 1 y el estudio de ruido."""
    from sklearn.model_selection import StratifiedGroupKFold
    from codigo import datos

    ruta = RAIZ_E2 / "resultados" / "particiones" / f"{SUBCONJUNTO_FASE1}.npz"
    if not ruta.exists():
        h = datos.heredados()
        ent = h["ids"]["entrenamiento"]
        bloques = datos.crudo()[1]
        _, elegido = next(StratifiedGroupKFold(int(round(1 / fraccion_bloques)), shuffle=True, random_state=CONFIG["busqueda_arquitectura"]["semilla_fase1"]).split(ent, h["y"][ent], bloques[ent]))
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ruta, ent=np.sort(ent[elegido]))
    return SUBCONJUNTO_FASE1


def plan_ruido():
    hp = mejores_hp()
    r, b = CONFIG["ruido_semilla_busqueda"], CONFIG["busqueda_arquitectura"]
    t = []
    for m in CONFIG["modelos"]["con_busqueda"]:
        for a in muestra_arquitecturas():
            for s in r["semillas"]:
                t.append(trabajo("ruido", m, hp[m], s, "_" + modelos.identificador(a), arquitectura=list(a), max_epocas=b["epocas_fase1"], subconjunto_ent=subconjunto_fase1()))
    return t


def plan_fase1():
    hp = mejores_hp()
    b = CONFIG["busqueda_arquitectura"]
    sub = subconjunto_fase1()
    return [trabajo("fase1", m, hp[m], b["semilla_fase1"], "_" + modelos.identificador(a), arquitectura=list(a), max_epocas=b["epocas_fase1"], subconjunto_ent=sub,
                    costo_relativo=modelos.contar_mac(modelos.construir(m, a)))
            for m in CONFIG["modelos"]["con_busqueda"] for a in modelos.espacio_arquitecturas()]


def k_finalistas():
    """Regla preregistrada para k, con el ruido por semilla del estudio de la fase 1."""
    ruido = leer("ruido")
    if not ruido:
        raise RuntimeError("Falta el estudio de ruido por semilla")
    razones = {}
    for m in CONFIG["modelos"]["con_busqueda"]:
        por_arq = {}
        for r in ruido:
            if r["modelo"] == m:
                por_arq.setdefault(tuple(r["arquitectura"]), []).append(r["f1_val"])
        medias = [np.mean(v) for v in por_arq.values()]
        dentro = np.sqrt(np.mean([np.var(v, ddof=1) for v in por_arq.values()]))
        razones[m] = float(dentro / np.std(medias, ddof=1))
    k = 20 if max(razones.values()) >= 0.5 else 10
    return k, razones


def finalistas(k):
    hp = mejores_hp()
    salida = {}
    for m in CONFIG["modelos"]["con_busqueda"]:
        f1 = sorted((r for r in leer("fase1") if r["modelo"] == m and r.get("f1_val") is not None), key=lambda r: (-r["f1_val"], modelos.identificador(r["arquitectura"])))
        elegidas = [tuple(r["arquitectura"]) for r in f1[:k]]
        referencia = tuple(CONFIG["busqueda_arquitectura"]["referencia"])
        if referencia not in elegidas:
            elegidas.append(referencia)
        salida[m] = elegidas
    return salida, hp


def plan_fase2():
    k, _ = k_finalistas()
    fin, hp = finalistas(k)
    b = CONFIG["busqueda_arquitectura"]
    return [trabajo("fase2", m, hp[m], s, "_" + modelos.identificador(a), arquitectura=list(a)) for m, lista in fin.items() for a in lista for s in b["semillas_fase2"]]


def plan_definitiva(R, arquitecturas):
    hp = mejores_hp()
    semillas = [CONFIG["definitiva"]["semilla_inicial"] + i for i in range(R)]
    t = []
    for m in profundos():
        extra = {"arquitectura": list(arquitecturas[m])} if m in arquitecturas else {}
        t += [trabajo("definitiva", m, hp[m], s, evaluar=True, guardar_prob_val=True, guardar_pesos=True, **extra) for s in semillas]
    for m in ("LightGBM", "Logística"):
        t += [trabajo("definitiva", m, hp[m], s, evaluar=True, guardar_prob_val=True, guardar_pesos=True) for s in semillas]
    t.append(trabajo("definitiva", "Trivial", {}, semillas[0], evaluar=True, guardar_prob_val=True))
    return t


def arquitectura_final(m, pre):
    return {"arquitectura": list(pre["arquitecturas_ganadoras"][m])} if m in pre.get("arquitecturas_ganadoras", {}) else {}


def adenda():
    return json.loads((RAIZ_E2 / "resultados" / "preregistro_e2_adenda.json").read_text(encoding="utf-8"))


VARIANTES = {"sin ponderación de clases": {"ponderar": False}, "sin log(1 + x)": {"transformacion": "estandar"}, "sin dropout": {"dropout": False}}


def plan_ablacion(modelo, variantes=VARIANTES):
    pre = adenda()
    hp = mejores_hp()
    semillas = [CONFIG["definitiva"]["semilla_inicial"] + i for i in range(pre["R"])]
    return [trabajo("ablacion", modelo, hp[modelo], s, "_" + nombre_archivo(n).replace("(", "").replace(")", "").replace(" ", "_").replace("+", ""), variante=v, variante_nombre=n,
                    evaluar=True, guardar_prob_eva=False, **arquitectura_final(modelo, pre)) for n, v in variantes.items() for s in semillas]


def particiones_5x2():
    """Validación cruzada 5 × 2 agrupada por bloques sobre entrenamiento + validación; la prueba no se toca.
    En cada pliegue, un 20 % de los bloques de la mitad de entrenamiento se reserva para la parada temprana."""
    from sklearn.model_selection import StratifiedGroupKFold
    from codigo import datos

    h = datos.heredados()
    bloques = datos.crudo()[1]
    y = h["y"]
    pool = np.concatenate([h["ids"]["entrenamiento"], h["ids"]["validacion"]])
    carpeta = RAIZ_E2 / "resultados" / "particiones"
    carpeta.mkdir(parents=True, exist_ok=True)
    nombres = []
    semilla = CONFIG["validacion_cruzada_5x2"]["semilla"]
    for rep in range(5):
        for j, (a, b) in enumerate(StratifiedGroupKFold(2, shuffle=True, random_state=semilla + rep).split(pool, y[pool], bloques[pool])):
            entrenar, evaluar = pool[a], pool[b]
            e, v = next(StratifiedGroupKFold(5, shuffle=True, random_state=semilla + 100 + rep).split(entrenar, y[entrenar], bloques[entrenar]))
            nombre = f"cv5x2_r{rep}_f{j}"
            np.savez(carpeta / f"{nombre}.npz", ent=entrenar[e], val=entrenar[v], eva=evaluar)
            nombres.append(nombre)
    return nombres


def plan_cv5x2():
    pre = adenda()
    hp = mejores_hp()
    nombres = particiones_5x2()
    t = []
    for m in profundos() + ["LightGBM", "Logística"]:
        for n in nombres:
            t.append(trabajo("cv5x2", m, hp[m], CONFIG["definitiva"]["semilla_inicial"], "_" + n, particion=n, evaluar=True, guardar_prob_eva=False, **arquitectura_final(m, pre)))
    return t


def plan_curvas(modelos_cc, fracciones=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), semillas=(7101, 7102, 7103, 7104, 7105)):
    pre = adenda()
    hp = mejores_hp()
    return [trabajo("curvas", m, hp[m], s, f"_p{int(100 * f):03d}", fraccion_ent=f, **arquitectura_final(m, pre)) for m in modelos_cc for f in fracciones for s in semillas]


def plan_permutacion(modelos_p, B=200):
    pre = adenda()
    hp = mejores_hp()
    return [trabajo("permutacion", m, hp[m], CONFIG["definitiva"]["semilla_inicial"], f"_b{b:03d}", permutar_etiquetas=9000 + b, evaluar=True, guardar_prob_eva=False,
                    **arquitectura_final(m, pre)) for m in modelos_p for b in range(B)]


def guardar(nombre, trabajos):
    TRABAJOS.mkdir(parents=True, exist_ok=True)
    ruta = TRABAJOS / f"{nombre}.jsonl"
    ruta.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in trabajos), encoding="utf-8")
    print(f"{ruta.name}: {len(trabajos)} trabajos")
    return ruta


if __name__ == "__main__":
    fase = sys.argv[1]
    if fase == "definitiva":
        pre = json.loads((RAIZ_E2 / "resultados" / "preregistro_e2_adenda.json").read_text(encoding="utf-8"))
        guardar(fase, plan_definitiva(pre["R"], {m: tuple(a) for m, a in pre["arquitecturas_ganadoras"].items()}))
    elif fase == "ablacion":
        guardar(fase, plan_ablacion(sys.argv[2]))
    elif fase == "cv5x2":
        guardar(fase, plan_cv5x2())
    elif fase == "curvas":
        guardar(fase, plan_curvas(sys.argv[2].split(",")))
    elif fase == "permutacion":
        guardar(fase, plan_permutacion(sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 200))
    else:
        guardar(fase, {"ajuste": plan_ajuste, "piloto": plan_piloto, "ruido": plan_ruido, "fase1": plan_fase1, "fase2": plan_fase2}[fase]())
