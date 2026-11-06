"""Hoja de extracción de los estudios leídos en texto completo y tabla de frecuencia de modelos de referencia.

Cada campo proviene de la lectura dirigida del texto. Lo no verificado queda como "no reportado" o "no identificado".
"""
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
CARPETA = Path(__file__).parent
UA = {"User-Agent": "Mozilla/5.0"}
SIN_DOI = {"sin-doi:dragon": "Deep Reinforced Adaptive Graph Optimization Network", "sin-doi:gatv2": "Edge-Aware GATv2-BiLSTM Framework",
           "sin-doi:ensemble": "Explainable Deep Ensemble Learning for Multi-Class Cyberattack", "sin-doi:iod": "Lightweight Intrusion Detection for IoD Infrastructure"}

# modelos: arquitecturas de Deep Learning usadas como referencia o comparación, normalizadas por familia
ESTUDIOS = [
    {"doi": "10.1109/cns66487.2025.11194990", "dataset": "UAVIDS-2025", "entrada": "flujos de red simulados", "propuesto": "conjunto de datos y comparación de modelos", "modelos": ["DNN/MLP", "CNN", "LSTM"],
     "detalle": "DNN 16-128-64-32-5; CNN 1D de 32 y 64 filtros; LSTM de 64 unidades; las tres con dropout de 0,2", "valores": "F1: DNN 93 %, CNN 93 %, LSTM 94 %; regresión logística 82 %, XGBoost 96 %",
     "particion": "aleatoria estratificada 80/20", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "El artículo reporta los resultados; la arquitectura está en el código de los autores"},
    {"doi": "10.3390/s19245529", "dataset": "KDD Cup 99", "entrada": "flujos de red", "propuesto": "DBN optimizada con PSO", "modelos": ["DNN/MLP"], "detalle": "ANN y DNN", "valores": "Exactitud: ANN 90,79 %, DNN 91,36 %; propuesto 92,44 %",
     "particion": "aleatoria; usa 10 % del archivo", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Conjunto genérico, no de drones"},
    {"doi": "10.1038/s41598-023-45065-8", "dataset": "CICIDS2017 y otros", "entrada": "flujos de red", "propuesto": "marco de RNN para Internet de drones", "modelos": ["RNN", "LSTM", "BiLSTM", "GRU", "DNN/MLP"], "detalle": "RNN, LSTM y Bi-LSTM de 64 unidades, GRU y perceptrón",
     "valores": "Exactitud: Bi-LSTM 91,36 %, LSTM 91,00 %, RNN 90,16 %, GRU 89,13 %, perceptrón 85,02 %", "particion": "aleatoria 70/30", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Conjunto genérico, no de drones"},
    {"doi": "10.1109/access.2024.3359043", "dataset": "ToN-IoT y Edge-IIoTset", "entrada": "flujos de red", "propuesto": "red híbrida con optimización", "modelos": ["LSTM", "GRU"], "detalle": "LSTM y GRU", "valores": "Exactitud: LSTM de 96,20 a 96,95 %, GRU de 96,00 a 96,50 %",
     "particion": "aleatoria 70/30", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Conjuntos de IoT"},
    {"doi": "10.1109/access.2024.3349469", "dataset": "UAV con ataques de GPS", "entrada": "datos del sistema", "propuesto": "ANN con algoritmo genético", "modelos": ["DNN/MLP"], "detalle": "Red de retropropagación, con y sin una capa adicional", "valores": "Exactitud: BPN 0,895, BPN con una capa más 0,9324",
     "particion": "5 pliegues", "repeticiones": "5 pliegues", "dispersion": "no", "prueba": "ninguna", "nota": ""},
    {"doi": "10.3389/fenrg.2024.1491332", "dataset": "KDD Cup 99, NSL-KDD, WSN-DS y CICIDS", "entrada": "flujos de red", "propuesto": "ConvLSTM", "modelos": ["CNN", "LSTM", "CNN-LSTM"], "detalle": "CNN, LSTM y CNN-LSTM", "valores": "Exactitud en una de las tablas: CNN 89,10 %, LSTM 92,25 %, CNN-LSTM 95,11 %; propuesto 99,99 %",
     "particion": "aleatoria 70/30", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "t de Student entre los dos mejores", "nota": "Conjuntos genéricos"},
    {"doi": "10.1109/access.2024.3392618", "dataset": "no identificado", "entrada": "flujos de red", "propuesto": "red con optimización de mangosta", "modelos": ["DBN"], "detalle": "IDBN y DPC-DBN", "valores": "Exactitud: IDBN 95,32 %, DPC-DBN 94,39 %",
     "particion": "aleatoria 80/20 y 70/30", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": ""},
    {"doi": "10.1109/access.2024.3386570", "dataset": "no identificado", "entrada": "flujos de red", "propuesto": "red con depredadores marinos", "modelos": ["DBN"], "detalle": "IDBN y DPC-DBN", "valores": "Exactitud: IDBN 95,32 %, DPC-DBN 94,39 %; propuesto 99,63 %",
     "particion": "aleatoria 80/20 y 70/30", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Repite las cifras de comparación de 10.1109/access.2024.3392618"},
    {"doi": "10.22266/ijies2025.0531.33", "dataset": "CIC-IDS 2017", "entrada": "flujos de red", "propuesto": "xLSTM", "modelos": ["CNN-LSTM", "GRU"], "detalle": "CNN-GRU y LSTM-CNN, tomados de trabajos publicados", "valores": "Exactitud: CNN-GRU 98,73 %, LSTM-CNN 99,21 %; propuesto: F1 99,28 %",
     "particion": "no identificada", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Conjunto genérico; compara con cifras publicadas"},
    {"doi": "10.3389/fcomp.2025.1545282", "dataset": "datos ciberfísicos de GPS", "entrada": "datos del sistema", "propuesto": "aprendizaje por transferencia", "modelos": ["DNN/MLP", "CNN", "LSTM", "ResNet"], "detalle": "MLP, CNN, LSTM y ResNet", "valores": "Exactitud: ResNet 90,99 %, MLP 89,49 %, CNN 86,36 %, LSTM 82,45 %",
     "particion": "aleatoria 70/30", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": ""},
    {"doi": "10.7546/crabs.2024.08.09", "dataset": "no identificado", "entrada": "flujos de red", "propuesto": "red RBF", "modelos": ["LSTM", "CNN"], "detalle": "LSTM y CNN", "valores": "Exactitud: LSTM 93 %, CNN 93 %; propuesto 95 %", "particion": "aleatoria 80/20", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": ""},
    {"doi": "10.3934/math.2024348", "dataset": "NSL-KDD y TON_IoT", "entrada": "flujos de red", "propuesto": "red con optimización aritmética", "modelos": ["LSTM", "CNN"], "detalle": "LSTM-RNN y CNN", "valores": "Exactitud en NSL-KDD: LSTM-RNN 95,91 %, CNN 86,54 %; propuesto 99,32 %", "particion": "aleatoria 80/20", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Conjuntos genéricos"},
    {"doi": "10.1109/dasc66011.2025.11257309", "dataset": "UAV IDS, UKM IDS y otros", "entrada": "flujos de red", "propuesto": "esquema con preservación de privacidad", "modelos": ["Autoencoder", "CNN"], "detalle": "MLP-AE, L-MADE y FFCNN", "valores": "Exactitud en UAV IDS: FFCNN 99,99 %, MLP-AE 98,81 %; propuesto 99,99 %",
     "particion": "aleatoria 80/20", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": ""},
    {"doi": "10.1109/cits70307.2026.11637173", "dataset": "UAVIDS-2025", "entrada": "flujos de red simulados", "propuesto": "explicabilidad y ensambles", "modelos": ["DNN/MLP"], "detalle": "MLP y RealMLP", "valores": "F1 de prueba: MLP 90,66 %, RealMLP 94,28 %", "particion": "aleatoria estratificada 80/20, con 10 pliegues en entrenamiento",
     "repeticiones": "10 pliegues", "dispersion": "no", "prueba": "permutación sobre atributos, no sobre modelos", "nota": "Codifica las IP por frecuencia"},
    {"doi": "10.1038/s41598-026-55446-4", "dataset": "T-ITS, ciberfísico de UAV", "entrada": "datos del sistema", "propuesto": "Bi-LSTM con 1D-CNN", "modelos": ["DNN/MLP"], "detalle": "DNN", "valores": "Exactitud: DNN 96,43 ± 0,22 %; propuesto 98,74 ± 0,07 %", "particion": "entrenamiento y prueba, no identifica agrupación", "repeticiones": "10", "dispersion": "sí", "prueba": "t de Student", "nota": ""},
    {"doi": "10.3389/fphy.2025.1660104", "dataset": "ISCXVPN2016, CICIDS-2018, TON IoT y CIC IoT 2023", "entrada": "flujos de red", "propuesto": "CNN con LSTM", "modelos": ["CNN", "LSTM", "Transformer", "DNN/MLP"], "detalle": "CNN, LSTM, Transformer y BPNN de 512, 256 y 128", "valores": "Exactitud en ISCXVPN2016: LSTM 95,9 %, CNN 94,95 %", "particion": "5 pliegues",
     "repeticiones": "5 pliegues", "dispersion": "no", "prueba": "ninguna", "nota": "Conjuntos genéricos"},
    {"doi": "10.1038/s41598-026-52524-5", "dataset": "UAVIDS-2025", "entrada": "flujos de red simulados", "propuesto": "CNN 1D residual", "modelos": ["CNN", "LSTM", "RNN", "DNN/MLP", "BiLSTM"], "detalle": "CNN, RNN, LSTM, ANN y BiLSTM", "valores": "LSTM de dos capas: F1 98,60 %; propuesto: exactitud 99,68 ± 0,12 %", "particion": "aleatoria, con validación cruzada",
     "repeticiones": "5 semillas", "dispersion": "sí", "prueba": "t de Student pareada", "nota": "No menciona excluir FlowID ni las IP; sus cifras superan a las de los autores"},
    {"doi": "10.17762/ijritcc.v11i9.8338", "dataset": "CICIDS2017", "entrada": "flujos de red", "propuesto": "CNN con aprendizaje activo", "modelos": ["CNN-LSTM", "CNN", "DNN/MLP"], "detalle": "CNN-LSTM, CNN y EDFFNN", "valores": "No se pudieron asociar las cifras a cada modelo en el texto", "particion": "aleatoria 70/30", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": ""},
    {"doi": "10.47000/tjmcs.1828534", "dataset": "UAVIDS-2025 y T-ITS", "entrada": "flujos de red simulados", "propuesto": "aprendizaje federado con MLP", "modelos": ["DNN/MLP", "CNN", "LSTM"], "detalle": "MLP compacto federado; DNN, CNN y LSTM de los autores del conjunto de datos", "valores": "F1: FedAvg 93 %, FedProx 84 %; referencias de los autores: DNN 93 %, CNN 93 %, LSTM 94 %", "particion": "aleatoria 80/20, con 176 clientes por IP de origen",
     "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna sobre modelos", "nota": ""},
    {"doi": "10.1109/pimrc59610.2024.10817244", "dataset": "UAV-IDS-2020", "entrada": "flujos convertidos a matriz", "propuesto": "CNN 2D", "modelos": ["CNN"], "detalle": "CNN 1D y CNN 1D con PCA", "valores": "CNN 1D: exactitud cerca de 75 % en modo BF; propuesto 99,99 %", "particion": "aleatoria", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": ""},
    {"doi": "10.48550/arxiv.2607.17025", "dataset": "UNSW-NB15 y Edge-IIoT", "entrada": "flujos de red", "propuesto": "aprendizaje federado con destilación", "modelos": ["DNN/MLP", "CNN", "CNN-LSTM"], "detalle": "DNN, CNN 1D y CNN 1D con LSTM", "valores": "Mejora de exactitud de 0,83 a 4,98 % en UNSW-NB15", "particion": "aleatoria 80/20", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Preprint; conjuntos genéricos"},
    {"doi": "10.70102/afts.2026.1836.229", "dataset": "UAVIDS-2025 y UAV-GCS", "entrada": "flujos de red simulados", "propuesto": "comparación de modelos", "modelos": ["DNN/MLP", "LSTM", "CNN"], "detalle": "FFN de 128 y 64, LSTM de 64 y CNN 1D", "valores": "En UAVIDS-2025, el texto da 97,65 % de exactitud para FFN y 100 % para LightGBM", "particion": "aleatoria 80/20", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna",
     "nota": "No menciona excluir FlowID ni las IP, y un 100 % sugiere fuga"},
    {"doi": "sin-doi:dragon", "dataset": "UAVIDS-2025", "entrada": "flujos de red simulados", "propuesto": "red de grafos con refuerzo", "modelos": ["DNN/MLP", "CNN-LSTM"], "detalle": "ANN-GA y ConvLSTM, tomados de trabajos publicados", "valores": "Exactitud: ANN-GA 91,8 %, ConvLSTM 95,3 %; propuesto 98,7 %", "particion": "no identificada", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Compara con cifras publicadas de otros conjuntos"},
    {"doi": "sin-doi:gatv2", "dataset": "UAVIDS-2025 y T-ITS", "entrada": "flujos de red simulados", "propuesto": "GATv2 con BiLSTM y XGBoost", "modelos": ["BiLSTM", "GNN", "Autoencoder"], "detalle": "BiLSTM, GNN y autoencoder", "valores": "F1 macro: BiLSTM 77,87 %, GNN 94,60 %; propuesto 98,18 %", "particion": "aleatoria estratificada, 72 % de entrenamiento", "repeticiones": "no reportadas", "dispersion": "no", "prueba": "ninguna", "nota": "Usa las IP como nodos del grafo"},
]

EXCLUIDOS_TEXTO_COMPLETO = [
    {"doi": "10.3390/s23198077", "motivo": "compara con trabajos previos sin identificar sus modelos"},
    {"doi": "10.3389/fdata.2021.782902", "motivo": "redes de IoT, no de drones"},
    {"doi": "10.1109/tvt.2023.3302814", "motivo": "señal de radio 5G, no datos de tráfico"},
    {"doi": "10.21203/rs.3.rs-271920/v1", "motivo": "aprendizaje por refuerzo, sin modelo de referencia"},
    {"doi": "10.32604/jcs.2023.042486", "motivo": "sin cifras asociables a un modelo de referencia"},
    {"doi": "10.17762/ijritcc.v11i6s.6960", "motivo": "enrutamiento, sin modelo de referencia de Deep Learning"},
    {"doi": "10.1109/gcwkshps68340.2025.11590964", "motivo": "aprendizaje cuántico frente a SVM, sin Deep Learning de referencia"},
    {"doi": "10.47000/tjmcs.1568820", "motivo": "sin cifras asociables a un modelo de Deep Learning"},
    {"doi": "10.55041/ijsrem68442", "motivo": "redes autónomas en general, sin cifras de referencia"},
    {"doi": "sin-doi:ensemble", "motivo": "sin cifras asociables a un modelo de referencia"},
    {"doi": "sin-doi:iod", "motivo": "clasificadores clásicos, sin Deep Learning de referencia"},
]


# Estudios Q1 leídos con el acceso de la biblioteca de la Universidad del Norte
BIBLIOTECA = json.loads((CARPETA / "estudios_biblioteca.json").read_text(encoding="utf-8"))
ESTUDIOS += BIBLIOTECA["incluidos"]
EXCLUIDOS_TEXTO_COMPLETO += BIBLIOTECA["excluidos"]
# Estudios Q1 de la verificación de cobertura de sinónimos, que no están entre los registros de las bases
COBERTURA = json.loads((CARPETA / "estudios_cobertura.json").read_text(encoding="utf-8"))
ESTUDIOS += COBERTURA["incluidos"]
EXCLUIDOS_TEXTO_COMPLETO += COBERTURA["excluidos"]


def clave(titulo):
    return re.sub(r"[^a-z0-9]", "", (titulo or "").lower())[:70]


# Decisiones de cribado y de recuperación dentro de la hoja de registros, de donde salen los conteos de PRISMA
ruta_registros = CARPETA / "registros_depurados.csv"
registros = list(csv.DictReader(ruta_registros.open(encoding="utf-8-sig")))
cribado = {clave(r["titulo"]): r for r in csv.DictReader((CARPETA / "cribado_titulo_resumen.csv").open(encoding="utf-8-sig"))}
recuperados = {clave(r["titulo"]): int(r["caracteres"]) for r in csv.DictReader((CARPETA / "recuperacion_texto.csv").open(encoding="utf-8-sig"))}
for r in registros:
    k = clave(r["titulo"])
    r["cribado"] = cribado[k]["decision"] if k in cribado else ""
    r["motivo_cribado"] = cribado[k]["motivo"] if k in cribado else ""
    r["texto_completo"] = ("recuperado" if recuperados.get(k, 0) > 3000 else "no recuperado") if r["cribado"] == "elegible" else ""
    if r["doi"] == "10.1109/cns66487.2025.11194990":
        # El artículo del conjunto de datos no menciona Deep Learning en el resumen, y el filtro automático lo retiró
        r["cribado"], r["motivo_cribado"], r["texto_completo"] = "elegible", "artículo del conjunto de datos, rescatado del filtro automático", "recuperado"

ruta_meta = CARPETA / "estudios_metadatos.json"
metadatos = json.loads(ruta_meta.read_text(encoding="utf-8")) if ruta_meta.exists() else {}
for e in ESTUDIOS + EXCLUIDOS_TEXTO_COMPLETO:
    doi = e["doi"]
    if doi in metadatos and "error" not in metadatos[doi]:
        continue
    try:
        if doi in SIN_DOI:
            url = "https://api.openalex.org/works?per-page=1&filter=" + urllib.parse.quote("title.search:" + SIN_DOI[doi])
            w = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read())["results"][0]
        else:
            url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="/.")
            w = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read())
        b = w.get("biblio") or {}
        ubicacion = w.get("primary_location") or {}
        metadatos[doi] = {"titulo": w.get("title"), "anio": w.get("publication_year"), "autores": [a["author"]["display_name"] for a in w.get("authorships", [])],
                          "fuente": (ubicacion.get("source") or {}).get("display_name"), "tipo": w.get("type"), "volumen": b.get("volume"), "numero": b.get("issue"),
                          "paginas": "-".join(x for x in (b.get("first_page"), b.get("last_page")) if x), "url": ubicacion.get("landing_page_url") or (w.get("open_access") or {}).get("oa_url")}
    except Exception as error:
        metadatos[doi] = {"error": repr(error)[:80]}
    time.sleep(1.2)
ruta_meta.write_text(json.dumps(metadatos, ensure_ascii=False, indent=1), encoding="utf-8")

with (CARPETA / "extraccion_estudios.csv").open("w", encoding="utf-8-sig", newline="") as f:
    campos = ["autores", "anio", "titulo", "fuente", "doi", "dataset", "entrada", "propuesto", "modelos", "detalle", "valores", "particion", "repeticiones", "dispersion", "prueba", "nota"]
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    for e in ESTUDIOS:
        m = metadatos.get(e["doi"], {})
        w.writerow({**{k: e.get(k, "") for k in campos}, "modelos": "; ".join(e["modelos"]), "autores": "; ".join(m.get("autores", [])), "anio": m.get("anio", ""), "titulo": m.get("titulo", ""), "fuente": m.get("fuente", "")})

frecuencia = Counter(mod for e in ESTUDIOS for mod in e["modelos"])
con_dataset = [e for e in ESTUDIOS if "UAVIDS-2025" in e["dataset"]]
frecuencia_dataset = Counter(mod for e in con_dataset for mod in e["modelos"])
total = len(ESTUDIOS)
filas = [{"modelo": k, "estudios": v, "porcentaje": round(100 * v / total, 1), "estudios_uavids": frecuencia_dataset[k], "porcentaje_uavids": round(100 * frecuencia_dataset[k] / len(con_dataset), 1)} for k, v in frecuencia.most_common()]
with (CARPETA / "frecuencia_modelos.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["modelo", "estudios", "porcentaje", "estudios_uavids", "porcentaje_uavids"])
    w.writeheader()
    w.writerows(filas)

with ruta_registros.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(registros[0]))
    w.writeheader()
    w.writerows(registros)

elegibles = [r for r in registros if r["cribado"] == "elegible"]
evaluados = [r for r in elegibles if r["texto_completo"] != "no recuperado"]
print("estudios incluidos:", total, "| excluidos a texto completo:", len(EXCLUIDOS_TEXTO_COMPLETO), "| con UAVIDS-2025:", len(con_dataset))
n_cobertura = len(COBERTURA["incluidos"]) + len(COBERTURA["excluidos"])
assert len(evaluados) + n_cobertura == total + len(EXCLUIDOS_TEXTO_COMPLETO), (len(evaluados), n_cobertura, total + len(EXCLUIDOS_TEXTO_COMPLETO))
print("PRISMA | depurados:", len(registros), "| filtro automático:", sum(1 for r in registros if r["cribado"] == ""), "| cribados:", sum(1 for r in registros if r["cribado"]),
      "| excluidos en cribado:", sum(1 for r in registros if r["cribado"] == "excluido"), "| buscados a texto completo:", len(elegibles), "| no recuperados:", len(elegibles) - len(evaluados), "| evaluados:", len(evaluados))
print("no recuperados por editorial:", Counter(r["doi"].split("/")[0] if r["doi"] else "sin DOI" for r in elegibles if r["texto_completo"] == "no recuperado").most_common(4))
print("metadatos con error:", [d for d, m in metadatos.items() if "error" in m])
for fila in filas:
    print(f"{fila['modelo']:<14} {fila['estudios']:>3}  {fila['porcentaje']:>5} % | UAVIDS {fila['estudios_uavids']:>2}  {fila['porcentaje_uavids']:>5} %")
print("con alguna prueba:", sum(1 for e in ESTUDIOS if not e["prueba"].startswith("ninguna")), "| con dispersión:", sum(1 for e in ESTUDIOS if e["dispersion"] == "sí"), "| con agrupación:", sum(1 for e in ESTUDIOS if "agrupa" in e["particion"] and "no identifica" not in e["particion"]))
for e in ESTUDIOS:
    m = metadatos.get(e["doi"], {})
    print(" -", (m.get("autores") or ["?"])[0], "|", len(m.get("autores") or []), "|", m.get("anio"), "|", (m.get("titulo") or "?")[:52], "|", (m.get("fuente") or "?")[:30])
