"""Resume, por cada texto recuperado, los pasajes donde el estudio compara contra modelos de referencia.

Uso: python pasajes_referencia.py <carpeta_textos> <salida.txt>. Corta la lista de referencias para no contar obras
solo citadas. Es una ayuda de lectura: la decisión y los valores de extraccion_estudios.py salen de revisar cada pasaje.
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
CARPETA = Path(__file__).parent
TEXTOS = Path(sys.argv[1])
SALIDA = Path(sys.argv[2])

FAMILIAS = {"DNN/MLP": r"\bDNNs?\b|\bMLPs?\b|multi-?layer perceptron|deep neural network|fully[- ]connected (?:deep )?(?:neural )?network|\bANNs?\b|\bFNNs?\b|\bDFFN\w*", "CNN": r"\b(?:1D-?|2D-?)?CNNs?\b|convolutional neural network",
            "LSTM": r"\bLSTMs?\b|long short-term memory", "BiLSTM": r"\bBi-?LSTMs?\b", "GRU": r"\bGRUs?\b|\bBi-?GRUs?\b", "RNN": r"\bRNNs?\b|recurrent neural network", "CNN-LSTM": r"CNN[- +]+(?:Bi-?)?LSTM|ConvLSTM|LSTM[- +]+CNN",
            "Autoencoder": r"[Aa]uto-?encoder", "Transformer": r"Transformer|\bViT\b", "GNN": r"\bGNNs?\b|\bGCNs?\b|\bGATs?\b|graph neural", "TabNet": r"TabNet", "DBN": r"\bDBNs?\b|deep belief", "ResNet": r"Res[Nn]et[- ]?\d*"}
PATRON_FAMILIA = re.compile("|".join(FAMILIAS.values()))
COMPARA = re.compile(r"compar|baseline|benchmark|state-of-the-art|outperform|existing (?:method|model|work|approach)|other (?:model|method|network|approach)", re.I)
ESTADISTICA = re.compile(r"Wilcoxon|t-test|McNemar|Friedman|ANOVA|DeLong|statistically significant|confidence interval", re.I)
PARTICION = re.compile(r"\d+[- ]?fold|cross[- ]validation|\d+\s?%\s*(?:/|:|for training|of the (?:data|dataset)|training)|hold-?out|train(?:ing)?[/ -]+(?:validation|test)", re.I)
REPETICION = re.compile(r"(?:repeated|repeat|runs?|trials?|seeds?)\b[^.]{0,40}\b(?:\d+|five|ten|three|times)\b|\b(?:\d+|five|ten|three)\b[^.]{0,20}\b(?:runs|trials|seeds|times)\b", re.I)
DATOS = re.compile(r"UAVIDS(?:-2025)?|UAV-?NIDD|UAV-?IDS\S*|CIC-?IDS\S*|CSE-?CIC\S*|NSL-?KDD|UNSW-?NB15|KDD-??(?:Cup)?99|TON[-_]IoT|Bot-?IoT|WSN-?DS|IoT-?23|Edge-?IIoT\S*|ALFA|N-?BaIoT|own dataset|simulat\w+ (?:data|dataset)|NS-?3", re.I)


def cuerpo(texto):
    posiciones = [m.start() for m in re.finditer(r"\n\s*(?:R ?E ?F ?E ?R ?E ?N ?C ?E ?S|References|Bibliography)\s*\n", texto)]
    return texto[:posiciones[-1]] if posiciones and posiciones[-1] > len(texto) * 0.4 else texto


registros = list(csv.DictReader((CARPETA / "recuperacion_texto.csv").open(encoding="utf-8-sig")))
bloques = []
for r in registros:
    if int(r["caracteres"]) == 0:
        continue
    ruta = TEXTOS / r["carpeta"] / "doc.txt"
    if not ruta.exists():
        continue
    texto = cuerpo(ruta.read_text(encoding="utf-8", errors="replace"))
    plano = re.sub(r"\s+", " ", texto)
    conteo = Counter()
    for familia, patron in FAMILIAS.items():
        conteo[familia] = len(re.findall(patron, plano))
    frases = re.split(r"(?<=[.;:])\s+(?=[A-Z])", plano)
    pasajes = [f for f in frases if COMPARA.search(f) and PATRON_FAMILIA.search(f) and 40 < len(f) < 420]
    pasajes.sort(key=lambda f: -(len(set(PATRON_FAMILIA.findall(f))) + 2 * bool(re.search(r"\d+\.\d+\s*%?", f))))
    extra = {"datos": Counter(m.group(0) for m in DATOS.finditer(plano)).most_common(3), "particion": [m.group(0)[:40] for m in PARTICION.finditer(plano)][:3],
             "repeticion": [m.group(0)[:50] for m in REPETICION.finditer(plano)][:2], "dispersion": "±" in plano, "estadistica": sorted({m.group(0) for m in ESTADISTICA.finditer(plano)})}
    bloques.append({"r": r, "conteo": {k: v for k, v in conteo.most_common() if v}, "pasajes": pasajes[:4], "extra": extra})

with SALIDA.open("w", encoding="utf-8") as s:
    for n, b in enumerate(bloques):
        r = b["r"]
        s.write(f"\n[{r['carpeta']}] {r['anio']} {r['titulo'][:90]} | {r['doi']}\n  familias: {b['conteo']}\n  extra: {b['extra']}\n")
        for p in b["pasajes"]:
            s.write(f"  > {p[:330]}\n")
print("textos resumidos:", len(bloques), "de", len(registros), "registros")
