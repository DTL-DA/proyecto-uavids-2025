"""Cribado por título y resumen de los registros que pasaron el filtro automático de depurar_registros.py.

Elegible: el estudio detecta o clasifica intrusiones o ataques en redes o sistemas de drones (UAV, FANET, Internet of Drones)
a partir de datos tabulares de tráfico o del sistema, usando Deep Learning. La decisión se tomó leyendo el título de cada
registro y queda por registro en cribado_titulo_resumen.csv. Cuando entren registros nuevos (exportaciones de IEEE, Scopus
o WoS), quedan "sin cribar" para que se decidan y no queden incluidos ni excluidos por omisión.

No debe volver a ejecutarse: identifica las decisiones por su posición en el archivo original. Las decisiones de las bases nuevas
están en cribado_bases_nuevas.csv y se agregan con integrar_bases_nuevas.py.
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
CARPETA = Path(__file__).parent
ENTRADA = CARPETA / "registros_depurados.csv"
SALIDA = CARPETA / "cribado_titulo_resumen.csv"


def clave(titulo):
    return re.sub(r"[^a-z0-9]", "", (titulo or "").lower())[:70]


filas = list(csv.DictReader(ENTRADA.open(encoding="utf-8-sig")))

# Posiciones en el orden con que se cribó registros_depurados.csv el 03/11/2026; abajo se convierten en títulos
ELEGIBLES = [1, 3, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 30, 33, 35, 36, 37, 38, 39, 42, 44, 46, 47, 48, 49, 50, 51, 53, 59, 60, 61, 62,
             69, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 92, 94, 95, 97, 98, 99, 100, 103, 104, 105, 106, 111, 115, 116, 118, 119, 121, 122, 124, 125, 126,
             128, 131, 135, 137, 138, 141, 142, 143, 145, 146, 148, 149, 153, 154, 155, 157, 158, 159, 160, 161, 163, 165, 168, 169, 174, 177, 186, 187, 191, 192, 193, 196, 198,
             200, 201, 202, 203, 212, 213, 214, 216, 219, 220, 221, 223, 227, 231, 232, 238, 242, 243, 245, 247, 248, 250, 251,
             255, 265, 272, 276, 281, 289, 298, 307, 314, 329, 346, 414, 416, 417, 418, 420, 421, 422, 423, 426, 430, 431, 432, 435, 437, 439, 441, 442, 443, 444, 445, 450, 465, 466, 468, 472, 483, 505]
elegibles = {clave(filas[i]["titulo"]) for i in ELEGIBLES}

# Motivos de exclusión de los registros de tema afín; el resto del conjunto no trata de intrusiones en redes de drones
AFINES = {
    2: "revisión", 5: "revisión", 57: "capítulo general", 43: "revisión de forense", 73: "capítulo general", 299: "taxonomía, no estudio primario", 449: "revisión",
    12: "solo suplantación de GPS", 34: "solo suplantación de GPS", 41: "solo suplantación de GPS", 45: "solo suplantación de GPS", 52: "solo suplantación de GPS", 70: "solo suplantación de GPS", 133: "solo suplantación de GNSS",
    32: "imagen de GPS, no datos tabulares", 91: "cyber-physical con imágenes", 134: "visión, no datos tabulares", 4: "solo suplantación e interferencia, sin Deep Learning", 58: "solo interferencia de GPS",
    93: "señales RF, no datos tabulares", 113: "señales RF, no datos tabulares", 136: "señales RF, no datos tabulares", 16: "señales RF, no intrusiones", 222: "solo interferencia", 359: "solo interferencia", 376: "espectrogramas RF", 451: "espectrogramas RF",
    56: "intrusión física, con cámara", 67: "intrusión física, con cámara", 64: "intrusión física en video", 139: "intrusión física", 204: "intrusión física", 206: "intrusión física, con radar", 237: "detección acústica de drones", 170: "detección acústica de drones",
    63: "red móvil ad hoc, no de drones", 66: "estaciones de carga", 181: "vehículos conectados, no drones", 184: "vehículos conectados, no drones", 178: "artículo de datos, sin modelo de referencia", 185: "artículo de datos, sin modelo de referencia", 249: "repositorio de código", 489: "repositorio de código",
}
afines = {clave(filas[i]["titulo"]): motivo for i, motivo in AFINES.items()}
assert not elegibles & set(afines), elegibles & set(afines)

salida, conteo = [], Counter()
for f in filas:
    if f["exclusion_filtro"] and not f["exclusion_filtro"].startswith("sin resumen"):
        continue
    k = clave(f["titulo"])
    if k in elegibles:
        decision, motivo = "elegible", ""
    elif k in afines:
        decision, motivo = "excluido", afines[k]
    else:
        decision, motivo = "excluido", "no detecta intrusiones en redes de drones con datos tabulares"
    conteo[decision] += 1
    salida.append({"titulo": f["titulo"], "anio": f["anio"], "doi": f["doi"], "fuente_publicacion": f["fuente_publicacion"], "citas": f["citas"], "bases": f["bases"], "cadenas": f["cadenas"],
                   "acceso_abierto": f["acceso_abierto"], "decision": decision, "motivo": motivo})

with SALIDA.open("w", encoding="utf-8-sig", newline="") as s:
    w = csv.DictWriter(s, fieldnames=list(salida[0]))
    w.writeheader()
    w.writerows(salida)
print("cribados:", len(salida), dict(conteo))
print("motivos:", Counter(x["motivo"] for x in salida if x["decision"] == "excluido").most_common(6))
