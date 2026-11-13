from .base import code, md

md("s3", r"""
## 3. Revisión sistemática del estado del arte

En esta sección se extiende la revisión del Entregable 1, que identificó los tres modelos base, para seleccionar con un puntaje explícito el Top 10 de modelos de Deep Learning, describirlos uno por uno, compararlos críticamente y formular el *gap* de investigación. La revisión del Entregable 1 se conserva completa en su cuaderno; aquí se resume lo necesario y se documenta lo nuevo: la actualización de las cadenas, una cadena ampliada por familias de modelos, la búsqueda transversal de mecanismos, dos campos nuevos de la hoja de extracción (código público y líneas base ajustadas) y el puntaje de selección.

### 3.1 Estrategia de búsqueda

**Cadenas, bases y fechas.** El Entregable 1 ejecutó tres cadenas sobre título y resumen, de 2019 a 2026: C1, general; C2, centrada en los ataques del conjunto de datos, y C3, sobre el uso de UAVIDS-2025. Se corrieron en OpenAlex y arXiv el 3 de noviembre de 2026 y en IEEE Xplore, Scopus y Web of Science el 4 de noviembre, con el acceso de la biblioteca de la Universidad del Norte, y se completaron con una verificación de cobertura de sinónimos el 5 de noviembre. Para este entregable se agregan, el 12 de noviembre de 2026 en OpenAlex y arXiv, la repetición de C1 a C3, para captar lo publicado después; C4, que amplía el concepto de método con las familias candidatas al Top 10, y cuatro cadenas transversales, T1 a T4, que buscan los mecanismos en otras modalidades, como pide el enunciado. Las cadenas nuevas para IEEE Xplore, Scopus y Web of Science están listas en `literatura/consultas_e2_wos_scopus_ieee.md`; su ejecución queda pendiente, y se declara como limitación, porque esas bases exigen el acceso institucional.

| Cadena | Consulta | Periodo |
|---|---|---|
| C1 | (drone OR UAV OR "unmanned aerial" OR FANET) AND ("intrusion detection" OR "attack detection") AND ("deep learning" OR "neural network" OR CNN OR LSTM OR MLP) | 2019–2026 |
| C2 | (drone OR UAV OR "unmanned aerial" OR FANET) AND (swarm OR "ad hoc" OR "network traffic") AND (blackhole OR wormhole OR sybil OR flooding OR cyberattack) AND ("deep learning" OR "neural network" OR CNN OR LSTM OR MLP) | 2019–2026 |
| C3 | UAVIDS OR "UAVIDS-2025" | 2019–2026 |
| C4 | (drone OR UAV OR "unmanned aerial" OR FANET OR "Internet of Drones") AND ("intrusion detection" OR "attack detection" OR "anomaly detection") AND (GRU OR "gated recurrent" OR BiLSTM OR "bidirectional LSTM" OR transformer OR TabNet OR "Kolmogorov-Arnold" OR autoencoder OR "deep belief" OR ResNet OR residual OR "CNN-LSTM" OR "graph neural" OR "temporal convolutional" OR mamba OR "state space") | 2019–2026 |
| T1 | ("supervised contrastive" OR "contrastive learning") AND (image OR "computer vision" OR visual) AND (transfer OR adaptation) | 2018–2026 |
| T2 | ("mixture of experts" OR "gated experts") AND ("natural language" OR "language model" OR vision) AND (transfer OR adaptation) | 2018–2026 |
| T3 | ("domain generalization" OR "invariant risk minimization" OR "distributionally robust") AND (image OR vision OR speech) AND (transfer OR adaptation) | 2018–2026 |
| T4 | ("Kolmogorov-Arnold" OR "feature tokenizer" OR "tabular transformer") AND ("time series" OR vision OR image OR language) AND (transfer OR adaptation) | 2018–2026 |

**Periodo.** Las cadenas cubren los últimos ocho años. Los trabajos fundacionales anteriores, como Elman (1990) para la RNN o Hinton et al. (2006) para la DBN, se citan como fuente de cada arquitectura y no entran al conteo de frecuencia, porque no abordan el problema.

**Criterios de inclusión y exclusión.** Son los del Entregable 1: se incluye un estudio si detecta o clasifica intrusiones o ataques en redes o sistemas de drones a partir de datos tabulares de tráfico o del sistema, usa Deep Learning, reporta métricas verificables y compara contra al menos un modelo de Deep Learning; se excluye si no es un estudio primario o si trabaja con imágenes, señales de radio, solo suplantación de GPS o anomalías de vuelo. La calidad de la fuente se registra con el cuartil SJR 2025 de SCImago (2026), y la disponibilidad de código y de detalles para reproducir, en los campos nuevos de la hoja de extracción.
""")

code("s3-conteos", r"""
import csv
LIT1, LIT2 = D.E1 / "literatura", RAIZ_E2 / "literatura"
leer_csv = lambda ruta: list(csv.DictReader(ruta.open(encoding="utf-8-sig")))
REG_E1, COB_E1, REG_E2 = leer_csv(LIT1 / "registros_depurados.csv"), leer_csv(LIT1 / "cribado_cobertura_sinonimos.csv"), leer_csv(LIT2 / "registros_nuevos_e2.csv")
EXT = leer_csv(LIT1 / "extraccion_estudios.csv") + leer_csv(LIT2 / "extraccion_estudios_e2.csv")
REPRO = {r["doi"]: r for r in leer_csv(LIT2 / "reproducibilidad_estudios.csv")}
resumen_e2 = json.loads((LIT2 / "exportaciones" / "2026-11-12_resumen_e2.json").read_text(encoding="utf-8"))
conteo_e2 = pd.DataFrame(resumen_e2["resultados"]).pivot(index="fuente", columns="cadena", values="total")
conteo_e1 = pd.DataFrame({"C1": [323, 11, 481, 132, 523], "C2": [40, 3, 40, 11, 18], "C3": [220, 29, 81, 16, 26]}, index=["OpenAlex", "arXiv", "IEEE Xplore", "Web of Science", "Scopus"])
conteos = pd.concat({"Entregable 1 (3 y 4 de noviembre)": conteo_e1, "Entregable 2 (12 de noviembre)": conteo_e2}, axis=1).reindex(["OpenAlex", "arXiv", "IEEE Xplore", "Scopus", "Web of Science"])
tabla(conteos.reset_index(names="Base").map(lambda v: "pendiente" if pd.isna(v) else (f"{int(v):,}".replace(",", " ") if isinstance(v, (int, float)) else v)), indice=False)
elegibles_e1 = [r for r in REG_E1 if r["cribado"] == "elegible"]
cob_eleg = [r for r in COB_E1 if r["decision"] in ("elegible", "dudoso")]
PRISMA = {"exportados_e1": 583 + 42 + 602 + 567 + 159, "depurados_e1": len(REG_E1), "filtro_e1": sum(r["cribado"] == "" for r in REG_E1), "excluidos_e1": sum(r["cribado"] == "excluido" for r in REG_E1),
          "elegibles_e1": len(elegibles_e1), "no_recuperados_e1": sum(r["texto_completo"] == "no recuperado" for r in elegibles_e1), "cob_elegibles": len(cob_eleg), "cob_incluidos": 4,
          "unicos_e2": 860, "ya_revisados_e2": 860 - len(REG_E2), "nuevos_e2": len(REG_E2), "filtro_e2": sum(r["cribado"] == "" for r in REG_E2), "excluidos_e2": sum(r["cribado"] == "excluido" for r in REG_E2),
          "elegibles_e2": sum(r["cribado"] in ("elegible", "dudoso") for r in REG_E2), "no_recuperados_e2": sum(r["texto_completo"] == "no recuperado" for r in REG_E2),
          "incluidos_e2": sum(r["texto_completo"] == "incluido" for r in REG_E2), "incluidos": len(EXT)}
PRISMA["evaluados_e1"] = PRISMA["elegibles_e1"] - PRISMA["no_recuperados_e1"]
PRISMA["incluidos_e1_bases"] = 38
""")

code("s3-prisma", r"""
fig, eje = plt.subplots(figsize=(12.5, 6.8))
eje.axis("off")
eje.set(xlim=(0, 15), ylim=(0, 11))
def caja(x, yy, texto, ancho=3.4, alto=1.05, relleno="white"):
    eje.add_patch(plt.Rectangle((x - ancho / 2, yy - alto / 2), ancho, alto, facecolor=relleno, edgecolor=NEGRO, lw=0.8))
    eje.text(x, yy, texto, ha="center", va="center", fontsize=7.2)
def flecha(a, b):
    eje.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=NEGRO, lw=0.8))
P = PRISMA
for etiqueta_fase, yy in (("Identificación", 10), ("Cribado", 7.6), ("Elegibilidad", 5.2), ("Inclusión", 2.2)):
    eje.text(0.2, yy, etiqueta_fase, rotation=90, va="center", fontsize=8, color=GRIS_OSCURO)
caja(2.6, 10, f"Entregable 1: {P['exportados_e1']:,} registros\nde 5 bases (C1 a C3)".replace(",", " "), relleno=GRIS_CLARO)
caja(2.6, 7.6, f"{P['depurados_e1']:,} sin duplicados\n{P['filtro_e1']} retirados por el filtro automático\n{P['excluidos_e1']} excluidos por título y resumen".replace(",", " "), alto=1.3)
caja(2.6, 5.2, f"{P['elegibles_e1']} elegibles\n{P['no_recuperados_e1']} sin texto completo\n{P['evaluados_e1']} evaluados", alto=1.3)
caja(6.6, 10, f"Verificación de cobertura\n(C1 y C2 ampliadas)", relleno=GRIS_CLARO)
caja(6.6, 6.4, f"{P['cob_elegibles']} elegibles o dudosos\n5 Q1 leídos", alto=1.1)
caja(10.6, 10, f"Entregable 2: {P['unicos_e2']} únicos\nen C1 a C4 (OpenAlex y arXiv)", relleno=GRIS_CLARO)
caja(10.6, 7.6, f"{P['ya_revisados_e2']} ya revisados en el E1\n{P['nuevos_e2']} nuevos: {P['filtro_e2']} por el filtro,\n{P['excluidos_e2']} por título y resumen", alto=1.3)
caja(10.6, 5.2, f"{P['elegibles_e2']} elegibles o dudosos\n{P['no_recuperados_e2']} sin texto completo\n{P['incluidos_e2']} evaluado", alto=1.3)
caja(6.6, 2.2, f"{P['incluidos']} estudios incluidos\n{P['incluidos_e1_bases']} de las bases del E1, {P['cob_incluidos']} de la cobertura\ny {P['incluidos_e2']} de la actualización del E2", ancho=4.6, alto=1.3, relleno=GRIS_CLARO)
caja(13.4, 2.2, f"Búsqueda transversal T1 a T4\nmecanismos: {len(leer_csv(LIT2 / 'mecanismos_transferibles.csv'))}\n(no entran al conteo)", ancho=2.6, alto=1.3)
for a, b in (((2.6, 9.45), (2.6, 8.27)), ((2.6, 6.93), (2.6, 5.87)), ((2.6, 4.53), (5.0, 2.85)), ((6.6, 9.45), (6.6, 6.97)), ((6.6, 5.83), (6.6, 2.87)),
             ((10.6, 9.45), (10.6, 8.27)), ((10.6, 6.93), (10.6, 5.87)), ((10.6, 4.53), (8.2, 2.85))):
    flecha(a, b)
eje.set_title("Diagrama de flujo PRISMA de la revisión, Entregables 1 y 2", fontsize=10)
plt.show()
""")

md("s3-prisma-int", r"""
**Interpretación.** La actualización del 12 de noviembre devolvió en C1 y C2 los mismos conteos del 3 de noviembre, de modo que en nueve días no aparecieron estudios nuevos con las cadenas originales. Lo nuevo vino de C3 y de la cadena ampliada C4: de 860 registros únicos, 766 ya se habían revisado en el Entregable 1 y 94 eran nuevos; el filtro automático retiró 52, el cribado por título y resumen excluyó 34, casi todos por tratar anomalías de vuelo, de sensores o de imágenes, y quedaron 5 elegibles y 3 dudosos. De esos 8, solo uno tenía texto completo accesible (Tasfia et al., 2026), que se incluye; los otros 7 quedan sin evaluar. El total de estudios incluidos pasa de 42 a 43.

Hay que tener en cuenta que entre los no recuperados hay dos estudios relevantes para este benchmark: Di et al. (2026), que según su resumen comparan FT-Transformer, MLP y árboles en IDS de UAV con 5 semillas y corrección de Holm, y el de TopoXNet, que usa UAVIDS-2025. Es decir, la revisión tiene un sesgo de acceso, que ya declaró el Entregable 1, y que aquí podría afectar el último puesto del Top 10, como se cuantifica en la sección 3.2.

Por tal motivo, se mantiene la regla del Entregable 1, según la cual solo cuentan los estudios evaluados a texto completo, y se reporta la sensibilidad del Top 10 a incluir los no recuperados con lo que dicen sus resúmenes. Las decisiones de cada registro están en `literatura/registros_nuevos_e2.csv`.

**Quién realizó la selección.** Al ser un trabajo de autor único, hubo un solo revisor, de modo que no aplica el acuerdo de Cohen entre revisores; es una limitación frente a un cribado doble. El cribado y la extracción se hicieron con apoyo de un asistente de inteligencia artificial (Claude, de Anthropic) bajo la supervisión del autor, con búsqueda dirigida de las tablas y los pasajes de cada texto.
""")

md("s3-extraccion", r"""
**Hoja de extracción.** La hoja tiene los campos fijos que pide el enunciado: arquitectura, dataset, partición, métricas, repeticiones o semillas, pruebas estadísticas, código público y líneas base ajustadas. Los dos últimos son nuevos en este entregable: el código se buscó en el texto completo de cada estudio, con una revisión manual de cada enlace hallado, y se considera ajustada una línea base solo si el estudio declara haberla ajustado con un procedimiento comparable al del modelo propuesto.
""")

code("s3-extraccion-tabla", r"""
import unicodedata

METADATOS = json.loads((LIT1 / "estudios_metadatos.json").read_text(encoding="utf-8"))
METADATOS["10.22266/ijies2025.0531.33"]["autores"] = ["Aws Hamed Hamad", "Noor Khalid Hussein", "Abdulrahman M. Abdulghani"]
METADATOS["sin-doi:dragon"]["autores"] = ["P. SureshBabu", "M. Geetha"]
METADATOS["10.1007/s10207-026-01353-8"] = {"autores": ["Rifa Tasfia", "Anitha Chennamaneni", "Abhijit Kumar Nag"], "anio": 2026, "titulo": "Revisiting adversarial robustness in NIDS: a multi-dataset evaluation of test-time evasion attacks",
                                           "fuente": "International Journal of Information Security", "volumen": "25", "numero": None, "paginas": "184", "tipo": "article"}
COMPUESTOS = {"Canan Batur Şahin": ("Batur Şahin", "Canan")}
PARTICULAS = {"van", "von", "de", "der", "den", "del", "da", "du", "le", "la"}
def partes_nombre(nombre):
    # Apellido y nombres; las partículas (van, de, von...) van con el apellido: Bart van Merriënboer → van Merriënboer, Bart
    if nombre in COMPUESTOS:
        return COMPUESTOS[nombre]
    f = nombre.replace("‐", "-").split()
    i = len(f) - 2
    while i >= 1 and f[i].lower() in PARTICULAS:
        i -= 1
    return " ".join(f[i + 1:]), " ".join(f[:i + 1])
def cita_base(doi):
    m = METADATOS[doi]
    a = [partes_nombre(x)[0] for x in m["autores"]]
    autor = a[0] if len(a) == 1 else f"{a[0]} y {a[1]}" if len(a) == 2 else f"{a[0]} et al."
    return f"{autor} ({m['anio']})"
def clave_orden(t):
    # Clave de orden alfabético sin tildes ni signos diacríticos (la ı turca se ordena como i)
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower().replace("ı", "i")) if not unicodedata.combining(c))
def iniciales(nombres):
    return " ".join("-".join(p[0] + "." for p in n.split("-") if p) for n in nombres.split())
def autor_apa(apellido, nombres):
    return f"{apellido}, {iniciales(nombres)}" if nombres.strip() else apellido
def orden_autores(doi):
    # Mismo orden que la lista de referencias: autores completos y luego título
    m = METADATOS[doi]
    return clave_orden(", ".join(autor_apa(a, n) for a, n in map(partes_nombre, m["autores"])) + " " + m["titulo"])
SUFIJO = {}
for base in {cita_base(e["doi"]) for e in EXT}:
    iguales = sorted((e["doi"] for e in EXT if cita_base(e["doi"]) == base), key=orden_autores)
    if len(iguales) > 1:
        SUFIJO.update({d: "abcdef"[i] for i, d in enumerate(iguales)})
citar = lambda doi: cita_base(doi)[:-1] + SUFIJO.get(doi, "") + ")"
agrupa = lambda p: "por sesión" in p or "por nodo" in p
hoja = pd.DataFrame([{"Estudio": citar(e["doi"]), "Arquitecturas de DL": e["modelos"], "Dataset": e["dataset"], "Partición": e["particion"], "Métricas y valores": e["valores"],
                      "Repeticiones o semillas": e["repeticiones"], "Dispersión": e["dispersion"], "Prueba estadística": e["prueba"],
                      "Código público": REPRO[e["doi"]]["codigo_publico"], "Líneas base ajustadas": REPRO[e["doi"]]["baselines_ajustados"],
                      "Riesgo de fuga": "No evaluable" if e["particion"] == "no identificada" else ("Menor: agrupa filas relacionadas" if agrupa(e["particion"]) else "Alto: no agrupa filas relacionadas")}
                     for e in sorted(EXT, key=lambda e: cita_base(e["doi"]).lower())])
hoja.to_csv(LIT2 / "hoja_extraccion_e2.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_colwidth", 70):
    tabla(hoja.drop(columns=["Métricas y valores"]))
n = len(hoja)
calidad_lit = {"con código": int((hoja["Código público"] == "sí").sum()), "sin texto": int((hoja["Código público"] == "no verificable").sum()),
               "datos públicos": sum(REPRO[e["doi"]]["datos_publicos"] == "sí" for e in EXT), "agrupan": int(hoja["Riesgo de fuga"].str.startswith("Menor").sum()),
               "sin repeticiones": int((hoja["Repeticiones o semillas"] == "no reportadas").sum()), "con dispersión": int((hoja["Dispersión"] == "sí").sum()),
               "con prueba": int((~hoja["Prueba estadística"].str.startswith("ninguna")).sum()), "baselines publicados": int((hoja["Líneas base ajustadas"] == "no").sum())}
interpretar(f'''**Interpretación.** De los {n} estudios incluidos, {calidad_lit["con código"]} publica su código, y en {calidad_lit["sin texto"]} no se pudo verificar por falta de texto completo; {calidad_lit["datos públicos"]} usan conjuntos de datos públicos. En cuanto al protocolo, {calidad_lit["agrupan"]} agrupan las filas relacionadas al partir, {calidad_lit["sin repeticiones"]} no reportan repeticiones, {calidad_lit["con dispersión"]} reportan dispersión y {calidad_lit["con prueba"]} aplican alguna prueba estadística. Ninguno declara haber ajustado sus líneas base con un presupuesto equivalente al del modelo propuesto, y {calidad_lit["baselines publicados"]} se comparan directamente con cifras tomadas de otros trabajos.

Esto significa que la reproducibilidad de la literatura es muy baja: el único código disponible es el de los autores del conjunto de datos (Zeng, 2025), que su artículo ni siquiera cita, y la mayoría de las comparaciones enfrenta un modelo propuesto, ajustado, contra líneas base sin ajustar o tomadas de otros protocolos. En ese sentido, las brechas de desempeño publicadas no son comparables entre sí y tienden a favorecer al modelo propuesto.

Por lo anterior, conviene usar las cifras de la literatura solo como contexto y no como objetivo, y reimplementar los diez modelos con el mismo pipeline, la misma partición y el mismo presupuesto de ajuste, que es lo que hace la sección 4.''')
""")

md("s3-calidad", r"""
**Valoración de la calidad metodológica.** Cada estudio se valoró en tres ejes que pide el enunciado: el riesgo de fuga, según si la partición agrupa las filas relacionadas; el número de repeticiones, y la equidad en el ajuste de las líneas base. La tabla resume la distribución de los 43 estudios en esos ejes, y la valoración de cada uno está en la hoja de extracción.
""")

code("s3-calidad-tabla", r"""
ejes_calidad = pd.DataFrame({"Eje": ["Riesgo de fuga alto (no agrupa filas relacionadas)", "Riesgo de fuga menor (agrupa por sesión o nodo)", "Partición no identificada",
                                      "Sin repeticiones reportadas", "Con dispersión entre repeticiones", "Con alguna prueba estadística", "Líneas base tomadas de cifras publicadas",
                                      "Ajuste de líneas base no reportado", "Código público"],
                             "Estudios": [int(hoja["Riesgo de fuga"].str.startswith("Alto").sum()), calidad_lit["agrupan"], int((hoja["Riesgo de fuga"] == "No evaluable").sum()),
                                          calidad_lit["sin repeticiones"], calidad_lit["con dispersión"], calidad_lit["con prueba"], calidad_lit["baselines publicados"],
                                          int((hoja["Líneas base ajustadas"] == "no reportado").sum()), calidad_lit["con código"]]})
ejes_calidad["% de los incluidos"] = 100 * ejes_calidad["Estudios"] / n
tabla(ejes_calidad, {"% de los incluidos": "{:.1f}"})
""")

md("s3-transversal", r"""
**Búsqueda transversal de mecanismos.** Las cuatro cadenas T1 a T4 devolvieron miles de registros en total, porque cubren modalidades enteras, de modo que se revisaron por título y resumen los 100 más citados de cada una en OpenAlex y en arXiv, y de ellos se identificaron los mecanismos que comparten una propiedad estructural con el problema. Esta tabla no compromete la elección del modelo original del Entregable 3, pero es su insumo: cada mecanismo apunta a una de las dos debilidades que el Entregable 1 dejó abiertas, la confusión entre Blackhole y Wormhole y la caída del desempeño entre series no vistas, o a la fragilidad ante atributos faltantes que reporta la literatura.
""")

code("s3-mecanismos", r"""
mecanismos = pd.DataFrame(leer_csv(LIT2 / "mecanismos_transferibles.csv"))
tabla(mecanismos.rename(columns={"mecanismo": "Mecanismo", "familia": "Familia", "cadena": "Cadena", "aplicacion_origen": "Aplicación de origen", "fuente_primaria": "Fuente primaria", "doi": "DOI",
                                 "propiedad_compartida": "Propiedad compartida con el problema", "limitacion_que_atacaria": "Limitación que atacaría", "transferibilidad": "Transferibilidad preliminar"}))
""")

md("s3-2", r"""
### 3.2 Selección del Top 10 de modelos

**Candidatos y puntaje.** Los candidatos son las familias de Deep Learning de la tabla de frecuencia de los 43 estudios incluidos, sin las tres de los modelos base, que entran aparte. El puntaje se definió antes de calcularlo y combina los cinco criterios del enunciado más la frecuencia, que fue la regla del Entregable 1, llevando cada uno a [0, 1] al dividir por el máximo entre candidatos:

$$P = 0{,}30\,f + 0{,}15\,c + 0{,}15\,r + 0{,}15\,u + 0{,}10\,i + 0{,}15\,s$$

donde $f$ es la frecuencia en los estudios incluidos; $c$, la citación del artículo fundacional normalizada por antigüedad, $\log_{10}(1 + \text{citas por año})$, según OpenAlex; $r$, la proporción de los estudios de la familia publicados en 2025 o 2026; $u$, los estudios que la evalúan en UAVIDS-2025, que es la evidencia de desempeño en datos comparables; $i$, la disponibilidad de una implementación de referencia (1, o 0,5 si hay que programarla desde el artículo), y $s$, la representatividad de la familia (1 si aporta un sesgo inductivo que los modelos base no tienen, 0,5 si es una variante de una familia base). La frecuencia pesa más porque es la evidencia directa de uso en el problema; las citas usan logaritmo porque se distribuyen con cola pesada y, sin él, ResNet y el Transformer aplastarían a los demás. Un candidato solo entra si es aplicable a flujos individuales con los 16 atributos de tráfico, sin los identificadores que funcionan como atajo.
""")

code("s3-puntaje", r"""
puntaje = pd.read_csv(LIT2 / "puntaje_top10.csv", encoding="utf-8-sig")
columnas_p = {"posicion": "Posición", "modelo": "Modelo", "estudios": "Estudios", "estudios_uavids": "Con UAVIDS-2025", "recientes": "2025–2026", "articulo_fundacional": "Fundacional",
              "citas": "Citas", "citas_por_anio": "Citas por año", "c_frecuencia": "f", "c_citacion": "c", "c_recencia": "r", "c_uavids": "u", "c_implementacion": "i", "c_representatividad": "s",
              "puntaje": "Puntaje", "aplicable": "Aplicable", "seleccionado": "Top 10"}
tabla(puntaje[list(columnas_p)].rename(columns=columnas_p), {"Citas": "{:,.0f}", "Citas por año": "{:,.0f}", **{k: "{:.3f}" for k in ["f", "c", "r", "u", "i", "s", "Puntaje"]}})
PESOS_ALT = {"Pesos del preregistro": {"c_frecuencia": .30, "c_citacion": .15, "c_recencia": .15, "c_uavids": .15, "c_implementacion": .10, "c_representatividad": .15},
             "Pesos iguales": {k: 1 / 6 for k in ["c_frecuencia", "c_citacion", "c_recencia", "c_uavids", "c_implementacion", "c_representatividad"]},
             "Solo frecuencia": {"c_frecuencia": 1.0}}
sensibilidad = {}
aplicables = puntaje[puntaje["aplicable"] == "sí"]
for nombre, pesos in PESOS_ALT.items():
    total = sum(aplicables[k] * w for k, w in pesos.items())
    sensibilidad[nombre] = set(aplicables.loc[total.sort_values(ascending=False).index[:10], "modelo"])
# Sensibilidad al sesgo de acceso: los no recuperados con lo que dice su resumen (FT-Transformer y TCN, una mención cada uno)
ft = {"estudios": 1, "uavids": 0, "recencia": 1.0, "citas_anio": 615 / 6, "impl": 1.0, "repr": 1.0}
max_f, max_c, max_u = puntaje["estudios"].max(), np.log10(1 + puntaje["citas_por_anio"]).max(), puntaje["estudios_uavids"].max()
p_ft = .30 * ft["estudios"] / max_f + .15 * np.log10(1 + ft["citas_anio"]) / max_c + .15 * ft["recencia"] + .15 * 0 + .10 + .15
decimo = aplicables.sort_values("puntaje", ascending=False).iloc[9]
interpretar(f'''**Interpretación.** El Top 10 queda formado por {enumerar(aplicables.sort_values("puntaje", ascending=False)["modelo"].head(10))}. La GNN obtiene el tercer puntaje ({num(puntaje.loc[puntaje["modelo"] == "GNN", "puntaje"].item(), 3)}), pero no es aplicable, porque los tres estudios que la usan arman el grafo con direcciones IP, nodos o paquetes, y la GAN, con {num(puntaje.loc[puntaje["modelo"] == "GAN", "puntaje"].item(), 3)}, tampoco, porque se usa para generar datos y no para clasificar. Con pesos iguales, el Top 10 {"es el mismo" if sensibilidad["Pesos iguales"] == sensibilidad["Pesos del preregistro"] else "cambia en " + enumerar(sensibilidad["Pesos iguales"] ^ sensibilidad["Pesos del preregistro"])}, y con solo la frecuencia {"también es el mismo" if sensibilidad["Solo frecuencia"] == sensibilidad["Pesos del preregistro"] else "cambia en " + enumerar(sensibilidad["Solo frecuencia"] ^ sensibilidad["Pesos del preregistro"])}, porque los candidatos aplicables son exactamente diez.

Sin embargo, el décimo puesto es frágil frente al sesgo de acceso: si se contara el estudio no recuperado de Di et al. (2026) con lo que dice su resumen, FT-Transformer tendría un puntaje de {num(p_ft, 3)}, por encima de la {decimo["modelo"]} ({num(decimo["puntaje"], 3)}), que además no tiene estudios en 2025 o 2026. Es decir, la selección es estable frente a los pesos, pero no frente a la literatura que no se pudo leer.

Por tal motivo, se mantiene el Top 10 que resulta de la regla preregistrada, con la {decimo["modelo"]}, y FT-Transformer queda declarado como el primer excluido: su tokenización por atributo se registra como mecanismo candidato para el Entregable 3 (sección 3.1), y su costo en este equipo, de 64 a 75 segundos por época, unas 30 veces el de la DNN, también pesa en contra de agregarlo como undécimo modelo.''')
""")

md("s3-2b", r"""
**Diversidad arquitectónica.** El Top 10 cubre siete familias: recurrentes (RNN, GRU y BiLSTM), híbridas convolucionales y recurrentes (CNN-LSTM), basadas en atención (Transformer y TabNet), residuales (ResNet), de codificación y decodificación (Autoencoder), generativas preentrenadas capa a capa (DBN) y redes de Kolmogorov-Arnold (KAN). Cumple así el mínimo de cuatro familias distintas, sin modelos de espacio de estados ni preentrenados con *fine-tuning*, que no aparecen en la literatura del problema.

**Líneas base de referencia.** Además del Top 10 entran los tres modelos base del Entregable 1 (DNN, CNN 1D y LSTM), un baseline clásico fuerte (LightGBM, porque el enunciado pide árboles potenciados como comparador en datos tabulares), la regresión logística penalizada y un baseline trivial de clase mayoritaria.

**Modelos excluidos más citados.** Por citas de su artículo fundacional, los excluidos con más peso son la GAN, la GNN y FT-Transformer, y las tres razones ya se expusieron: la GAN no clasifica, la GNN depende de un grafo armado con atajos y FT-Transformer no tiene estudios evaluados en la literatura del problema.
""")

md("s3-3", r"""
### 3.3 Análisis detallado por modelo

Cada ficha reúne la referencia del artículo fundacional y de la fuente de la arquitectura, la familia, la descripción técnica con la formulación de sus bloques centrales, el problema y los datos de los estudios que la usan, las métricas que reportan, las fortalezas y limitaciones, la complejidad, el sesgo de reporte y si la arquitectura está definida. La figura resume el flujo de datos de los 13 modelos profundos, y la tabla, su complejidad calculada sobre la implementación de este entregable con 16 atributos y 5 clases. Las referencias completas, con DOI, están en la bibliografía.
""")

code("s3-diagramas", r"""
DIAGRAMAS = {
    "DNN": ["16", "Densa 128", "Densa 64", "Densa 32", "5"],
    "CNN 1D": ["1 × 16", "Conv 32", "Conv 64", "Max-pool", "Densa 64", "5"],
    "LSTM": ["16 pasos", "LSTM 64", "5"],
    "RNN": ["16 pasos", "RNN 128", "RNN 64", "BN", "Densa 128", "5"],
    "GRU": ["16 pasos", "GRU 128", "BN", "GRU 64", "BN", "Densa 128", "5"],
    "BiLSTM": ["16 pasos", "BiLSTM 2×128", "BiLSTM 2×128", "5"],
    "Transformer": ["16", "Densa 128", "16 fichas × 8", "Atención 2 cabezas", "Promedio", "Densa 64", "5"],
    "TabNet": ["16", "BN", "3 pasos: máscara\nsparsemax + GLU", "Suma de\ndecisiones", "5"],
    "KAN": ["16", "Compuerta", "KAN 28", "KAN 16", "5"],
    "ResNet": ["16", "Bloque\nresidual w₁", "Bloque\nresidual w₂", "Bloque\nresidual w₃", "5"],
    "Autoencoder": ["16", "w₁", "w₂", "Código w₃", "5 (y espejo\nhasta 16)"],
    "DBN": ["16", "RBM w₁", "RBM w₂", "RBM w₃", "5"],
    "CNN-LSTM": ["1 × 16", "Conv w₁", "Conv w₂ y pool", "LSTM w₃", "5"],
}
fig, ejes = plt.subplots(7, 2, figsize=(12, 10.5))
for eje, (nombre, bloques_d) in zip(ejes.ravel(), DIAGRAMAS.items()):
    eje.axis("off")
    eje.set(xlim=(0, len(bloques_d)), ylim=(0, 1))
    for i, b in enumerate(bloques_d):
        relleno = GRIS_OSCURO if i == len(bloques_d) - 1 else ("white" if i == 0 else GRIS_CLARO)
        eje.add_patch(plt.Rectangle((i + 0.08, 0.2), 0.84, 0.6, facecolor=relleno, edgecolor=NEGRO, lw=0.7, ls="--" if "w" in b else "-"))
        eje.text(i + 0.5, 0.5, b, ha="center", va="center", fontsize=6.4, color="white" if i == len(bloques_d) - 1 else NEGRO)
        if i:
            eje.annotate("", xy=(i + 0.08, 0.5), xytext=(i - 0.08, 0.5), arrowprops=dict(arrowstyle="->", lw=0.6))
    eje.set_title(nombre + ("  (arquitectura por búsqueda)" if MOD.CATALOGO[nombre]["busqueda"] else ""), fontsize=8.5, loc="left")
ejes.ravel()[-1].axis("off")
fig.suptitle("Flujo de datos de los modelos profundos; los bloques con borde discontinuo salen de la búsqueda de arquitectura", fontsize=10)
plt.tight_layout()
plt.show()
complejidad_modelos = []
for nombre in CONFIG["modelos"]["base_e1"] + CONFIG["modelos"]["top10"]:
    m = MOD.construir(nombre)
    complejidad_modelos.append({"Modelo": nombre, "Familia": MOD.CATALOGO[nombre]["familia"], "Parámetros": MOD.parametros(m), "MAC por muestra": MOD.contar_mac(m),
                                "Arquitectura": "Búsqueda (referencia 64, 64, 64)" if MOD.CATALOGO[nombre]["busqueda"] else "Definida en la literatura"})
complejidad_modelos = pd.DataFrame(complejidad_modelos)
complejidad_modelos["FLOPs por muestra"] = 2 * complejidad_modelos["MAC por muestra"]
tabla(complejidad_modelos, {"Parámetros": "{:,.0f}", "MAC por muestra": "{:,.0f}", "FLOPs por muestra": "{:,.0f}"})
""")
