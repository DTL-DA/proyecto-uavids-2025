"""Genera el guion de defensa del Entregable 2 en Word, con cuadros al estilo booktabs y notas al pie.

El texto hablado coincide con las notas del orador de la presentación, y los cuadros leen resultados/ cuando la cifra
está en un archivo; las demás se transcriben del cuaderno ejecutado, con la sección indicada en la nota al pie.
Requiere python-docx, que no está en requirements.txt porque no interviene en el benchmark.

Uso:
    python Entregable_2_UAVIDS_2025/herramientas/crear_guion_defensa.py
"""
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import XmlPart
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

E2 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
SALIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else E2 / "defensa" / "Guion de defensa - Entregable 2 - UAVIDS-2025.docx"
RES = E2 / "resultados"
FUENTE = "Times New Roman"
NEGRO = RGBColor(0, 0, 0)

# ---------------------------------------------------------------- utilidades


def num(x, d=4):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def pval(p):
    if p >= 0.0001:
        return num(p, 4)
    m, e = f"{p:.2e}".split("e")
    return f"{m.replace('.', ',')}e−{abs(int(e)):02d}"


def fuente_run(run, tam=None, cursiva=None, negrita=None, sup=False):
    run.font.name = FUENTE
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), FUENTE)
    if tam:
        run.font.size = Pt(tam)
    if cursiva is not None:
        run.italic = cursiva
    if negrita is not None:
        run.bold = negrita
    if sup:
        run.font.superscript = True
    run.font.color.rgb = NEGRO
    return run


def texto_marcado(par, texto, tam=None, base_cursiva=False):
    """Escribe texto con *cursiva* y **negrita** simples."""
    import re
    partes = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", texto)
    for p in partes:
        if not p:
            continue
        if p.startswith("**"):
            fuente_run(par.add_run(p[2:-2]), tam, cursiva=base_cursiva, negrita=True)
        elif p.startswith("*"):
            fuente_run(par.add_run(p[1:-1]), tam, cursiva=not base_cursiva)
        else:
            fuente_run(par.add_run(p), tam, cursiva=base_cursiva)
    return par


def parrafo(doc, texto, tam=12, alinear=WD_ALIGN_PARAGRAPH.JUSTIFY, antes=0, despues=6, cursiva=False, sangria=None, junto=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = alinear
    pf.space_before = Pt(antes)
    pf.space_after = Pt(despues)
    pf.line_spacing = 1.15
    if sangria:
        pf.left_indent = Cm(sangria)
    pf.keep_with_next = junto
    import re
    texto = re.sub(r"(?<![\w*])p(?= [=<]| TOST| de \d)", "*p*", texto)
    texto_marcado(p, texto, tam, base_cursiva=cursiva)
    return p


def titulo(doc, texto, nivel):
    tam = {1: 14, 2: 12}[nivel]
    p = doc.add_paragraph(style=f"Heading {nivel}")
    pf = p.paragraph_format
    pf.space_before = Pt(14 if nivel == 1 else 10)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    fuente_run(p.add_run(texto), tam, negrita=True, cursiva=False)
    return p


# ---------------------------------------------------------------- notas al pie


class Notas:
    def __init__(self, doc):
        xml = (f'<w:footnotes {nsdecls("w", "r")}>'
               '<w:footnote w:type="separator" w:id="-1"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:separator/></w:r></w:p></w:footnote>'
               '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:continuationSeparator/></w:r></w:p></w:footnote>'
               '</w:footnotes>')
        self.parte = XmlPart(PackURI("/word/footnotes.xml"),
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml",
                             parse_xml(xml), doc.part.package)
        doc.part.relate_to(self.parte, RT.FOOTNOTES)
        self.n = 0

    def agregar(self, par, texto):
        self.n += 1
        i = self.n
        rpr = f'<w:rPr><w:rFonts w:ascii="{FUENTE}" w:hAnsi="{FUENTE}" w:cs="{FUENTE}" w:eastAsia="{FUENTE}"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        rpr_sup = f'<w:rPr><w:rFonts w:ascii="{FUENTE}" w:hAnsi="{FUENTE}" w:cs="{FUENTE}" w:eastAsia="{FUENTE}"/><w:vertAlign w:val="superscript"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        texto = texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        nota = parse_xml(
            f'<w:footnote {nsdecls("w")} w:id="{i}"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
            f'<w:r>{rpr_sup}<w:footnoteRef/></w:r><w:r>{rpr}<w:t xml:space="preserve"> {texto}</w:t></w:r></w:p></w:footnote>')
        self.parte.element.append(nota)
        ref = parse_xml(
            f'<w:r {nsdecls("w")}><w:rPr><w:rFonts w:ascii="{FUENTE}" w:hAnsi="{FUENTE}" w:cs="{FUENTE}" w:eastAsia="{FUENTE}"/><w:vertAlign w:val="superscript"/></w:rPr><w:footnoteReference w:id="{i}"/></w:r>')
        par._p.append(ref)


# ---------------------------------------------------------------- cuadros


def insertar_antes(padre, nuevo, etiquetas):
    for hijo in padre:
        if hijo.tag in [qn(t) for t in etiquetas]:
            hijo.addprevious(nuevo)
            return
    padre.append(nuevo)


class Cuadros:
    def __init__(self, doc, notas):
        self.doc, self.notas, self.n = doc, notas, 0

    def __call__(self, leyenda, encabezado, filas, anchos, nota, centrar=None, tam=10, resaltar=()):
        self.n += 1
        doc = self.doc
        cap = doc.add_paragraph()
        cap.paragraph_format.space_before = Pt(10)
        cap.paragraph_format.space_after = Pt(4)
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        texto_marcado(cap, f"Cuadro {self.n}: {leyenda}", 11)
        self.notas.agregar(cap, nota)
        centrar = centrar if centrar is not None else [j > 0 for j in range(len(encabezado))]
        t = doc.add_table(rows=1 + len(filas), cols=len(encabezado))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tblPr = t._tbl.tblPr
        bordes = parse_xml(
            f'<w:tblBorders {nsdecls("w")}><w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:left w:val="nil"/><w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>')
        insertar_antes(tblPr, bordes, ["w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription"])
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(anchos[j] * 567)))
        for i, fila in enumerate([encabezado] + filas):
            row = t.rows[i]
            trPr = row._tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            if i == 0:
                trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            for j, valor in enumerate(fila):
                c = row.cells[j]
                c.width = Cm(anchos[j])
                if i == 0:
                    c._tc.get_or_add_tcPr().append(parse_xml(
                        f'<w:tcBorders {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>'))
                p = c.paragraphs[0]
                p.paragraph_format.space_before = Pt(1.5)
                p.paragraph_format.space_after = Pt(1.5)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.keep_with_next = i < len(filas)
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if centrar[j] else WD_ALIGN_PARAGRAPH.LEFT
                if i > 0 and (i - 1) in resaltar:
                    texto_marcado(p, str(valor), tam, base_cursiva=False)
                    for r in p.runs:
                        r.italic = True
                else:
                    texto_marcado(p, str(valor), tam)
        esp = doc.add_paragraph()
        esp.paragraph_format.space_after = Pt(4)
        return t


# ---------------------------------------------------------------- documento

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
for lado in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, lado, Cm(2.5))

normal = doc.styles["Normal"]
normal.font.name = FUENTE
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FUENTE)
for nombre in ("Heading 1", "Heading 2"):
    st = doc.styles[nombre]
    st.font.name = FUENTE
    st.font.color.rgb = NEGRO
    st.font.bold = True
    st.font.italic = False
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FUENTE)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]

# Número de página en el pie
pie = sec.footer.paragraphs[0]
pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
pie._p.append(parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"><w:r><w:rPr><w:rFonts w:ascii="{FUENTE}" w:hAnsi="{FUENTE}"/><w:sz w:val="20"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>'))

notas = Notas(doc)
cuadro = Cuadros(doc, notas)

# ---------------------------------------------------------------- portada breve

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
fuente_run(p.add_run("Guion de defensa del Entregable 2"), 18, negrita=True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(10)
fuente_run(p.add_run("Benchmark de Deep Learning para detectar intrusiones en redes de drones (UAVIDS-2025)"), 13, cursiva=True)
for linea in ("Donny Torres · Maestría en Analítica de Datos · Universidad del Norte",
              "Deep Learning · Proyecto final · 13 de noviembre de 2026"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    fuente_run(p.add_run(linea), 11)
doc.add_paragraph().paragraph_format.space_after = Pt(6)

# ---------------------------------------------------------------- 1. propósito

titulo(doc, "1. Propósito y estructura de la defensa", 1)
parrafo(doc, "Este guion acompaña la presentación oral que pide el enunciado, «Presentación oral con defensa técnica de las decisiones metodológicas», y que se evalúa dentro del criterio de reproducibilidad, redacción científica y defensa, con un peso del 10 %. El propósito es defender por qué el resultado es creíble: la unidad de independencia y la partición, el preregistro y sus desviaciones, el mismo presupuesto para todos los modelos y la inferencia estadística de cada pregunta de investigación, más que el número final.")
parrafo(doc, "La presentación tiene 16 diapositivas y dura cerca de 15 minutos, según el cuadro 1. El texto hablado de cada diapositiva está en la sección 2 y coincide con las notas del orador de la presentación; las cifras de apoyo, en la sección 3, y las preguntas probables del jurado con sus respuestas, en la sección 4. Toda cifra sale del cuaderno ejecutado «Entregable 2 - UAVIDS-2025.ipynb» o de los archivos de la carpeta resultados, y la nota al pie de cada cuadro indica de dónde.")

DIAPOS = [
    (1, "Portada", "0:20", "—"),
    (2, "Problema y decisiones heredadas", "1:00", "1.1, 1.3 y 2.1.4"),
    (3, "Cinco preguntas fijadas antes de entrenar", "0:45", "1.2"),
    (4, "Revisión sistemática y Top 10 por puntaje", "1:05", "3.1, 3.2 y 3.4"),
    (5, "Paper guía: réplica parcial con su propio código", "0:55", "5.1 y 5.2"),
    (6, "Diseño: mismo pipeline, mismo presupuesto", "1:05", "4.1 a 4.3.1"),
    (7, "Búsqueda de arquitectura en dos fases", "0:55", "4.3.2"),
    (8, "Benchmark: la tabla ordena, pero no decide", "0:55", "6.1"),
    (9, "RQ1: ningún modelo supera al baseline por más de δ", "1:05", "6.3"),
    (10, "RQ2: LightGBM gana en esta prueba, no en 5 × 2 cv", "1:05", "6.3"),
    (11, "La serie pesa más que la arquitectura", "1:05", "4.5 y 6.3"),
    (12, "Ruta F1: ajuste, calibración y adecuación", "0:55", "4.5 y 7"),
    (13, "Incertidumbre, techo y explicaciones", "0:55", "4.6 y 6.4"),
    (14, "Costo: más cómputo no compra F1", "0:45", "6.2 y 6.3"),
    (15, "Limitación compartida: Blackhole frente a Wormhole", "1:00", "4.5, 5.3 y 5.4"),
    (16, "Reproducibilidad y conclusiones", "1:05", "8 y 9.3"),
]
filas = [[str(n), t, tm, s] for n, t, tm, s in DIAPOS] + [["", "Total", "14:55", ""]]
cuadro("estructura de la presentación, tiempo por diapositiva y sección del cuaderno que la sustenta",
       ["N.º", "Diapositiva", "Tiempo", "Sección del cuaderno"], filas, [1.2, 9.0, 1.8, 4.0],
       "Elaboración propia para la defensa. Los tiempos son una estimación para una lectura pausada del texto de la sección 2 y no provienen del cuaderno; la última columna remite a las secciones del cuaderno «Entregable 2 - UAVIDS-2025.ipynb».",
       centrar=[True, False, True, True])

# ---------------------------------------------------------------- 2. texto hablado

titulo(doc, "2. Texto hablado por diapositiva", 1)
parrafo(doc, "El texto está escrito para decirse, no para leerse en voz alta palabra por palabra; conviene ensayarlo con cronómetro y recortar en las diapositivas 11 a 13 si el tiempo se alarga, porque las cifras de esas tres diapositivas están completas en los cuadros de la sección 3.")

# Sensibilidad del criterio de adecuación al umbral del ECE, que no estaba en el preregistro (sección 7 del cuaderno)
SENS_ECE = pd.read_csv(RES / "adecuacion_sensibilidad_ece.csv")
N_SENS = dict(zip(SENS_ECE["Umbral del ECE con temperatura"].round(2), SENS_ECE["Modelos adecuados"]))
TEXTO_SENS = f"con umbrales de 0,03 a 0,07, los modelos adecuados pasan de {N_SENS[0.03]} a {N_SENS[0.07]}"

HABLADO = {
    1: ["Buenos días. Presento el segundo entregable del proyecto: un benchmark de dieciséis modelos sobre UAVIDS-2025. Más que el número final, voy a defender las decisiones que lo hacen creíble: cómo se partieron los datos, cómo se igualó el esfuerzo entre modelos y cómo se probó cada diferencia."],
    2: ["El problema es clasificar cada flujo de red de un enjambre de drones en una de cinco clases, con los dieciséis atributos de tráfico. Las decisiones de fondo se heredan sin cambios del Entregable 1 y se fijaron antes de ver un solo resultado.",
        "La métrica es el F1 macro, porque en seguridad un ataque poco frecuente importa tanto como uno frecuente, y el margen de relevancia es δ = 0,01, que equivale a que el F1 de una sola clase cambie 0,05.",
        "La decisión más importante es la unidad de independencia: los flujos consecutivos son casi gemelos, de modo que se parte por bloques de mil flujos y no por fila. En el Entregable 1, la partición aleatoria subía el F1 en 0,0689, y esa diferencia es fuga, no desempeño. Por último, la prueba se selló con su suma SHA-256 y se toca una sola vez por corrida, con el modelo ya fijado."],
    3: ["Las preguntas quedaron en el preregistro antes del primer entrenamiento. Las tres confirmatorias tienen control del error por familia con Holm, y las dos exploratorias solo generan hipótesis: no se usan para elegir el modelo final.",
        "Un detalle de diseño que conviene resaltar está en RQ2: el mejor modelo profundo se elige con validación y no con prueba, porque elegir el máximo de trece medias de prueba sesgaría hacia arriba la diferencia frente a LightGBM, que es la maldición del ganador."],
    4: ["La revisión extiende la del Entregable 1 y llega a 43 estudios incluidos. El Top 10 no sale de un criterio personal: sale de un puntaje definido antes de calcularlo, donde la frecuencia en la literatura pesa 30 % y el resto lo reparten la citación, la actualidad, el uso en UAVIDS-2025, la disponibilidad de implementación y la representatividad. Con pesos iguales, o solo con la frecuencia, el Top 10 es el mismo.",
        "La GNN obtuvo el tercer puntaje, pero se excluye porque los tres estudios que la usan arman el grafo con direcciones IP, que en este conjunto son un atajo, y la GAN se usa para generar datos, no para clasificar. FT-Transformer queda como el primer excluido.",
        "El gap es verificable con la hoja de extracción: de los 12 estudios que usan UAVIDS-2025, solo uno agrupa las filas al partir, y ninguno ajusta sus líneas base con el mismo presupuesto ni corrige por comparaciones múltiples. Las cadenas transversales T1 a T4 se corrieron también en Scopus y en Web of Science, y C4, que es complementaria, se quedó en OpenAlex y arXiv porque no cambió los estudios incluidos."],
    5: ["El paper guía es Zeng et al. (2025), que define el conjunto de datos y es el único de los 43 estudios con código público, lo que permite replicarlo en sus condiciones exactas: partición aleatoria 80/20, semilla 42 y su mismo entrenamiento. El criterio de éxito quedó preregistrado.",
        "Cuatro de los cinco modelos se reproducen dentro de δ, lo que valida la implementación del pipeline. El resultado clave no: la LSTM obtiene 0,9285 en diez réplicas frente al 94 % reportado, ninguna réplica llega a 0,935, que es el menor valor que se redondea a 94, y además queda por debajo de la DNN y de la CNN, de modo que la ventaja de la recurrencia que sugiere el artículo no se sostiene.",
        "El artículo no dice cuántas corridas promedia ni su dispersión, así que no puedo distinguir entre una ejecución favorable y otra versión de los datos; por eso lo declaro como no reproducido."],
    6: ["Todos los modelos comparten pipeline, partición, semillas y número de intentos de ajuste: doce configuraciones cada uno, sin importar cuánto tarde, porque se iguala el número de intentos y no el tiempo. Todo cambio al protocolo quedó fechado en una adenda antes de usarlo.",
        "La desviación 1 es la más visible: la fase 1 de la búsqueda y el estudio de ruido se entrenaron con el 25 % de los bloques de entrenamiento, porque el protocolo completo medía unas cuarenta horas; se mantuvieron las mil arquitecturas y la validación completa, y la fase 2 volvió a los datos completos.",
        "El número de semillas sale del piloto: con la desviación más conservadora, 0,0177, detectar δ con potencia de 0,80 tras Bonferroni exigía 42 corridas. El tope de cómputo es 20, así que la diferencia mínima detectable es 0,0162, unas 1,6 veces δ, y lo declaro como limitación. Por eso los resultados no significativos se leen con TOST y con la prueba bayesiana, y no como ausencia de diferencias."],
    7: ["Cuatro modelos no tienen arquitectura definida en la literatura, así que pasan por la búsqueda de la sección 14: mil combinaciones de anchos y dropout, todas evaluadas en la fase 1, y veinte finalistas más la referencia, con cinco semillas, en la fase 2.",
        "La regla del preregistro fijó k = 20 y no 10, porque el ruido entre semillas llegó a ser casi tres veces la dispersión entre arquitecturas, y con un ruido así la probabilidad de capturar la mejor con diez finalistas ronda el 50 %. La correlación entre fases quedó entre −0,42 y 0,02, de modo que el cribado informó poco, y en tres de los cuatro modelos la regla de desempate eligió la propia referencia: la búsqueda no encontró nada que la supere en más de δ.",
        "Durante la fase 2 el equipo se suspendió siete horas; las seis corridas en curso conservaron su resultado y a su tiempo se le descontó la ventana que registró el sistema."],
    8: ["Con veinte semillas por modelo, LightGBM tiene el mayor F1 macro de prueba, 0,8748, seguido de CNN-LSTM, Autoencoder y Transformer; el baseline, la CNN 1D reentrenada con este protocolo, queda en 0,8584. Ningún modelo profundo alcanza el umbral de mejora de 0,8775 que fijó el Entregable 1.",
        "Pero hay que leer la columna de intervalos: el ancho mediano del IC por bloques es 0,191, y el de LightGBM se solapa con el de todos los demás modelos no triviales, porque refleja cuánto cambia el F1 según qué series caen en la prueba. Por eso la tabla ordena, pero no decide: la decisión sale de los contrastes pareados, que comparan los modelos sobre las mismas series y semillas."],
    9: ["RQ1 se responde con la regla preregistrada, no con el orden de la tabla. De los diez contrastes, tres son equivalentes al baseline dentro de ±δ según TOST: Autoencoder, Transformer y DBN. Tres son no concluyentes, CNN-LSTM, BiLSTM y ResNet, porque su intervalo ni excluye el cero ni cabe dentro de ±δ. RNN, GRU y TabNet quedan por debajo de la CNN 1D en más de δ, y KAN queda por debajo de forma significativa, pero por menos de δ. Entonces, H0 no se rechaza a favor del Top 10 en ningún contraste.",
        "El procedimiento global confirma que hay diferencias entre los dieciséis modelos, con Friedman y χ² de 184,8, pero el Model Confidence Set al 90 %, calculado sobre los 24 bloques de prueba, conserva diez modelos que estos datos no pueden separar. Por eso presento como ganador ese conjunto y no un solo modelo."],
    10: ["En RQ2, el mejor modelo profundo por validación es la CNN 1D, y LightGBM la supera por 0,0164, con t pareada y Wilcoxon por debajo de 0,001. Según la regla preregistrada, la diferencia es significativa y relevante, lo que coincide con la advertencia del enunciado de que los árboles potenciados son muy difíciles de batir en datos tabulares.",
         "Sin embargo, esa regla compara veinte semillas sobre las mismas 24 series de prueba. Cuando la validación cruzada 5 × 2 cambia los datos de ajuste y de evaluación, sin tocar la prueba, la diferencia cae a −0,0003, con p de 0,990, y el intervalo por bloques de la prueba contiene el cero. Es decir, la ventaja es real en estas series, pero no puedo afirmar que se repita en series nuevas.",
         "Por eso la respuesta queda acotada, y la justificación para preferir un modelo profundo tendría que venir de otro eje: robustez, calibración o costo."],
    11: ["Este es el hallazgo que más condiciona la lectura. Todos los modelos profundos caen unas siete centésimas de validación a prueba, y no es sobreajuste del entrenamiento: la brecha entre entrenamiento y validación no pasa de 0,0154, por debajo de δg. La validación adversarial agrupada da un AUC de 0,515, dentro de la distribución nula, de modo que tampoco es un desplazamiento propio de la partición: es la variación entre series.",
         "La descomposición de la varianza lo resume: el bloque de prueba explica el 93,1 % de la varianza del F1 y el modelo, el 0,3 %. En ese contexto, RQ3 da equivalencia: leer los atributos como una secuencia ni ayuda ni perjudica.",
         "LightGBM cae mucho menos, 0,0123, y mi hipótesis es que los umbrales por atributo son menos sensibles a los cambios de escala entre series que las combinaciones continuas de las redes. Por eso, para el Entregable 3 propongo elegir el modelo con validación cruzada por bloques y no con una sola partición de validación."],
    12: ["La ruta F1 del enunciado se siguió completa, con las tablas A, B y C en la ficha de ruta. Las pruebas de sanidad pasan: todos los modelos superan al trivial, y la permutación de etiquetas con B = 200, en los seis modelos cuyo costo lo permitió, da p = 0,005, el mínimo posible.",
         "El criterio de adecuación exige cuatro condiciones a la vez. Cumplen cinco modelos: CNN-LSTM, Autoencoder, Transformer, BiLSTM y la regresión logística. LightGBM falla por su brecha entre entrenamiento y validación, de 0,0858, propia de los árboles, que se ajustan casi por completo al entrenamiento. La CNN 1D falla por calibración, y aquí soy transparente: el preregistro no fijó un umbral para el ECE, y el de 0,05 lo fijé después de la ejecución definitiva, así que declaro el veredicto como exploratorio. "
         f"Además, {TEXTO_SENS}, y cinco modelos quedan por encima de 0,05 por menos de 0,005, de modo que el umbral decide más que la diferencia real.",
         "La temperatura ajustada en validación quedó entre 0,90 y 1,10 y no mejoró la prueba, por eso no se aplica; recalibrar con datos de la serie de destino queda como requisito de una puesta en producción."],
    13: ["Sobre incertidumbre, la predicción conforme muestra el costo de que validación y prueba no sean intercambiables: calibrada en validación al 90 %, LAC cubre solo del 81,7 al 82,6 % en prueba, y APS alcanza la cobertura con conjuntos de más de tres clases, que informan poco; en el peor bloque, la cobertura baja a 0,12. Por eso los conjuntos conformes se usan como alerta y no como garantía.",
         "El techo, con Cover y Hart y con aprendizaje confiado, deja un margen real: la distancia normalizada de LightGBM es de 0,665 a 0,711, donde cero sería alcanzar el techo.",
         "En robustez, el ruido gaussiano es lo que más degrada, y la robustez no sigue el orden del F1, así que es un eje propio. Las atribuciones de gradientes integrados son fieles en los tres mejores, pero en KAN son inestables entre semillas, y no las leo como mecanismo. La ablación señala la transformación logarítmica como el componente con más evidencia de aportar."],
    14: ["En costo, el resultado va contra la intuición: entre los modelos profundos, la correlación entre parámetros y F1 es de 0,07, y la BiLSTM, con más de medio millón de parámetros, no está en la frontera de Pareto.",
         "La pregunta exploratoria RQ5 es afirmativa: Autoencoder, DBN, DNN y Transformer son equivalentes a la CNN 1D dentro de ±δ con menor latencia, y el Autoencoder evalúa cada muestra 5,3 veces más rápido. Como es exploratoria y sin control del error por familia, la uso como referencia de costo para el Entregable 3 y no para elegir el modelo."],
    15: ["La limitación que cuantifiqué en el paper guía es la confusión entre Blackhole y Wormhole, dos ataques de enrutamiento que alteran de forma parecida el número de saltos y la entrega de paquetes. Con el protocolo de los autores, la LSTM los confunde en el 9,96 % de esos flujos; con la partición agrupada, en el 15,69 %, porque la partición aleatoria la subestima.",
         "Y no es propia de la LSTM: los diez modelos del Top 10 quedan entre 12,30 % y 18,51 %, y en la CNN 1D y la DBN esa confusión reúne casi la mitad de sus errores. Además, los tres mejores se equivocan en los mismos flujos, y combinarlos gana menos que δ.",
         "Esto apunta a una superposición en el espacio de los atributos y no a falta de capacidad, y es el hueco que traslado al Entregable 3: un mecanismo que separe explícitamente ese par, como la pérdida contrastiva supervisada, la mezcla de negativos difíciles o una etapa jerárquica dedicada."],
    16: ["Cierro con la reproducibilidad, que es parte del criterio de evaluación. En un entorno virtual nuevo, instalado solo con requirements.txt, volví a entrenar catorce corridas con sus mismas semillas, y el F1 coincidió hasta el último decimal; la tabla principal se recalcula desde las corridas guardadas con un error máximo de 2,22e−16. El cuaderno se regenera con un solo comando y no reentrena nada.",
         "En síntesis: ningún modelo del Top 10 supera al baseline por más de δ; LightGBM gana en estas series, pero empata en la validación cruzada; la serie pesa más que la arquitectura, y el margen real de mejora está en separar Blackhole de Wormhole, que es lo que abordará el modelo original del Entregable 3. Muchas gracias; quedo atento a sus preguntas."],
}
APOYO = {2: "cuadro 2", 5: "cuadro 3", 8: "cuadro 4", 9: "cuadro 5", 10: "cuadro 6", 11: "cuadro 7", 12: "cuadro 9",
         13: "cuadros 10 y 11", 14: "cuadro 8", 15: "cuadro 12", 16: "cuadro 13"}

for n, t, tm, _ in DIAPOS:
    titulo(doc, f"Diapositiva {n}. {t} ({tm})", 2)
    for txt in HABLADO[n]:
        parrafo(doc, txt)
    if n in APOYO:
        parrafo(doc, f"Cifras de apoyo: {APOYO[n]}.", tam=11, cursiva=True, despues=4)

# ---------------------------------------------------------------- 3. cifras

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
titulo(doc, "3. Cifras clave para sostener la defensa", 1)
parrafo(doc, "Los cuadros reúnen las cifras que el jurado puede pedir durante la exposición o en las preguntas. Siguen el orden de la presentación, y la nota al pie de cada uno indica la sección del cuaderno y el archivo de donde sale.")

cuadro("decisiones fijadas a priori y heredadas del Entregable 1",
       ["Decisión", "Valor"],
       [["Tipo de problema", "Clasificación tabular multiclase de cinco categorías (ruta F1)"],
        ["Métrica primaria", "F1 macro en el conjunto de prueba"],
        ["Margen de relevancia δ", "0,01 (equivale a que el F1 de una sola clase cambie 0,05)"],
        ["Margen de sobreajuste δg", "0,05 entre entrenamiento y validación"],
        ["Nivel α", "0,05, con Holm dentro de cada familia"],
        ["Unidad de independencia", "Serie de flujos según FlowID, aproximada con bloques de 1000 flujos"],
        ["Partición", "64/16/20 %, estratificada y agrupada, semilla 42: 78, 19 y 24 bloques"],
        ["Filas", "73 267 de entrenamiento, 17 802 de validación y 22 883 de prueba"],
        ["Conjunto de prueba", "Sellado con SHA-256 antes de entrenar; una evaluación por corrida"],
        ["Umbral de mejora del Entregable 1", "0,8775: baseline de 0,8549 más su diferencia mínima detectable, 0,0226"]],
       [5.0, 11.0],
       "Cuaderno, secciones 1.1, 1.3 y 2.1.4, y celda de configuración del entorno. El número de bloques por partición (78, 19 y 24) se cuenta en resultados/particion.npz del Entregable 1, y el umbral de 0,8775 con su desglose sale de resultados/baseline.json del Entregable 1; el cuaderno del Entregable 2 cita el umbral, pero no ese desglose.",
       centrar=[False, False])

cuadro("réplica del paper guía con el código y la partición de los autores (F1 macro, 10 réplicas)",
       ["Modelo", "Media", "DE", "IC 95 %", "Reportado", "Diferencia", "Éxito"],
       [["Regresión logística", "0,8191", "0,0023", "[0,8175; 0,8208]", "0,82", "−0,0009", "sí"],
        ["XGBoost", "0,9575", "0,0008", "[0,9569; 0,9581]", "0,96", "−0,0025", "sí"],
        ["DNN", "0,9353", "0,0021", "[0,9338; 0,9368]", "0,93", "+0,0053", "sí"],
        ["CNN", "0,9365", "0,0028", "[0,9345; 0,9385]", "0,93", "+0,0065", "sí"],
        ["LSTM (resultado clave)", "0,9285", "0,0044", "[0,9254; 0,9317]", "0,94", "−0,0115", "no"]],
       [3.7, 1.6, 1.5, 3.0, 2.1, 1.9, 1.3],
       "Cuaderno, sección 5.2; archivo resultados/replica.csv. Criterio preregistrado: diferencia absoluta menor que δ o valor reportado dentro del IC 95 %.")

bm = pd.read_csv(RES / "tabla_benchmark.csv", index_col=0)
filas = []
for m, r in bm.iterrows():
    filas.append([m, r["Grupo"], r["F1 macro (media ± DE)"].replace("-", "−"), r["IC 95 % (bootstrap por bloques)"],
                  num(r["ECE"], 4), "—" if pd.isna(r["Robustez (índice)"]) else num(r["Robustez (índice)"], 3),
                  num(r["Latencia (ms)"], 3), num(r["Rango promedio"], 2)])
cuadro("tabla principal del benchmark en la prueba, ordenada por rango promedio sobre 20 semillas",
       ["Modelo", "Grupo", "F1 macro (media ± DE)", "IC 95 % por bloques", "ECE", "Robustez", "Latencia (ms)", "Rango"],
       filas, [2.4, 1.5, 2.8, 2.9, 1.3, 1.8, 1.7, 1.3],
       "Cuaderno, sección 6.1 (plantilla A.2 del enunciado); archivo resultados/tabla_benchmark.csv. El IC remuestrea con reemplazo los 24 bloques de prueba; el trivial no tiene índice de robustez. En cursiva, el baseline del Entregable 1, reentrenado con este protocolo.",
       tam=9, resaltar=(4,))

rq1 = pd.read_csv(RES / "contrastes_rq1.csv")
filas = []
for _, r in rq1.iterrows():
    ic = r["IC 95 % t"].replace("np.float64(", "").replace(")", "").replace("(", "").split(",")
    filas.append([r["a"], num(r["Diferencia media"], 4), f"[{num(float(ic[0]), 4)}; {num(float(ic[1]), 4)}]",
                  pval(r["p (i) Holm"]), pval(r["p Wilcoxon Holm"]), pval(r["p TOST"]), r["Decisión"]])
cuadro("contrastes de RQ1: cada modelo del Top 10 menos la CNN 1D, F1 macro de prueba sobre 20 semillas pareadas",
       ["Modelo", "Diferencia", "IC 95 %", "*p* Holm (i)", "*p* Wilcoxon Holm", "*p* TOST", "Decisión frente a δ"],
       filas, [2.4, 1.8, 3.1, 1.9, 2.0, 1.6, 3.6],
       "Cuaderno, sección 6.3 (RQ1, plantilla A.3); archivo resultados/contrastes_rq1.csv. La prueba (i) es la t pareada o el bootstrap pareado si Shapiro-Wilk rechaza la normalidad; el IC es el de la t. Holm se aplica dentro de la familia de 10 contrastes. Una diferencia negativa indica que el modelo queda por debajo de la CNN 1D, de modo que «significativa y relevante» en RNN, GRU y TabNet es en contra del Top 10.",
       tam=9)

cuadro("RQ2: CNN 1D, el mejor modelo profundo por validación, frente a LightGBM",
       ["Prueba", "Resultado", "*p*"],
       [["t pareada (20 semillas, misma prueba)", "Diferencia −0,0164; IC 95 % [−0,0188; −0,0139]", pval(1.8321105032782385e-11)],
        ["Wilcoxon (20 semillas)", "Mismo signo en las 20 semillas (δ de Cliff = −1,00)", pval(1.9073486328125e-06)],
        ["TOST con ±δ", "No equivalentes", "1,0000"],
        ["d de Cohen pareada", "−3,13", "—"],
        ["IC 95 % por bloques de la prueba", "[−0,0553; 0,0257]: contiene el cero", "—"],
        ["t de Nadeau y Bengio (5 × 2 cv agrupada)", "Diferencia media −0,0003; t = −0,01", "0,9902"],
        ["F de Alpaydin (5 × 2 cv agrupada)", "F = 0,61", "0,7645"],
        ["McNemar (promedio de semillas)", "b = 555, c = 978; supone filas independientes", "4,37e−27"],
        ["Decisión de la regla preregistrada", "Significativa y relevante, acotada a las series de esta prueba", "—"]],
       [5.6, 8.4, 2.0],
       "Cuaderno, sección 6.3 (RQ2); archivo resultados/contrastes_rq2.csv. La validación cruzada 5 × 2 reparte los 97 bloques de entrenamiento y validación sin tocar la prueba. El δ de Cliff de −1,00 indica que la CNN 1D queda por debajo de LightGBM en todas las semillas.",
       centrar=[False, False, True])

cuadro("RQ3: modelos que leen los atributos como secuencia frente a los que no suponen orden, y descomposición de la varianza del F1 por bloque",
       ["Elemento", "Estimación", "IC 95 %", "*p*"],
       [["t pareada sobre semillas (secuencia − sin orden)", "−0,0042", "[−0,0087; 0,0002]", "0,0605"],
        ["Wilcoxon sobre semillas", "−0,0042", "[−0,0083; −0,0001]", "0,0484"],
        ["Modelo mixto, con bloque y semilla aleatorios", "−0,0030", "[−0,0050; −0,0009]", "0,0054"],
        ["Varianza explicada por el bloque de prueba", "93,1 %", "—", "—"],
        ["Varianza explicada por el modelo (efecto fijo)", "0,3 %", "—", "—"],
        ["Varianza explicada por la semilla", "0,0 %", "—", "—"],
        ["Varianza residual", "6,5 %", "—", "—"],
        ["Decisión de la regla preregistrada", "Equivalente", "—", "—"]],
       [7.4, 2.6, 3.8, 2.2],
       "Cuaderno, sección 6.3 (RQ3). La regla preregistrada exige que la t pareada y Wilcoxon rechacen a la vez; la t no rechaza (p = 0,0605) y TOST respalda la equivalencia dentro de ±δ. El modelo mixto detecta una diferencia de −0,0030, menor que δ, porque pondera por igual cada bloque y promedia el F1 sobre las clases presentes.",
       tam=9.5)

cuadro("RQ5 (exploratoria): modelos con menor latencia que la CNN 1D y su equivalencia dentro de ±δ",
       ["Modelo", "Latencia (ms)", "Parámetros", "Diferencia con CNN 1D", "IC 95 %", "*p* TOST", "Equivalente"],
       [["Autoencoder", "0,298", "19 093", "+0,0007", "[−0,0043; 0,0056]", "0,0004", "sí"],
        ["Logística", "0,311", "85", "−0,0554", "[−0,0578; −0,0531]", "1,0000", "no"],
        ["DBN", "0,362", "9 733", "−0,0016", "[−0,0058; 0,0027]", "0,0003", "sí"],
        ["DNN", "0,400", "12 677", "−0,0008", "[−0,0045; 0,0026]", "0,0000", "sí"],
        ["ResNet", "0,890", "14 245", "−0,0075", "[−0,0142; −0,0007]", "0,2215", "no"],
        ["Transformer", "1,348", "3 381", "−0,0008", "[−0,0069; 0,0053]", "0,0026", "sí"],
        ["KAN", "1,477", "4 845", "−0,0096", "[−0,0159; −0,0033]", "0,4473", "no"],
        ["LightGBM", "1,585", "32 916", "+0,0164", "[0,0139; 0,0188]", "1,0000", "no, es mejor"]],
       [2.4, 1.9, 1.9, 2.5, 3.3, 1.6, 2.4],
       "Cuaderno, sección 6.3 (RQ5). La latencia de la CNN 1D es de 1,594 ms por muestra; los valores p no se corrigen por multiplicidad porque la pregunta es exploratoria, y la latencia se midió en un solo equipo con 2 hilos por corrida.",
       tam=9)

ad = pd.read_csv(RES / "adecuacion.csv")
filas = [[r["Modelo"], r["(i) Supera al trivial"], r["(ii) Brecha < δg"], r["(iii) ECE con temperatura < 0,05"],
          num(r["Pendiente media"], 3), r["(iv) Prueba consistente con validación"], r["Adecuado"]] for _, r in ad.iterrows()]
cuadro("criterio de adecuación de la sección 7.2 del enunciado: el modelo es adecuado solo si cumple las cuatro condiciones",
       ["Modelo", "(i) Supera al trivial", "(ii) Brecha < δg", "(iii) ECE con T < 0,05", "Pendiente media", "(iv) Prueba consistente", "Adecuado"],
       filas, [2.6, 2.1, 2.1, 2.3, 2.1, 2.3, 1.9],
       "Cuaderno, sección 7 (criterio de adecuación); archivos resultados/adecuacion.csv y resultados/adecuacion_sensibilidad_ece.csv. El umbral de 0,05 del ECE no estaba en el preregistro, de modo que la condición (iii) y el veredicto son exploratorios; "
       f"{TEXTO_SENS}. La pendiente de calibración media menor que 1 indica sobreconfianza y mayor que 1, subconfianza; LightGBM es el único con pendiente mayor que 1.",
       tam=9)

cuadro("techo de desempeño y distancia normalizada γ, medida desde el baseline del Entregable 1 (exactitud de prueba)",
       ["Cota", "Exactitud máxima", "CNN 1D", "CNN-LSTM", "γ CNN-LSTM", "LightGBM", "γ LightGBM"],
       [["Aprendizaje confiado", "0,9246", "0,8646", "0,8711", "0,891", "0,8847", "0,665"],
        ["Cover y Hart (1-NN en validación)", "0,9341", "0,8646", "0,8711", "0,906", "0,8847", "0,711"]],
       [4.2, 2.2, 1.6, 1.8, 1.9, 1.8, 2.0],
       "Cuaderno, sección 6.4; archivo resultados/techo.csv. γ = (techo − mejor) / (techo − baseline): 0 es alcanzar el techo y 1, quedarse en el baseline. La CNN 1D es el propio baseline, de modo que su γ es 1 por construcción.",
       tam=9.5)

cf = pd.read_csv(RES / "conforme.csv")
filas = [[r["Modelo"], r["Puntaje"], num(r["Nominal"], 2), num(r["Cobertura empírica"], 4), num(r["Tamaño medio"], 2),
          num(r["Cobertura mínima por clase"], 4), r["Clase con menor cobertura"], num(r["Cobertura mínima por bloque"], 4)]
         for _, r in cf.iterrows() if abs(r["Nominal"] - 0.9) < 1e-9]
cuadro("predicción conforme dividida al 90 %, calibrada en validación y evaluada en prueba",
       ["Modelo", "Puntaje", "Nominal", "Cobertura", "Tamaño medio", "Mínima por clase", "Clase", "Mínima por bloque"],
       filas, [1.9, 1.6, 1.6, 1.8, 1.9, 2.4, 2.2, 2.4],
       "Cuaderno, sección 6.4 (incertidumbre predictiva); archivo resultados/conforme.csv, que también trae el nivel de 95 %. La garantía supone intercambiabilidad entre calibración y prueba, que no se cumple entre series distintas.",
       tam=9)

lim = pd.read_csv(RES / "limitacion_paper_guia.csv")
filas = [[r["Escenario"].replace("Réplica, partición aleatoria 80/20", "Réplica, aleatoria 80/20").replace("Benchmark, partición agrupada (prueba)", "Benchmark, agrupada"),
          r["Modelo"], num(r["Tasa media"], 2), num(r["DE entre semillas"], 2), r["IC 95 % por bloques"]] for _, r in lim.iterrows()]
cuadro("tasa de confusión entre Blackhole y Wormhole (%) con el protocolo del paper y con la partición agrupada",
       ["Escenario", "Modelo", "Tasa media (%)", "DE (pp)", "IC 95 % por bloques"],
       filas, [4.6, 2.8, 2.6, 2.0, 3.6],
       "Cuaderno, sección 5.3; archivo resultados/limitacion_paper_guia.csv. La tasa es (n B→W + n W→B) / (n B + n W) sobre los flujos de esas dos clases; el IC agrega las confusiones de todas las semillas en cada bloque de 1000 flujos.",
       tam=9.5, centrar=[False, False, True, True, True])

ent = [("DNN", "0,831046"), ("CNN 1D", "0,856619"), ("LSTM", "0,812688"), ("CNN-LSTM", "0,879750"), ("Autoencoder", "0,858838"),
       ("RNN", "0,827471"), ("Transformer", "0,861018"), ("ResNet", "0,846919"), ("KAN", "0,854693"), ("TabNet", "0,810258"),
       ("DBN", "0,851675"), ("LightGBM", "0,873772"), ("Logística", "0,802986"), ("Trivial", "0,069822")]
cuadro("reproducción en un entorno limpio: corridas de la semilla 6001 reentrenadas con requirements.txt",
       ["Modelo", "F1 original", "F1 en entorno limpio", "Diferencia absoluta"],
       [[m, f, f, "0,00e+00"] for m, f in ent],
       [3.6, 3.2, 3.8, 3.6],
       "Cuaderno, sección 8 (prueba de entorno limpio); archivo resultados/reproduccion_entorno_limpio.json. Quedaron fuera, por costo, BiLSTM y GRU; la tabla principal recalculada desde las corridas guardadas coincide con un error máximo de 2,22e−16.",
       tam=9.5)

# ---------------------------------------------------------------- 4. preguntas

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
titulo(doc, "4. Preguntas probables del jurado y respuestas", 1)
parrafo(doc, "Las respuestas siguen el mismo orden: primero la respuesta directa, después la evidencia con cifras y al final la limitación que se declara. Conviene responder en menos de un minuto y ofrecer el cuadro de apoyo si el jurado pide el detalle.")

PREGUNTAS = [
    ("¿Por qué la fase 1 de la búsqueda se entrenó con solo el 25 % de los bloques? ¿No sesga la elección de la arquitectura?",
     ["Porque el protocolo completo medía unas 40 horas de cómputo en CPU, y la desviación 1 se declaró con fecha en la adenda antes de usarla, no después de ver resultados. La fase 1 y el estudio de ruido entrenan con un subconjunto fijo de 20 bloques (19 351 flujos), pero se conservan las 1000 arquitecturas, las 6 épocas y la validación completa, y la fase 2 vuelve al entrenamiento completo.",
      "El riesgo existe y se midió: con menos datos, el F1 de la fase 1 es más ruidoso y puede favorecer arquitecturas pequeñas. Por eso se reporta la correlación de rangos entre fases, que quedó entre −0,42 y 0,02, y por eso la regla fijó k = 20 finalistas y no 10. La conclusión sobre los cuatro modelos con búsqueda se limita a su mejor finalista, y así queda en las amenazas a la validez interna (sección 9.2). Aun con la desviación, la búsqueda costó 55,1 horas de proceso."]),
    ("¿Por qué R = 20 semillas si el análisis de potencia pedía 42?",
     ["Porque 20 es el tope de cómputo declarado en el preregistro. El piloto, con cinco semillas en cuatro modelos, dio una desviación de las diferencias pareadas de 0,0070, y la regla toma el máximo entre esa y la del Entregable 1 en prueba, 0,0177. Con α′ = 0,05/10 y potencia de 0,80, detectar δ exigía 42 corridas.",
      "Con 20, la diferencia mínima detectable por contraste es 0,0162, unas 1,6 veces δ, y lo declaro como limitación estadística. La cifra es conservadora, porque Holm es menos exigente que Bonferroni salvo en el primer contraste, y los resultados no significativos no se leen como ausencia de diferencias, sino con TOST y con la prueba bayesiana. Además, 20 semillas superan con holgura el mínimo de seis que pide el enunciado para pruebas por rangos."]),
    ("Si cada bloque de prueba tiene solo una o dos clases, ¿tiene sentido partir por bloques? ¿Cómo se calcula el F1 por bloque?",
     ["Sí, porque la alternativa tiene fuga. La unidad de independencia es la serie de flujos consecutivos: los vecinos son casi idénticos y el archivo no identifica la simulación de origen, de modo que la serie se aproxima con bloques de 1000 flujos contiguos. Partir por filas mezcla flujos de una misma serie entre entrenamiento y prueba, y en el Entregable 1 eso subía el F1 en 0,0689. La auditoría confirma cero bloques compartidos y cero duplicados exactos entre particiones.",
      "La consecuencia es que, de los 24 bloques de prueba, 14 tienen una clase y 10 tienen dos. Por eso, según la aclaración 2 de la adenda, fijada antes de ver la prueba, el F1 macro de un bloque, de un segmento o de un remuestreo se promedia sobre las clases presentes: promediar sobre las cinco contaría como cero el F1 de una clase ausente, lo que no es un error del modelo. Sobre la prueba completa, con las cinco clases, las dos definiciones coinciden."]),
    ("LightGBM tiene el mayor F1 de prueba y la diferencia es significativa. ¿Por qué no lo declara ganador?",
     ["La regla preregistrada sí lo declara: frente a la CNN 1D, la diferencia es de 0,0164, significativa y relevante (t pareada, p = 1,83e−11; Wilcoxon, p = 1,91e−06), y no cambio esa decisión. Lo que agrego es su alcance. Esa regla compara 20 semillas sobre las mismas 24 series de prueba, de modo que la variabilidad que mide es la del entrenamiento, no la de las series.",
      "Cuando la validación cruzada 5 × 2 agrupada cambia los datos de ajuste y de evaluación, sin tocar la prueba, la diferencia es de −0,0003 (Nadeau y Bengio, p = 0,990; Alpaydin, p = 0,764), y el IC por bloques de la prueba, [−0,0553; 0,0257], contiene el cero. Es decir, la ventaja es real en estas series, pero no se puede afirmar que se repita en series nuevas. McNemar y DeLong también rechazan, pero son optimistas porque tratan los flujos como independientes. Además, el Model Confidence Set al 90 % conserva 10 modelos, LightGBM entre ellos."]),
    ("¿Por qué todos los modelos profundos caen unas siete centésimas de validación a prueba? ¿Es sobreajuste?",
     ["No es sobreajuste del entrenamiento: la brecha entre entrenamiento y validación de los profundos va de −0,0019 a 0,0154, y la prueba de no inferioridad confirma que es menor que δg = 0,05. Tampoco es un desplazamiento propio de la partición: la validación adversarial con pliegues agrupados da un AUC de 0,515 frente a validación y de 0,451 frente a prueba, dentro de la distribución nula.",
      "Lo que explica la caída es la variación entre series: el bloque de prueba explica el 93,1 % de la varianza del F1 por bloque. La caída depende además del tipo de modelo: mediana de 0,0715 en los profundos, 0,0937 en la logística y 0,0123 en LightGBM; la hipótesis es que los umbrales por atributo de los árboles son menos sensibles a los cambios de escala entre series. El IC por bloques de la diferencia entre validación y prueba contiene el cero en los 16 modelos, y por eso el desempeño esperado en series nuevas se reporta con ese intervalo y no con el F1 de validación."]),
    ("¿Por qué Holm y no Bonferroni o FDR? ¿Para qué TOST y ROPE si ya hay pruebas de significancia?",
     ["Holm controla el error por familia igual que Bonferroni, pero ajusta de forma escalonada, de modo que rechaza todo lo que rechaza Bonferroni y a veces más. Se aplica dentro de familias definidas a priori: los 10 contrastes de RQ1, y RQ2 y RQ3 como contrastes únicos. No se usa FDR porque las decisiones de RQ1 son confirmatorias, y el enunciado pide controlar las comparaciones múltiples dentro de familias definidas a priori.",
      "TOST es necesario porque la ausencia de significancia no demuestra equivalencia: para declarar dos modelos equivalentes, las dos pruebas unilaterales deben rechazar que la diferencia exceda ±δ. Así, Autoencoder, Transformer y DBN son equivalentes a la CNN 1D (p TOST de 0,0003 a 0,0026), mientras que CNN-LSTM, BiLSTM y ResNet quedan no concluyentes. La prueba bayesiana con ROPE de ±δ da la probabilidad de que la diferencia esté dentro del margen práctico; complementa la lectura, sin control del error por familia."]),
    ("¿Es válida la predicción conforme si validación y prueba no son intercambiables?",
     ["No del todo, y lo declaro. La garantía marginal de la predicción conforme dividida supone que calibración y prueba son intercambiables; aquí la calibración usa validación, que es disjunta de entrenamiento y de prueba, pero son series distintas. El resultado lo muestra: al 90 %, LAC cubre en prueba de 0,8173 a 0,8259, por debajo de la nominal, y en el peor bloque la cobertura baja a 0,1226; APS alcanza la cobertura, pero con conjuntos de 3,03 a 3,46 de las 5 clases, que informan poco.",
      "Por eso los conjuntos se usan como alerta, por ejemplo para escalar a revisión los flujos con dos clases de enrutamiento en su conjunto, y no como garantía por clase o por serie. El enunciado indica que, con dependencia entre observaciones, se usen variantes adaptadas, por bloques o con ponderación, y se declare la limitación; esa variante, calibrada con datos de la serie de destino, conviene evaluarla en el Entregable 3."]),
    ("¿Qué significa el γ de Cover y Hart y por qué el de la CNN 1D es 1?",
     ["γ es la distancia normalizada al techo de desempeño, γ = (techo − mejor) / (techo − baseline), con el baseline del Entregable 1 como referencia: vale 0 si el modelo alcanza el techo y 1 si se queda en el baseline. El techo se estima con dos cotas: la de Cover y Hart, que con el error del vecino más cercano en validación, 0,1264, acota por debajo el error de Bayes, de modo que la exactitud alcanzable no pasa de 0,9341, y la del aprendizaje confiado, 0,9246.",
      "LightGBM, con exactitud de 0,8847, obtiene γ de 0,665 y 0,711, y la CNN-LSTM, el mejor profundo en prueba, de 0,891 y 0,906. La CNN 1D es el propio baseline, de modo que su γ es 1 por construcción, y por eso se reporta el γ de los mejores en prueba. Las dos cotas son aproximadas: el 1-NN de validación incluye el cambio entre series y el aprendizaje confiado mezcla ruido con superposición, de modo que el techo verdadero podría ser más alto."]),
    ("¿Qué pasó con la suspensión del equipo? ¿Afecta los resultados?",
     ["El 13 de noviembre de 2026, entre las 02:03 y las 09:16, el portátil entró en suspensión durante 7 h 13 min (25 987 segundos), según los eventos Kernel-Power del registro de Windows. Seis corridas de la DBN de la fase 2 estaban en curso; al reanudar, el proceso continuó y su resultado no cambia.",
      "Lo único afectado es el tiempo de reloj de esas seis corridas, que se corrige al leerlas con el descuento registrado en resultados/corridas/suspensiones.json, y la nota queda en la tabla del costo de la búsqueda (sección 4.3.2). Ninguna métrica de desempeño cambia, y si volviera a ocurrir, basta con agregar la ventana a ese archivo."]),
    ("¿Por qué se excluyeron la GNN y la GAN si tienen un buen puntaje?",
     ["Porque el puntaje mide relevancia en la literatura, pero un candidato solo entra si es aplicable al problema declarado: flujos individuales con los 16 atributos de tráfico, sin los identificadores que funcionan como atajo. La GNN obtuvo el tercer puntaje (0,794), pero los tres estudios que la usan construyen el grafo con direcciones IP, nodos o paquetes, que en UAVIDS-2025 delatan el escenario. La GAN (0,419) se usa en la literatura para generar datos, no para clasificar.",
      "Con pesos iguales, o solo con la frecuencia, el Top 10 es el mismo, porque los candidatos aplicables son exactamente diez. El décimo puesto sí es frágil frente al sesgo de acceso: si se contara un estudio no recuperado, FT-Transformer (0,520) superaría a la DBN (0,450); se mantuvo la regla preregistrada, y FT-Transformer quedó declarado como el primer excluido y como mecanismo candidato para el Entregable 3."]),
    ("¿En qué bases se corrieron las cadenas nuevas del Entregable 2?",
     ["Las ocho cadenas nuevas se ejecutaron por programa el 12 de noviembre de 2026 en OpenAlex y arXiv. Las transversales T1 a T4, que son las que exige el enunciado, se corrieron además el 13 de noviembre con el acceso de la biblioteca en Scopus (4 046, 334, 1 181 y 95 registros) y en Web of Science (1 023, 99, 631 y 86), de modo que cumplen el mínimo de dos bases principales. IEEE Xplore no se usó para ellas, porque el primer intento devolvió un error del servidor y el mínimo ya se cumplía.",
      "La repetición de C1 a C3 no se corrió en esas bases, porque las mismas cadenas ya se habían corrido allí el 4 de noviembre, en el Entregable 1, y la actualización en OpenAlex devolvió en C1 y C2 los mismos conteos. C4 es una cadena complementaria que no exige el enunciado: aportó 87 de los 94 registros nuevos, pero ninguno de sus estudios elegibles tuvo texto completo accesible, así que no cambió los 43 estudios incluidos ni el Top 10. Las consultas exactas están en literatura/consultas_e2_wos_scopus_ieee.md."]),
    ("¿Por qué la LSTM del paper guía no se reproduce? ¿No será un error de la réplica?",
     ["La réplica usa el código de los autores, también en PyTorch, con su partición aleatoria 80/20, sus semillas y su mismo entrenamiento, y cuatro de los cinco modelos se reproducen dentro de δ: regresión logística, XGBoost, DNN y CNN. Eso valida la implementación. La LSTM obtiene 0,9285 ± 0,0044 en 10 réplicas, ninguna llega a 0,935 (la mejor, 0,9338) y queda por debajo de la DNN y de la CNN.",
      "El artículo no reporta cuántas corridas promedia ni su dispersión, de modo que el 94 % podría venir de una sola ejecución favorable o de otra versión del conjunto de datos, y la réplica no puede distinguir entre esas explicaciones. Por eso se declara como no reproducido, sin atribuir la diferencia a un error de los autores."]),
    ("¿Por qué no se usó validación anidada para estimar el optimismo de la selección?",
     ["Porque la validación anidada completa repetiría el ajuste dentro de cada pliegue externo y multiplicaría por 12 el costo de todo el benchmark. Se reemplaza por dos mediciones del mismo optimismo: la comparación entre el puntaje de validación usado para seleccionar y el de prueba (tabla B), donde el interno supera al externo en 0,0796 en promedio sobre los 13 modelos profundos (p < 0,001), y la validación cruzada 5 × 2 agrupada del contraste de RQ2, que mide la variabilidad por partición."]),
    ("¿Por qué no se recalibraron las probabilidades con temperature scaling?",
     ["Porque no mejora en series nuevas. La temperatura ajustada en validación queda entre 0,90 y 1,10, muy cerca de 1, de modo que en validación los modelos ya están casi calibrados, y con ella el ECE medio de prueba de los profundos pasa de 0,0523 a 0,0534; la versión adaptativa confirma la lectura. Lo que vuelve las probabilidades demasiado extremas es el paso a series distintas: 14 de los 15 modelos tienen pendiente de calibración menor que 1 en prueba.",
      "La condición (iii) del criterio de adecuación se evalúa con el ECE tras la temperatura, con un umbral de 0,05 que no estaba en el preregistro, de modo que ese veredicto es exploratorio, y recalibrar con datos de la serie de destino queda como requisito de una puesta en producción."]),
    ("¿El umbral de 0,05 del ECE en el criterio de adecuación estaba en el preregistro?",
     ["No. El preregistro fijó δ, δg y el ECE como métrica secundaria, pero no un umbral para la condición de calibración; el de 0,05 se fijó al redactar la sección 7, después de la ejecución definitiva. "
      f"Por eso el veredicto de adecuación se declara exploratorio, no se usa para elegir el modelo final y se reporta su sensibilidad: {TEXTO_SENS}.",
      "Las conclusiones confirmatorias no dependen de ese umbral, porque se apoyan en los contrastes preregistrados de RQ1 a RQ3, con Holm dentro de cada familia. Para el Entregable 3, el umbral de calibración se fija en el preregistro antes de entrenar."]),
    ("¿Por qué la CNN 1D da 0,8584 aquí y 0,8549 en el Entregable 1?",
     ["Porque se reentrenó con el protocolo de este entregable: las mismas 12 configuraciones de ajuste que los demás modelos, las semillas 6001 a 6020 y, como corrección declarada en la sección 4.2, el dropout exacto del código de los autores, que el Entregable 1 había tratado como desviación con un análisis de sensibilidad. El umbral de mejora de 0,8775 se conserva tal como lo fijó el Entregable 1, y ningún modelo profundo lo alcanza."]),
]
for i, (preg, resp) in enumerate(PREGUNTAS, 1):
    parrafo(doc, f"**P{i}. {preg}**", antes=8, despues=4, junto=True)
    for r in resp:
        parrafo(doc, r)

# Nota sobre la fuente del dato del Entregable 1 en la última respuesta
ult = doc.paragraphs[-1]
notas.agregar(ult, "El F1 de 0,8549 de la CNN 1D del Entregable 1 sale de resultados/baseline.json de ese entregable (7 semillas); no aparece en el cuaderno del Entregable 2.")

zoom = doc.settings.element.find(qn("w:zoom"))
if zoom is not None and zoom.get(qn("w:percent")) is None:
    zoom.set(qn("w:percent"), "100")

SALIDA.parent.mkdir(parents=True, exist_ok=True)
doc.save(SALIDA)
print("ok", SALIDA.name, "cuadros:", cuadro.n, "notas:", notas.n)
