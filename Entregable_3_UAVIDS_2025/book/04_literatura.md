---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Revisión de literatura y evidencia de novedad

La revisión sistemática del Entregable 1, actualizada en el Entregable 2, sostiene el Top 10. En el E1 se exportaron 1 953 registros de OpenAlex, arXiv, IEEE Xplore, Scopus y Web of Science, y se incluyeron 42 estudios; en el E2 la actualización llevó el total a 43. Este capítulo se ocupa de otra pregunta, la que exige la condición R1: si el mecanismo del modelo original ya se usó para detectar intrusiones en redes de UAV y, en caso de que exista algo parecido, si la diferencia es de mecanismo o solo de aplicación (sección 6.3 del enunciado).

```{code-cell} ipython3
:tags: [hide-input]
import json
from pathlib import Path
import pandas as pd
RAIZ = Path("..").resolve()
N = RAIZ / "experiments/novedad"
pd.set_option("display.max_colwidth", 300)
```

## Cadenas de búsqueda

Se usaron siete cadenas, que combinan el mecanismo con el problema de aplicación, buscan el mecanismo en otras aplicaciones para ubicar su origen, buscan el origen de la propiedad demostrada e incluyen sinónimos en español. La primera consulta se hizo el 22/11/2026 en OpenAlex y arXiv, por programa (`experiments/novedad/buscar_novedad.py`). El 23/11/2026 se agregaron IEEE Xplore (N1) y la búsqueda por citación y en repositorios de código (`experiments/novedad/citacion_y_repositorios.py`). La búsqueda completa se repitió el 23/11/2026 con el mismo programa, y los conteos coinciden con los del 22/11 salvo OpenAlex N7, que pasa de 22 a 23 por BRIDGE (ICML 2025), una mezcla de expertos en un modelo fundacional de grafos que trata otro problema y no cambia la declaración.

```{code-cell} ipython3
:tags: [hide-input]
resumen = sorted(p for p in N.glob("exportaciones/*_resumen_novedad.json") if p.name[:10].replace("-", "").isdigit() and p.name[10:] == "_resumen_novedad.json")[-1]
r = json.loads(resumen.read_text(encoding="utf-8"))
cadenas = pd.DataFrame([{"Clave": k, "Cadena": v} for k, v in r["cadenas"].items()])
conteos = pd.DataFrame(r["resultados"]).pivot(index="cadena", columns="fuente", values="total")
print("Fecha de la consulta:", r["fecha"])
cadenas.merge(conteos, left_on="Clave", right_index=True)
```

## Matriz de novedad

```{code-cell} ipython3
:tags: [hide-input]
texto = (N / "matriz_novedad.md").read_text(encoding="utf-8")
def tabla(seccion, siguiente):
    lineas = texto.split(seccion)[1].split(siguiente)[0].strip().splitlines()
    filas = [[c.strip() for c in l.strip("|").split("|")] for l in lineas if l.startswith("|") and not l.startswith("|---")]
    return pd.DataFrame(filas[1:], columns=filas[0])
tabla("## Matriz de novedad (cuadro 2 del enunciado)", "## Clasificación")
```

## Trabajos más cercanos y clasificación de la diferencia

```{code-cell} ipython3
:tags: [hide-input]
tabla("## Clasificación de la diferencia frente a cada trabajo cercano", "## Búsqueda por citación")
```

**Las siete cadenas devuelven pocos registros: para la mezcla de expertos aplicada a intrusiones en UAV (N1), 3 en OpenAlex, 2 en IEEE Xplore, 4 en Scopus, 1 en Web of Science y ninguno en arXiv ni en español; para los híbridos de árboles y redes en UAV (N5), 13 en OpenAlex, 28 en Scopus, de los cuales 18 son actas de congreso, y 3 en Web of Science. La búsqueda por citación sobre 56 obras que citan a los trabajos más cercanos, los 27 repositorios de código revisados y las cadenas N1, N2 y N5 en Scopus agregan siete trabajos cercanos, y Web of Science solo devuelve trabajos ya encontrados; ninguno combina un experto de escalones suaves con uno suave mediante una compuerta aprendida por flujo.**

Así mismo, el trabajo más cercano en el problema, el estudio de UAVIDS-2025 de {cite:t}`zarkadis2026statistical`, compara árboles, redes y *stacking*, que es una regla de fusión estándar y una de las reglas contra las que se contrasta la compuerta propia, y llega por otro camino a la misma limitación: Blackhole y Wormhole comparten el soporte de su distribución. El trabajo más cercano en la motivación es PS-CAHE {cite:p}`ma2026pscahe`, que sobre UAVIDS-2025 fija pesos de fusión para los pares de clases que se confunden; lo hace con pesos por par de clases sobre modelos externos, no con una compuerta aprendida por flujo, y su ventaja desaparece con nodos disjuntos. El más cercano en el mecanismo es ACGE {cite:p}`abdelnaby2026acge`, que en redes fog-IoT pondera por muestra un LightGBM y un Transformer con una regresión logística sobre cuatro rasgos de confianza del LightGBM, ajustada en validación después de entrenar los dos modelos; el mismo autor la presenta como una adaptación del *stacking* y de la mezcla de expertos, no como un mecanismo nuevo. Sin embargo, su experto de umbrales es un árbol externo, que no se entrena por gradiente y es discontinuo, mientras que la propuesta lleva el sesgo de umbral dentro de la red con escalones suaves, su compuerta ve el flujo, se entrena con los expertos bajo el esquema cooperativo y la pendiente de la mezcla queda acotada. Le siguen las mezclas de expertos homogéneos: para intrusiones en 5G (arXiv 2412.03483), con enrutamiento disperso, y en Switch IDS {cite:p}`wang2025switchids`, para repartir el tráfico entre nodos de borde asistidos por aeronaves. La fusión de XGBoost y un MLP con un peso global {cite:p}`albalasmeh2026xgbmlp`, la compuerta ajustada fuera de muestra {cite:p}`alessa2026vgmoe` y la mezcla de un experto por conjunto de datos con *stacking* de U-Flow MoE-IDS {cite:p}`bhaskaran2026uflow` completan el grupo. En todos los casos la diferencia es de mecanismo o de esquema de entrenamiento, que son las dos categorías que sustentan R1.

Por lo anterior, conviene declarar la novedad en los términos de la sección siguiente, con OpenAlex, IEEE Xplore y arXiv como las tres bases de la sección 6.3 del enunciado y Scopus y Web of Science como confirmación.

## Declaración de novedad

Al mejor de nuestro conocimiento y según la búsqueda descrita, no se encontraron estudios que apliquen a la detección de intrusiones en redes de UAV una compuerta por flujo entre un experto de escalones suaves con umbrales aprendibles y un experto neuronal suave, entrenada con un esquema cooperativo y con cota de Lipschitz demostrada.

Los trabajos más cercanos, y la diferencia de cada uno con la propuesta, son:

- **PS-CAHE sobre UAVIDS-2025** {cite:p}`ma2026pscahe`: comparte la motivación, que son los pares de clases que se confunden, pero fija pesos de fusión por par sobre modelos externos; la propuesta aprende una compuerta por flujo entre dos expertos internos de sesgo distinto, con entrenamiento cooperativo.
- **ACGE en fog-IoT** {cite:p}`abdelnaby2026acge`: es el más cercano en el mecanismo, porque pondera por muestra un árbol (LightGBM) y un Transformer con una compuerta aprendida; la propuesta reemplaza el árbol externo por un experto neuronal de escalones suaves, lo que permite entrenar la compuerta y los expertos de forma cooperativa y demostrar una cota de Lipschitz para la mezcla, que con un árbol discontinuo no existe.
- **Switch IDS** {cite:p}`wang2025switchids` **y la mezcla de expertos para intrusiones en 5G:** usan expertos homogéneos; la propuesta difiere en el mecanismo y en el esquema de entrenamiento.
- **El estudio de *stacking* sobre UAVIDS-2025** {cite:p}`zarkadis2026statistical`: difiere en el mecanismo, porque la propuesta reemplaza una regla de fusión estándar por una compuerta aprendida por flujo.
- **Las fuentes de las piezas** {cite:p}`jacobs1991adaptive,gorishniy2022embeddings`: las cuatro modificaciones algorítmicas están en `experiments/procedencia.md`.

## Verificación por un tercero

Los profesores del curso indicaron el 23/11/2026 que esta verificación queda a su cargo, por lo que se omite en el informe. Para facilitarla, las cadenas exactas están en `experiments/novedad/buscar_novedad.py` (OpenAlex y arXiv) y en `experiments/novedad/cadenas_scopus_wos.md` (Scopus, Web of Science y los enlaces directos de OpenAlex para N1 y N5), y los conteos de cada consulta, con su fecha, en `experiments/novedad/exportaciones/`.
