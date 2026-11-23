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

Se usaron siete cadenas, que combinan el mecanismo con el problema de aplicación, buscan el mecanismo en otras aplicaciones para ubicar su origen, buscan el origen de la propiedad demostrada e incluyen sinónimos en español. La primera consulta se hizo el 22/11/2026 en OpenAlex y arXiv, por programa (`experiments/novedad/buscar_novedad.py`), y se repite antes de la entrega.

```{code-cell} ipython3
:tags: [hide-input]
resumen = sorted(N.glob("exportaciones/*_resumen_novedad.json"))[-1]
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

**Las siete cadenas devuelven pocos registros: 3 para la mezcla de expertos aplicada a intrusiones en UAV en OpenAlex, ninguno en arXiv ni en español, y 13 para los híbridos de árboles y redes en UAV. Ninguno de los trabajos cercanos combina un experto de escalones suaves con uno suave mediante una compuerta por flujo.**

Así mismo, el trabajo más cercano en el problema, el estudio de UAVIDS-2025 de {cite:t}`zarkadis2026statistical`, compara árboles, redes y *stacking*, que es una regla de fusión estándar y una de las reglas contra las que se contrasta la compuerta propia, y llega por otro camino a la misma limitación: Blackhole y Wormhole comparten el soporte de su distribución. El trabajo más cercano en el mecanismo, la mezcla de expertos para intrusiones en 5G (arXiv 2412.03483), usa expertos homogéneos sobre una representación convolucional, enrutamiento disperso y entrenamiento conjunto. En todos los casos la diferencia es de mecanismo o de esquema de entrenamiento, que son las dos categorías que sustentan R1.

Por lo anterior, conviene declarar la novedad en los términos de la sección siguiente y dejarla sujeta a la confirmación de Scopus, Web of Science y el verificador externo.

## Declaración de novedad

Al mejor de nuestro conocimiento y según la búsqueda descrita, no se encontraron estudios que apliquen a la detección de intrusiones en redes de UAV una compuerta por flujo entre un experto de escalones suaves con umbrales aprendibles y un experto neuronal suave, entrenada con un esquema cooperativo y con cota de Lipschitz demostrada.

Los trabajos más cercanos, y la diferencia de cada uno con la propuesta, son:

- **La mezcla de expertos homogéneos para intrusiones en 5G:** la propuesta difiere en el mecanismo y en el esquema de entrenamiento.
- **El estudio de *stacking* sobre UAVIDS-2025** {cite:p}`zarkadis2026statistical`: difiere en el mecanismo, porque la propuesta reemplaza una regla de fusión estándar por una compuerta aprendida por flujo.
- **Las fuentes de las piezas** {cite:p}`jacobs1991adaptive,gorishniy2022embeddings`: las cuatro modificaciones algorítmicas están en `experiments/procedencia.md`.

## Verificación por un tercero

Una persona ajena al diseño repite las cadenas N1 y N5 y anota la fecha, el número de resultados y su concepto. El concepto se anexa en el capítulo 11.
