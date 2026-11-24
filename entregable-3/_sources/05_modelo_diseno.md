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

# Modelo original (MoE-ES): limitación, fuente, hipótesis, teoría y diseño

Este capítulo explica por qué se propone el modelo, de dónde viene su mecanismo, qué hipótesis pone a prueba, qué propiedad se demuestra y cómo se tomaron las decisiones de diseño sin tocar los datos de confirmación.

```{code-cell} ipython3
:tags: [hide-input]
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
RAIZ = Path("..").resolve()
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "training"))
pd.set_option("display.precision", 4)
```

## Limitación identificada con evidencia

El paper guía {cite:p}`zeng2025uavids` reporta que Blackhole y Wormhole son las clases que más se confunden en UAVIDS-2025, y {cite:t}`zarkadis2026statistical` atribuyen esa confusión a que las dos clases comparten el soporte de su distribución en casi todos los atributos. En el Entregable 2 esa limitación quedó medida con el benchmark completo: en la prueba sellada, LightGBM confunde Blackhole con Wormhole en el 10,0 % de esos flujos y las redes del Top 10 entre el 12,3 y el 18,5 %, mientras que las redes reconocen mejor Sybil (F1 de hasta 0,864 frente a 0,834 de LightGBM). Es decir, los errores del sesgo de árbol, que parte el espacio con umbrales por eje, y los del sesgo neuronal, que ajusta funciones suaves, son **complementarios**, y una regla de fusión fija le da el mismo peso a cada modelo en todos los flujos.

La cuantificación se hizo en la partición de desarrollo, que solo usa filas de entrenamiento del E2 (sección 7.2 del enunciado):

```{code-cell} ipython3
:tags: [hide-input]
d01 = pd.read_csv(RAIZ / "experiments/diseno/resultados/d01/resumen.csv")
d01 = d01.rename(columns={"modelo": "Modelo", "f1_medio": "F1 macro medio", "f1_de": "DE", "conf_BW": "Confusión B–W", "desacuerdo": "Flujos en desacuerdo (%)"})
d01
```

**El oráculo que elige flujo por flujo entre LightGBM y el DNN llega a un F1 macro de 0,871, frente a 0,845 de LightGBM solo, aunque los dos modelos discrepan en apenas el 5,8 % de los flujos.** El promedio simple de probabilidades se queda en 0,849.

Esto significa que hay un margen de +0,026 que ninguna regla fija aprovecha, y que está concentrado en pocos flujos. Al revisar quién acierta cuando discrepan, la red acierta en Sybil y LightGBM en las demás clases; elegir por máxima confianza no aporta (0,848), pero una regla que confía en la red solo cuando predice Sybil sube a 0,856.

Por tal motivo, se opta por un mecanismo que decida **por flujo** cuánto pesa cada sesgo inductivo, en lugar de una regla global.

## Fuente del mecanismo y transferencia

El mecanismo es propio, con tres piezas tomadas de otras aplicaciones y reimplementadas desde su formulación (`experiments/procedencia.md`):

- la mezcla de expertos con red de compuerta de {cite:t}`jacobs1991adaptive`, propuesta para reconocimiento de vocales y patrones;
- la codificación de atributos numéricos de {cite:t}`gorishniy2022embeddings`, que le da a un MLP parte del sesgo de umbral de los árboles en datos tabulares generales;
- la regularización de balance de la compuerta de {cite:t}`shazeer2017outrageously`, propuesta para modelos de lenguaje.

La propiedad estructural compartida es que la frontera entre clases en UAVIDS-2025 tiene partes alineadas con los ejes (umbrales en pérdida, retardo y tasa) y partes suaves (combinaciones de atributos), que es justo la situación en la que {cite:t}`grinsztajn2022tree` muestran que los árboles y las redes fallan de formas distintas. Las modificaciones algorítmicas son cuatro: escalones sigmoides con umbrales y temperaturas aprendibles con temperatura mínima, entrenamiento cooperativo en lugar del conjunto, mezcla de probabilidades en lugar de logits y un regularizador KL de balance para dos expertos densos.

## Hipótesis del mecanismo

La hipótesis, preregistrada en `protocolo.md`, es causal y falsable: **el codificador de escalones suaves da al modelo el sesgo de umbral por eje de los árboles y reduce la confusión Blackhole–Wormhole frente a las redes de sesgo suave, y la compuerta lo combina flujo por flujo con un experto suave que reconoce mejor Sybil; por tanto, el F1 macro del modelo supera al de la mejor red del Top 10 en al menos δ = 0,01, con la ventaja concentrada en Blackhole y Wormhole.**

Las predicciones secundarias declaradas a priori son cinco:

- **P1.** Sin el codificador, la confusión Blackhole–Wormhole sube.
- **P2.** La compuerta pesa más el experto de umbrales en los flujos Blackhole y Wormhole que en los Sybil.
- **P3.** La ventaja frente a la mejor red se concentra en esas dos clases.
- **P4.** La variante con menor cota de Lipschitz se degrada menos ante ruido.
- **P5.** El modelo tampoco generaliza a un escenario de simulación no visto.

## Formulación

El modelo original se llama **MoE-ES**, por mezcla de expertos con escalones suaves; en el código es `ModeloOriginal` y en las tablas, "Original". Sea $x \in \mathbb{R}^{d}$ el flujo transformado con $\log(1+x)$ y estandarizado ($d = 16$), y $K_c = 5$ el número de clases. El **experto de umbrales** codifica cada atributo con $T$ escalones suaves:

$$
s_{j,t}(x) = \sigma\!\left(\frac{x_j - \theta_{j,t}}{\tau_j}\right), \qquad \tau_j = \tau_{\min} + \operatorname{softplus}(\rho_j), \qquad t = 1,\dots,T,
$$ (eq-escalones)

donde los umbrales $\theta_{j,t}$ se inicializan en los cuantiles del entrenamiento y se aprenden, y la temperatura nunca baja de $\tau_{\min}$. Con $s(x) \in \mathbb{R}^{dT}$, los dos expertos y la compuerta son

$$
p_u(x) = \operatorname{softmax}\big(\mathrm{MLP}_u(s(x))\big), \qquad p_s(x) = \operatorname{softmax}\big(\mathrm{MLP}_s(x)\big), \qquad g(x) = \operatorname{softmax}\big(\mathrm{MLP}_g(x)\big) \in \Delta^2,
$$ (eq-expertos)

y la salida es la mezcla de probabilidades

$$
p(y \mid x) = g_u(x)\, p_u(y \mid x) + g_s(x)\, p_s(y \mid x).
$$ (eq-mezcla)

El módulo devuelve $\log p(y \mid x)$; como las probabilidades suman uno, la entropía cruzada estándar sobre esa salida es exactamente la log-verosimilitud negativa de la mezcla, y el modelo entra sin cambios al motor de entrenamiento del E2. El **entrenamiento cooperativo** minimiza, en cada lote y con los pesos de clase $w$ del E2,

$$
\mathcal{L} = \mathrm{CE}_w\!\big(\log(g_u\,\bar p_u + g_s\,\bar p_s),\, y\big) + \mathrm{CE}_w(\log p_u, y) + \mathrm{CE}_w(\log p_s, y) + \gamma\,\mathrm{KL}\big(\bar g \,\|\, \mathcal{U}_2\big),
$$ (eq-perdida)

donde $\bar p_u$ y $\bar p_s$ indican gradiente detenido: el primer término solo entrena la compuerta y los dos siguientes mantienen a cada experto como un clasificador completo. El gradiente del primer término respecto de los logits $a$ de la compuerta es $\partial\mathcal{L}_1/\partial a_k = w_y\,(g_k - r_k)$, con $r_k = g_k \bar p_k(y) / \sum_m g_m \bar p_m(y)$ la responsabilidad posterior del experto $k$: la compuerta se mueve hacia el experto que mejor explica la etiqueta de cada flujo.

## Propiedad demostrada: cota de Lipschitz

Se demuestra una cota de estabilidad (cuadro 3 del enunciado, primera fila). Las tres herramientas son conocidas; el aporte es la cota explícita para esta arquitectura, que depende de $T$ y de $\tau_{\min}$, los hiperparámetros propios del mecanismo.

**Lema 1 (codificador).** Para todo $x, x'$, $\;\|s(x) - s(x')\|_2 \le \dfrac{\sqrt{T}}{4\,\tau_*}\,\|x - x'\|_2$, con $\tau_* = \min_j \tau_j \ge \tau_{\min}$.

*Demostración.* Cada $s_{j,t}$ depende solo de $x_j$. Como $\sigma'(u) = \sigma(u)(1 - \sigma(u)) \le 1/4$, el teorema del valor medio da $|s_{j,t}(x) - s_{j,t}(x')| \le |x_j - x'_j| / (4\tau_j)$. Sumando los cuadrados sobre $t$ y $j$: $\|s(x) - s(x')\|_2^2 \le \sum_j T\,|x_j - x'_j|^2 / (16\tau_j^2) \le T\,\|x - x'\|_2^2/(16\tau_*^2)$. $\square$

**Lema 2 (MLP con ReLU).** Si $\mathrm{MLP}(z) = W_L\,\phi(\cdots\phi(W_1 z + b_1)\cdots) + b_L$ con $\phi = \mathrm{ReLU}$, entonces es Lipschitz con constante $\prod_l \|W_l\|_2$, porque ReLU es 1-Lipschitz y la constante de una composición es a lo sumo el producto de las constantes. $\square$

**Lema 3 (softmax).** $\|\operatorname{softmax}(z) - \operatorname{softmax}(z')\|_2 \le \tfrac12\,\|z - z'\|_2$.

*Demostración.* El jacobiano es $J = \operatorname{diag}(p) - pp^\top$, que es la matriz de covarianza del vector indicador $e_Y$ con $Y \sim p$. Para $\|v\|_2 = 1$, $v^\top J v = \operatorname{Var}(v_Y) \le (\max_i v_i - \min_i v_i)^2/4$ por la desigualdad de Popoviciu, y $|v_i - v_j| \le \sqrt{2}$ cuando $v_i^2 + v_j^2 \le 1$. Entonces $\|J\|_2 \le 1/2$ y el valor medio a lo largo del segmento entre $z$ y $z'$ concluye. $\square$

**Teorema (cota de Lipschitz de la mezcla).** Con $L_u = \tfrac12\prod_l\|W^{u}_l\|_2\cdot\sqrt{T}/(4\tau_*)$, $L_s = \tfrac12\prod_l\|W^{s}_l\|_2$ y $L_g = \tfrac12\prod_l\|W^{g}_l\|_2$, para todo $x, x'$:

$$
\|p(x) - p(x')\|_2 \;\le\; K\,\|x - x'\|_2, \qquad K = \sqrt{2}\,L_g + \max(L_u, L_s).
$$ (eq-cota)

*Demostración.* Por los lemas 1 a 3, $p_u$, $p_s$ y $g$ son Lipschitz con constantes $L_u$, $L_s$ y $L_g$. Escribiendo $\Delta$ para la diferencia entre $x$ y $x'$,

$$
\Delta p = \sum_k \Delta g_k\, p_k(x) + \sum_k g_k(x')\,\Delta p_k ,
$$

de modo que $\|\Delta p\|_2 \le \sum_k |\Delta g_k|\,\|p_k(x)\|_2 + \sum_k g_k(x')\,\|\Delta p_k\|_2$. Como $\|p_k\|_2 \le \|p_k\|_1 = 1$ y $\sum_k g_k(x') = 1$, queda $\|\Delta p\|_2 \le \|\Delta g\|_1 + \max_k\|\Delta p_k\|_2 \le \sqrt{2}\,\|\Delta g\|_2 + \max_k L_k \|\Delta x\|_2 \le (\sqrt{2}L_g + \max_k L_k)\,\|\Delta x\|_2$. $\square$

**Lema 4 (Jensen de la mezcla).** Para cada flujo, $-\log\sum_k g_k\,p_k(y) \le \sum_k g_k\,(-\log p_k(y))$, por la convexidad de $-\log$. La pérdida de la mezcla nunca supera la media ponderada de las pérdidas de los expertos y, si la compuerta elige al mejor experto del flujo, no supera la del mejor. $\square$

**Supuestos y honestidad teórica.** El teorema no necesita supuestos sobre los datos: vale para cualquier entrada, en modo de evaluación (sin dropout). Tiene tres límites que se declaran:

- la cota es global y de peor caso, así que puede ser holgada; por eso se reporta el cociente empírico frente a $K$;
- se expresa en el espacio transformado y estandarizado, no en las unidades originales de los atributos;
- acota la sensibilidad de las probabilidades, no el error: una predicción estable puede ser estable y equivocada.

La pertinencia práctica, es decir, que una $K$ menor se traduzca en menos degradación ante ruido, se contrasta en el capítulo 8 (P4).

### Verificación numérica

Las pruebas unitarias verifican la cota en datos sintéticos para las cinco configuraciones de compuerta (`tests/test_original_model.py`). Aquí se verifica sobre los modelos definitivos entrenados con las semillas 6001 a 6005, con pares de flujos de prueba perturbados en tres escalas:

```{code-cell} ipython3
:tags: [hide-input]
import torch, registro
from ejecutar_original import CONFIGURACION
from codigo import datos as D, modelos as M
registro.registrar(**CONFIGURACION)
h = D.heredados()
x = torch.from_numpy(h["X"][h["ids"]["prueba"]][:4000])
filas = []
for semilla in range(6001, 6006):
    ruta = RAIZ / f"experiments/resultados/corridas/definitiva/definitiva_Original_{semilla}.pt"
    if not ruta.exists():
        continue
    m = M.construir("Original"); m.load_state_dict(torch.load(ruta, map_location="cpu")); m.eval()
    K = m.cota_lipschitz()
    g = torch.Generator().manual_seed(semilla)
    for escala in (1e-3, 1e-2, 1e-1):
        dx = escala * torch.randn(x.shape, generator=g)
        with torch.no_grad():
            cociente = ((m(x).exp() - m(x + dx).exp()).norm(dim=1) / dx.norm(dim=1)).max().item()
        filas.append({"Semilla": semilla, "Escala de la perturbación": escala, "Cociente empírico máximo": cociente, "Cota K": K,
                      "Cociente / K": cociente / K, "¿Se cumple?": cociente <= K})
verificacion = pd.DataFrame(filas)
verificacion
```

**En las 15 combinaciones de semilla y escala, el cociente empírico máximo (entre 1,80 y 2,68) queda por debajo de la cota $K$ (entre 24,3 y 34,8), como exige el teorema. La relación cociente/$K$ va de 0,06 a 0,09: la cota es entre 11 y 17 veces más holgada que lo observado.**

Esto quiere decir que la garantía se cumple sin excepciones y que, como toda cota global obtenida con normas espectrales, es conservadora: el modelo es mucho más estable en la práctica de lo que la cota asegura en el peor caso.

Por ello, se usa $K$ como garantía de peor caso y como guía de diseño (una $\tau_{\min}$ mayor la reduce), y la robustez real se mide con las perturbaciones controladas del capítulo 8.

## Pseudocódigo

```text
Algoritmo: Compuerta cooperativa de sesgos complementarios
Entrada: entrenamiento {(x_i, y_i)}, validación, T = 8, τ_min = 0,05, γ = 0,01, hiperparámetros de optimización (lr, wd, lote)
Salida: parámetros Θ = (θ, ρ, MLP_u, MLP_s, MLP_g) y la cota K
1. θ_{j,t} ← cuantil t/(T+1) del atributo j en el entrenamiento;  ρ_j ← softplus⁻¹(0,25 − τ_min)
2. para época = 1, …, 30:
3.     para cada lote B:
4.         s ← σ((x − θ)/τ);  p_u ← softmax(MLP_u(s));  p_s ← softmax(MLP_s(x));  g ← softmax(MLP_g(x))
5.         L ← CE_w(log(g_u·sg(p_u) + g_s·sg(p_s)), y) + CE_w(log p_u, y) + CE_w(log p_s, y) + γ·KL(media(g) ‖ U)
6.         Θ ← Adam(Θ, ∇L) con recorte del gradiente en norma 5
7.     F1_val ← F1 macro de p = g_u p_u + g_s p_s en validación;  ReduceLROnPlateau(pérdida de validación)
8.     si F1_val no mejora en 6 épocas: detener y restaurar el mejor Θ
9. K ← √2·L_g + max(L_u, L_s)   (ecuación de la cota)
```

## Disciplina de diseño

El diseño se tomó en dos particiones de desarrollo hechas solo con filas de entrenamiento del E2: una que deja fuera un escenario de simulación completo y otra por bloques. La validación y la prueba del E2 son el conjunto de confirmación. La bitácora completa está en `experiments/design_log.md`; el número total de variantes evaluadas es **Nd = 10**.

```{code-cell} ipython3
:tags: [hide-input]
d02 = pd.read_csv(RAIZ / "experiments/diseno/resultados/d02/resumen.csv")
d03 = pd.read_csv(RAIZ / "experiments/diseno/resultados/d03/resumen.csv")
d3 = pd.read_csv(RAIZ / "experiments/bloque_a/prueba_d3/_resumen.csv")
diseno = pd.concat([d3.assign(Partición="por escenario (9 pliegues)").rename(columns={"metodo": "variante", "acierto_fuera_medio": "métrica"})[["variante", "Partición", "métrica"]],
                    pd.concat([d02, d03]).assign(Partición="por bloques (3 reparticiones)").rename(columns={"f1": "métrica"})[["variante", "Partición", "métrica"]]])
diseno["Métrica"] = np.where(diseno["Partición"].str.startswith("por escenario"), "acierto en el escenario no visto", "F1 macro de diseño")
diseno[["variante", "Partición", "Métrica", "métrica"]]
```

**La opción de invariancia por escenario (V-REx condicional por clase) bajó el acierto en el escenario no visto con los tres valores de λ y quedó descartada por la regla fijada antes de correr. En la partición por bloques, el experto de escalones suaves solo llega a 0,843 (LightGBM, 0,845), la compuerta entrenada en conjunto empeora a 0,832 porque los expertos se especializan, y el entrenamiento cooperativo la recupera a 0,842.**

Hay que tener en cuenta que la compuerta, en desarrollo, no capturó el margen del oráculo: aprende con filas en las que los dos expertos aciertan casi siempre y recibe poca señal de en qué falla cada uno fuera de muestra. Esto se declara antes de ver la confirmación y forma parte de la discusión del capítulo 10.

Con base en esto, se congeló la variante 10 (compuerta cooperativa que ve el flujo; T = 8, τ_min = 0,05, γ = 0,01) el 22/11/2026, antes del protocolo con etiqueta `e3-protocolo`. Su sesgo de selección es acotado porque Nd es pequeño, y la confirmación se hace en datos que no participaron en el diseño.
