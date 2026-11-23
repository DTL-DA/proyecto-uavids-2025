# Guía de verificación externa — Entregable 3, UAVIDS-2025

**Verificadora:** Imera Logreira · **Autor del proyecto:** Donny Torres · **Fecha sugerida:** 25/11/2026

Gracias por ayudar. El enunciado del curso exige que una persona que **no participó en el diseño ni en el código** haga dos verificaciones:

1. **Novedad:** repetir dos cadenas de búsqueda y confirmar o refutar la declaración de novedad (sección 6.3 del enunciado).
2. **Reproducción:** ejecutar el proyecto desde cero dentro de un contenedor de Docker, regenerar la tabla principal de resultados y reportar la discrepancia máxima (secciones 11 y 14.5.7).

Las dos las haces tú sola, en tu computador. Donny solo te entrega este paquete. Si algo falla, **no lo arregles ni dejes que nadie lo arregle**: anota el error tal cual y envíaselo a Donny. Ese registro también cuenta como evidencia. Tiempo total: entre 2 y 3 horas, casi todo esperando a que el computador termine.

---

## Tarea 1. Verificar la novedad (unos 45 minutos)

**Haz esta tarea antes de abrir cualquier otro archivo del paquete**, en especial `experiments/novedad/matriz_novedad.md`, para que tu búsqueda sea independiente de la nuestra.

**La declaración que vas a verificar:**

> Al mejor de nuestro conocimiento, no se encontraron estudios que apliquen a la detección de intrusiones en redes de UAV (drones) una compuerta por flujo entre un experto de "escalones suaves" con umbrales aprendibles y un experto neuronal suave, entrenada con un esquema cooperativo y con cota de Lipschitz demostrada.

En palabras simples: un modelo que, para cada flujo de red, decide cuánto confiar en un experto tipo "árbol suave" y cuánto en una red neuronal, aplicado a ataques en redes de drones.

**Pasos:**

1. Abre en el navegador estos dos enlaces de OpenAlex (base bibliográfica abierta; no pide cuenta). Cada uno devuelve un JSON: el número de resultados está en `"count"` y los títulos en `"results"`.
   - **N1** (mecanismo + drones):
     https://api.openalex.org/works?per-page=50&select=title,publication_year,doi&filter=title_and_abstract.search%3A%28%22mixture%20of%20experts%22%20OR%20%22gating%20network%22%20OR%20%22gated%20fusion%22%20OR%20%22dynamic%20ensemble%20selection%22%29%20AND%20%28%22intrusion%20detection%22%20OR%20%22attack%20detection%22%29%20AND%20%28UAV%20OR%20drone%20OR%20FANET%20OR%20%22unmanned%20aerial%22%29%2Cfrom_publication_date%3A2015-01-01
   - **N5** (híbridos árbol-red en drones):
     https://api.openalex.org/works?per-page=50&select=title,publication_year,doi&filter=title_and_abstract.search%3A%28UAV%20OR%20drone%20OR%20FANET%20OR%20%22unmanned%20aerial%22%29%20AND%20%22intrusion%20detection%22%20AND%20%28hybrid%20OR%20ensemble%20OR%20fusion%29%20AND%20%28%22gradient%20boosting%22%20OR%20LightGBM%20OR%20XGBoost%20OR%20%22random%20forest%22%20OR%20%22decision%20tree%22%29%20AND%20%28%22neural%20network%22%20OR%20%22deep%20learning%22%29%2Cfrom_publication_date%3A2015-01-01
2. *(Opcional, si tienes acceso a Scopus por la biblioteca.)* En **Advanced document search** pega las cadenas N1 y N5 de `experiments/novedad/cadenas_scopus_wos.md`.
3. Lee los títulos y, si alguno se parece a la declaración, abre su resumen (por el DOI).
4. Llena el formato `concepto_novedad.md`: fecha, base, número de resultados, cuántos revisaste, el trabajo más parecido y tu concepto (confirma o refuta). Fírmalo.

---

## Tarea 2. Reproducir los resultados (de 1,5 a 2,5 horas)

### Requisitos

- Windows 10/11, macOS o Linux, con al menos 8 GB de RAM y 15 GB libres en disco.
- **Docker Desktop** instalado y abierto (https://www.docker.com/products/docker-desktop/). En Windows te pedirá activar WSL 2; acepta y reinicia si lo pide.
- Internet para construir el contenedor (descarga cerca de 2 GB una sola vez).

### Pasos

Abre **PowerShell** (Windows) o una terminal (macOS/Linux) y copia los comandos uno por uno. Anota en el reporte el tiempo de cada paso y cualquier mensaje de error.

**1. Descomprime el paquete en una carpeta corta**, por ejemplo `C:\uavids` (las rutas largas de Windows o de OneDrive pueden fallar), y entra en ella:

```
cd C:\uavids
docker --version
```

Anota la versión de Docker.

**2. Construye el contenedor** (de 10 a 25 minutos):

```
docker build -t uavids-e3 -f Entregable_3_UAVIDS_2025/Dockerfile .
```

Debe terminar sin la palabra `ERROR`.

**3. Regenera la tabla principal y reentrena tres corridas desde cero** (de 10 a 30 minutos):

```
docker run --name repro uavids-e3 python verificacion_externa/reproducir.py
docker cp repro:/proyecto/Entregable_3_UAVIDS_2025/verificacion_externa/salida ./salida_reproduccion
docker rm repro
```

Al final imprime un bloque `===== RESULTADO =====`. Copia sus cuatro líneas en el reporte:

- `datos íntegros`: debe ser `true`.
- `discrepancia máxima de la tabla principal`: la discrepancia que pide el enunciado.
- `alguna decisión cambia`: si la conclusión estadística cambia en tu equipo.
- `discrepancia máxima de F1 al reentrenar`: diferencia entre el F1 publicado y el que obtuvo tu equipo al entrenar de nuevo.

La carpeta `salida_reproduccion` queda en `C:\uavids` con el detalle completo.

**4. Corre las pruebas unitarias** (de 2 a 5 minutos):

```
docker run --rm uavids-e3 python -m pytest tests
```

Anota la línea final, del tipo `34 passed`.

**5. Compila el libro desde cero** (de 20 a 60 minutos; ejecuta todo el código del libro):

```
docker run --name libro uavids-e3 jupyter-book build book/
docker cp libro:/proyecto/Entregable_3_UAVIDS_2025/book/_build/html ./libro_html
docker rm libro
```

Anota si al final aparece `Finished generating HTML for book` y si hubo errores de ejecución de celdas. Abre `libro_html\index.html`, ve al capítulo 7 ("Validación") y compara a ojo la tabla de resultados confirmatorios con `salida_reproduccion\tabla_contrastes_regenerada.csv`.

**6. Prueba de humo del dashboard** (5 minutos):

```
docker run --rm -p 8050:8050 uavids-e3
```

Abre http://localhost:8050, recorre las páginas, elige un flujo en la página de modelos y verifica que dé una predicción. Cierra con `Ctrl + C`. Si Donny te envió el enlace del dashboard publicado en Render, haz la misma revisión allí.

**7. Llena y firma** el formato `reporte_reproduccion.md`.

---

## Qué le envías a Donny

1. `concepto_novedad.md` y `reporte_reproduccion.md` firmados (PDF o foto de la firma).
2. La carpeta `salida_reproduccion` comprimida.
3. Si algo falló, una captura o el texto completo del error.
