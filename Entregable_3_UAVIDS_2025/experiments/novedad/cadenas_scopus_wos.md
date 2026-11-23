# Cadenas N1, N2 y N5 en Scopus y Web of Science

Son las mismas cadenas de `matriz_novedad.md`, escritas con la sintaxis de cada base. En Scopus, el campo es título, resumen y palabras clave (`TITLE-ABS-KEY`). En Web of Science, el campo *Topic* (`TS`) cubre título, resumen, palabras clave del autor y Keywords Plus. No se filtra por año, tipo de documento ni idioma.

## Scopus (Advanced document search)

**N1**

```
TITLE-ABS-KEY(("mixture of experts" OR "gating network" OR "gated fusion" OR "dynamic ensemble selection") AND ("intrusion detection" OR "attack detection") AND (UAV OR drone OR FANET OR "unmanned aerial"))
```

**N2**

```
TITLE-ABS-KEY(("mixture of experts" OR "gating network") AND "intrusion detection" AND ("decision tree" OR "gradient boosting" OR "tree-based" OR "soft tree"))
```

**N5**

```
TITLE-ABS-KEY((UAV OR drone OR FANET OR "unmanned aerial") AND "intrusion detection" AND (hybrid OR ensemble OR fusion) AND ("gradient boosting" OR LightGBM OR XGBoost OR "random forest" OR "decision tree") AND ("neural network" OR "deep learning"))
```

## Web of Science Core Collection (Advanced Search)

**N1**

```
TS=(("mixture of experts" OR "gating network" OR "gated fusion" OR "dynamic ensemble selection") AND ("intrusion detection" OR "attack detection") AND (UAV OR drone OR FANET OR "unmanned aerial"))
```

**N2**

```
TS=(("mixture of experts" OR "gating network") AND "intrusion detection" AND ("decision tree" OR "gradient boosting" OR "tree-based" OR "soft tree"))
```

**N5**

```
TS=((UAV OR drone OR FANET OR "unmanned aerial") AND "intrusion detection" AND (hybrid OR ensemble OR fusion) AND ("gradient boosting" OR LightGBM OR XGBoost OR "random forest" OR "decision tree") AND ("neural network" OR "deep learning"))
```

## OpenAlex (enlaces directos)

Son N1 y N5 sobre título y resumen, desde 2015, sin cuenta. Cada enlace devuelve un JSON: el número de resultados está en `meta.count` y los títulos en `results`. Las siete cadenas completas, con OpenAlex y arXiv, están en `buscar_novedad.py`.

- **N1** (mecanismo + drones): https://api.openalex.org/works?per-page=50&select=title,publication_year,doi&filter=title_and_abstract.search%3A%28%22mixture%20of%20experts%22%20OR%20%22gating%20network%22%20OR%20%22gated%20fusion%22%20OR%20%22dynamic%20ensemble%20selection%22%29%20AND%20%28%22intrusion%20detection%22%20OR%20%22attack%20detection%22%29%20AND%20%28UAV%20OR%20drone%20OR%20FANET%20OR%20%22unmanned%20aerial%22%29%2Cfrom_publication_date%3A2015-01-01
- **N5** (híbridos árbol-red en drones): https://api.openalex.org/works?per-page=50&select=title,publication_year,doi&filter=title_and_abstract.search%3A%28UAV%20OR%20drone%20OR%20FANET%20OR%20%22unmanned%20aerial%22%29%20AND%20%22intrusion%20detection%22%20AND%20%28hybrid%20OR%20ensemble%20OR%20fusion%29%20AND%20%28%22gradient%20boosting%22%20OR%20LightGBM%20OR%20XGBoost%20OR%20%22random%20forest%22%20OR%20%22decision%20tree%22%29%20AND%20%28%22neural%20network%22%20OR%20%22deep%20learning%22%29%2Cfrom_publication_date%3A2015-01-01

## Registro

Por cada base y cadena se anota: la fecha, el número de resultados, cuántos se revisaron por título y resumen, cuántos son relevantes y el trabajo más cercano. Los resultados se exportan a `exportaciones/` (Scopus: CSV; Web of Science: "Plain text file" o "Excel").
