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

## Registro

Por cada base y cadena se anota: la fecha, el número de resultados, cuántos se revisaron por título y resumen, cuántos son relevantes y el trabajo más cercano. Los resultados se exportan a `exportaciones/` (Scopus: CSV; Web of Science: "Plain text file" o "Excel").
