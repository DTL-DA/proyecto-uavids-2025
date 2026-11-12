# Consultas para Web of Science, Scopus e IEEE Xplore

Estas consultas completan la sección 3.1 del informe. Son las mismas tres cadenas que se ejecutaron en OpenAlex y arXiv, adaptadas a la sintaxis de cada base, de modo que los resultados se puedan unir y depurar con el mismo procedimiento. El periodo es de 2019 a 2026.

Se usa el campo más cercano al de OpenAlex en cada base. Scopus y Web of Science incluyen además las palabras clave, e IEEE Xplore busca en todos los metadatos, por lo que los conteos pueden ser algo más amplios; esa diferencia se declara en el informe.

## OpenAlex y arXiv

Se consultaron por programa con `buscar_literatura.py` el 3 de noviembre de 2026. En OpenAlex, cada cadena va en el filtro `title_and_abstract.search`, desde el 1 de enero de 2019. En arXiv, cada término o frase lleva el prefijo `abs:`, por ejemplo `abs:drone OR abs:UAV OR abs:"unmanned aerial"`, y se conservan las publicaciones desde 2019.

| Base | C1 | C2 | C3 | Fecha |
|---|---|---|---|---|
| OpenAlex | 323 | 40 | 220 | 3 de noviembre de 2026 |
| arXiv | 11 | 3 | 29, de los cuales 28 desde 2019 | 3 de noviembre de 2026 |

## IEEE Xplore

Se entra por *Advanced Search*, pestaña *Command Search*, se pega cada cadena y luego se limita el año de publicación de 2019 a 2026.

- **C1:** `(drone OR UAV OR "unmanned aerial" OR FANET) AND ("intrusion detection" OR "attack detection") AND ("deep learning" OR "neural network" OR CNN OR LSTM OR MLP)`
- **C2:** `(drone OR UAV OR "unmanned aerial" OR FANET) AND (swarm OR "ad hoc" OR "network traffic") AND (blackhole OR wormhole OR sybil OR flooding OR cyberattack) AND ("deep learning" OR "neural network" OR CNN OR LSTM OR MLP)`
- **C3:** `UAVIDS OR "UAVIDS-2025"`

Exportación: *Export*, pestaña *Search Results*, formato CSV.

## Scopus

Se entra por *Advanced document search* y se pega cada cadena.

- **C1:** `TITLE-ABS-KEY((drone OR uav OR "unmanned aerial" OR fanet) AND ("intrusion detection" OR "attack detection") AND ("deep learning" OR "neural network" OR cnn OR lstm OR mlp)) AND PUBYEAR > 2018 AND PUBYEAR < 2027`
- **C2:** `TITLE-ABS-KEY((drone OR uav OR "unmanned aerial" OR fanet) AND (swarm OR "ad hoc" OR "network traffic") AND (blackhole OR wormhole OR sybil OR flooding OR cyberattack) AND ("deep learning" OR "neural network" OR cnn OR lstm OR mlp)) AND PUBYEAR > 2018 AND PUBYEAR < 2027`
- **C3:** `TITLE-ABS-KEY(uavids OR "UAVIDS-2025") AND PUBYEAR > 2018 AND PUBYEAR < 2027`

Exportación: *Export*, formato CSV, con la información de citación y con el resumen y las palabras clave.

## Web of Science

Se entra por *Advanced Search*, en la colección principal (*Core Collection*), y se pega cada cadena.

- **C1:** `TS=((drone OR UAV OR "unmanned aerial" OR FANET) AND ("intrusion detection" OR "attack detection") AND ("deep learning" OR "neural network" OR CNN OR LSTM OR MLP)) AND PY=(2019-2026)`
- **C2:** `TS=((drone OR UAV OR "unmanned aerial" OR FANET) AND (swarm OR "ad hoc" OR "network traffic") AND (blackhole OR wormhole OR sybil OR flooding OR cyberattack) AND ("deep learning" OR "neural network" OR CNN OR LSTM OR MLP)) AND PY=(2019-2026)`
- **C3:** `TS=(UAVIDS OR "UAVIDS-2025") AND PY=(2019-2026)`

Exportación: *Export*, formato Excel, con el registro completo (*Full Record*).

## Dónde dejar los archivos

Cada exportación se guarda en `literatura/exportaciones/` con el nombre `AAAA-MM-DD_Base_Cadena`, por ejemplo `2026-11-04_IEEE_C1.csv`, `2026-11-04_Scopus_C2.csv` o `2026-11-04_WoS_C3.xlsx`.

Por cada consulta también se anota la fecha y el número de resultados que muestra la base, porque el enunciado los pide. Las exportaciones se hicieron con el acceso de la biblioteca de la Universidad del Norte:

| Base | C1 | C2 | C3 | Fecha |
|---|---|---|---|---|
| IEEE Xplore | 481 | 40 | 81 | 4 de noviembre de 2026 |
| Scopus | 523 | 18 | 26 | 4 de noviembre de 2026 |
| Web of Science | 132 | 11 | 16 | 4 de noviembre de 2026 |

## Verificación de cobertura de sinónimos

Se ejecutó en OpenAlex el 5 de noviembre de 2026 con `verificar_cobertura_sinonimos.py`, en el filtro `title_and_abstract.search`, del 1 de enero de 2019 al 4 de noviembre de 2026. No se repitió en Scopus, Web of Science ni IEEE Xplore.

**Plataforma (P):** (drone OR UAV OR UAS OR "unmanned aerial" OR "unmanned aircraft" OR FANET OR "flying ad hoc" OR "Internet of Drones" OR IoD OR quadcopter)

**Método (M):** ("deep learning" OR "neural network" OR CNN OR LSTM OR MLP OR GRU OR RNN OR transformer OR autoencoder)

| Cadena | Consulta | Registros |
|---|---|---|
| C1 ampliada | P AND ("intrusion detection" OR "attack detection" OR "anomaly detection" OR IDS OR "cyberattack detection" OR "threat detection") AND M | 851 |
| C2 ampliada | P AND (swarm OR "ad hoc" OR "network traffic") AND (blackhole OR "black hole" OR wormhole OR sybil OR flooding OR DDoS OR cyberattack) AND M | 49 |

De los registros de ambas cadenas, 523 no estaban entre los 1109 depurados y 422 superaron el mismo filtro automático del proyecto. El cribado por título y resumen, en `cribado_cobertura_sinonimos.csv`, dejó 20 elegibles y 3 dudosos. Se leyeron los 5 que están en revistas Q1, cuya extracción está en `estudios_cobertura.json`: se incluyen 4 y se excluye 1.
