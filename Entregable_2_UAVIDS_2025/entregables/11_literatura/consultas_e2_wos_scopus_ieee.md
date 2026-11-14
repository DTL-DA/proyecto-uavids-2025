# Consultas del Entregable 2 para Web of Science, Scopus e IEEE Xplore

Estas consultas extienden las del Entregable 1 (`Entregable_1_UAVIDS_2025/literatura/consultas_wos_scopus_ieee.md`). C1 a C3 son las mismas cadenas, que en OpenAlex y arXiv se volvieron a ejecutar el 12 de noviembre de 2026 para captar lo publicado después del 3 de noviembre; C4 amplía la cadena con las familias candidatas al Top 10, y T1 a T4 son la búsqueda transversal de mecanismos en otras modalidades que pide la sección 3.1 del enunciado. El periodo es de 2019 a 2026 para C1 a C4 y de 2018 a 2026 para T1 a T4.

## OpenAlex y arXiv (ejecutadas por programa)

`buscar_literatura_e2.py`, 12 de noviembre de 2026. Conteos en `exportaciones/2026-11-12_resumen_e2.json`.

## Cadenas

- **C4:** `(drone OR UAV OR "unmanned aerial" OR FANET OR "Internet of Drones") AND ("intrusion detection" OR "attack detection" OR "anomaly detection") AND (GRU OR "gated recurrent" OR BiLSTM OR "bidirectional LSTM" OR transformer OR TabNet OR "Kolmogorov-Arnold" OR autoencoder OR "deep belief" OR ResNet OR residual OR "CNN-LSTM" OR "graph neural" OR "temporal convolutional" OR mamba OR "state space")`
- **T1:** `("supervised contrastive" OR "contrastive learning") AND (image OR "computer vision" OR visual) AND (transfer OR adaptation)`
- **T2:** `("mixture of experts" OR "gated experts") AND ("natural language" OR "language model" OR vision) AND (transfer OR adaptation)`
- **T3:** `("domain generalization" OR "invariant risk minimization" OR "distributionally robust") AND (image OR vision OR speech) AND (transfer OR adaptation)`
- **T4:** `("Kolmogorov-Arnold" OR "feature tokenizer" OR "tabular transformer") AND ("time series" OR vision OR image OR language) AND (transfer OR adaptation)`

## Qué se corre en las bases con acceso institucional

Solo las cadenas transversales T1 a T4, en Scopus y en Web of Science, que cumplen el mínimo de dos bases principales del enunciado:

- **C1 a C3** no se repiten, porque ya se corrieron en IEEE Xplore, Scopus y Web of Science el 4 de noviembre de 2026 (Entregable 1).
- **C4** se queda en OpenAlex y arXiv: es una cadena complementaria que no exige el enunciado, y aportó 87 de los 94 registros nuevos, pero ninguno de sus estudios elegibles tuvo texto completo accesible, de modo que no modificó los estudios incluidos.
- **IEEE Xplore** no se usó para T1 a T4: el primer intento devolvió un error del servidor, y el mínimo ya se cumplía con Scopus y Web of Science.

## Scopus

*Advanced document search*, cada cadena por separado: `TITLE-ABS-KEY(` cadena `) AND PUBYEAR > 2017 AND PUBYEAR < 2027`.

## Web of Science

*Advanced Search*, colección principal, cada cadena por separado: `TS=(` cadena `) AND PY=(2018-2026)`.

## Conteos

Con el acceso de la biblioteca de la Universidad del Norte; se anotó el número de resultados que muestra cada base, sin exportar registros. También están en `exportaciones/2026-11-13_conteos_scopus_wos_e2.json`.

| Base | T1 | T2 | T3 | T4 | Fecha |
|---|---|---|---|---|---|
| Scopus | 4 046 | 334 | 1 181 | 95 | 13 de noviembre de 2026 |
| Web of Science | 1 023 | 99 | 631 | 86 | 13 de noviembre de 2026 |
