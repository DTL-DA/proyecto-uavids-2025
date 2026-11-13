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

## IEEE Xplore

*Advanced Search* → *Command Search*; año de publicación de 2019 a 2026 (C4) o de 2018 a 2026 (T1 a T4). Exportación en CSV.

## Scopus

`TITLE-ABS-KEY(` cadena `) AND PUBYEAR > 2018 AND PUBYEAR < 2027` para C4, y `PUBYEAR > 2017` para T1 a T4. Exportación en CSV con resumen.

## Web of Science

Colección principal: `TS=(` cadena `) AND PY=(2019-2026)` para C4, y `PY=(2018-2026)` para T1 a T4. Exportación en Excel.

Las exportaciones se guardan en `exportaciones/` con el nombre `2026-11-DD_<Base>_<cadena>.<extensión>`.
