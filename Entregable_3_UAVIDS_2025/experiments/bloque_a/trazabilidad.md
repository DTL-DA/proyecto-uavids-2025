# Bloque A: correcciones del Entregable 2

El docente no hizo observaciones escritas al E2 (D1, 18/11/2026). La tabla se construyó con una **autoauditoría** del E2 frente a su enunciado y frente a los criterios de este (sección 4). Si llegan observaciones del docente, se agregan como filas nuevas. Una observación sin fila se considera no corregida.

## Tabla de trazabilidad

| N.° | Observación (autoauditoría) | Acción realizada | Ubicación en el Jupyter Book | Evidencia | Impacto en los resultados |
|---|---|---|---|---|---|
| A1 | La partición por bloques de 1000 flujos se declaró en el E1 como aproximación de la serie. No se había verificado si los **escenarios de simulación** cruzan particiones (dependencia L3.2 de Kapoor y Narayanan, 2023) | Auditoría de fuga ampliada: bloques compartidos, duplicados, preprocesamiento, atributos ilegítimos y tramos de escenario que cruzan particiones; error según la vecindad en filas y en atributos; experimento natural con los escenarios de Sybil aislados | Cap. 1, sección "Auditoría de fuga" | `bloque_a/auditoria_particion.py` y `auditoria_particion/*.csv` | No hay fuga L1 ni L2 (0 bloques compartidos; identificadores excluidos). Sí hay dependencia L3.2: el 70,5 % de las filas de prueba pertenecen a un escenario con filas en entrenamiento. No se cambia la partición congelada, por comparabilidad entre E1, E2 y E3; la dependencia se reporta y se mide su efecto (A2) |
| A2 | Falta el **ranking antes y después** de controlar la dependencia | F1 de los 15 modelos con y sin los segmentos de los dos escenarios de Sybil aislados, sobre las predicciones guardadas | Cap. 1, tabla "Antes y después" | `auditoria_particion/f1_sin_escenarios_sybil.csv` | **Prueba:** todos los modelos suben unos 0,04 y el orden se mantiene (LightGBM 0,875 → 0,915; CNN-LSTM 0,863 → 0,904). **Validación:** el orden cambia; LightGBM pasa de 0,887 a 0,936 y queda primero. La ventaja de las redes en validación se debía al escenario de 695 flujos. Cambio desfavorable para las redes, reportado sin ocultarlo |
| A3 | No se había medido si algún modelo **generaliza a un escenario no visto** | Prueba con partición por escenario (9 pliegues, solo filas de entrenamiento) para el DNN, LightGBM y una penalización de invariancia (V-REx condicional por clase) | Cap. 1 y cap. 5 (motivación del diseño) | `bloque_a/prueba_d3/` | Ningún modelo generaliza: el Sybil nuevo se predice como Flooding (85 a 97 %) y el Wormhole nuevo como Blackhole (82 %). LightGBM no generaliza mejor que el DNN (0,401 frente a 0,475) |
| A4 | **Determinismo** y semillas documentados solo en el código | Se documentan las semillas (Python, NumPy y PyTorch en `motor.fijar_semilla`), los hilos por proceso, la versión de PyTorch (2.6.0+cpu) y que todo corrió en CPU. Las corridas en GPU, si las hay, fijan `use_deterministic_algorithms` y se declaran | Cap. 11 (reproducibilidad) | `protocolo.md`, sección 5.1 | Sin cambios en las cifras |
| A5 | Los benchmarks no tenían un **hash de congelamiento** | SHA-256 del conjunto ordenado de las 301 corridas definitivas del E2 (JSON y probabilidades) fijado en el protocolo antes de las corridas del E3 | Cap. 3 | `protocolo.md`, sección 5.1: `a35f8535…1f05f03` | Sin cambios; los rivales no se reajustan |
| A6 | En el E2 las probabilidades y los pesos por corrida quedaron fuera del repositorio, lo que impide la reproducción de las pruebas por instancia | Se copiaron a la copia de trabajo del E3, se verificó que los registros JSON coinciden (SHA-256 idéntico) y se publicarán en la *release* del E3 con DOI de Zenodo | Cap. 11 | Hash de los JSON: `a0126570…41948f18` en ambas copias | Sin cambios |
| A7 | La familia de contrastes del E2 (RQ1) era Top 10 frente al *baseline* del E1. El E3 exige el Top 10 **y todos los *baselines* fuertes** | Familia de m = 12 (Top 10, LightGBM y CNN 1D), definida con las etiquetas del preregistro del E2 antes de ver resultados del E3 | Cap. 7 | `estandar.md` | Más contrastes y semillas (48 para CNN-LSTM, 38 para TabNet) |
| A8 | El E2 corrió la cadena C4 solo en OpenAlex y arXiv | La revisión de novedad del E3 se corre en OpenAlex, arXiv, Scopus y Web of Science, y la verifica un tercero | Cap. 4 | `novedad/matriz_novedad.md` | — |
| A9 | La limitación del paper guía (confusión Blackhole–Wormhole) quedó registrada en el E2 sin una propuesta que la atacara | El modelo original se diseña contra esa limitación, con predicciones secundarias P1 a P3 preregistradas | Cap. 5 | `protocolo.md`, sección 2 | Se verifica en el cap. 7 |

## Ranking antes y después (prueba congelada; F1 macro medio de 20 semillas)

Antes: prueba completa. Después: sin los segmentos de los escenarios de Sybil aislados.

| Modelo | Antes | Después | Posición antes | Posición después |
|---|---|---|---|---|
| LightGBM | 0,875 | 0,915 | 1 | 1 |
| CNN-LSTM | 0,863 | 0,904 | 2 | 2 |
| Autoencoder | 0,859 | 0,900 | 3 | 3 |
| CNN 1D | 0,858 | 0,899 | 4 | 5 |
| DNN | 0,858 | 0,899 | 5 | 6 |
| Transformer | 0,858 | 0,899 | 6 | 4 |
| DBN | 0,857 | 0,897 | 7 | 7 |
| ResNet | 0,851 | 0,892 | 8 | 8 |
| KAN | 0,849 | 0,889 | 9 | 10 |
| GRU | 0,847 | 0,889 | 10 | 9 |
| BiLSTM | 0,846 | 0,886 | 11 | 11 |
| RNN | 0,837 | 0,877 | 12 | 12 |
| LSTM | 0,829 | 0,869 | 13 | 13 |
| TabNet | 0,825 | 0,863 | 14 | 14 |
| Logística | 0,803 | 0,841 | 15 | 15 |
