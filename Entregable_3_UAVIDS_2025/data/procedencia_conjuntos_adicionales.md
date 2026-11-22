# Procedencia de los conjuntos adicionales (sección 8.3 del enunciado)

Descargados el 22/11/2026, antes de congelar el protocolo y sin entrenar ningún modelo con ellos. Los datos no se versionan; este archivo sí, con la fuente, la licencia, la versión y el SHA-256 de cada archivo.

## UAV-NIDD

- **Fuente:** repositorio de GitHub `CyberSaR-KAUST/UAV-Intrusion-Detection-Dataset`, rama `main`, commit `1d6948882ef9c2a0e4708462614880656c1126bf` (25/03/2026). Los PCAP originales están en Figshare (DOI 10.6084/m9.figshare.25486462, versión 2, 21/07/2025) y no se descargaron.
- **Licencia:** CC BY 4.0, la que declara Figshare. El repositorio de GitHub no trae archivo LICENSE.
- **Cita:** Hadi, H. J., Cao, Y., Khan, M. K., Ahmad, N., Hu, Y. y Fu, C. (2025). UAV-NIDD: A Dynamic Dataset for Cybersecurity and Intrusion Detection in UAV Networks. *IEEE Transactions on Network Science and Engineering*, 12(4), 2739–2758. https://doi.org/10.1109/TNSE.2025.3553442

| Archivo | Bytes | SHA-256 |
|---|---|---|
| `UAV-Case1-Label.zip` | 22 958 880 | `8326d56543ccf9645355c7d70057b9507e5c236c1584727b513fe18c0553482c` |
| `Access Point Case2 Label.zip` | 13 043 311 | `5ba10af3d372c996c8324b84ea403c6d8b50f99e41bba3c9f07c3047c5412aa0` |
| `GCS-Case 3.zip` | 4 865 042 | `59f443c8fc9090959608745ff1784a75592e2a2398372c626c57648076484e88` |
| `README.md` | 3 496 | `9777418b8f549cc80ccbf90255b9ed234370b3315f71ed744b19105bcda311fb` |

## UNSW-NB15

- **Fuente:** carpeta oficial "UNSW-NB15 dataset/CSV Files" en el OneDrive de UNSW, enlazada desde https://research.unsw.edu.au/projects/unsw-nb15-dataset. Los archivos tienen fecha de modificación del 2/8/2024. Se bajaron como `UNSW-NB15.zip` y la prueba de integridad del ZIP pasó.
- **Licencia:** uso libre para investigación académica, a perpetuidad, con la obligación de citar los cinco artículos de la página. El uso comercial requiere acuerdo con los autores.
- **Registros:** la página declara 2 540 044, pero los cuatro CSV suman **2 540 047** (700 001 + 700 001 + 700 001 + 440 044), sin cabecera ni líneas vacías. La diferencia de 3 registros se declara en `DATA_STATEMENT.md`.
- **Atributos:** 49 con la etiqueta (`NUSW-NB15_features.csv`), incluidos `Stime` y `Ltime`, que sirven para la partición agrupada por tiempo, y las IP, que se excluyen como atributos.

| Archivo | Bytes | SHA-256 |
|---|---|---|
| `UNSW-NB15.zip` (descarga original) | 140 687 406 | `a22f4877e5ead699fb29265cb7f52590984c8ad65a7bef3deab7550b33cf5ce6` |
| `NUSW-NB15_features.csv` | 4 044 | `c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce` |
| `The UNSW-NB15 description.pdf` | 189 112 | `20784d87509db0185b5d816f01a4ea04e60be180798e2d38cabb500f6dd13999` |
| `UNSW-NB15_1.csv` | 168 979 718 | `7d851bbeabd27894ce39c8e78835c73341fc946652fb7743b9eff193b55eb511` |
| `UNSW-NB15_2.csv` | 165 221 021 | `6130ad02873cc6069ae695cf2844f2e8c2e9a9a1b7532dd82ab8f202757cacf8` |
| `UNSW-NB15_3.csv` | 154 588 103 | `ae990a96c3dfcd425ce2801aadb1727a34d5e0ae6d8215dbcdae60dedfaef640` |
| `UNSW-NB15_4.csv` | 97 588 754 | `cdf563692d51d405541dd659ddcdad9fa01f001f05fe9fc4b67f00ca12fbc96a` |

## UNR-IDD (tercer conjunto, reemplaza a UAV-CAS)

- **Por qué reemplaza a UAV-CAS:** UAV-CAS (IEEE DataPort, DOI 10.21227/zgrg-z865, CC BY) está publicado como "conjunto de datos estándar" y exige una suscripción de pago. Eso choca con el requisito de repositorio público y con la reproducción independiente. UNR-IDD era la reserva aprobada; el cambio lo decidió Donny el 22/11/2026, sin haber visto resultados en ninguno de los dos.
- **Fuente:** enlace de Google Drive (`id=1ANnT1g7NjbGGOvL1uHDHEQF_Dzx9kK3f`) publicado en la página oficial https://www.tapadhirdas.com/das-lab/datasets/unr-idd. El archivo servido es `UNR-IDD.csv`.
- **Licencia y términos:** la página permite redistribuir, republicar y replicar el conjunto "in any form", con la condición de citar el artículo de UNR-IDD (IEEE Xplore, documento 10059640).
- **Cita:** Das, T., Hamdan, O. A., Shukla, R. M., Sengupta, S. y Arslan, E. (2023). UNR-IDD: Intrusion Detection Dataset using Network Port Statistics. *IEEE Consumer Communications & Networking Conference (CCNC)*. https://ieeexplore.ieee.org/document/10059640
- **Contenido verificado:**
  - 37 411 registros y 34 columnas de estadísticos de puertos de un banco SDN (ONOS, Mininet, OpenFlow) con 12 conmutadores.
  - Etiqueta `Label` de 6 clases: PortScan 9 500, TCP-SYN 9 081, Blackhole 8 420, Diversion 5 615, Normal 3 773 y Overflow 1 022.
  - 9 columnas son constantes (contadores de errores y descartes en 0, `is_valid`, `Table ID` y `Max Size`).
  - `Switch ID`, `Port Number` y `Connection Point` son identificadores del puerto y se excluyen como atributos.

| Archivo | Bytes | SHA-256 |
|---|---|---|
| `UNR-IDD.csv` | 5 390 181 | `94c776fef49091a308bff2b5d1f925648689be6d0f8b653d7e4782a9b2a369cd` |
