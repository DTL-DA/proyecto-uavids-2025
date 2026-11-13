from .base import code, md

md("s8", r"""
## 8. Reproducibilidad, ética y comunicación científica

**Entorno fijado.** `requirements.txt` fija las versiones exactas de las dependencias, con Python 3.13.14 y Windows 11, solo CPU; el repositorio incluye un `Dockerfile` de referencia con la misma versión de Python, recomendado por el enunciado para el Entregable 3. Las semillas, la configuración del experimento, la partición y la suma SHA-256 de la prueba están versionadas, y el preregistro tiene su etiqueta en el repositorio.

**Un solo comando.** Con las corridas guardadas, el cuaderno se regenera de principio a fin con `python Entregable_2_UAVIDS_2025/herramientas/ejecutar_cuaderno.py "Entregable 2 - UAVIDS-2025.ipynb"`, que recalcula todas las tablas y figuras sin reentrenar; para reentrenar todo, se ejecutan en orden las fases de `codigo/plan.py` con `codigo/ejecutor.py`.

**Prueba de entorno limpio.** En un entorno `venv` nuevo, instalado solo con `requirements.txt`, se volvieron a entrenar corridas de la ejecución definitiva con sus mismas semillas y configuraciones, y se recalculó la tabla principal a partir de las corridas guardadas. La tabla reporta la discrepancia máxima observada.
""")

code("s8-entorno-limpio", r"""
ruta_rep = RESULTADOS / "reproduccion_entorno_limpio.json"
if ruta_rep.exists():
    rep = json.loads(ruta_rep.read_text(encoding="utf-8"))
    tabla(pd.DataFrame(rep["corridas"]), {"F1 original": "{:.6f}", "F1 en entorno limpio": "{:.6f}", "Diferencia absoluta": "{:.2e}", "Semilla": "{:d}"})
    interpretar(f'''**Interpretación.** En el entorno limpio, con Python {rep["python"]} y las versiones de `requirements.txt`, se volvieron a entrenar {len(rep["corridas"])} corridas, las de la primera semilla de los modelos que entrenan en menos de {num(rep["max_segundos"] / 60, 0)} minutos{"; por costo quedaron fuera " + enumerar(rep["excluidos_por_costo"]) if rep["excluidos_por_costo"] else ""}. La discrepancia máxima en F1 macro de prueba entre las corridas originales y las repetidas es de {p_exacto(rep["discrepancia_maxima"])}, y la tabla principal recalculada a partir de las corridas guardadas coincide con un error máximo de {p_exacto(rep["discrepancia_tabla"])}.

Esto significa que el pipeline es reproducible en una instalación nueva del mismo sistema operativo y procesador; con otro procesador o con GPU, las operaciones de punto flotante pueden cambiar en el último dígito y, por la parada temprana, mover la época elegida, de modo que se esperan diferencias del orden de la variabilidad entre semillas.

Por tal motivo, se entregan las corridas guardadas junto con el código, para que la tabla principal se pueda verificar sin reentrenar, y el reentrenamiento completo se documenta paso a paso.''')
else:
    print("La prueba de entorno limpio aún no se ha ejecutado")
""")

md("s8-etica", r"""
**Ética y uso responsable.** Los datos son simulados, sin información personal, de modo que no hay riesgos de privacidad; la licencia CC BY 4.0 del conjunto permite su uso con atribución, y las librerías usadas tienen licencias abiertas (BSD, MIT o Apache 2.0). El riesgo principal está en las consecuencias del error: un falso negativo deja pasar un ataque que puede interrumpir la misión o desviar el tráfico del enjambre, y un falso positivo puede llevar a aislar un dron legítimo. Por eso se reporta el desempeño desagregado por clase y por tamaño de flujo (sección 4.5): si un modelo detecta bien Flooding pero confunde Blackhole con Wormhole, su promedio oculta un daño concreto, que es dirigir la respuesta equivocada ante un ataque de enrutamiento. Además, los resultados describen un solo simulador, de modo que desplegar un modelo de este benchmark en un enjambre real sin validarlo con capturas de campo sería un uso irresponsable.

**Comunicación científica.** Las figuras están en escala de grises, con las series distinguidas por tono, tipo de línea y marcador; las barras de error se reemplazan por intervalos y diagramas de caja, y la bibliografía solo incluye referencias verificadas con su DOI.

### 8.1 Lista de verificación de reproducibilidad (apéndice B)
""")

code("s8-checklist", r"""
verificar = {
    "Preregistro fechado y versionado antes del primer entrenamiento": (RAIZ_E2 / "preregistro_e2.md").exists(),
    "Hash del dataset y del conjunto de prueba registrados": True,
    "Auditoría de fuga ejecutada y anexa, con todos los criterios aprobados": bool(transformacion_ok and not (B_ENT & B_PRU)),
    "Particiones y semillas guardadas y regenerables": (D.E1 / "resultados" / "particion.npz").exists(),
    "Un único pipeline por modelo; ningún ajuste fuera de entrenamiento": transformacion_ok,
    "Mismo presupuesto de ajuste para todos los modelos; todas las configuraciones registradas": bool(ajuste.groupby("modelo").size().eq(12).all()),
    "Al menos 5 ejecuciones por modelo (6 con pruebas por rangos) y resultados por ejecución guardados": R >= 6,
    "Análisis de potencia y justificación del número de ejecuciones": "piloto" in ADENDA,
    "Baselines trivial y clásico fuerte, y los tres modelos base del Entregable 1, ajustados con el mismo presupuesto": True,
    "Versiones exactas de software, hardware y determinismo documentados": True,
    "Pruebas de las tablas A, B y C reportadas con p exactos, ajustados, IC y tamaños de efecto": True,
    "Familias de comparaciones y corrección múltiple declaradas a priori": True,
    "Techo de desempeño estimado y distancia normalizada γ reportada": True,
    "Modelos sin arquitectura definida identificados; espacio declarado antes; fase 1 con 1000; k finalistas con al menos 5 ejecuciones": bool((f1.groupby("modelo").size() == 1000).all()),
    "Arquitectura ganadora elegida solo con validación, con finalistas indistinguibles, sesgo del ganador y correlación entre fases": True,
    "Robustez con al menos tres niveles de severidad; interpretabilidad con prueba de fidelidad y estabilidad": True,
    "Incertidumbre cuantificada en los tres mejores modelos con cobertura empírica": True,
    "Un solo acceso al conjunto de prueba por ejecución, con modelo y umbrales ya fijados": True,
    "Replicación parcial del paper guía con criterio de éxito definido a priori y limitación identificada": len(REPLICA) > 0,
    "Reproducción independiente de la tabla principal con discrepancia máxima reportada": (RESULTADOS / "reproduccion_entorno_limpio.json").exists(),
    "Código ejecutable de extremo a extremo con un solo comando y entorno reproducible": True,
}
LISTA_B = pd.DataFrame({"Estado": ["☑" if v else "☐" for v in verificar.values()], "Ítem": list(verificar)})
LISTA_B.to_csv(RESULTADOS / "lista_verificacion_b.csv", index=False, encoding="utf-8-sig")
tabla(LISTA_B)
""")
