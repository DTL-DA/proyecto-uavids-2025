# Adenda del preregistro del Entregable 2

Esta adenda registra, con fecha, las cifras que el preregistro dejó en función del piloto y de la búsqueda, y toda desviación del protocolo. Ninguna se decidió después de ver resultados de prueba.

## Desviación 1 · 12 de noviembre de 2026, 13:58

**Qué cambia.** La fase 1 de la búsqueda de arquitectura (sección 10 del preregistro) y el estudio de ruido por semilla que fija k entrenan con un subconjunto fijo del 25 % de los bloques de entrenamiento, estratificado por clase (20 bloques, 19 351 flujos, `resultados/particiones/fase1_25_bloques.npz`, semilla 5000), en lugar de con todo el entrenamiento. Se mantienen las 1000 arquitecturas, las 6 épocas, la semilla, la parada temprana, la estandarización con los estadísticos de todo el entrenamiento y la validación completa. La fase 2, la ejecución definitiva y los demás experimentos usan el entrenamiento completo.

**Por qué.** Las primeras corridas del ajuste midieron el costo real con varios procesos en paralelo: 39 minutos por BiLSTM con 15 épocas, más del doble de lo estimado en el preregistro, que se calculó con un solo proceso. Con esas velocidades, el protocolo completo tomaría unas 40 horas de cómputo continuo, y la fase 1 de la CNN-LSTM sola, unas 20. El autor decidió reducir la fase 1, como admite la sección 14.4 del enunciado si la desviación se declara.

**Efecto potencial sobre las conclusiones.** Con menos datos, el F1 de la fase 1 es más ruidoso y puede favorecer arquitecturas pequeñas, que convergen antes con pocos ejemplos, de modo que el mejor verdadero tiene más probabilidad de quedar fuera de los finalistas. Para medirlo, el estudio de ruido usa el mismo subconjunto, la regla de k no cambia y se reporta la correlación de rangos entre la fase 1 y la fase 2. La conclusión sobre cada modelo con búsqueda se limita a su mejor finalista, como ya declaraba el preregistro.

**Fijado antes de:** el estudio de ruido, la fase 1 y todo lo que sigue.
