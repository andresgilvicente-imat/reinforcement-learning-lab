# Registro de cambios respecto al borrador original

El archivo `report_original.tex` permite inspeccionar el diff exacto.
Se mantienen intactas las 8 figuras, 4 tablas y la estructura de 6 apartados.

## Correcciones académicas
- Se evita llamar GLIE, sin justificación de cobertura, al calendario
  `epsilon_k = 0.2 / (1 + k/1000)`. Se conserva el experimento y se
  distingue su resultado empírico de un teorema de convergencia.
- Se explicita por qué Q-learning puede aproximarse al valor óptimo incluso
  con alpha constante en el MDP *determinista y acíclico*, si hay infinitas
  visitas; se evita extrapolar la propiedad a entornos estocásticos.
- Se matiza que SARSA con alpha alto puede alterar su política y no solo
  oscilar alrededor del punto fijo teórico.
- Se distingue una aproximación numérica en 20 000 episodios de
  convergencia exacta en todos los pares.
- Se limita temporalmente la trampa de Q0=-5, epsilon bajo a la ventana
  realmente estudiada, sin afirmar literalmente «para siempre».
- Se corrige la afirmación de que ambas Q de s10 alcanzan 3.5 exacto:
  coinciden entre sí por redondeo (3.4999999999999956 en la semilla 0),
  lo que desencadena el desempate, sin alcanzar exactamente 3.5.

## Precision experimental
- La estadística de evaluaciones subóptimas (11 %) se atribuye
  explícitamente a la semilla 0 del ejemplo de alpha=0.5.
- Se explicita que la dificultad de Q(s6,arriba-derecha) depende de
  las visitas *y* de la propagación previa de los valores sucesores.
- Se evita afirmar que los valores se detienen de mejorar simultáneamente
  para ambos algoritmos a partir del episodio 10 000.
- La leyenda de eficiencia muestral distingue diez semillas por barrido
  de veinte semillas en la configuración basal.

## Presentación y trazabilidad
- Se mantiene portada, títulos, numeración, figuras, tablas, ecuaciones
  y referencias cruzadas. Se cambia una fecha de entrega sin día conocido
  por la fecha comprobable de esta revisión (26/10/2026).
- Se actualiza la declaración de apoyo de IA para reflejar el trabajo
  de Claude y ChatGPT sin atribuir validaciones no ejecutadas.
- Se incluye un comprobador matemático independiente, código fuente de
  lectura, evidencias resumidas y guía de compilación.
- Compilación final: 13 páginas, sin referencias indefinidas ni
  overfull boxes. Recuento conservador 3760 / 4000 palabras.

## Cambios NO realizados
- Ningún cambio a `sarsa.py`, `q_learning.py` o `env.py`.
- Ningún reentrenamiento ni sustitución de datos, gráficos o
  valores tabulados.
- No se han añadido teorías o estados ajenos al enunciado.
