# Auditoría final del informe de la Práctica 2 de RL

**Fecha de revisión:** 26 de octubre de 2026  
**Documento base:** `report.tex` aportado; copia literal preservada en `report_original.tex`.  
**Alcance:** lectura del enunciado oficial, de las dos evaluaciones de P1, del código y resultados archivados del ZIP, auditoría matemática, revisión selectiva de cada figura, edición conservadora del `.tex`, compilación y revisión del PDF resultante.

## A. Aspectos del borrador que ya estaban bien

Se han **conservado** la portada, el índice, las seis secciones principales, las tres subsecciones dedicadas a cada algoritmo, las **ocho figuras originales**, las **cuatro tablas originales** y sus datos, la numeración y referencias cruzadas.

- **Trabajo previo:** los 40 números \(q_*(s,a)\) de los veinte estados y dos acciones ya eran correctos. Se han comprobado de forma independiente, reconstruyendo las transiciones del mapa original y resolviendo Bellman para \(\gamma=0.9\), \(r_s=-1\), \(r_{obs}=-5\), \(r_{goal}=5\); todos concuerdan hasta el redondeo de la tabla.
- **Implementaciones:** SARSA usa la acción efectivamente seleccionada en \(s'\); Q-learning usa \(\max_a Q(s',a)\); ambos omiten correctamente el bootstrap terminal en los escenarios analizados. La regla \(\varepsilon\)-greedy, el desglose de pasos y retornos, el código de recopilación y los resultados son consistentes. Las implementaciones originales no se han editado.
- **Resultados relevantes:** la diferencia entre \(q_\varepsilon(s_{13},\nearrow)=1.2995\) y \(q_*(s_{13},\nearrow)=2.15\); la varianza temporal de SARSA para \(\alpha\) moderadas; la eficiencia condicionada a visitas; la ruta de \(s_{10}\); y la trampa empírica de inicialización pesimista ya están razonadas y se apoyan en datos.
- **Comparación cuantitativa:** los CSV y los NPZ respaldan la ganancia basal de SARSA en el intervalo [500, 10000): media pareada **+0.1846**, IC bootstrap 95 % **[0.1204, 0.2431]** y 15/20 semillas con diferencia positiva. El cálculo exacto del retorno sin descuento bajo la conducta \(\varepsilon\)-greedy produce **−0.7410** para la ruta \(s_6\) y **−1.0326** para la ruta \(s_{14}\).
- **Correcciones de P1 tenidas en cuenta:** el borrador ya explicaba el mecanismo de propagación, distinguía política estable de valores numéricos y comentaba curvas por fases. En P1 se penalizó no razonar mecanismos ni examinar suficientemente la evolución; no se ha trasladado de manera automática la teoría de Policy Iteration/Value Iteration al aprendizaje TD.

## B. Errores o imprecisiones corregidos

| Prioridad | Apartado | Problema y corrección aplicada |
|---|---|---|
| **P0 conceptual** | 3.3 | Se presentaba como **GLIE** el calendario \(\varepsilon_k=0.2/(1+k/1000)\). Que \(\varepsilon_k\to0\) no basta para garantizar visitas infinitas a cada par: trayectorias que precisan varias decisiones exploratorias pueden tener probabilidades sumables. Ahora se denomina **exploración decreciente** y se conserva sin cambios la observación \(Q(s_{13},\nearrow)\approx2.10\). |
| **P1 matemático** | 4.1 y 6 | Se formulaba la convergencia exacta de Q-learning demasiado categóricamente. Ahora se diferencia entre los errores finitos de las 20 000 iteraciones experimentales y la justificación asintótica, **específica del MDP determinista y acíclico**, por inducción y suponiendo visitas infinitas. |
| **P1 matemático** | 3.1 y 6 | Se aclaró que la varianza AR(1) de SARSA supone una política greedy local estable y sucesores prácticamente aprendidos. Con \(\alpha\) grande pueden cambiar las decisiones y la media, por lo que la aproximación deja de describir el proceso. |
| **P1 numérico** | 5 | Se decía que ambas entradas de Q-learning en \(s_{10}\) alcanzan «3.5 exacto». Al inspeccionar las matrices finales, **son iguales entre sí**, pero su valor en la semilla 0 es `3.4999999999999956`. Se ha corregido la frase: es un empate **entre dos números de coma flotante**, no igualdad literal con el número real \(3.5\). Se conserva correctamente la explicación del desempate de `argmax`. |
| **P1 precisión estadística** | 3.3 | El 11 % de evaluaciones subóptimas se refería a **la semilla 0** del caso \(\alpha=0.5\), no a todo el barrido. Queda identificado. |
| **P1 precisión empírica** | 3.2 y 3.3 | La lentitud de \(Q(s_6,\nearrow)\) se debe a la escasez de visitas **y** al retraso en aprender sus sucesores. Se elimina la afirmación tajante de que después de 10 000 episodios ningún valor puede seguir mejorando. |
| **P1 exceso de generalización** | 6 | La afirmación «choca para siempre» con \(Q_0=-5\), \(\varepsilon\le0.05\) se limita a los **20 000 episodios observados**. |
| **P2 editorial** | Portada, figuras, uso de IA | Se sustituye una fecha de entrega no especificada por la fecha de revisión conocida. Se matiza la cantidad de semillas en la figura de eficiencia y se actualiza la declaración de asistencia de IA. |

## C. Análisis mejorados y auditoría por secciones

**1. Introducción y metodología.** Se mantiene la definición de retorno *sin descontar* para las gráficas frente al factor \(\gamma=0.9\) usado por las actualizaciones. Se matiza la trazabilidad: 32 pruebas aparecen **documentadas como realizadas anteriormente**; no se afirma que se hayan podido ejecutar íntegramente en este entorno de revisión. Conservados: semillas pareadas, evaluación greedy independiente y 400 entrenamientos archivados.

**2. Trabajo previo.** Se verificó cada par de la tabla mediante el modelo obtenido del mapa. Los estados inalcanzables son efectivamente \(s_2,s_{16},s_{17}\), dado \(s_{init}=\{s_0,s_4,s_8,s_{12}\}\). Sin cambios en los números.

**3. SARSA.** Preservada la distinción entre el operador de política \(\varepsilon\)-greedy y la optimalidad Bellman. Se restringen afirmaciones asintóticas, la denominación GLIE y una comparación de tiempos de aprendizaje. Se conserva la explicación específica de \(s_6\), \(s_{13}\), \(\alpha\), \(\varepsilon\), episodios y \(Q_0\).

**4. Q-learning.** Añadida una justificación compacta de por qué, en el entorno elegido, un \(\alpha\) constante no obliga a fluctuaciones permanentes cuando los targets de transición son deterministas. Se precisa que el algoritmo se aproxima a \(q_*\) solo donde tiene cobertura, sin atribuir una garantía universal a valores altos de \(\alpha\).

**5. Comparación.** Conservados los cuatro criterios del profesor y las pruebas numéricas. Se corrigió el mecanismo de empate en \(s_{10}\) para describir exactamente lo que hay en los datos. La ventaja intermedia es empírica, no una propiedad universal de SARSA.

**6. Conclusiones.** Reducidas afirmaciones absolutas sobre convergencia, \(\alpha\) e inicialización. Se conserva la síntesis original.

## Revisión específica de las ocho figuras

| Figura | Comprobación y decisión |
|---|---|
| Mapa | Correspondencia de estados, objetivo, transiciones, iniciales; **conservar**. |
| Evolución \(Q\) en \(s_6,s_{13}\) | Distingue acciones, algoritmos, referencias \(q_*\), \(q_\varepsilon\) y pasos globales; **conservar**. |
| Barrido de \(\alpha\) en \(s_{13}\) | Permite diferenciar fluctuación de SARSA, target determinista y ejemplo de colapso; **conservar**. |
| Barrido de \(\varepsilon\) | Separa retorno durante entrenamiento, greedy y estimadores; **conservar**. |
| Inicialización \(Q_0\) | Contrasta exploración insuficiente y optimismo, ejes legibles; **conservar**. |
| Errores frente a visitas | Muestra el papel de la cobertura y el suelo de ruido; **conservar**, con interpretación de convergencia finita más precisa. |
| Retornos y acción en \(s_{10}\) | Fundamental para explicar la ventaja de SARSA y el empate de coma flotante; **conservar**. |
| Eficiencia | Pasos, no solo episodios, y objetivos de política vs valores; **conservar**; pie de figura precisado. |

## D. Contenido incorporado

No se añadieron experimentos ni figuras: los originales eran suficientes. Se añadieron una justificación de convergencia **condicionada** de Q-learning para el DAG, la distinción formal entre \(\varepsilon\) decreciente y GLIE, la precisión numérica del empate en \(s_{10}\), la identificación correcta de la semilla del colapso y una formulación más defendible de los límites de evidencia. Se prepararon además instrucciones de compilación, registro de cambios, un verificador independiente y copias del código y datos agregados en el ZIP.

## E. Verificación final y límites

**Verificado directamente en esta revisión:**
- Enunciado: trabajo previo completo y respuestas a SARSA, Q-learning e interrogantes comparativos; también los cuatro criterios cualitativos comunicados por el profesor.
- Reconstrucción independiente de las 40 entradas \(q_*\); además \(q_\varepsilon(s_{13},\nearrow)=1.2995\), \(q_\varepsilon(s_{13},\searrow)=-5.5855\).
- Reglas de actualización reales de SARSA y Q-learning, así como bootstrap terminal, en entorno de juguete sin Gymnasium.
- Archivos `npz` de los 20 entrenamientos basales por algoritmo: tabla teórica incluida, retorno por fase, acción final \(s_{10}\), y valores medios finales en \(s_{13}\); comprobación independiente de la diferencia basal y su IC.
- Correspondencia visual de las ocho figuras presentes en el PDF, tablas, referencias y maquetación.
- Compilación con `pdflatex` repetida; **13 páginas**, sin errores, desbordamientos ni referencias indefinidas. Control conservador final: **3760/4000 palabras**.
- Compilabilidad del ZIP autónomo y comprobación de integridad del archivo comprimido.

**No se presenta como revalidado aquí:**
- No se reejecutaron los **400 entrenamientos** porque los datos originales se conservaron completos y se pudo hacer una validación selectiva; no habría añadido evidencia imprescindible.
- Las **32 verificaciones** del historial `validation.txt` aparecen todas con `PASS`, pero no pudieron reejecutarse íntegramente: falta la biblioteca `gymnasium` en el entorno de revisión y, además, el verificador original referencia `lab1/src`, carpeta que **no estaba en el ZIP entregado**. El nuevo verificador realiza pruebas matemáticas y funcionales alternativas, pero **no sustituye** esos 32 ensayos.
- En `sarsa.py` el código solo consulta `terminated` y no `truncated`, mientras que Q-learning distingue ambos. `env.py` devuelve siempre `truncated=False`, por lo que **no afecta** al experimento evaluado; fuera de este entorno convendría manejar truncamientos de manera explícita.
- No se comprobó contra un PDF/código íntegro de P1 más allá de sus dos evaluaciones y el modelo de P2: la equivalencia de la tabla con PI/VI de P1 figura documentada en los registros originales; la tabla sí se recalculó sin depender de ellos.

**Valoración académica:** no se han identificado contradicciones pendientes en los requisitos obligatorios ni en las ocho figuras examinadas. Los límites más importantes que siguen vigentes son los de alcance experimental (un entorno determinista, diez o veinte semillas y presupuestos finitos), no errores corregibles del informe.

**Revisión humana antes de entregar:** confirmar los nombres, la fecha real de entrega si la asignatura exige reflejarla y la exactitud de la declaración de uso de IA según lo efectivamente realizado por los autores.
