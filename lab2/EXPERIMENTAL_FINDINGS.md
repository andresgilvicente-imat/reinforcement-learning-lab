# Hallazgos experimentales — Práctica 2 (material interno)

Documento de trabajo, **no** forma parte del informe. Resume qué se ha hecho, qué se encontró, qué entró en `report/report.pdf` y qué se quedó fuera.

## 1. Estructura y reproducción

```text
lab2/
├── src/                      código de la práctica (sin modificar fuera de START/END CODE HERE)
├── experiments/
│   ├── common.py             referencias teóricas (q*, q_eps, alcanzabilidad) + runner instrumentado
│   ├── loaders.py            carga de resultados
│   ├── validate_algorithms.py  32 comprobaciones (todas PASS)
│   ├── run_experiments.py    400 entrenamientos (≈2.5 min con 14 procesos)
│   ├── analyze_results.py    figuras, tablas LaTeX y cifras (results/summaries/numbers.json)
│   ├── exploration.ipynb     inspección de resultados
│   └── analisis_hiperparametros.ipynb + 0X_*.png   trabajo previo (conservado, no usado)
├── results/
│   ├── configurations/experiments.json   configuraciones y semillas
│   ├── raw/<grupo>/<config>/<algo>_s<semilla>.npz   (73 MB)
│   └── summaries/            CSV/JSON con todas las cifras citadas + validation.txt
├── figures/{baseline,states,hyperparameters,comparisons}/   PDF + PNG
└── report/                   report.tex, report.pdf, tables/, figures/, build.py
```

Reproducir todo (desde `lab2/experiments`, con el `.venv` del repositorio):

```bash
python validate_algorithms.py      # 32 PASS
python run_experiments.py          # o: python run_experiments.py base alpha
python analyze_results.py
cd ../report && python build.py    # compila en un directorio temporal y cuenta palabras
```

`build.py` compila fuera de la carpeta sincronizada porque Synology/OneDrive bloqueó `report.pdf` durante un `pdflatex` in situ (quedaron `report.aux/.log/.out/.toc` bloqueados en `report/`, se pueden borrar).

**Recuento final de palabras:** 3 570 (conservador) = texcount 3 494 (texto 2 731 + títulos 38 + leyendas 282 + 438 fórmulas en línea contadas como palabras + 5 ecuaciones) + 76 palabras en tablas. Límite: 4 000. Los listados de código y las cifras de las tablas no se cuentan.

## 2. Validación (validate_algorithms.py)

- `epsilon_greedy`: P(greedy) = 1 − ε/2 (0.8993 con ε = 0.2); empate → acción 0 (↗).
- Entorno de juguete con azar guionizado: SARSA usa Q(s′,a′) de la a′ ejecutada; Q-learning usa max; transición terminal sin bootstrap; Q(T,·) nunca se toca.
- El runner instrumentado reproduce **bit a bit** `sarsa()`/`q_learning()` (6 configuraciones).
- ε = 0 ⇒ SARSA ≡ Q-learning exactamente (mismo consumo del rng, s ≠ s′ siempre).
- N episodios = prefijo de M > N con la misma semilla (el «número de episodios» solo decide dónde se para).
- s2, s16, s17 sin visitas y con Q = Q0; q* = V de PI/VI de la P1.

## 3. Configuraciones

γ = 0.9, r = (−1, −5, +5), s_init uniforme en {s0, s4, s8, s12}, 20 000 episodios en todas.

| Grupo | Configuraciones | Semillas |
|---|---|---|
| base | α = 0.1, ε = 0.2, Q0 = 0 | 0–19 |
| alpha | α ∈ {0.01, 0.05, 0.5, 1.0} | 0–9 |
| epsilon | ε ∈ {0, 0.05, 0.1, 0.5} | 0–9 |
| q_init | Q0 ∈ {−5, +5} | 0–9 |
| inter | (α, ε) = (0.5, 0.5), (0.05, 0.5); (ε, Q0) = (0, −5), (0, +5), (0.05, +5), (0.05, −5) | 0–9 |
| ext | GLIE ε_k = 0.2/(1+k/1000); GLIE + α_k = 0.1(1+k/1000)^−0.6 | 0–9 |

Ambos algoritmos usan las mismas semillas (comparaciones pareadas).

## 4. Hallazgos principales (todos en el informe)

1. **Estructura:** s2, s16, s17 inalcanzables (sus Q quedan en Q0); s1, s15, s18, s19 sin ruta al objetivo; empates de misma transición en s1, s3, s7, s15, s19 (y s16); empate de rutas distintas en s10 (3.5 = 3.5).
2. **SARSA estima q_ε (punto fijo de T_ε), no q*.** Q(s13,↗): predicción 1.2995, medido 1.302 (20 semillas). Q(s13,↘): −5.586 vs −5.583. s6 no distingue los algoritmos (sin bifurcación aguas abajo).
3. **Ruido de SARSA = AR(1):** σ = sqrt(α σ_X²/(2−α)), con σ_X = 2.55. Teórica/empírica: 0.18/0.17 (α = 0.01), 0.41/0.41 (0.05), 0.59/0.58 (0.1). Falla con α ≥ 0.5.
4. **Colapso de SARSA con α grande:** α = 0.5, semilla 0, ep. 1 444. Dos exploraciones en s11 hunden Q(s13,↗); s9, s4 y s12 pasan a chocar; s13 casi no se visita durante ~4 800 episodios. 11 % de evaluaciones subóptimas. Con α = 1 todos los s_init chocan (retorno greedy −4.9). Q-learning con α = 1 es el más rápido (política estable en 75 episodios).
5. **Q-learning → q* exacto** en pares visitados; error ∝ (1−α)^N hasta 1e−16 (fig_visits_error). Residuo ≤ 0.08 en el par menos visitado de s3.
6. **El hallazgo clave de la comparación: s10.**
   - Q-learning sigue la ruta s14 (↘) el 67 % del tiempo entre los episodios 500 y 10k; SARSA, el 6 %.
   - Explorar en s14 cuesta 11 y en s6 cuesta 2. Retorno exacto de la conducta ε-greedy: −0.741 vs −1.033.
   - Ventaja de SARSA medida: +0.18 [0.12, 0.24], p < 0.001; predicha: 0.67 × 0.29 ≈ 0.20.
   - Causa del sesgo de Q-learning: con Q0 = 0, el desempate de argmax hacia ↗ es la acción-objetivo en s14 pero el rodeo en s6. La preferencia se autorrefuerza.
   - Desaparece hacia el episodio 9 300 por un **artefacto de coma flotante**: la acción no greedy llega a 3.5 exacto tras ~390 visitas y argmax desempata a ↗.
   - Con ε = 0.05 el empate no llega y la ventaja de SARSA persiste (+0.08, p = 0.004).
7. **Fase inicial:** Q-learning algo mejor en los episodios 0–100 (−0.19, p = 0.008); la diferencia greedy (episodios 20–200) no es significativa (p = 0.08).
8. **ε:** el retorno de entrenamiento sigue el valor exacto ε-greedy de π*; el greedy es siempre óptimo. Q(s13,↗) de SARSA sigue la recta q_ε(ε). Q-learning necesita exploración para cubrir pares no greedy (ε = 0.05 → Q(s6,↗) = 0.08 tras 20k episodios).
9. **Q0:** Q0 = 0 ya es optimista para 19 de 34 pares. Q0 = −5 retrasa (5 790 / 2 840 episodios). ε ∈ {0, 0.05} con Q0 = −5 es una trampa permanente (éxito ≈ 0 %). ε = 0 con Q0 = +5 aprende la política en 130 episodios, pero deja 3 pares sin visitar y acciones erróneas fuera de la trayectoria.
10. **La política se estabiliza mucho antes que los valores:** ~1 500 episodios frente a ~10 000 (Q-learning). SARSA nunca se detiene.
11. **GLIE:** Q(s13,↗) de SARSA → 2.10 ± 0.15 (q* = 2.15).

## 5. Correcciones a nuestras notas («respuestas y observaciones lab2 RL.pdf»)

- s6 ↘: la actualización es Q ← Q + **α**(5 − Q), no γ.
- s13 ↘ en SARSA converge a **−5.59** (q_ε), no a −5.5. El −6.355 que calculamos es el target de la rama exploratoria, correcto como retorno de esa trayectoria.
- s13 ↗ en SARSA no «intenta pegarse a 2.15»: oscila alrededor de **1.30**. Las caídas son muestras del target −6.355.
- «Q-learning converge a q*»: solo en pares visitados y con visitas suficientes (s3, s6↗ con ε pequeño; s2/s16/s17 nunca).
- El notebook previo (`analisis_hiperparametros.ipynb`) promediaba SARSA y Q-learning juntos en las gráficas de barrido, usaba éxitos acumulados (que crecen siempre) y el eje x de la primera semilla. Se conserva pero no se usa.

## 6. Hallazgos que no entraron por espacio

- Con r_obs hasta −50 y ε hasta 0.5, el argmax de q_ε solo difiere del de q* en s10: no hay un caso tipo *cliff walking* con políticas estrictamente distintas en este entorno.
- Q-learning con Q0 = +5 sigue la ruta s14 solo el 6 % del tiempo: el sesgo de s10 depende de la inicialización (con optimismo, la acción más visitada baja hacia 3.5 y deja de ser greedy).
- La media temporal de SARSA en s10↘ (2.32 frente a q_ε = 2.56) está sesgada a la baja: el par se actualiza más cuando es alto (es greedy), así que pasa más tiempo bajo. Es un sesgo de muestreo dependiente del estado.
- Las acciones de los estados con empate de misma transición (s1, s15, s19) alternan hasta el final del entrenamiento.
- GLIE + α decreciente: σ(s13,↗) = 0.055, pero los pares poco visitados aprenden más lento (Q(s6,↗) = 1.72) y la política tarda ~17 800 episodios.
- Un `main.py --episodes 20000` (semilla 0) termina con Q(s13,↗) de SARSA = −1.04: es una muestra instantánea tras una caída, no la media (1.30).

## 7. Limitaciones pendientes

- Un solo entorno determinista: no se observa el sesgo de maximización de Q-learning (requiere targets ruidosos). El entorno estocástico de la P1 sería la extensión natural.
- 10 semillas en los barridos (20 en la basal); los IC de las configuraciones secundarias son más anchos.
- La evaluación greedy se hace cada 20 episodios: resolución de ~80 pasos en las métricas de eficiencia.
- La fecha de entrega en la portada está como «octubre de 2026»: ajustarla.
- `.gitignore` ignora `*.png`: las figuras PDF sí se versionarían, las PNG no. `results/raw` ocupa 73 MB (valorar no versionarlo).
