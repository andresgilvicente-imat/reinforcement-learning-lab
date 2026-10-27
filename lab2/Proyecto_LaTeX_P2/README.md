# Proyecto LaTeX revisado — Práctica 2 de Aprendizaje por Refuerzo

Proyecto autónomo de **compilación del informe**: no necesita rutas absolutas,
carpetas sincronizadas ni archivos exteriores al ZIP.

## Contenido

- `report.tex`: documento revisado (archivo principal).
- `report_original.tex`: copia literal del borrador entregado.
- `tables/`: cuatro tablas LaTeX originales.
- `figures/`: ocho figuras vectoriales PDF originales.
- `build.py`: compila dos veces con pdflatex en un directorio temporal y cuenta palabras.
- `CHANGELOG.md`: registro de modificaciones.
- `codigo_original/`: copia de lectura del código Python analizado, sin modificaciones.
- `evidencias/`: resúmenes CSV/JSON y registro histórico de 32 validaciones.
- `verificacion_independiente.py`: comprueba 40 q*, algunos valores q_epsilon,
  las actualizaciones reales SARSA/Q-learning sobre un entorno mínimo y cifras base.
  No requiere Gymnasium ni reentrena agentes.

## Compilar

Requisitos: distribución LaTeX con `pdflatex` y los paquetes indicados en `report.tex`;
Python 3.10-3.13, y `texcount` para el control de extensión (opción 1).

**Opción 1** (recomendada, terminal en la carpeta raíz del ZIP):

    python build.py

**Opción 2** (compilación sin Python):

    pdflatex -interaction=nonstopmode -halt-on-error report.tex
    pdflatex -interaction=nonstopmode -halt-on-error report.tex

Ambas emplean solo las rutas relativas a `figures/` y `tables/`.

Validación matemática alternativa (requiere únicamente numpy además de Python):

    python verificacion_independiente.py

## Alcance de la reproducción experimental

Las figuras del ZIP son las del proyecto original, **no se han generado
nuevos entrenamientos**. Se adjuntan sus resúmenes y las configuraciones.
La reproducción íntegra de los **400 entrenamientos originales** requiere
además los archivos brutos `.npz` del ZIP `lab2(1).zip` aportado por los
autores, y las dependencias `numpy`, `gymnasium` y `matplotlib`.
El validador histórico original necesita también código de `lab1/src` para
dos comparaciones PI/VI, código que no figura en `lab2(1).zip`.
Por eso `verificacion_independiente.py` verifica esos valores directamente
con las transiciones definidas en `env.py`, sin depender de `lab1`.

## Extensión

Último control `texcount` conservador: **3760 palabras** (incluye palabras
en tablas, títulos, pies de figura y ecuaciones contadas de forma conservadora).
Límite declarado: 4000 palabras.

Fecha de revisión editorial: 26 de octubre de 2026.
