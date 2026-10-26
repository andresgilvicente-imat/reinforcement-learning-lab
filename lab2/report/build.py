"""Compila report.tex y cuenta palabras.

Uso (desde lab2/report):  python build.py [--count-only]

La compilación se hace en un directorio temporal y el PDF se copia al final:
la carpeta sincronizada (Synology/OneDrive) puede bloquear report.pdf mientras
pdflatex lo escribe.

Recuento conservador = texcount (texto + títulos + leyendas + fórmulas en
línea) + palabras con letras dentro de las tablas (texcount no cuenta las
celdas; los números de las tablas se excluyen). Los listados de código no se
cuentan.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT = 4000


def table_words():
    total = 0
    for f in sorted((HERE / "tables").glob("*.tex")):
        txt = f.read_text(encoding="utf-8")
        txt = re.sub(r"\$[^$]*\$", " ", txt)              # fórmulas de cabecera
        txt = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})*", " ", txt)  # comandos
        words = [w for w in re.split(r"[\s&/(),\[\]{}]+", txt) if re.search(r"[A-Za-zÁÉÍÓÚáéíóúñ]", w)]
        total += len(words)
    return total


def count():
    out = subprocess.run(["texcount", "-sum", "-inc", "-utf8", "report.tex"], cwd=HERE,
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    m = re.search(r"Sum count: (\d+)", out)
    tc = int(m.group(1)) if m else -1
    tw = table_words()
    for line in out.splitlines():
        if any(k in line for k in ("Words in text", "Words in headers", "outside text", "math inlines", "math displayed")):
            print("  " + line.strip())
    total = tc + tw
    print(f"texcount (suma): {tc} | palabras en tablas: {tw} | TOTAL conservador: {total} / {LIMIT}")
    return total


def build():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for item in ("report.tex", "tables", "figures"):
            src = HERE / item
            (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp / item)
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                               cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=600)
        log = (tmp / "report.log").read_text(encoding="utf-8", errors="replace")
        (HERE / "build.log").write_text(log, encoding="utf-8")
        issues = [l for l in log.splitlines() if re.search(r"^!|Overfull|undefined", l)]
        print("\n".join(issues[:20]) or "Sin errores, referencias indefinidas ni cajas desbordadas.")
        if r.returncode != 0 or not (tmp / "report.pdf").exists():
            sys.exit("Fallo de compilación")
        shutil.copy(tmp / "report.pdf", HERE / "report.pdf")
        pages = re.search(r"Output written on report.pdf \((\d+) pages", log)
        print(f"report.pdf generado ({pages.group(1) if pages else '?'} páginas)")


if __name__ == "__main__":
    if "--count-only" not in sys.argv:
        build()
    total = count()
    sys.exit(0 if total <= LIMIT else 2)
