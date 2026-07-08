# Compilazione

```
cd report/latex
latexmk -pdf main.tex
```

In alternativa, senza `latexmk`:

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Le figure in `res/` sono copie di `results/report_figures/`, generate da
`scripts/plot_report_figures.py` (dalla radice del repository). Per
rigenerarle e ricopiarle:

```
.venv/bin/python scripts/plot_report_figures.py
cp results/report_figures/*.pdf report/latex/res/
```
