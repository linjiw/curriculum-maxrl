LATEXMK ?= latexmk
PYTHON ?= python3

.PHONY: all check package clean

all:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main_iclr.tex

check: all
	$(PYTHON) check_sync.py
	@if rg -n 'Overfull|Undefined control sequence|undefined references|duplicate destination|Package .* Error|Emergency stop|Fatal error' main.log main_iclr.log; then \
		echo "LaTeX validation failed"; exit 1; \
	fi

package: check
	cp main.pdf ../docs/paper-draft.pdf
	cp main_iclr.pdf ../docs/paper-iclr.pdf

clean:
	$(LATEXMK) -C main.tex
	$(LATEXMK) -C main_iclr.tex
