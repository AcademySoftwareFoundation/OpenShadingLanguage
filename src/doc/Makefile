PDFLATEX_FLAGS = -interaction=errorstopmode   ###  "--shell-escape"


# by default, just make the document
all: languagespec.pdf


# document making rule: use pdflatex
languagespec.pdf: *.tex languagespec.aux macros.aux Figures/*.pdf Figures/*png
	pdflatex ${PDFLATEX_FLAGS} languagespec.tex



# special command 'make index' to regenerate the index
index: languagespec_index

languagespec_index:
	pdflatex ${PDFLATEX_FLAGS} languagespec.tex
	pdflatex ${PDFLATEX_FLAGS} languagespec.tex
	makeindex languagespec.tex
	pdflatex ${PDFLATEX_FLAGS} languagespec.tex

