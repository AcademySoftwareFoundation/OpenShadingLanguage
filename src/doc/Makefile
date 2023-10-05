# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

PDFLATEX_FLAGS = -interaction=errorstopmode   ###  "--shell-escape"


# by default, just make the document
all: languagespec.pdf


# document making rule: use pdflatex
languagespec.pdf: *.tex Figures/*.pdf Figures/*png
	pdflatex ${PDFLATEX_FLAGS} languagespec.tex



# special command 'make index' to regenerate the index
index: languagespec_index

languagespec_index:
	pdflatex ${PDFLATEX_FLAGS} languagespec.tex
	makeindex languagespec
	pdflatex ${PDFLATEX_FLAGS} languagespec.tex languagespec.aux macros.aux
	pdflatex ${PDFLATEX_FLAGS} languagespec.tex languagespec.aux macros.aux languagespec.toc

clean:
	rm -f languagespec.aux languagespec.idx languagespec.ilg languagespec.ind languagespec.log
	rm -f languagespec.out languagespec.pdf languagespec.toc macros.aux

.PHONY: clean


doxygen:
	echo "make doxygen TBD"
	doxygen Doxyfile
	touch doxygen.turd

doxygen.turd: doxygen

doxygenclean:
	cmake -E remove_directory ../../build/doxygen
	cmake -E remove doxygen.turd
	cmake -E remove_directory ../../build/sphinx


# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
SPHINXBUILDDIR      = ../../build/sphinx


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
sphinx : Makefile conf.py
	# @$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(SPHINXBUILDDIR)" $(SPHINXOPTS) $(O)
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(SPHINXBUILDDIR)" $(SPHINXOPTS)

sphinxpdf : Makefile conf.py
	# @$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(SPHINXBUILDDIR)" $(SPHINXOPTS) $(O)
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(SPHINXBUILDDIR)/latex" $(SPHINXOPTS)
	$(MAKE) -C $(SPHINXBUILDDIR)/latex all-pdf
# 	$(SPHINXBUILD) -b latexpdf "$(SOURCEDIR)" "$(SPHINXBUILDDIR)/latex" $(SPHINXOPTS)
