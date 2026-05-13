# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


# by default, make the sphinx documentation
all: doxygen sphinx


clean: doxygenclean

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
