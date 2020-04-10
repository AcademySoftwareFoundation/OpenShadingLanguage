#!/bin/bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# This script, which assumes it is runnign on a Mac OSX with Homebrew
# installed, does a "brew install" in all packages reasonably needed by
# OIIO.

if [[ `uname` != "Darwin" ]] ; then
    echo "Don't run this script unless you are on Mac OSX"
    exit 1
fi

if [[ `which brew` == "" ]] ; then
    echo "You need to install Homebrew before running this script."
    echo "See http://brew.sh"
    exit 1
fi


brew update >/dev/null
echo ""
echo "Before my brew installs:"
brew list --versions

brew install --display-times gcc ccache ninja boost python pybind11 && true
brew link --overwrite gcc
brew upgrade --display-times python cmake && true
brew link --overwrite python
brew install --display-times flex bison
brew install --display-times libtiff ilmbase openexr
brew install --display-times opencolorio partio
brew install --display-times freetype libpng
brew install --display-times llvm${LLVMBREWVER}
brew install --display-times qt

echo ""
echo "After brew installs:"
brew list --versions

# Needed on some systems
pip install numpy

# Set up paths. These will only affect the caller if this script is
# run with 'source' rather than in a separate shell.
export PATH=/usr/local/opt/qt5/bin:$PATH ;
export PATH=/usr/local/opt/python/libexec/bin:$PATH ;
export PYTHONPATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH ;
export PATH=/usr/local/Cellar/llvm/9.0.0*/bin:$PATH ;
