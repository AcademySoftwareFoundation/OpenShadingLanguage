#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# This script, which assumes it is runnign on a Mac OSX with Homebrew
# installed, does a "brew install" in all packages reasonably needed by
# OSL and its dependencies.

if [[ `uname` != "Darwin" ]] ; then
    echo "Don't run this script unless you are on Mac OSX"
    exit 1
fi

if [[ `which brew` == "" ]] ; then
    echo "You need to install Homebrew before running this script."
    echo "See http://brew.sh"
    exit 1
fi


#brew update >/dev/null
echo ""
echo "Before my brew installs:"
brew list --versions

brew install --display-times -q gcc ccache cmake ninja boost || true
brew link --overwrite gcc
brew unlink python@2.7 || true
brew unlink python@3.9 || true
brew unlink python@3.8 || true
brew link --overwrite --force python@${PYTHON_VERSION} || true
brew upgrade --display-times -q cmake || true
brew install --display-times -q libtiff ilmbase openexr
brew install --display-times -q libpng giflib webp jpeg-turbo freetype
brew install --display-times -q opencolorio partio pugixml
brew install --display-times -q pybind11 numpy || true
brew install --display-times -q tbb || true
brew install --display-times -q flex bison
brew install --display-times -q llvm${LLVMBREWVER}
brew install --display-times -q qt

echo ""
echo "After brew installs:"
brew list --versions

# Needed on some systems
if [[ $PYTHON_VERSION != "2.7" ]] ; then
    pip3 install numpy
else
    pip install numpy
fi

# Set up paths. These will only affect the caller if this script is
# run with 'source' rather than in a separate shell.
export PATH=/usr/local/opt/qt5/bin:$PATH
export PATH=/usr/local/opt/python/libexec/bin:$PATH
export PYTHONPATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH
export PATH=/usr/local/opt/llvm${LLVMBREWVER}/bin:$PATH
export LLVM_DIRECTORY=/usr/local/opt/llvm${LLVMBREWVER}
export LLVM_ROOT=/usr/local/opt/llvm${LLVMBREWVER}
export PATH=/usr/local/opt/flex/bin:/usr/local/opt/bison/bin:$PATH

# Save the env for use by other stages
src/build-scripts/save-env.bash
