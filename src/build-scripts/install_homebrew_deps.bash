#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# This script, which assumes it is running on a Mac OSX with Homebrew
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

# set -ex

if [[ "${DO_BREW_UPDATE:=0}" != "0" ]] ; then
    brew update >/dev/null
fi
echo ""
echo "Before my brew installs:"
brew list --versions

if [[ "$OSL_BREW_INSTALL_PACKAGES" == "" ]] ; then
    OSL_BREW_INSTALL_PACKAGES=" \
        bison \
        ccache \
        expat \
        flex \
        fmt \
        imath \
        llvm${LLVMBREWVER} \
        ninja \
        numpy \
        opencolorio \
        openexr \
        partio \
        ptex \
        pugixml \
        pybind11 \
        robin-map \
        tbb \
        "
    if [[ "${USE_OPENVDB:=1}" != "0" ]] && [[ "${INSTALL_OPENVDB:=1}" != "0" ]] ; then
        OSL_BREW_INSTALL_PACKAGES+=" openvdb"
    fi
    if [[ "${USE_QT:=1}" != "0" ]] && [[ "${INSTALL_QT:=1}" != "0" ]] ; then
        OSL_BREW_INSTALL_PACKAGES+=" qt${QT_VERSION}"
    fi
    if [[ "${EXTRA_BREW_PACKAGES}" != "" ]] ; then
        brew install --display-times -q ${EXTRA_BREW_PACKAGES}
    fi
fi
brew install --display-times -q $OSL_BREW_INSTALL_PACKAGES $OSL_BREW_EXTRA_INSTALL_PACKAGES || true

echo ""
echo "After brew installs:"
brew list --versions

# Needed on some systems
pip${PYTHON_VERSION} install numpy || true

# Set up paths. These will only affect the caller if this script is
# run with 'source' rather than in a separate shell.
export PATH=${HOMEBREW_PREFIX}/opt/qt5/bin:$PATH
export PATH=${HOMEBREW_PREFIX}/opt/python/libexec/bin:$PATH
export PYTHONPATH=${HOMEBREW_PREFIX}/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH
echo "PYTHONPATH=$PYTHONPATH"
ls ${HOMEBREW_PREFIX}/lib/python${PYTHON_VERSION}
# export PATH=${HOMEBREW_PREFIX}/opt/llvm${LLVMBREWVER}/bin:$PATH
export LLVM_DIRECTORY=${HOMEBREW_PREFIX}/opt/llvm${LLVMBREWVER}
export LLVM_ROOT=${HOMEBREW_PREFIX}/opt/llvm${LLVMBREWVER}
export PATH=$LLVM_ROOT/bin:$PATH
echo LLVM_ROOT=${LLVM_ROOT}
# ls $LLVM_ROOT
export PATH=${HOMEBREW_PREFIX}/opt/flex/bin:${HOMEBREW_PREFIX}/opt/bison/bin:$PATH

# Save the env for use by other stages
src/build-scripts/save-env.bash
