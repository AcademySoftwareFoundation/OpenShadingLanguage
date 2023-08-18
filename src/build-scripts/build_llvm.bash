#!/usr/bin/env bash

# Utility script to download and build LLVM & clang
#
# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Exit the whole script if any command fails.
set -ex

echo "Building LLVM"
uname


if [[ `uname` == "Linux" ]] ; then
    : ${LLVM_VERSION:=14.0.0}
    : ${LLVM_INSTALL_DIR:=${PWD}/llvm-install}
    : ${LLVM_DISTRO_NAME:=ubuntu-18.04}
    LLVMTAR=clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-${LLVM_DISTRO_NAME}.tar.xz
    echo LLVMTAR = $LLVMTAR
    if [[ "$LLVM_VERSION" == "9.0.0" ]] || [[ "$LLVM_VERSION" == "9.0.1" ]] ;
    then
        # old -- get rid of this when LLVM 10 is the newest we allow
        curl --location http://releases.llvm.org/${LLVM_VERSION}/${LLVMTAR} -o $LLVMTAR
    else
        # new
        curl --location https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${LLVMTAR} -o $LLVMTAR
    fi
    ls -l $LLVMTAR
    tar xf $LLVMTAR
    rm -f $LLVMTAR
    echo "Installed ${LLVM_VERSION} in ${LLVM_INSTALL_DIR}"
    mkdir -p $LLVM_INSTALL_DIR && true
    mv clang+llvm*/* $LLVM_INSTALL_DIR
    export LLVM_DIRECTORY=$LLVM_INSTALL_DIR
    export PATH=${LLVM_INSTALL_DIR}/bin:$PATH
    # ls -a $LLVM_DIRECTORY
fi
