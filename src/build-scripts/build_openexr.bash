#!/usr/bin/env bash

# Utility script to download and build OpenEXR & IlmBase
#
# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


# Exit the whole script if any command fails.
set -ex

# Which OpenEXR to retrieve, how to build it
OPENEXR_REPO=${OPENEXR_REPO:=https://github.com/AcademySoftwareFoundation/openexr.git}
OPENEXR_VERSION=${OPENEXR_VERSION:=v3.1.1}

# Where to install the final results
LOCAL_DEPS_DIR=${LOCAL_DEPS_DIR:=${PWD}/ext}
OPENEXR_SOURCE_DIR=${OPENEXR_SOURCE_DIR:=${LOCAL_DEPS_DIR}/openexr}
OPENEXR_BUILD_DIR=${OPENEXR_BUILD_DIR:=${LOCAL_DEPS_DIR}/openexr-build}
OPENEXR_INSTALL_DIR=${OPENEXR_INSTALL_DIR:=${LOCAL_DEPS_DIR}/dist}
OPENEXR_BUILD_TYPE=${OPENEXR_BUILD_TYPE:=Release}
OPENEXR_CMAKE_FLAGS=${OPENEXR_CMAKE_FLAGS:=""}
OPENEXR_CXX_FLAGS=${OPENEXR_CXX_FLAGS:=""}
BASEDIR=$PWD

pwd
echo "Building OpenEXR ${OPENEXR_VERSION}"
echo "OpenEXR build dir will be: ${OPENEXR_BUILD_DIR}"
echo "OpenEXR install dir will be: ${OPENEXR_INSTALL_DIR}"
echo "OpenEXR Build type is ${OPENEXR_BUILD_TYPE}"
echo "CMAKE_PREFIX_PATH is ${CMAKE_PREFIX_PATH}"

# Clone OpenEXR project (including IlmBase) from GitHub and build
if [[ ! -e ${OPENEXR_SOURCE_DIR} ]] ; then
    echo "git clone ${OPENEXR_REPO} ${OPENEXR_SOURCE_DIR}"
    git clone ${OPENEXR_REPO} ${OPENEXR_SOURCE_DIR}
fi
mkdir -p ${OPENEXR_INSTALL_DIR} && true
mkdir -p ${OPENEXR_BUILD_DIR} && true

pushd ${OPENEXR_SOURCE_DIR}
git checkout ${OPENEXR_VERSION} --force

if [[ ${OPENEXR_VERSION} == "v2.3.0" ]] ; then
    # Simplified setup for 2.3+
    cd ${OPENEXR_BUILD_DIR}
    cmake -DCMAKE_BUILD_TYPE=${OPENEXR_BUILD_TYPE} \
            -DCMAKE_INSTALL_PREFIX="${OPENEXR_INSTALL_DIR}" \
            -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
            -DILMBASE_PACKAGE_PREFIX="${OPENEXR_INSTALL_DIR}" \
            -DOPENEXR_BUILD_UTILS=0 \
            -DOPENEXR_BUILD_TESTS=0 \
            -DOPENEXR_BUILD_PYTHON_LIBS=0 \
            -DCMAKE_CXX_FLAGS="${OPENEXR_CXX_FLAGS}" \
            ${OPENEXR_CMAKE_FLAGS} ${OPENEXR_SOURCE_DIR}
    time cmake --build . --target install --config ${OPENEXR_BUILD_TYPE}
else
    # Simplified setup for 2.4+
    cd ${OPENEXR_BUILD_DIR}
    cmake -DCMAKE_BUILD_TYPE=${OPENEXR_BUILD_TYPE} \
            -DCMAKE_INSTALL_PREFIX="${OPENEXR_INSTALL_DIR}" \
            -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
            -DILMBASE_PACKAGE_PREFIX="${OPENEXR_INSTALL_DIR}" \
            -DOPENEXR_BUILD_UTILS=0 \
            -DBUILD_TESTING=0 \
            -DPYILMBASE_ENABLE=0 \
            -DOPENEXR_VIEWERS_ENABLE=0 \
            -DINSTALL_OPENEXR_EXAMPLES=0 \
            -DOPENEXR_INSTALL_EXAMPLES=0 \
            -DCMAKE_INSTALL_LIBDIR=lib \
            -DCMAKE_CXX_FLAGS="${OPENEXR_CXX_FLAGS}" \
            ${OPENEXR_CMAKE_FLAGS} ${OPENEXR_SOURCE_DIR}
    time cmake --build . --target install --config ${OPENEXR_BUILD_TYPE}
fi

popd

#ls -R ${OPENEXR_INSTALL_DIR}

# Set up paths. These will only affect the caller if this script is
# run with 'source' rather than in a separate shell.
export ILMBASE_ROOT=$OPENEXR_INSTALL_DIR
export OPENEXR_ROOT=$OPENEXR_INSTALL_DIR
export ILMBASE_LIBRARY_DIR=$OPENEXR_INSTALL_DIR/lib
export OPENEXR_LIBRARY_DIR=$OPENEXR_INSTALL_DIR/lib
export LD_LIBRARY_PATH=$OPENEXR_ROOT/lib:$LD_LIBRARY_PATH

