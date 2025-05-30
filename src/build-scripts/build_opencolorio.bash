#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Utility script to download and build OpenColorIO

# Exit the whole script if any command fails.
set -ex

# Which OCIO to retrieve, how to build it
OPENCOLORIO_REPO=${OPENCOLORIO_REPO:=https://github.com/AcademySoftwareFoundation/OpenColorIO.git}
OPENCOLORIO_VERSION=${OPENCOLORIO_VERSION:=v2.3.2}

# Where to install the final results
LOCAL_DEPS_DIR=${LOCAL_DEPS_DIR:=${PWD}/ext}
OPENCOLORIO_SOURCE_DIR=${OPENCOLORIO_SOURCE_DIR:=${LOCAL_DEPS_DIR}/OpenColorIO}
OPENCOLORIO_BUILD_DIR=${OPENCOLORIO_BUILD_DIR:=${LOCAL_DEPS_DIR}/OpenColorIO-build}
OPENCOLORIO_INSTALL_DIR=${OPENCOLORIO_INSTALL_DIR:=${LOCAL_DEPS_DIR}/dist}
OPENCOLORIO_BUILD_TYPE=${OPENCOLORIO_BUILD_TYPE:=Release}
# Just need libs:
OPENCOLORIO_CMAKE_FLAGS+=" -DOCIO_BUILD_APPS=OFF -DOCIO_BUILD_NUKE=OFF \
                       -DOCIO_BUILD_DOCS=OFF -DOCIO_BUILD_TESTS=OFF \
                       -DOCIO_BUILD_GPU_TESTS=OFF \
                       -DOCIO_BUILD_PYTHON=OFF -DOCIO_BUILD_PYGLUE=OFF \
                       -DOCIO_BUILD_JAVA=OFF \
                       -DBUILD_SHARED_LIBS=${OPENCOLORIO_BUILD_SHARED_LIBS:=ON}"

# Fix yaml-cpp which breaks against cmake 4.0 because of too-old cmake min.
# Remove when yaml-cpp is fixed to declare its own minimum high enough.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

BASEDIR=`pwd`
pwd
echo "OpenColorIO install dir will be: ${OPENCOLORIO_INSTALL_DIR}"

mkdir -p ${LOCAL_DEPS_DIR}
pushd ${LOCAL_DEPS_DIR}

# Clone OpenColorIO project from GitHub and build
if [[ ! -e ${OPENCOLORIO_SOURCE_DIR} ]] ; then
    echo "git clone ${OPENCOLORIO_REPO} ${OPENCOLORIO_SOURCE_DIR}"
    git clone ${OPENCOLORIO_REPO} ${OPENCOLORIO_SOURCE_DIR}
fi
cd ${OPENCOLORIO_SOURCE_DIR}

echo "git checkout ${OPENCOLORIO_VERSION} --force"
git checkout ${OPENCOLORIO_VERSION} --force
echo "Building OpenColorIO from commit" `git rev-parse --short HEAD`

cmake -S ${OPENCOLORIO_SOURCE_DIR} -B ${OPENCOLORIO_BUILD_DIR} \
           -DCMAKE_BUILD_TYPE=${OPENCOLORIO_BUILD_TYPE} \
           -DCMAKE_INSTALL_PREFIX=${OPENCOLORIO_INSTALL_DIR} \
           ${OPENCOLORIO_CMAKE_FLAGS}
time cmake --build ${OPENCOLORIO_BUILD_DIR} --target install --config ${OPENCOLORIO_BUILD_TYPE}
popd

# ls -R ${OPENCOLORIO_INSTALL_DIR}

#echo "listing .."
#ls ..

# Set up paths. These will only affect the caller if this script is
# run with 'source' rather than in a separate shell.
export OpenColorIO_ROOT=$OPENCOLORIO_INSTALL_DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OPENCOLORIO_INSTALL_DIR}/lib

