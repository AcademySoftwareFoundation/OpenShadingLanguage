#!/bin/bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Install OpenImageIO

# Exit the whole script if any command fails.
set -ex

OPENIMAGEIO_REPO=${OPENIMAGEIO_REPO:=AcademySoftwareFoundation/OpenImageIO}
OPENIMAGEIO_VERSION=${OPENIMAGEIO_VERSION:=release}

LOCAL_DEPS_DIR=${LOCAL_DEPS_DIR:=${PWD}/ext}
OPENIMAGEIO_SOURCE_DIR=${OPENIMAGEIO_SOURCE_DIR:=${LOCAL_DEPS_DIR}/OpenImageIO}
OPENIMAGEIO_BUILD_DIR=${OPENIMAGEIO_BUILD_DIR:=${OPENIMAGEIO_SOURCE_DIR}/build}
OPENIMAGEIO_INSTALL_DIR=${OPENIMAGEIO_INSTALL_DIR:=${LOCAL_DEPS_DIR}/dist}
OPENIMAGEIO_BUILD_TYPE=${OPENIMAGEIO_BUILD_TYPE:=Release}
OPENIMAGEIO_CMAKE_FLAGS=${OPENIMAGEIO_CMAKE_FLAGS:=""}
OPENIMAGEIO_CXXFLAGS=${OPENIMAGEIO_CXXFLAGS:=""}
BASEDIR=$PWD

pwd
echo "Building OpenImageIO ${OPENIMAGEIO_VERSION}"
echo "OpenImageIO build dir will be: ${OPENIMAGEIO_BUILD_DIR}"
echo "OpenImageIO install dir will be: ${OPENIMAGEIO_INSTALL_DIR}"
echo "OpenImageIO Build type is ${OPENIMAGEIO_BUILD_TYPE}"
echo "CXX: '${CXX}'"
echo "CC: '${CC}'"
echo "CMAKE_CXX_COMPILER: '${CMAKE_CXX_COMPILER}'"

if [ ! -e $OPENIMAGEIO_SOURCE_DIR ] ; then
    git clone https://github.com/${OPENIMAGEIO_REPO} $OPENIMAGEIO_SOURCE_DIR
fi
mkdir -p ${OPENIMAGEIO_INSTALL_DIR} && true
mkdir -p ${OPENIMAGEIO_BUILD_DIR} && true

pushd $OPENIMAGEIO_SOURCE_DIR
git fetch --all -p
git checkout $OPENIMAGEIO_VERSION --force
echo "Building OpenImageIO from commit" `git rev-parse --short HEAD`

if [[ "$USE_SIMD" != "" ]] ; then
    OPENIMAGEIO_CMAKE_FLAGS+=" -DUSE_SIMD=$USE_SIMD"
fi


cmake   -S ${OPENIMAGEIO_SOURCE_DIR} -B ${OPENIMAGEIO_BUILD_DIR} \
        -DCMAKE_BUILD_TYPE="${OPENIMAGEIO_BUILD_TYPE}" \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
        -DCMAKE_INSTALL_PREFIX="${OPENIMAGEIO_INSTALL_DIR}" \
        -DPYTHON_VERSION="${PYTHON_VERSION}" \
        -DCMAKE_INSTALL_LIBDIR="${OPENIMAGEIO_INSTALL_DIR}/lib" \
        -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}" \
        ${OPENIMAGEIO_CMAKE_FLAGS} -DVERBOSE=1
echo "Parallel build $CMAKE_BUILD_PARALLEL_LEVEL"
time cmake --build ${OPENIMAGEIO_BUILD_DIR} --target install --config ${OPENIMAGEIO_BUILD_TYPE}

popd

export OpenImageIO_ROOT=$OPENIMAGEIO_INSTALL_DIR
export DYLD_LIBRARY_PATH=$OpenImageIO_ROOT/lib:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OpenImageIO_ROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$OpenImageIO_ROOT/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH

echo "DYLD_LIBRARY_PATH = $DYLD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "OpenImageIO_ROOT $OpenImageIO_ROOT"
# ls -R $OpenImageIO_ROOT
