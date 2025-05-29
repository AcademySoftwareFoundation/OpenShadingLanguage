#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Important: set -ex causes this whole script to terminate with error if
# any command in it fails. This is crucial for CI tests.
set -ex

OSL_CMAKE_FLAGS="$MY_CMAKE_FLAGS $OSL_CMAKE_FLAGS"
export OSL_SRC_DIR=${OSL_SRC_DIR:=$PWD}
export OSL_BUILD_DIR=${OSL_BUILD_DIR:=${OSL_SRC_DIR}/build}
export OSL_INSTALL_DIR=${OSL_INSTALL_DIR:=${OSL_SRC_DIR}/dist}
export OSL_CMAKE_BUILD_TYPE=${OSL_CMAKE_BUILD_TYPE:=${CMAKE_BUILD_TYPE:=Release}}

if [[ "$USE_SIMD" != "" ]] ; then
    OSL_CMAKE_FLAGS="$OSL_CMAKE_FLAGS -DUSE_SIMD=$USE_SIMD"
fi

if [[ -n "$CODECOV" ]] ; then
    OSL_CMAKE_FLAGS="$OSL_CMAKE_FLAGS -DCODECOV=${CODECOV}"
fi

cmake -S ${OSL_SRC_DIR} -B ${OSL_BUILD_DIR} -G "$CMAKE_GENERATOR" \
        -DCMAKE_BUILD_TYPE="${OSL_CMAKE_BUILD_TYPE}" \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
        -DCMAKE_INSTALL_PREFIX="$OSL_ROOT" \
        -DUSE_PYTHON="${USE_PYTHON:=1}" \
        -DPYTHON_VERSION="$PYTHON_VERSION" \
        -DUSE_BATCHED="${USE_BATCHED:=0}" \
        -DCMAKE_INSTALL_LIBDIR="$OSL_ROOT/lib" \
        -DCMAKE_CXX_STANDARD="$CMAKE_CXX_STANDARD" \
        $OSL_CMAKE_FLAGS -DVERBOSE=1

# Save a copy of the generated files for debugging broken CI builds.
mkdir ${OSL_BUILD_DIR}/cmake-save || true
cp -r ${OSL_BUILD_DIR}/CMake* ${OSL_BUILD_DIR}/*.cmake ${OSL_BUILD_DIR}/cmake-save

: ${BUILDTARGET:=install}
if [[ "$BUILDTARGET" != "none" ]] ; then
    echo "Parallel build ${CMAKE_BUILD_PARALLEL_LEVEL} of target ${BUILDTARGET}"
    time ${OSL_CMAKE_BUILD_WRAPPER} cmake --build ${OSL_BUILD_DIR} --target ${BUILDTARGET} --config ${OSL_CMAKE_BUILD_TYPE}
    ccache --show-stats
fi

if [[ "${DEBUG_CI:=0}" != "0" ]] ; then
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "PYTHONPATH=$PYTHONPATH"
    echo "ldd testshade"
    ldd $OSL_ROOT/bin/testshade
fi

if [[ "$BUILDTARGET" == clang-format ]] ; then
    echo "Running " `which clang-format` " version " `clang-format --version`
    git diff --color
    THEDIFF=`git diff`
    if [[ "$THEDIFF" != "" ]] ; then
        echo "git diff was not empty. Failing clang-format or clang-tidy check."
        exit 1
    fi
fi
