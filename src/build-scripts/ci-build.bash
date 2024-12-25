#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Important: set -ex causes this whole script to terminate with error if
# any command in it fails. This is crucial for CI tests.
set -ex

export OSL_CMAKE_BUILD_TYPE=${OSL_CMAKE_BUILD_TYPE:=${CMAKE_BUILD_TYPE:=Release}}

if [[ "$USE_SIMD" != "" ]] ; then
    OSL_CMAKE_FLAGS="$OSL_CMAKE_FLAGS -DUSE_SIMD=$USE_SIMD"
fi

if [[ -n "$CODECOV" ]] ; then
    OSL_CMAKE_FLAGS="$OSL_CMAKE_FLAGS -DCODECOV=${CODECOV}"
fi

pushd build
cmake .. -G "$CMAKE_GENERATOR" \
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
mkdir cmake-save || /bin/true
cp -r CMake* *.cmake cmake-save

: ${BUILDTARGET:=install}
if [[ "$BUILDTARGET" != "none" ]] ; then
    echo "Parallel build ${CMAKE_BUILD_PARALLEL_LEVEL} of target ${BUILDTARGET}"
    time ${OSL_CMAKE_BUILD_WRAPPER} cmake --build . --target ${BUILDTARGET} --config ${CMAKE_BUILD_TYPE}
fi
popd

if [[ "${DEBUG_CI:=0}" != "0" ]] ; then
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "PYTHONPATH=$PYTHONPATH"
    echo "ldd testshade"
    ldd $OSL_ROOT/bin/testshade
fi

if [[ "$BUILDTARGET" == clang-format ]] ; then
    git diff --color
    THEDIFF=`git diff`
    if [[ "$THEDIFF" != "" ]] ; then
        echo "git diff was not empty. Failing clang-format or clang-tidy check."
        exit 1
    fi
fi
