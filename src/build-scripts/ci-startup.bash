#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# This script is run when CI system first starts up.
# Since it sets many env variables needed by the caller, it should be run
# with 'source', not in a separate shell.

# Figure out the platform
if [[ $TRAVIS_OS_NAME == osx || $RUNNER_OS == macOS ]] ; then
      export ARCH=macosx
fi
if [[ $TRAVIS_OS_NAME == linux || $RUNNER_OS == Linux || $CIRCLECI == true ]] ; then
      export ARCH=linux64
fi
if [[ $RUNNER_OS == Windows ]] ; then
      export ARCH=windows64
fi
export PLATFORM=$ARCH

if [[ "$DEBUG" == 1 ]] ; then
    export PLATFORM=${PLATFORM}.debug
fi

echo "Architecture is $ARCH"
echo "Build platform name is $PLATFORM"

# Environment variables we always need
export USE_CCACHE=1
export CCACHE_CPP2=1
export OSL_ROOT=$PWD/dist/$PLATFORM
export DYLD_LIBRARY_PATH=$OSL_ROOT/lib:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OSL_ROOT/lib:$LD_LIBRARY_PATH
export OIIO_LIBRARY_PATH=$OSL_ROOT/lib:${OIIO_LIBRARY_PATH}
export LSAN_OPTIONS=suppressions=$PWD/src/build-scripts/nosanitize.txt
export ASAN_OPTIONS=print_suppressions=0

export PYTHON_VERSION=${PYTHON_VERSION:="2.7"}
export PYTHONPATH=$OSL_ROOT/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH
export BUILD_MISSING_DEPS=${BUILD_MISSING_DEPS:=1}
export COMPILER=${COMPILER:=gcc}
export CXX=${CXX:=g++}
export CI=true
export USE_NINJA=1
export CMAKE_GENERATOR=${CMAKE_GENERATOR:=Ninja}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}

if [[ $TRAVIS == true ]] ; then
    export PARALLEL=2
elif [[ $CIRCLECI == true ]] ; then
    export PARALLEL=4
elif [[ $GITHUB_ACTIONS == true ]] ; then
    export PARALLEL=4
fi
export PARALLEL=${PARALLEL:=4}
export PAR_MAKEFLAGS=-j${PARALLEL}
export CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL}
export CTEST_PARALLEL_LEVEL=${PARALLEL}

export LOCAL_DEPS_DIR=${LOCAL_DEPS_DIR:=$HOME/ext}

uname -a
uname -n
pwd
ls
env | sort

if [[ $ARCH == linux64 ]] ; then
    head -40 /proc/cpuinfo
elif [[ $ARCH == macosx ]] ; then
    sysctl machdep.cpu.features
fi
