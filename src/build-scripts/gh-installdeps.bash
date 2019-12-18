#!/usr/bin/env bash
#

set -ex

#dpkg --list

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
time sudo apt-get update

time sudo apt-get -q install -y \
    git \
    cmake \
    ninja-build \
    g++ \
    ccache \
    libboost-dev libboost-thread-dev \
    libboost-filesystem-dev libboost-regex-dev \
    libtiff-dev \
    libilmbase-dev libopenexr-dev \
    python-dev python-numpy \
    libgif-dev \
    libpng-dev \
    flex bison libbison-dev \
    opencolorio-tools


# Disable libheif on CI for now... seems to make crashes in CI tests.
# Works fine for me in real life. Investigate.
#if [[ "$USE_LIBHEIF" != "0" ]] ; then
#    sudo add-apt-repository ppa:strukturag/libde265
#    sudo add-apt-repository ppa:strukturag/libheif
#    time sudo apt-get -q install -y libheif-dev
#fi

if [[ "$CXX" == "g++-4.8" ]] ; then
    time sudo apt-get install -y g++-4.8
elif [[ "$CXX" == "g++-6" ]] ; then
    time sudo apt-get install -y g++-6
elif [[ "$CXX" == "g++-7" ]] ; then
    time sudo apt-get install -y g++-7
elif [[ "$CXX" == "g++-8" ]] ; then
    time sudo apt-get install -y g++-8
elif [[ "$CXX" == "g++-9" ]] ; then
    time sudo apt-get install -y g++-9
fi

# time sudo apt-get install -y clang
# time sudo apt-get install -y llvm
#time sudo apt-get install -y libopenjpeg-dev
#time sudo apt-get install -y libjpeg-turbo8-dev

#dpkg --list

export CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu:$CMAKE_PREFIX_PATH

# Build or download LLVM
export LLVM_DISTRO_NAME=ubuntu-18.04
source src/build-scripts/build_llvm.bash

# Build OpenEXR
CXX="ccache $CXX" source src/build-scripts/build_openexr.bash

# We don't need OCIO, but if we ever want it, turn this on:
#CXX="ccache $CXX" source src/build-scripts/build_ocio.bash

# There are many parts of OIIO we don't need to build
export ENABLE_iinfo=0 ENABLE_iv=0 ENABLE_igrep=0 ENABLE_iconvert=0 ENABLE_testtex=0
export ENABLE_cineon=0 ENABLE_DDS=0 ENABLE_DPX=0 ENABLE_FITS=0
export ENABLE_iff=0 ENABLE_jpeg2000=0 ENABLE_PNM=0 ENABLE_PSD=0
export ENABLE_RLA=0 ENABLE_SGI=0 ENABLE_SOCKET=0 ENABLE_SOFTIMAGE=0
export ENABLE_TARGA=0 ENABLE_WEBP=0 ENABLE_TARGA=0
export OPENIMAGEIO_MAKEFLAGS="OIIO_BUILD_TESTS=0 USE_PYTHON=0 USE_OPENGL=0"
source src/build-scripts/build_openimageio.bash
