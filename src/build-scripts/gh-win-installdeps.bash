#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

if [[ ! -e build/$PLATFORM ]] ; then
    mkdir -p build/$PLATFORM
fi
if [[ ! -e dist/$PLATFORM ]] ; then
    mkdir -p dist/$PLATFORM
fi

# DEP_DIR="$PWD/ext/dist"
DEP_DIR="$PWD/dist/$PLATFORM"
mkdir -p "$DEP_DIR"
INT_DIR="build/$PLATFORM"
VCPKG_INSTALLATION_ROOT=/c/vcpkg

export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:=.}
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;$DEP_DIR"
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;$BOOST_ROOT"
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;$VCPKG_INSTALLATION_ROOT/installed/x64-windows"
export PATH="$PATH:$DEP_DIR/bin:$DEP_DIR/lib:$VCPKG_INSTALLATION_ROOT/installed/x64-windows/bin:/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$DEP_DIR/bin:$VCPKG_INSTALLATION_ROOT/installed/x64-windows/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$DEP_DIR/lib:$VCPKG_INSTALLATION_ROOT/installed/x64-windows/lib"

# export OSL_CMAKE_FLAGS="$OSL_CMAKE_FLAGS -DCMAKE_TOOLCHAIN_FILE=$VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake"
# export OPENEXR_CMAKE_FLAGS="$OPENEXR_CMAKE_FLAGS -DCMAKE_TOOLCHAIN_FILE=$VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake"

ls -l "C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC"

vcpkg list
vcpkg update

# vcpkg install zlib:x64-windows
vcpkg install tiff:x64-windows
vcpkg install libpng:x64-windows
#vcpkg install giflib:x64-windows
#vcpkg install freetype:x64-windows
# vcpkg install openexr:x64-windows
# vcpkg install libjpeg-turbo:x64-windows

# vcpkg install libraw:x64-windows
# vcpkg install openjpeg:x64-windows
# vcpkg install ffmpeg:x64-windows   # takes FOREVER!
# vcpkg install webp:x64-windows  # No such vcpkg package?a

echo "$VCPKG_INSTALLATION_ROOT"
ls "$VCPKG_INSTALLATION_ROOT"
echo "$VCPKG_INSTALLATION_ROOT/installed/x64-windows"
ls "$VCPKG_INSTALLATION_ROOT/installed/x64-windows"
echo "$VCPKG_INSTALLATION_ROOT/installed/x64-windows/lib"
ls "$VCPKG_INSTALLATION_ROOT/installed/x64-windows/lib"
echo "$VCPKG_INSTALLATION_ROOT/installed/x64-windows/bin"
ls "$VCPKG_INSTALLATION_ROOT/installed/x64-windows/bin"

echo "All VCPkg installs:"
vcpkg list

# curl --location https://ffmpeg.zeranoe.com/builds/win64/shared/ffmpeg-4.2.1-win64-shared.zip -o ffmpeg-libs.zip
# unzip ffmpeg-libs.zip
curl --location https://ffmpeg.zeranoe.com/builds/win64/dev/ffmpeg-4.2.1-win64-dev.zip -o ffmpeg-dev.zip
unzip ffmpeg-dev.zip
ls
ls -R *ffmpeg*
FFmpeg_ROOT=$PWD/ffmpeg-4.2.1-win64-dev

echo "CMAKE_PREFIX_PATH = $CMAKE_PREFIX_PATH"

mkdir ext

# ZLib
pushd ext
git clone -b v1.2.11 https://github.com/madler/zlib.git
cd zlib
mkdir -p $INT_DIR
cd $INT_DIR
cmake ../.. -G "$CMAKE_GENERATOR" -DCMAKE_CONFIGURATION_TYPES="$CMAKE_BUILD_TYPE" -DCMAKE_PREFIX_PATH="$DEP_DIR" -DCMAKE_INSTALL_PREFIX="$DEP_DIR"
cmake --build . --config $CMAKE_BUILD_TYPE --target install
popd
export OSL_CMAKE_FLAGS="$OSL_CMAKE_FLAGS -DZLIB_LIBRARY=$DEP_DIR/lib/zlib.lib"
export OPENEXR_CMAKE_FLAGS="$OPENEXR_CMAKE_FLAGS -DZLIB_LIBRARY=$DEP_DIR/lib/zlib.lib"

source src/build-scripts/build_pybind11.bash

OPENEXR_CXX_FLAGS=" /W1 /EHsc /DWIN32=1 "
#OPENEXR_BUILD_TYPE=$CMAKE_BUILD_TYPE
OPENEXR_INSTALL_DIR=$DEP_DIR
OPENEXR_BRANCH=v2.4.0
source src/build-scripts/build_openexr.bash
export PATH="$OPENEXR_INSTALL_DIR/bin:$OPENEXR_INSTALL_DIR/lib:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PATH
# the above line is admittedly sketchy

cp $DEP_DIR/lib/*.lib $DEP_DIR/bin
cp $DEP_DIR/bin/*.dll $DEP_DIR/lib


#OPENIMAGEIO_REPO=lgritz/oiio
#OPENIMAGEIO_BRANCH=lg-cmake312
OPENIMAGEIO_INSTALLDIR=$DEP_DIR
OPENIMAGEIO_CMAKE_FLAGS+=" -DOIIO_BUILD_TESTS=0 -DOIIO_BUILD_TOOLS=0"
OPENIMAGEIO_CMAKE_FLAGS+=" -DUSE_OPENGL=0 -DUSE_GENERATED_EXPORT_HEADER=1"
OPENIMAGEIO_CMAKE_FLAGS+=" -DENABLE_DPX=0 -DENABLE_CINEON=0 -DENABLE_DDS=0"
OPENIMAGEIO_CMAKE_FLAGS+=" -DENABLE_IFF=0 -DENABLE_ICO=0 -DENABLE_PSD=0"
OPENIMAGEIO_CMAKE_FLAGS+=" -DENABLE_PNM=0 -DENABLE_ZFILE=0 -DENABLE_SOFTIMAGE=0"
OPENIMAGEIO_CMAKE_FLAGS+=" -DLINKSTATIC=1 -DBUILD_SHARED_LIBS=0"
export OPENIMAGEIO_CMAKE_FLAGS
source src/build-scripts/build_openimageio.bash


cp $DEP_DIR/lib/*.lib $DEP_DIR/bin
cp $DEP_DIR/bin/*.dll $DEP_DIR/lib
echo "after OIIO install, DEP_DIR $DEP_DIR :"
ls -R -l "$DEP_DIR"


# export PATH="$PATH:$DEP_DIR/bin:$VCPKG_INSTALLATION_ROOT/installed/x64-windows/bin"
export PATH="$DEP_DIR/lib:$DEP_DIR/bin:$PATH:$VCPKG_INSTALLATION_ROOT/installed/x64-windows/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VCPKG_INSTALLATION_ROOT/installed/x64-windows/lib:$DEP_DIR/lib:$DEP_DIR/bin"


# source src/build-scripts/build_openexr.bash
# export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;$ILMBASE_ROOT;$OPENEXR_ROOT"
# source src/build-scripts/build_ocio.bash


if [[ "$PYTHON_VERSION" == "3.6" ]] ; then
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;/c/hostedtoolcache/windows/Python/3.6.8/x64"
fi

# For CI on Windows, prefer to pick up static libs where possible
OSL_CMAKE_FLAGS="$OSL_CMAKE_FLAGS -DLINKSTATIC=1 -DBUILD_SHARED_LIBS=0"
