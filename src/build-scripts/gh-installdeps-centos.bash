#!/usr/bin/env bash
#

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set -ex

echo "Which g++ " `which g++`
g++ --version && true
ls /usr/bin/g++* && true
/usr/bin/g++ --version && true

export PATH=/opt/rh/devtoolset-6/root/usr/bin:/usr/local/bin:$PATH
#ls /opt/rh/devtoolset-6/root/usr/bin && true
#ls /usr/local/bin

ls /etc/yum.repos.d

#sudo yum install -y giflib giflib-devel && true
sudo yum install -y Field3D Field3D-devel && true

#sudo rpm -v --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro && true
#sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm && true
#sudo yum install -y ffmpeg ffmpeg-devel && true


if [[ "$OPTIX_VERSION" != "" ]] ; then
    echo "Requested OPTIX_VERSION = '${OPTIX_VERSION}'"
    mkdir -p $LOCAL_DEPS_DIR/dist/include/internal
    OPTIXLOC=https://developer.download.nvidia.com/redist/optix/v${OPTIX_VERSION}
    for f in optix.h optix_device.h optix_function_table.h \
             optix_function_table_definition.h optix_host.h \
             optix_stack_size.h optix_stubs.h optix_types.h optix_7_device.h \
             optix_7_host.h optix_7_types.h \
             internal/optix_7_device_impl.h \
             internal/optix_7_device_impl_exception.h \
             internal/optix_7_device_impl_transformations.h
        do
        curl --retry 100 -m 120 --connect-timeout 30 \
            $OPTIXLOC/include/$f > $LOCAL_DEPS_DIR/dist/include/$f
    done
    export OptiX_ROOT=$LOCAL_DEPS_DIR/dist
fi

source src/build-scripts/build_pybind11.bash

source src/build-scripts/build_pugixml.bash

# Only build OpenEXR if a specific version is requested
if [[ "$OPENEXR_VERSION" != "" ]] ; then
    source src/build-scripts/build_openexr.bash
fi

# Only build OpenColorIO if a specific version is requested
if [[ "$OPENCOLORIO_VERSION" != "" ]] ; then
    # Temporary (?) fix: GH ninja having problems, fall back to make
    CMAKE_GENERATOR="Unix Makefiles" \
    source src/build-scripts/build_opencolorio.bash
fi

# Only build OpenImageIO if a specific version is requested
if [[ "$OPENIMAGEIO_VERSION" != "" ]] ; then
    # There are many parts of OIIO we don't need to build
    export ENABLE_iinfo=0 ENABLE_iv=0 ENABLE_igrep=0 ENABLE_iconvert=0 ENABLE_testtex=0
    export ENABLE_cineon=0 ENABLE_DDS=0 ENABLE_DPX=0 ENABLE_FITS=0
    export ENABLE_iff=0 ENABLE_jpeg2000=0 ENABLE_PNM=0 ENABLE_PSD=0
    export ENABLE_RLA=0 ENABLE_SGI=0 ENABLE_SOCKET=0 ENABLE_SOFTIMAGE=0
    export ENABLE_TARGA=0 ENABLE_WEBP=0
    export OPENIMAGEIO_MAKEFLAGS="OIIO_BUILD_TESTS=0 USE_OPENGL=0"
    source src/build-scripts/build_openimageio.bash
fi
