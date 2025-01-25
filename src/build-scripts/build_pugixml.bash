#!/usr/bin/env bash

# Utility script to download and build pugixml
#
# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Exit the whole script if any command fails.
set -ex

# Repo and branch/tag/commit of pugixml to download if we don't have it yet
PUGIXML_REPO=${PUGIXML_REPO:=https://github.com/zeux/pugixml.git}
PUGIXML_VERSION=${PUGIXML_VERSION:=v1.11.4}

LOCAL_DEPS_DIR=${LOCAL_DEPS_DIR:=${PWD}/ext}
PUGIXML_SOURCE_DIR=${PUGIXML_SOURCE_DIR:=${LOCAL_DEPS_DIR}/pugixml}
PUGIXML_BUILD_DIR=${PUGIXML_BUILD_DIR:=${PUGIXML_SOURCE_DIR}/build}
PUGIXML_INSTALL_DIR=${PUGIXML_INSTALL_DIR:=${LOCAL_DEPS_DIR}/dist}
PUGIXML_BUILD_TYPE=${PUGIXML_BUILD_TYPE:=Release}

pwd
echo "Building Pugixml ${PUGIXML_VERSION}"
echo "Pugixml source dir will be: ${PUGIXML_SOURCE_DIR}"
echo "Pugixml build dir will be: ${PUGIXML_BUILD_DIR}"
echo "Pugixml install dir will be: ${PUGIXML_INSTALL_DIR}"
echo "Pugixml build type is ${PUGIXML_BUILD_TYPE}"
echo "CMAKE_PREFIX_PATH is ${CMAKE_PREFIX_PATH}"

# Clone pugixml project from GitHub and build
if [[ ! -e ${PUGIXML_SOURCE_DIR} ]] ; then
    echo "git clone ${PUGIXML_REPO} ${PUGIXML_SOURCE_DIR}"
    git clone ${PUGIXML_REPO} ${PUGIXML_SOURCE_DIR}
fi
mkdir -p ${PUGIXML_INSTALL_DIR} && true

pushd ${PUGIXML_SOURCE_DIR}
echo "git checkout ${PUGIXML_VERSION} --force"
git checkout ${PUGIXML_VERSION} --force
echo "Building pugixml from commit" `git rev-parse --short HEAD`

if [[ -z $DEP_DOWNLOAD_ONLY ]]; then
    time cmake -S ${PUGIXML_SOURCE_DIR} -B ${PUGIXML_BUILD_DIR} \
               -DCMAKE_BUILD_TYPE=${PUGIXML_BUILD_TYPE} \
               -DCMAKE_INSTALL_PREFIX=${PUGIXML_INSTALL_DIR} \
               -DBUILD_SHARED_LIBS=${PUGIXML_LOCAL_BUILD_SHARED_LIBS:=ON} \
               -DBUILD_TESTS=OFF \
               ${PUGIXML_CMAKE_FLAGS}
    time cmake --build ${PUGIXML_BUILD_DIR} --target install --config ${PUGIXML_BUILD_TYPE}
fi

# ls -R ${PUGIXML_INSTALL_DIR}
popd

#echo "listing .."
#ls ..

# Set up paths. These will only affect the caller if this script is
# run with 'source' rather than in a separate shell.
export pugixml_ROOT=$PUGIXML_INSTALL_DIR
export LD_LIBRARY_PATH=$pugixml_ROOT/lib:$pugixml_ROOT/lib64:$LD_LIBRARY_PATH
