#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Print a cache key for the dependencies that gh-win-installdeps.bash
# builds from source. The key changes when anything that affects those
# builds changes: the dependency versions, the build settings, or the
# build scripts. Branch names (like OIIO's "release") are resolved to their
# current commit, so the deps are rebuilt when the branch moves.
#
# Usage: gh-win-deps-cache-key.bash <prefix>

set -e

# Resolve a branch name to its current commit. Tags and commit hashes are
# printed unchanged.
resolve_ref () {
    local repo=$1 ref=$2
    if [[ "$ref" == "" ]] ; then
        return
    fi
    # If ls-remote fails, fail the script rather than keying on a branch
    # name, which could keep restoring stale builds.
    local refs
    refs=$(git ls-remote "$repo" "refs/heads/$ref") || exit 1
    local sha=$(echo "$refs" | cut -f1)
    echo "${sha:-$ref}"
}

openexr=$(resolve_ref ${OPENEXR_REPO:-https://github.com/AcademySoftwareFoundation/openexr.git} ${OPENEXR_VERSION})
opencolorio=$(resolve_ref ${OPENCOLORIO_REPO:-https://github.com/AcademySoftwareFoundation/OpenColorIO.git} ${OPENCOLORIO_VERSION})
openimageio=$(resolve_ref https://github.com/${OPENIMAGEIO_REPO:-AcademySoftwareFoundation/OpenImageIO} ${OPENIMAGEIO_VERSION})
scripts=$(cat src/build-scripts/gh-win-installdeps.bash src/build-scripts/build_*.bash | sha256sum)

desc=$(
    echo "zlib=${ZLIB_VERSION}"
    echo "libpng=${LIBPNG_VERSION}"
    echo "pybind11=${PYBIND11_VERSION}"
    echo "pugixml=${PUGIXML_VERSION}"
    echo "openexr=${openexr}"
    echo "opencolorio=${opencolorio}"
    echo "openimageio=${openimageio}"
    echo "OpenImageIO_BUILD_MISSING_DEPS=${OpenImageIO_BUILD_MISSING_DEPS}"
    echo "OPENIMAGEIO_UNITY=${OPENIMAGEIO_UNITY}"
    echo "CMAKE_GENERATOR=${CMAKE_GENERATOR}"
    echo "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    echo "PYTHON_VERSION=${PYTHON_VERSION}"
    echo "USE_SIMD=${USE_SIMD}"
    echo "scripts=${scripts}"
)

# Show the inputs in the log, to help explain cache misses
echo "$desc" | sed 's/^/    /' >&2

echo "$1-$(echo "$desc" | sha256sum | cut -c1-16)"
