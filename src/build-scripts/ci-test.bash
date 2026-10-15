#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Important: set -ex causes this whole script to terminate with error if
# any command in it fails. This is crucial for CI tests.
set -ex

: ${CTEST_EXCLUSIONS:="broken"}
: ${CTEST_TEST_TIMEOUT:=60}

$OSL_ROOT/bin/testshade --help

echo ; echo "Results of testshade --buildinfo:"
$OSL_ROOT/bin/testshade --buildinfo | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
# Note: the above both echos the --buildinfo output, and also appends it to the
# message that will be visible in the GHA summary log page so it's easy to
# see at a glance which build chain & dependencies were used across all the
# tests.
echo

echo "Parallel test ${CTEST_PARALLEL_LEVEL}"
echo "Default timeout ${CTEST_TEST_TIMEOUT}"
echo "Test exclusions '${CTEST_EXCLUSIONS}'"
echo "CTEST_ARGS '${CTEST_ARGS}'"

pushd build

ctest -C ${CMAKE_BUILD_TYPE} --force-new-ctest-process --output-on-failure \
    --timeout ${CTEST_TEST_TIMEOUT} -E "${CTEST_EXCLUSIONS}" ${CTEST_ARGS} \
  || \
ctest -C ${CMAKE_BUILD_TYPE} --force-new-ctest-process \
    --rerun-failed --repeat until-pass:5 -R render --output-on-failure \
    --timeout ${CTEST_TEST_TIMEOUT} -E "${CTEST_EXCLUSIONS}" ${CTEST_ARGS}
# The weird construct above allows the render-* tests to run multiple times

popd


# if [[ "$CODECOV" == 1 ]] ; then
#     bash <(curl -s https://codecov.io/bash)
# fi
