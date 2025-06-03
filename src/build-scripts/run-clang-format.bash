#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Important: set -ex causes this whole script to terminate with error if
# any command in it fails. This is crucial for CI tests.
set -ex

CLANG_FORMAT_EXE=${CLANG_FORMAT_EXE:="clang-format"}
echo "Running " `which clang-format` " version " `${CLANG_FORMAT_EXE} --version`

files=`find ./{src,testsuite} \( -name '*.h' -o -name '*.cpp' -o -name '*.cu' \) -print \
       | grep -Ev 'testsuite/.*\.h|src/shaders'`


${CLANG_FORMAT_EXE}  -i -style=file $files
git diff --color
THEDIFF=`git diff`
if [[ "$THEDIFF" != "" ]] ; then
    echo "git diff was not empty. Failing clang-format check."
    exit 1
fi
