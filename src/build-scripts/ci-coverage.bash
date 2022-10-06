#!/usr/bin/env bash

# Run code coverage analysis
# This assumes that the build occurred with CODECOV=1 andtests have already
# fully run.

set -ex

echo "Performing code coverage analysis"
mkdir _coverage
pushd _coverage

ls -R ../build/src/

# The sed command below converts from:
#   ../build/src/liboslexec/CMakeFiles/oslexec.dir/foo.gcno
# to:
#   ../src/liboslexec/foo.cpp

for g in $(find ../build -name "*.gcno" -type f); do
    echo "Processing $g"
    echo "dirname $g = $(dirname $g) to " `$(echo "$g" | sed -e 's/\/build\//\//' -e 's/\.gcno/\.cpp/' -e 's/\.cpp\.cpp/\.cpp/' -e 's/\/CMakeFiles.*\.dir\//\//')`
    gcov -l -p -o $(dirname "$g") $(echo "$g" | sed -e 's/\/build\//\//' -e 's/\.gcno/\.cpp/' -e 's/\/CMakeFiles.*\.dir\//\//')
done
popd
