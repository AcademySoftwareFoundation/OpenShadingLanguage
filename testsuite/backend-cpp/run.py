#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# End-to-end test of the BackendCpp path. With debug_output_cpp=3, testshade
# generates the C++ source, compiles it to a DSO, loads the DSO, and routes
# shader execution through it (skipping the JIT). A passing run therefore
# proves the whole pipeline: the generated source is valid C++ (it compiled),
# exports the expected ABI (the DSO loaded), and is correct (out.tif matches
# the reference, produced by the JIT path). No textual inspection of the
# generated .cpp is needed -- if anything were wrong, this run would fail.

command = testshade("--groupname mygroup"
                    + " --options debug_output_cpp=3,cpp_output_dir=."
                    + " -layer layer_a layer_a"
                    + " -layer backend_cpp_test_0 backend_cpp_test"
                    + " --connect layer_a out_val backend_cpp_test_0 in_val"
                    + " -o backend_cpp_test_0.Cout out.tif")

# Edge case: a shader with no renderer-output params.  The optimizer marks it
# as an empty instance, so the generated group entry dispatches zero layers.
# This verifies that BackendCpp produces a valid (compilable) translation unit
# even when no layer dispatch is emitted.  Use debug_output_cpp=2 so the DSO
# is compiled (validating the C++ is well-formed) without attempting to load
# and run it (there is no group entry to call with -o outputs).
command += testshade("--groupname edgedead"
                     + " --options debug_output_cpp=2,cpp_output_dir=."
                     + " -layer dl edge_dead")

# Edge case: a shader with an unsized-array parameter.  The optimizer resolves
# the array size from the default initializer before BackendCpp runs, so the
# generate_groupdata_struct() guard (// UNIMPLEMENTED) is a safety net only;
# the actual generated code should be valid C++ that compiles and runs
# correctly (output matches the JIT reference).
command += testshade("--groupname edgevararray"
                     + " --options debug_output_cpp=3,cpp_output_dir=."
                     + " -layer ev edge_vararray"
                     + " -o ev.Cout edge_vararray_out.tif")

# T040: ABI-mismatch test.
# Compile a stub DSO that returns the wrong ABI version (-999) from
# osl_cpp_abi_version().  Place it at the path that testshade would write a
# freshly compiled DSO, then run testshade with OSL_CPP_SKIP_COMPILE=1 so
# compile_to_dso() is a no-op (our stub is not overwritten) and load_dso()
# reads the stub instead of a valid DSO.  The expected outcome is a clear
# error message in out.txt; testshade exits 0 because edge_dead has no
# renderer outputs and no shading is attempted.
# The DSO filename includes the group's unique id; this single-group testshade
# run always assigns it id 1 (group-cpp-abimismatch_1).
import platform, os, subprocess as _sp
_dso_suffix = (".dylib" if platform.system() == "Darwin"
               else ".dll" if platform.system() == "Windows" else ".so")
_stub_src  = "wrong_abi_stub.cpp"
_stub_dso  = "group-cpp-abimismatch_1" + _dso_suffix
with open(_stub_src, "w") as _f:
    _f.write('extern "C" __attribute__((visibility("default")))\n'
             'int osl_cpp_abi_version() { return -999; }\n')
_flags = "-shared -fPIC"
if platform.system() == "Darwin":
    _flags += " -dynamiclib -undefined dynamic_lookup"
_sp.call("c++ " + _flags + " -o " + _stub_dso + " " + _stub_src, shell=True)
command += ("OSL_CPP_SKIP_COMPILE=1 "
            + testshade("--groupname abimismatch"
                        + " --options debug_output_cpp=3,cpp_output_dir=."
                        + " -layer dl edge_dead"))

# Normalize platform- and release-specific tokens in the ABI-mismatch error
# before comparison: the generated DSO suffix varies by OS (.dylib/.so/.dll)
# and the runtime ABI version embeds the OSL major/minor version (so it changes
# every release). The test only asserts that a clear mismatch error is shown.
# Written as a script file (not python -c) to avoid shell-quoting trouble.
with open("redact_abi.py", "w") as _f:
    _f.write('import re\n'
             's = open("out.txt").read()\n'
             's = re.sub(r"abimismatch_1[.](dylib|so|dll)", "abimismatch_1.DSO", s)\n'
             's = re.sub(r"runtime ABI version [0-9]+", "runtime ABI version NNN", s)\n'
             'open("out.txt", "w").write(s)\n')
command += (pythonbin + " redact_abi.py ;\n")

outputs = [ "out.txt", "out.tif", "edge_vararray_out.tif" ]
