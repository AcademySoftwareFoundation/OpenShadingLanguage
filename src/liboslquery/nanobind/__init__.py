# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Package init for the nanobind build of the oslquery module, used only when
# OSL_PYTHON_BINDINGS_BACKEND=both and we therefore need two modules to
# coexist. The pybind11 module keeps the plain `oslquery` name, so this one is
# built as `_oslquery` inside a package that re-exports it -- meaning
# `import oslquery` works the same either way, and which one you get depends
# only on what's on sys.path.
#
# When nanobind is the only backend, the module is named `oslquery` outright
# and the ordinary ../__init__.py is used instead of this file.

import os, sys, platform

# This works around the python 3.8 change to stop loading DLLs from PATH on Windows.
# We reproduce the old behaviour by manually tokenizing PATH, checking that the directories exist and are not ".",
# then add them to the DLL load path.
# This behaviour can be disabled by setting the environment variable "OSL_LOAD_DLLS_FROM_PATH" to "0"
if sys.version_info >= (3, 8) and platform.system() == "Windows" and os.getenv("OSL_LOAD_DLLS_FROM_PATH", "1") == "1":
    for path in os.getenv("PATH", "").split(os.pathsep):
        if os.path.exists(path) and path != ".":
            os.add_dll_directory(path)

# MSVC multi-config builds put _oslquery.pyd in a per-configuration subdirectory
# (oslquery/Release/, oslquery/Debug/, ...) while this file stays in oslquery/.
# Extending the package search path here is simpler and less fragile than
# trying to make CMake flatten the output layout.
if platform.system() == "Windows":
    _here = os.path.abspath(os.path.dirname(__file__))
    for _cfg in ("Release", "Debug", "RelWithDebInfo", "MinSizeRel"):
        _subdir = os.path.join(_here, _cfg)
        if os.path.isdir(_subdir) and _subdir not in __path__:
            __path__.append(_subdir)

from ._oslquery import *

# `import *` skips names beginning with an underscore, so bring the module's
# dunder attributes over by hand -- otherwise `oslquery.__version__` would
# exist in a nanobind-only build (where the extension module is imported
# directly) but not here. See the same note in ../__init__.py.
from ._oslquery import __version__
