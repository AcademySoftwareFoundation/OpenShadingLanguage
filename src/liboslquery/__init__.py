# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import os, sys, platform

# This works around the python 3.8 change to stop loading DLLs from PATH on Windows.
# We reproduce the old behaviour by manually tokenizing PATH, checking that the directories exist and are not ".",
# then add them to the DLL load path.
# This behaviour can be disabled by setting the environment variable "OSL_LOAD_DLLS_FROM_PATH" to "0"
if sys.version_info >= (3, 8) and platform.system() == "Windows" and os.getenv("OSL_LOAD_DLLS_FROM_PATH", "1") == "1":
    for path in os.getenv("PATH", "").split(os.pathsep):
        if os.path.exists(path) and path != ".":
            os.add_dll_directory(path)

from .oslquery import *

# `import *` skips names beginning with an underscore, so the module's dunder
# attributes have to be brought over by hand. Without this, `oslquery.__version__`
# works when importing the extension module straight out of the build tree but
# is missing from an installed OSL, where this package wraps it.
from .oslquery import __version__

