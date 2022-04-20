// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


// This file contains "declarations" for all the strings that might get used in
// JITed shader code or in renderer code. But the declaration itself is
// dependent on the RS_STRDECL macro, which should be declared by the outer file
// prior to including this file. Thus, this list may be repurposed and included
// multiple times, with different RS_STRDECL definitions.


#ifndef RS_STRDECL
#    error Do not include this file unless RS_STRDECL is defined
#endif


RS_STRDECL("perspective", perspective)
RS_STRDECL("raster", raster)
RS_STRDECL("myspace", myspace)