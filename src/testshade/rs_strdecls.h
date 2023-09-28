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


RS_STRDECL("osl:version", osl_version)
RS_STRDECL("camera:resolution", camera_resolution)
RS_STRDECL("camera:projection", camera_projection)
RS_STRDECL("camera:pixelaspect", camera_pixelaspect)
RS_STRDECL("camera:screen_window", camera_screen_window)
RS_STRDECL("camera:fov", camera_fov)
RS_STRDECL("camera:clip", camera_clip)
RS_STRDECL("camera:clip_near", camera_clip_near)
RS_STRDECL("camera:clip_far", camera_clip_far)
RS_STRDECL("camera:shutter", camera_shutter)
RS_STRDECL("camera:shutter_open", camera_shutter_open)
RS_STRDECL("camera:shutter_close", camera_shutter_close)
RS_STRDECL("perspective", perspective)
RS_STRDECL("raster", raster)
RS_STRDECL("myspace", myspace)
RS_STRDECL("options", options)
RS_STRDECL("blahblah", blahblah)
RS_STRDECL("shading:index", shading_index)
