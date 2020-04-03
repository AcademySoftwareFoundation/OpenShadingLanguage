/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// This file contains "declarations" for all the strings that might get used in
// JITed shader code or in renderer code. But the declaration itself is
// dependent on the STRDECL macro, which should be declared by the outer file
// prior to including this file. Thus, this list may be repurposed and included
// multiple times, with different STRDECL definitions.


#ifndef STRDECL
#error Do not include this file unless STRDECL is defined
#endif

#ifndef PUBLIC_STRDECL
#define PUBLIC_STRDECL STRDECL
#define UNDEF_PUBLIC_STRDECL
#endif

PUBLIC_STRDECL ("", _emptystring_)
PUBLIC_STRDECL ("common", common)
PUBLIC_STRDECL ("object", object)
PUBLIC_STRDECL ("shader", shader)
PUBLIC_STRDECL ("closest", closest)
PUBLIC_STRDECL ("linear", linear)
PUBLIC_STRDECL ("cubic", cubic)
PUBLIC_STRDECL ("smartcubic", smartcubic)
PUBLIC_STRDECL ("perlin", perlin)
PUBLIC_STRDECL ("uperlin", uperlin)
PUBLIC_STRDECL ("noise", noise)
PUBLIC_STRDECL ("snoise", snoise)
PUBLIC_STRDECL ("cell", cell)
PUBLIC_STRDECL ("gabor", gabor)
PUBLIC_STRDECL ("simplex", simplex)
PUBLIC_STRDECL ("usimplex", usimplex)
PUBLIC_STRDECL ("simplexnoise", simplexnoise)
PUBLIC_STRDECL ("usimplexnoise", usimplexnoise)
PUBLIC_STRDECL ("hash", hash)
PUBLIC_STRDECL ("null", null)
PUBLIC_STRDECL ("unull", unull)
PUBLIC_STRDECL ("catmull-rom", catmullrom)
PUBLIC_STRDECL ("bezier", bezier)
PUBLIC_STRDECL ("bspline", bspline)
PUBLIC_STRDECL ("hermite", hermite)
PUBLIC_STRDECL ("constant", constant)
PUBLIC_STRDECL ("end", end)
PUBLIC_STRDECL ("!!!uninitialized!!!", uninitialized_string)
PUBLIC_STRDECL ("unknown", unknown)



STRDECL ("camera", camera)
STRDECL ("screen", screen)
STRDECL ("NDC", NDC)
STRDECL ("rgb", rgb)
STRDECL ("RGB", RGB)
STRDECL ("hsv", hsv)
STRDECL ("hsl", hsl)
STRDECL ("YIQ", YIQ)
STRDECL ("XYZ", XYZ)
STRDECL ("xyz", xyz)
STRDECL ("xyY", xyY)
STRDECL ("default", default_)
STRDECL ("label", label)
STRDECL ("sidedness", sidedness)
STRDECL ("front", front)
STRDECL ("back", back)
STRDECL ("both", both)
STRDECL ("P", P)
STRDECL ("I", I)
STRDECL ("N", N)
STRDECL ("Ng", Ng)
STRDECL ("dPdu", dPdu)
STRDECL ("dPdv", dPdv)
STRDECL ("u", u)
STRDECL ("v", v)
STRDECL ("Ps", Ps)
STRDECL ("time", time)
STRDECL ("dtime", dtime)
STRDECL ("dPdtime", dPdtime)
STRDECL ("Ci", Ci)
STRDECL ("width", width)
STRDECL ("swidth", swidth)
STRDECL ("twidth", twidth)
STRDECL ("rwidth", rwidth)
STRDECL ("blur", blur)
STRDECL ("sblur", sblur)
STRDECL ("tblur", tblur)
STRDECL ("rblur", rblur)
STRDECL ("wrap", wrap)
STRDECL ("swrap", swrap)
STRDECL ("twrap", twrap)
STRDECL ("rwrap", rwrap)
STRDECL ("black", black)
STRDECL ("clamp", clamp)
STRDECL ("periodic", periodic)
STRDECL ("mirror", mirror)
STRDECL ("firstchannel", firstchannel)
STRDECL ("fill", fill)
STRDECL ("alpha", alpha)
STRDECL ("errormessage", errormessage)
STRDECL ("mindist", mindist)
STRDECL ("maxdist", maxdist)
STRDECL ("shade", shade)
STRDECL ("traceset", traceset)
STRDECL ("interp", interp)
STRDECL ("cellnoise", cellnoise)
STRDECL ("pcellnoise", pcellnoise)
STRDECL ("hashnoise", hashnoise)
STRDECL ("phashnoise", phashnoise)
STRDECL ("pnoise", pnoise)
STRDECL ("psnoise", psnoise)
STRDECL ("genericnoise", genericnoise)
STRDECL ("genericpnoise", genericpnoise)
STRDECL ("gabornoise", gabornoise)
STRDECL ("gaborpnoise", gaborpnoise)
STRDECL ("anisotropic", anisotropic)
STRDECL ("direction", direction)
STRDECL ("do_filter", do_filter)
STRDECL ("bandwidth", bandwidth)
STRDECL ("impulses", impulses)
STRDECL ("dowhile", op_dowhile)
STRDECL ("for", op_for)
STRDECL ("while", op_while)
STRDECL ("exit", op_exit)
STRDECL ("subimage", subimage)
STRDECL ("subimagename", subimagename)
STRDECL ("missingcolor", missingcolor)
STRDECL ("missingalpha", missingalpha)
STRDECL ("useparam", useparam)
STRDECL ("raytype", raytype)
STRDECL ("color", color)
STRDECL ("point", point)
STRDECL ("vector", vector)
STRDECL ("normal", normal)
STRDECL ("matrix", matrix)
STRDECL ("Rec709", Rec709)
STRDECL ("sRGB", sRGB)
STRDECL ("NTSC", NTSC)
STRDECL ("EBU", EBU)
STRDECL ("PAL", PAL)
STRDECL ("SECAM", SECAM)
STRDECL ("SMPTE", SMPTE)
STRDECL ("HDTV", HDTV)
STRDECL ("CIE", CIE)
STRDECL ("AdobeRGB", AdobeRGB)
STRDECL ("colorsystem", colorsystem)

#ifdef UNDEF_PUBLIC_STRDECL
#undef PUBLIC_STRDECL
#endif
