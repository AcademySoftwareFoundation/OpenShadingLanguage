/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.  All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of Sony Pictures Imageworks nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////


#ifndef OSLUTIL_H
#define OSLUTIL_H

// Return wireframe opacity factor [0, 1] given a geometry type in
// ("triangles", "polygons" or "patches"), and a line_width in raster
// or world space depending on the last (raster) boolean argument.
//
float wireframe(string edge_type, float line_width, int raster)
{
   // ray differentials are so big in diffuse context that this function would always return "wire"
   if (raytype("path:diffuse")) return 0.0;

   int np = 0;
   point p[64];
   float pixelWidth = 1;

   if (edge_type == "triangles")
   {
      np = 3;
      if (!getattribute("geom:trianglevertices", p))
         return 0.0;
   }
   else if (edge_type == "polygons" || edge_type == "patches")
   {
      getattribute("geom:numpolyvertices", np);
      if (np < 3 || !getattribute("geom:polyvertices", p))
         return 0.0;
   }

   if (raster)
   {
      // Project the derivatives of P to the viewing plane defined
      // by I so we have a measure of how big is a pixel at this point
      float pixelWidthX = length(Dx(P) - dot(Dx(P), I) * I);
      float pixelWidthY = length(Dy(P) - dot(Dy(P), I) * I);
      // Take the average of both axis' length
      pixelWidth = (pixelWidthX + pixelWidthY) / 2;
   }

   // Use half the width as the neighbor face will render the
   // other half. And take the square for fast comparison
   pixelWidth *= 0.5 * line_width;
   pixelWidth *= pixelWidth;
   for (int i = 0; i < np; i++)
   {
      int i2 = i ? i - 1 : np - 1;
      vector dir = P - p[i];
      vector edge = p[i] - p[i2];
      vector crs = cross(edge, dir);
      // At this point dot(crs, crs) / dot(edge, edge) is
      // the square of area / length(edge) == square of the
      // distance to the edge.
      if (dot(crs, crs) < (dot(edge, edge) * pixelWidth))
         return 1;
   }
   return 0;
}

float wireframe(string edge_type, float line_width) { return wireframe(edge_type, line_width, 1); }
float wireframe(string edge_type) { return wireframe(edge_type, 1.0, 1); }
float wireframe() { return wireframe("polygons", 1.0, 1); }

float draw_string(string str, float s, float t, int wrap_s, int wrap_t, int jitter) {
   // Glyphs from http://font.gohu.org/ (8x14 version, most common ascii characters only)
   int glyph_pixel(int c_, int x_, int y) {
      int c = c_ - 33;   // nudge to origin
      int x = x_ - 1;
      if (c < 0  || x < 0 || y < 0 ) return 0;
      if (c > 93 || x > 6 || y > 13) return 0;
      int i = 98 * c + 7 * y + x;
      return ((getchar("0@P01248@00120000P49B0000000000000:DXlW2UoDX@10008@h;IR4n@R<Y?48000PYDF"
              "PP011J:U1000<T8QQQDAR4a50000@P012000000000000222448@P024@010028P0148@PP011100000"
              "ABELDU410000000048@l7124000000000000000H`01100000000n10000000000000000006<0000@P"
              "P011224488@00000`CXHY:=:D8?0000004<DT01248@000000l4:444444h700000`C8@Ph02D8?0000"
              "008HX89b?8@P000000n58`7@P05b300000`CP0O25:D8?00000POPP0112248000000l4:D8?Q25b300"
              "000`CX@Ql1244700000000H`0000<H00000000`P1000H`0110000044444@014@0000000n100PO000"
              "0000004@014@@@@@0000h948@@@@00120000`G`l5;F\\Lf0n100000l4:DXOQ25:400000hCX@Qn4:D"
              "X?000000?Q248@P0Ql000000N49DX@Q25i100000hGP01N48@PO00000PO124hAP012000000l4:@PLQ"
              "25b3000008DX@Qn5:DX@000000748@P0124L00000001248@P25b3000008DT456D8AT@00000P01248"
              "@P01n10000017G=IbP1364000008dXAU:U:E\\H000000?Q25:DX@Ql000000n4:DX?1248000000`CX"
              "@Q2U:E4GP0000P?Q25jCR8Q2100000l4:@0?P05b300000l71248@P01200000P@Q25:DX@Ql0000002"
              "5:D89BT`P1000004<HbT9[:BT800000P@QT8QQ49Q210000013:B4548@P000000h7888888@PO00000"
              "7248@P01248`10P0148P0148P0148000h01248@P0124>000015A000000000000000000000000h?00"
              "04@010000000000000000l0bGX@aL10000124XcX@Q25j300000000?Q248@8?000008@Pl5:DX@aL10"
              "000000`CX@o24`70000`AP01N48@P0100000000l5:DX@aL12T70124XcX@Q25:40000@P0P348@P01>"
              "00000240HP01248@P0a101248@T47B4940000HP01248@P01L00000000oBV<IbT910000000hCX@Q25"
              ":400000000?Q25:D8?00000000j<:DX@Qn48@00000`GX@Q25c58@P0000P>S248@P000000000l48P7"
              "@Pn0000048@`31248@030000000P@Q25:D<G0000000025:T49<H000000004<HbTE5920000000P@QT"
              "`@BX@0000000025:DX@aL12T70000h744444h70000PS01248>P0124`1001248@P01248@P0007@P01"
              "24`@P01R30000000S9S10000000", i / 6) - 48) >> (i % 6)) & 1;
   }
   int modp(int a, int b) {
      int x = a % b;
      return x < 0 ? x + b : x;
   }
   int len = strlen(str);
   int jit = jitter ? hash(str) : 0;
   int pix_w = len * 8;
   int pix_h = 14;
   int x = int(floor(s));
   int y = int(floor(t));
   if (wrap_s) x = modp(x + ( jitter        & 15), pix_w + pix_h);
   if (wrap_t) y = modp(y + ((jitter >> 16) & 15), pix_h + pix_h);
   return float(glyph_pixel(getchar(str, x / 8), x - (x / 8) * 8, y));
}


#endif /* OSLUTIL_H */
