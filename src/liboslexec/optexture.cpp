/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include "oslops.h"
#include "oslexec_pvt.h"

#include "OpenImageIO/sysutil.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


namespace {  // anonymous

inline TextureOptions::Wrap
decode_wrap (ustring w)
{
    if (w == Strings::black)
        return TextureOptions::WrapBlack;
    if (w == Strings::clamp)
        return TextureOptions::WrapClamp;
    if (w == Strings::periodic)
        return TextureOptions::WrapPeriodic;
    if (w == Strings::mirror)
        return TextureOptions::WrapMirror;
    return TextureOptions::WrapDefault;
}

};  // end anonymous namespace



DECLOP (OP_texture)
{
    // Grab the required arguments: result, filename, s, t
    DASSERT (nargs >= 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Filename (exec->sym (args[1]));
    DASSERT (Filename.typespec().is_string());
    Symbol &S (exec->sym (args[2]));
    Symbol &T (exec->sym (args[3]));
    DASSERT (S.typespec().is_float() && T.typespec().is_float());

    // Figure out if we are texture(name,s,t,...) or
    // texture(name,s,t,dsdx,dtdx,dsdy,dtdy).
    bool user_derivs = false;
    int first_optional_arg = 4;
    if (nargs > 4 && exec->sym(args[4]).typespec().is_float()) {
        user_derivs = true;
        first_optional_arg = 8;
        DASSERT (exec->sym(args[5]).typespec().is_float());
        DASSERT (exec->sym(args[6]).typespec().is_float());
        DASSERT (exec->sym(args[7]).typespec().is_float());
    }

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, true /* Assume texture always varies */);

    float zero = 0.0f;
    VaryingRef<float> result ((float *)Result.data(), Result.step());
    VaryingRef<float> s ((float *)S.data(), S.step());
    VaryingRef<float> t ((float *)T.data(), T.step());
    VaryingRef<ustring> filename ((ustring *)Filename.data(), Filename.step());
    VaryingRef<ustring> swrap (NULL), twrap (NULL);
    VaryingRef<int> firstchannel (NULL);
    VaryingRef<float> alpha (NULL);
    Symbol* Alpha = NULL;

    TextureSystem *texturesys = exec->texturesys ();
    TextureOptions options;
    options.firstchannel = 0;
    options.nchannels = Result.typespec().simpletype().aggregate;
    options.fill.init (&zero);

    // Set up derivs
    VaryingRef<float> dsdx, dsdy, dtdx, dtdy;
    if (user_derivs) {
        Symbol &Dsdx (exec->sym (args[4]));
        Symbol &Dsdy (exec->sym (args[6]));
        dsdx.init ((float *)Dsdx.data(), Dsdx.step());
        dsdy.init ((float *)Dsdy.data(), Dsdy.step());
    } else if (S.has_derivs()) {
        dsdx.init ((float *)S.data() + 1, S.step());
        dsdy.init ((float *)S.data() + 2, S.step());
    } else {
        dsdx.init (&zero);
        dsdy.init (&zero);
    }
    if (user_derivs) {
        Symbol &Dtdx (exec->sym (args[5]));
        Symbol &Dtdy (exec->sym (args[7]));
        dtdx.init ((float *)Dtdx.data(), Dtdx.step());
        dtdy.init ((float *)Dtdy.data(), Dtdy.step());
    } else if (T.has_derivs()) {
        dtdx.init ((float *)T.data() + 1, T.step());
        dtdy.init ((float *)T.data() + 2, T.step());
    } else {
        dtdx.init (&zero);
        dtdy.init (&zero);
    }

    // Parse all the optional arguments
    for (int a = first_optional_arg;  a < nargs;  ++a) {
        Symbol &Name (exec->sym (args[a]));
        DASSERT (Name.typespec().is_string() &&
                 "optional texture token must be a string");
        DASSERT (a+1 < nargs && "malformed argument list for texture");
        if (Name.is_varying()) {
            exec->warning ("optional texture argument is a varying string! Seems pretty fishy.");
        }
        ++a;  // advance to next argument
        Symbol &Val (exec->sym (args[a]));
        TypeDesc valtype = Val.typespec().simpletype ();
        ustring name = * (ustring *) Name.data();
        if (name == Strings::width && valtype == TypeDesc::FLOAT) {
            options.swidth.init ((float *)Val.data(), Val.step());
            options.twidth.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::swidth && valtype == TypeDesc::FLOAT) {
            options.swidth.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::twidth && valtype == TypeDesc::FLOAT) {
            options.twidth.init ((float *)Val.data(), Val.step());

        } else if (name == Strings::blur && valtype == TypeDesc::FLOAT) {
            options.sblur.init ((float *)Val.data(), Val.step());
            options.tblur.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::sblur && valtype == TypeDesc::FLOAT) {
            options.sblur.init ((float *)Val.data(), Val.step());
        } else if (name == Strings::tblur && valtype == TypeDesc::FLOAT) {
            options.tblur.init ((float *)Val.data(), Val.step());

        } else if (name == Strings::wrap && valtype == TypeDesc::STRING) {
            swrap.init ((ustring *)Val.data(), Val.step());
            twrap.init ((ustring *)Val.data(), Val.step());
        } else if (name == Strings::swrap && valtype == TypeDesc::STRING) {
            swrap.init ((ustring *)Val.data(), Val.step());
        } else if (name == Strings::twrap && valtype == TypeDesc::STRING) {
            twrap.init ((ustring *)Val.data(), Val.step());

        } else if (name == Strings::firstchannel && valtype == TypeDesc::INT) {
            firstchannel.init ((int *)Val.data(), Val.step());
        } else if (name == Strings::fill && valtype == TypeDesc::FLOAT) {
            options.fill.init ((float *)Val.data(), Val.step());

        } else if (name == Strings::alpha && valtype == TypeDesc::FLOAT) {
            exec->adjust_varying (Val, true);
            alpha.init ((float *)Val.data(), Val.step());
            Alpha = &Val;

        } else {
            exec->error ("Unknown texture optional argument: \"%s\", <%s> (%s:%d)",
                         name.c_str(),
                         valtype.c_str(),
                         exec->op().sourcefile().c_str(),
                         exec->op().sourceline());
        }
    }

    if (alpha)
        options.nchannels += 1;

    float *r = &result[0];
    bool tempresult = false;
    if (Result.has_derivs() || alpha) {
        tempresult = true;
        r = ALLOCA (float, endpoint*options.nchannels);
        // allocate some space to track the derivatives of the result
        // NOTE: even though OIIO doesn't need derivatives from S and T to
        // compute the gradients, we need them on the OSL side to be able to
        // rotate the gradients via the chain rule
        if (S.has_derivs() && T.has_derivs()) {
            options.dresultds = ALLOCA (float, endpoint*options.nchannels);
            options.dresultdt = ALLOCA (float, endpoint*options.nchannels);
        } else {
            // we won't be able to provide derivatives properly
            if (Result.has_derivs())
                exec->zero_derivs(Result);
            if (Alpha && Alpha->has_derivs())
                exec->zero_derivs(*Alpha);
        }
    }
    for (int i = beginpoint;  i < endpoint;  ++i) {
        // FIXME -- this calls texture system separately for each point!
        // We really want to batch it into groups that share the same texture
        // filename.
        if (runflags[i]) {
            if (swrap)
                options.swrap = decode_wrap (swrap[i]);
            if (twrap)
                options.twrap = decode_wrap (twrap[i]);
            if (firstchannel)
                options.firstchannel = firstchannel[i];

            bool ok = texturesys->texture (filename[i], options,
                                           runflags, i /*beginpoint*/, i+1 /*endpoint*/,
                                           s, t, dsdx, dtdx, dsdy, dtdy,
                                           r);
            if (! ok) {
                std::string err = texturesys->geterror ();
                if (err.length()) {
                    exec->error ("texture lookup failed (%s:%d): %s",
                        exec->op().sourcefile().c_str(),
                        exec->op().sourceline(),
                        err.c_str());
                }
            }
         }
    }

    if (tempresult) {
        // We need to re-copy results back to the right destinations.
        int resultchans = Result.typespec().simpletype().aggregate;
        for (int i = beginpoint;  i < endpoint;  ++i) {
            if (runflags[i])
                for (int c = 0;  c < resultchans;  ++c)
                    (&result[i])[c] = r[i*options.nchannels+c];
        }
        if (alpha) {
            for (int i = beginpoint;  i < endpoint;  ++i)
                if (runflags[i])
                    alpha[i] = r[i*options.nchannels+resultchans];
        }
        // now figure out derivatives (as needed)
        // we use the multi-variate chain rule:
        // dTdx = dTds * dsdx + dTdt * dtdx
        // dTdy = dTds * dsdy + dTdt * dtdy
        if (options.dresultds) {
            for (int i = beginpoint;  i < endpoint;  ++i) {
                if (runflags[i]) {
                    for (int c = 0;  c < resultchans;  ++c) {
                        (&result[i])[1 * resultchans + c] = options.dresultds[i*options.nchannels+c] * dsdx[i] + options.dresultdt[i*options.nchannels+c] * dtdx[i];
                        (&result[i])[2 * resultchans + c] = options.dresultds[i*options.nchannels+c] * dsdy[i] + options.dresultdt[i*options.nchannels+c] * dtdy[i];
                    }
                    if (alpha) {
                        (&alpha[i])[1] = options.dresultds[i*options.nchannels+resultchans] * dsdx[i] + options.dresultdt[i*options.nchannels+resultchans] * dtdx[i];
                        (&alpha[i])[2] = options.dresultds[i*options.nchannels+resultchans] * dsdy[i] + options.dresultdt[i*options.nchannels+resultchans] * dtdy[i];
                    }
                }
            }
        }
    }
}



DECLOP (OP_gettextureinfo)
{
    // Grab the required arguments: result, filename, dataname, output data
    DASSERT (nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    DASSERT (Result.typespec().is_int());
    Symbol &Filename (exec->sym (args[1]));
    DASSERT (Filename.typespec().is_string());
    Symbol &Dataname (exec->sym (args[2]));
    DASSERT (Dataname.typespec().is_string());
    Symbol &Data (exec->sym (args[3]));

    TextureSystem *texturesys = exec->texturesys ();

    // Adjust the result's uniform/varying status
    bool varying = Filename.is_varying() | Dataname.is_varying();
    exec->adjust_varying (Result, varying);
    exec->adjust_varying (Data, varying);
    if (Result.has_derivs())
       exec->zero_derivs (Result);
    if (Data.has_derivs())
       exec->zero_derivs (Data);
    varying |= Result.is_varying() | Data.is_varying();

    VaryingRef<int> result ((int *)Result.data(), Result.step());
    VaryingRef<ustring> filename ((ustring *)Filename.data(), Filename.step());
    VaryingRef<ustring> dataname ((ustring *)Dataname.data(), Dataname.step());
    VaryingRef<char> data ((char *)Data.data(), Data.step());
    
    for (int i = beginpoint;  i < endpoint;  ++i) {
        // FIXME -- this calls get_texture_info separately for each
        // point.  We should batch it into groups that share the same
        // texture filename and data name.  Though it being varying is
        // already probably a rare case, so it's not very high priority.
        if (runflags[i]) {
            result[i] = texturesys->get_texture_info (filename[i], dataname[i],
                                  Data.typespec().simpletype(), &data[i]);
            if (!result[i]) {
                std::string err = texturesys->geterror ();
                if (err.length()) {
                    exec->error ("gettextureinfo failed (%s:%d): %s",
                        exec->op().sourcefile().c_str(),
                        exec->op().sourceline(),
                        err.c_str());
                }
            }
        }
        if (! varying)
            break;
    }
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
