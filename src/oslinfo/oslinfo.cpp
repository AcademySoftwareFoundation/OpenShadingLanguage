/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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


The code in this file is based somewhat on code released by NVIDIA as
part of Gelato (specifically, gsoinfo.cpp).  That code had the following
copyright notice:

   Copyright 2004 NVIDIA Corporation.  All Rights Reserved.

and was distributed under BSD licensing terms identical to the
Sony Pictures Imageworks terms, above.
*/


#include <iostream>
#include <string>
#include <cstring>

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/argparse.h>

#include "OSL/oslquery.h"
using namespace OSL;


static std::string searchpath;
static bool verbose = false;
static bool help = false;
static std::string oneparam;



static void
print_default_string_vals (const OSLQuery::Parameter *p, bool verbose)
{
    size_t ne;
    if (p->varlenarray || p->type.arraylen < 0)
        ne = p->sdefault.size();
    else
        ne = p->type.numelements();
    if (verbose) {
        for (size_t a = 0;  a < ne;  ++a)
            std::cout << "\t\tDefault value: \"" << p->sdefault[a] << "\"\n";
    } else {
        for (size_t a = 0;  a < ne;  ++a)
            std::cout << "\"" << p->sdefault[a] << "\" ";
        std::cout << "\n";
    }
}



static void
print_default_int_vals (const OSLQuery::Parameter *p, bool verbose)
{
    size_t nf = p->type.aggregate;
    size_t ne;
    if (p->varlenarray || p->type.arraylen < 0)
        ne = p->idefault.size() / nf;
    else
        ne = p->type.numelements();
    if (verbose)
        std::cout << "\t\tDefault value:";
    if (p->type.arraylen || nf > 1)
        std::cout << " [";
    for (size_t a = 0;  a < ne;  ++a) {
        for (size_t f = 0;  f < nf; ++f)
            std::cout << ' ' << p->idefault[a*nf+f];
    }
    if (p->type.arraylen || nf > 1)
        std::cout << " ]";
    std::cout << std::endl;
}



static void
print_default_float_vals (const OSLQuery::Parameter *p, bool verbose)
{
    size_t nf = p->type.aggregate;
    size_t ne;
    if (p->varlenarray || p->type.arraylen < 0)
        ne = p->fdefault.size() / nf;
    else
        ne = p->type.numelements();
    if (verbose)
        std::cout << "\t\tDefault value:";
    if (p->type.arraylen || nf > 1)
        std::cout << " [";
    for (size_t a = 0;  a < ne;  ++a) {
        if (verbose && p->spacename.size() > a && ! p->spacename[a].empty())
            std::cout << " \"" << p->spacename[a] << "\"";
        for (size_t f = 0;  f < nf; ++f)
            std::cout << ' ' << p->fdefault[a*nf+f];
    }
    if (p->type.arraylen || nf > 1)
        std::cout << " ]";
    std::cout << std::endl;
}



static void
print_metadata (const OSLQuery::Parameter &m)
{
    std::string typestring (m.type.c_str());
    std::cout << "\t\tmetadata: " << typestring << ' ' << m.name << " =";
    for (unsigned int d = 0;  d < m.idefault.size();  ++d)
        std::cout << " " << m.idefault[d];
    for (unsigned int d = 0;  d < m.fdefault.size();  ++d)
        std::cout << " " << m.fdefault[d];
    for (unsigned int d = 0;  d < m.sdefault.size();  ++d)
        std::cout << " \"" << OIIO::Strutil::escape_chars(m.sdefault[d]) << "\"";
    std::cout << std::endl;
}



static void
oslinfo (const std::string &name)
{
    OSLQuery g;
    g.open (name, searchpath);
    std::string e = g.geterror();
    if (! e.empty()) {
        std::cout << "ERROR opening shader \"" << name << "\" (" << e << ")\n";
        return;
    }
    if (oneparam.empty()) {
        std::cout << g.shadertype() << " \"" << g.shadername() << "\"\n";
        if (verbose) {
            for (unsigned int m = 0;  m < g.metadata().size();  ++m)
                print_metadata (g.metadata()[m]);
        }
    }

    for (size_t i = 0;  i < g.nparams();  ++i) {
        const OSLQuery::Parameter *p = g.getparam (i);
        if (!p)
            break;
        if (oneparam.size() && oneparam != p->name)
            continue;
        std::string typestring;
        if (p->isstruct)
            typestring = "struct " + p->structname.string();
        else
            typestring = p->type.c_str();
        if (verbose) {
            std::cout << "    \"" << p->name << "\" \""
                      << (p->isoutput ? "output " : "") << typestring << "\"\n";
        } else {
            std::cout << (p->isoutput ? "output " : "") << typestring << ' ' 
                      << p->name << ' ';
        }
        if (p->isstruct) {
            if (verbose)
                std::cout << "\t\t";
            std::cout << "fields: {";
            for (size_t f = 0;  f < p->fields.size();  ++f) {
                if (f)
                    std::cout << ", ";
                std::string fieldname = p->name.string() + '.' + p->fields[f].string();
                const OSLQuery::Parameter *field = g.getparam (fieldname);
                if (field)
                    std::cout << field->type.c_str() << ' ' << p->fields[f];
                else
                    std::cout << "UNKNOWN";
            }
            std::cout << "}\n";
        }
        else if (! p->validdefault) {
            if (verbose)
                 std::cout << "\t\tUnknown default value\n";
            else std::cout << "nodefault\n";
        }
        else if (p->type.basetype == TypeDesc::STRING)
            print_default_string_vals (p, verbose);
        else if (p->type.basetype == TypeDesc::INT)
            print_default_int_vals (p, verbose);
        else
            print_default_float_vals (p, verbose);
        if (verbose) {
            for (unsigned int i = 0;  i < p->metadata.size();  ++i)
                print_metadata (p->metadata[i]);
        }
    }
}



static int
input_file (int argc, const char *argv[])
{
    for (int i = 0;  i < argc;  i++) {
        oslinfo (argv[i]);
    }
    return 0;
}



int
main (int argc, char *argv[])
{
    OIIO::ArgParse ap (argc, (const char **)argv);
    ap.options ("oslinfo -- list parameters of a compiled OSL shader\n"
                OSL_INTRO_STRING "\n"
                "Usage:  oslinfo [options] file0 [file1 ...]\n",
                "%*", input_file, "",
                "-h", &help, "Print help message",
                "--help", &help, "",
                "-v", &verbose, "Verbose",
                "-p %s", &searchpath, "Set searchpath for shaders",
                "--param %s", &oneparam, "Output information in just this parameter",
                NULL);

    if (ap.parse (argc, (const char **)argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
    } else if (help || argc <= 1) {
        ap.usage ();
    }
    return EXIT_SUCCESS;
}
