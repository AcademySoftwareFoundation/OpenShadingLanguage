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


The code in this file is based somewhat on code released by NVIDIA as
part of Gelato (specifically, gsoinfo.cpp).  That code had the following
copyright notice:

   Copyright 2004 NVIDIA Corporation.  All Rights Reserved.

and was distributed under BSD licensing terms identical to the
Sony Pictures Imageworks terms, above.
*/


#include <iostream>
#include <string>

#include "oslquery.h"
using namespace OSL;



static void
usage (void)
{
    std::cout << "oslinfo -- list parameters of a compiled OSL shader\n";
    std::cout << "Usage:  oslinfo [options] file0 [file1 ...]\n";
    std::cout << "Options:\n";
    std::cout << "       -v       Verbose\n";
    std::cout << "       -p %s    Set searchpath for shaders\n";
}



static void
print_default_string_vals (const OSLQuery::Parameter *p, bool verbose)
{
    if (verbose) {
        for (size_t a = 0;  a < p->type.numelements();  ++a)
            std::cout << "\t\tDefault value: \"" << p->sdefault[a] << "\"\n";
    } else {
        for (size_t a = 0;  a < p->type.numelements();  ++a)
            std::cout << "\"" << p->sdefault[a] << "\" ";
        std::cout << "\n";
    }
}



static void
print_default_int_vals (const OSLQuery::Parameter *p, bool verbose)
{
    size_t nf = p->type.aggregate;
    for (size_t a = 0;  a < p->type.numelements();  ++a) {
        if (verbose) {
            std::cout << "\t\tDefault value: ";
            if (p->spacename.size() > a && ! p->spacename[a].empty())
                std::cout << "\"" << p->spacename[a] << "\" ";
        }
        if (nf > 1)
            std::cout << "[ ";
        for (size_t f = 0;  f < nf; ++f) {
            std::cout << p->idefault[a*nf+f];
            if (f < nf-1)
                std::cout << ' ';
        }
        if (nf > 1)
            std::cout << " ]";
        std::cout << std::endl;
    }
}



static void
print_default_float_vals (const OSLQuery::Parameter *p, bool verbose)
{
    size_t nf = p->type.aggregate;
    for (size_t a = 0;  a < p->type.numelements();  ++a) {
        if (verbose) {
            std::cout << "\t\tDefault value: ";
            if (p->spacename.size() > a && ! p->spacename[a].empty())
                std::cout << "\"" << p->spacename[a] << "\" ";
        }
        if (nf > 1)
            std::cout << "[ ";
        for (size_t f = 0;  f < nf; ++f) {
            std::cout << p->fdefault[a*nf+f];
            if (f < nf-1)
                std::cout << ' ';
        }
        if (nf > 1)
            std::cout << " ]";
        std::cout << std::endl;
    }
}



static std::string
elaborate_escape_chars (const std::string &unescaped)
{
    std::string s = unescaped;
    for (size_t i = 0;  i < s.length();  ++i) {
        char c = s[i];
        if (c == '\n' || c == '\t' || c == '\v' || c == '\b' || 
            c == '\r' || c == '\f' || c == '\a' || c == '\\' || c == '\"') {
            s[i] = '\\';
            ++i;
            switch (c) {
            case '\n' : c = 'n'; break;
            case '\t' : c = 't'; break;
            case '\v' : c = 'v'; break;
            case '\b' : c = 'b'; break;
            case '\r' : c = 'r'; break;
            case '\f' : c = 'f'; break;
            case '\a' : c = 'a'; break;
            }
            s.insert (i, &c, 1);
        }
    }
    return s;
}



static void
print_metadata (const OSLQuery::Parameter &m)
{
    std::string typestring (m.type.c_str());
    std::cout << "\t\tmetadata: " << typestring << ' ' << m.name << " =";
    for (unsigned int d = 0;  d < m.fdefault.size();  ++d)
        std::cout << " " << m.fdefault[d];
    for (unsigned int d = 0;  d < m.sdefault.size();  ++d)
        std::cout << " \"" << elaborate_escape_chars(m.sdefault[d]) << "\"";
    std::cout << std::endl;
}



static void
oslinfo (const std::string &name, const std::string &path, bool verbose)
{
    OSLQuery g;
    g.open (name, path);
    std::string e = g.error();
    if (! e.empty()) {
        std::cout << "ERROR opening shader \"" << name << "\" (" << e << ")\n";
        return;
    }
    if (verbose)
         std::cout << g.shadertype() << " \"" << g.shadername() << "\"\n";
    else std::cout << g.shadertype() << " " << g.shadername() << "\n";
    if (verbose) {
        for (unsigned int m = 0;  m < g.metadata().size();  ++m)
            print_metadata (g.metadata()[m]);
    }

    for (size_t i = 0;  i < g.nparams();  ++i) {
        const OSLQuery::Parameter *p = g.getparam (i);
        if (!p)
            break;
        std::string typestring = p->type.c_str();
        if (verbose) {
            std::cout << "    \"" << p->name << "\" \""
                      << (p->isoutput ? "output " : "") << typestring << "\"\n";
        } else {
            std::cout << (p->isoutput ? "output " : "") << typestring << ' ' 
                      << p->name << ' ';
        }
        if (! p->validdefault) {
            if (verbose)
                 std::cout << "\t\tUnknown default value\n";
            else std::cout << "nodefault\n";
        }
        else if (p->type.basetype == PT_STRING)
            print_default_string_vals (p, verbose);
        else if (p->type.basetype == PT_INT)
            print_default_int_vals (p, verbose);
        else
            print_default_float_vals (p, verbose);
        if (verbose) {
            for (unsigned int i = 0;  i < p->metadata.size();  ++i)
                print_metadata (p->metadata[i]);
        }
    }
}



int
main (int argc, char *argv[])
{
    std::string path;
    bool verbose = false;
    for (int a = 1;  a < argc;  ++a) {
        if (! strcmp(argv[a],"-") || ! strcmp(argv[a],"-h") ||
            ! strcmp(argv[a],"-help") || ! strcmp(argv[a],"--h") ||
            ! strcmp(argv[a],"--help")) {
            usage();
            exit(0);
        } else if (! strcmp(argv[a], "-p")) {
            if (a == argc-1) {
                usage(); return(-1);
            }
            path = argv[++a];
        } else if (! strcmp (argv[a], "-v")) {
            verbose = true;
        } else {
            oslinfo (argv[a], path, verbose);
        }
    }
    return 0;
}
