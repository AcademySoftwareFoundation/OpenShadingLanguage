/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/


#include <vector>
#include <string>
#include <fstream>
#include <cstdio>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "osoreader.h"


#define yyFlexLexer osoFlexLexer
#include "FlexLexer.h"


namespace OSL {

namespace pvt {   // OSL::pvt


osoFlexLexer * OSOReader::osolexer = NULL;
OSOReader * OSOReader::osoreader = NULL;
mutex OSOReader::m_osoread_mutex;



bool
OSOReader::parse (const std::string &filename)
{
    // The lexer/parser isn't thread-safe, so make sure Only one thread
    // can actually be reading a .oso file at a time.
    lock_guard guard (m_osoread_mutex);

    std::fstream input (filename.c_str(), std::ios::in);
    if (! input.is_open()) {
        std::cerr << "File " << filename << " not found.\n";
        return false;
    }

    osoreader = this;
    osolexer = new osoFlexLexer (&input);
    assert (osolexer);
    bool ok = ! osoparse ();   // osoparse returns nonzero if error
    if (ok) {
//        std::cout << "Correctly parsed " << filename << "\n";
    } else {
        std::cout << "Failed parse of " << filename << "\n";
    }
    delete osolexer;
    osolexer = NULL;

    input.close ();
    return ok;
}



}; // namespace pvt
}; // namespace OSL
