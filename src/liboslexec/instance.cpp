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
#include <cstdio>

#include <boost/algorithm/string.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/filesystem.h"

#include "oslexec_pvt.h"
#include "osoreader.h"




namespace OSL {

namespace pvt {   // OSL::pvt


void
ShaderInstance::append (ShaderInstance::ref anotherlayer)
{
    ShaderInstance *inst = this;
    while (inst->next_layer ())
        inst = inst->next_layer ();
    DASSERT (inst != NULL && inst->next_layer() == NULL &&
             "we should be pointing to the last layer of the group");
    inst->m_nextlayer = anotherlayer;
    anotherlayer->m_firstlayer = false;
}


}; // namespace pvt
}; // namespace OSL
