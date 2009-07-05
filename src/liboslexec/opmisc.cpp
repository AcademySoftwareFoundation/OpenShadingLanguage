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

#include "oslops.h"


namespace OSL {
namespace pvt {



DECLOP (OP_missing)
{
    std::cerr << "Missing op!\n";
}



DECLOP (OP_end)
{
    std::cerr << "Executing end!\n";
}



}; // namespace pvt
}; // namespace OSL
