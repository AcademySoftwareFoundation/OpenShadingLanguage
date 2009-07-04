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

#ifndef OSLOPS_H
#define OSLOPS_H

#include "OpenImageIO/typedesc.h"

#include "oslexec.h"
#include "osl_pvt.h"


namespace OSL {
namespace pvt {


#define DECLOP(name) \
    void name (ShadingExecution *exec, int nargs, const int *args, \
               Runflag *runflags, int beginpoint, int endpoint)


DECLOP (OP_assign);


}; // namespace pvt
}; // namespace OSL


#endif /* OSLOPS_H */
