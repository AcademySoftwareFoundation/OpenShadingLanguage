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


/// Macro that defines the arguments to shading opcode implementations
///
#define OPARGSDECL     ShadingExecution *exec, int nargs, const int *args, \
                       Runflag *runflags, int beginpoint, int endpoint

/// Macro that defines the full declaration of a shading opcode
/// implementation
#define DECLOP(name)   void name (OPARGSDECL)


// Declarations of all our shader opcodes follow:

DECLOP (OP_assign);
DECLOP (OP_end);

DECLOP (OP_missing);


}; // namespace pvt
}; // namespace OSL


#endif /* OSLOPS_H */
