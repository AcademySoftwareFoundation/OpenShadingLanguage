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

#ifndef OSL_PVT_H
#define OSL_PVT_H


namespace OSL {
namespace pvt {


/// Kinds of shaders
///
enum ShaderType {
    ShadTypeUnknown, ShadTypeGeneric, ShadTypeSurface, 
    ShadTypeDisplacement, ShadTypeVolume, ShadTypeLight,
    ShadTypeLast
};



/// Kinds of symbols
///
enum SymType {
    SymTypeParam, SymTypeOutputParam,
    SymTypeLocal, SymTypeTemp, SymTypeGlobal, SymTypeConst,
    SymTypeFunction, SymTypeType
};




}; // namespace OSL::pvt
}; // namespace OSL


#endif /* OSL_PVT_H */
