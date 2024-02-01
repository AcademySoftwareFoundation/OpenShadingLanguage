// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of dictionary operations.
///
/////////////////////////////////////////////////////////////////////////

//#include <vector>
//#include <string>
//#include <cstdio>
//#include <cstdlib>
//#include <ctype.h>
//#include <unordered_map>
//
//#include <OpenImageIO/dassert.h>
//#include <OpenImageIO/strutil.h>
//
//
//#include "oslexec_pvt.h"
//#include "define_opname_macros.h"
//
//OSL_NAMESPACE_ENTER
//
//namespace __OSL_WIDE_PVT {
//
//OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include <OSL/oslconfig.h>

//#include <OSL/batched_rendererservices.h>
#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

//#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER

namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

OSL_BATCHOP int
__OSL_OP(dict_find_iis)(void* bsg_, int nodeID, void* query)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    return bsg->uniform.context->dict_find(
        nullptr /*causes errors be reported through ShadingContext*/, nodeID,
        USTR(query));
}



OSL_BATCHOP void
__OSL_MASKED_OP3(dict_find, Wi, Wi, Ws)(void* bsg_, void* wout, void* wnodeID,
                                        void* wquery, unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    Mask mask(mask_value);
    OSL_ASSERT(mask.any_on());

    Wide<const int> wNID(wnodeID);
    Wide<const ustring> wQ(wquery);
    Masked<int> wOut(wout, Mask(mask_value));

    mask.foreach ([=](ActiveLane lane) -> void {
        int nodeID    = wNID[lane];
        ustring query = wQ[lane];
        wOut[lane]    = bsg->uniform.context->dict_find(
            nullptr /*causes errors be reported through ShadingContext*/,
            nodeID, query);
    });
}



OSL_BATCHOP int
__OSL_OP(dict_find_iss)(void* bsg_, void* dictionary, void* query)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    return bsg->uniform.context->dict_find(
        nullptr /*causes errors be reported through ShadingContext*/,
        USTR(dictionary), USTR(query));
}



OSL_BATCHOP void
__OSL_MASKED_OP3(dict_find, Wi, Ws, Ws)(void* bsg_, void* wout,
                                        void* wdictionary, void* wquery,
                                        unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    Mask mask(mask_value);
    OSL_ASSERT(mask.any_on());

    Wide<const ustring> wD(wdictionary);
    Wide<const ustring> wQ(wquery);
    Masked<int> wOut(wout, Mask(mask_value));

    mask.foreach ([=](ActiveLane lane) -> void {
        ustring dictionary = wD[lane];
        ustring query      = wQ[lane];
        wOut[lane]         = bsg->uniform.context->dict_find(
            nullptr /*causes errors be reported through ShadingContext*/,
            dictionary, query);
    });
}


OSL_BATCHOP int
__OSL_OP(dict_next)(void* bsg_, int nodeID)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    return bsg->uniform.context->dict_next(nodeID);
}


OSL_BATCHOP void
__OSL_MASKED_OP(dict_next)(void* bsg_, void* wout, void* wNodeID,
                           unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    Mask mask(mask_value);
    OSL_ASSERT(mask.any_on());

    Wide<const int> wNID(wNodeID);
    Masked<int> wR(wout, Mask(mask_value));


    mask.foreach ([=](ActiveLane lane) -> void {
        int nodeID = wNID[lane];
        wR[lane]   = bsg->uniform.context->dict_next(nodeID);
    });
}



OSL_BATCHOP int
__OSL_OP(dict_value)(void* bsg_, int nodeID, void* attribname, long long type,
                     void* data)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    return bsg->uniform.context->dict_value(nodeID, USTR(attribname),
                                            TYPEDESC(type), data);
}

namespace {  // anonymous

template<typename ValueT, bool IsArrayT> struct DictValueGetter;

template<typename ValueT> struct DictValueGetter<ValueT, false> {
    static void get(ShadingContext* context, Wide<const int> wNID,
                    Wide<const ustring> wAttribName, MaskedData wdest,
                    Masked<int> wout)
    {
        Masked<ValueT> dest(wdest);
        wdest.mask().foreach ([=](ActiveLane lane) -> void {
            int nodeID         = wNID[lane];
            ustring attribname = wAttribName[lane];

            ValueT value;
            int result = context->dict_value(nodeID, attribname, wdest.type(),
                                             &value);
            wout[lane] = result;
            if (result) {
                dest[lane] = value;
            }
        });
    }
};

template<typename ValueT> struct DictValueGetter<ValueT, true> {
    typedef typename std::remove_all_extents<ValueT>::type ElementType;

    static void get(ShadingContext* context, Wide<const int> wNID,
                    Wide<const ustring> wAttribName, MaskedData wdest,
                    Masked<int> wout)
    {
        Masked<ElementType[]> dest_array(wdest);
        wdest.mask().foreach ([=](ActiveLane lane) -> void {
            int nodeID         = wNID[lane];
            ustring attribname = wAttribName[lane];
            ElementType value[dest_array.length()];
            int result = context->dict_value(nodeID, attribname, wdest.type(),
                                             &value[0]);
            auto dest  = dest_array[lane];
            for (int element = 0; element < dest.length(); ++element) {
                dest[element] = value[element];
            }
            wout[lane] = result;
        });
    }
};

template<typename DataT>
using wide_dict_value = DictValueGetter<DataT, std::is_array<DataT>::value>;

}  // namespace

OSL_BATCHOP void
__OSL_MASKED_OP(dict_value)(void* bsg_, void* wOut /*return int dict_value*/,
                            void* wNodeID, void* wAttribname, long long type,
                            void* wDataValue /*attribute value*/,
                            unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    Mask mask(mask_value);
    OSL_ASSERT(mask.any_on());

    Wide<const int> wNID(wNodeID);
    Wide<const ustring> wAttribName(wAttribname);
    MaskedData wdest(TYPEDESC(type), false /*has_derivs*/, mask, wDataValue);
    Masked<int> wout(wOut, mask);

    auto context = bsg->uniform.context;

    if (Masked<ustring>::is(wdest)) {
        wide_dict_value<ustring>::get(context, wNID, wAttribName, wdest, wout);
    } else if (Masked<int>::is(wdest)) {
        wide_dict_value<int>::get(context, wNID, wAttribName, wdest, wout);
    } else if (Masked<float>::is(wdest)) {
        wide_dict_value<float>::get(context, wNID, wAttribName, wdest, wout);
    } else if (Masked<Matrix44>::is(wdest)) {
        wide_dict_value<Matrix44>::get(context, wNID, wAttribName, wdest, wout);
    } else if (Masked<Vec3>::is(wdest)) {
        wide_dict_value<Vec3>::get(context, wNID, wAttribName, wdest, wout);
    }
    // We can't get oslc to even compile a shader with dict_value against an array
    // which doesn't match the documentation.  If array support is needed this
    // commented out code should implement it (add tests of course to exercise)
    /*else if (wdest.is<int[]>()) {
        wide_dict_value<int[]>::get(context, wNID, wAttribName, wdest, wout);
    } else if (wdest.is<float[]>()) {
        wide_dict_value<float[]>::get(context, wNID, wAttribName, wdest, wout);
    } else if (wdest.is<Matrix44[]>()) {
        wide_dict_value<Matrix44[]>::get(context, wNID, wAttribName, wdest, wout);
    } else if (wdest.is<Vec3[]>()) {
        wide_dict_value<Vec3[]>::get(context, wNID, wAttribName, wdest, wout);
    }*/

    else {
        OSL_ASSERT(0 && "Unsupported destination type");
    }
}

};  // namespace __OSL_WIDE_PVT

OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
