// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include <OSL/batched_rendererservices.h>
#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)
using WidthTag = OSL::WidthOf<__OSL_WIDTH>;

#include "define_opname_macros.h"

// vals points to a symbol with a total of ncomps floats (ncomps ==
// aggregate*arraylen).  If has_derivs is true, it's actually 3 times
// that length, the main values then the derivatives.  We want to check
// for nans in vals[firstcheck..firstcheck+nchecks-1], and also in the
// derivatives if present.  Note that if firstcheck==0 and nchecks==ncomps,
// we are checking the entire contents of the symbol.  More restrictive
// firstcheck,nchecks are used to check just one element of an array.
OSL_BATCHOP void
__OSL_OP(naninf_check)(int ncomps, const void* vals_, int has_derivs,
                       void* bsg_, ustring_pod sourcefile, int sourceline,
                       ustring_pod symbolname, int firstcheck, int nchecks,
                       ustring_pod opname)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;
    const float* vals   = (const float*)vals_;
    for (int d = 0; d < (has_derivs ? 3 : 1); ++d) {
        for (int c = firstcheck, e = c + nchecks; c < e; ++c) {
            int i = d * ncomps + c;
            if (!OIIO::isfinite(vals[i])) {
                ctx->errorfmt("Detected {} value in {}{} at {}:{} (op {})",
                              vals[i], d > 0 ? "the derivatives of " : "",
                              USTR(symbolname), USTR(sourcefile), sourceline,
                              USTR(opname));
                return;
            }
        }
    }
}



// Wide vals + mask, but uniform index
OSL_BATCHOP void
__OSL_MASKED_OP1(naninf_check_offset,
                 i)(int mask_value, int ncomps, const void* vals_,
                    int has_derivs, void* bsg_, ustring_pod sourcefile,
                    int sourceline, ustring_pod symbolname, int firstcheck,
                    int nchecks, ustring_pod opname)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;
    const float* vals   = (const float*)vals_;
    const Mask mask(mask_value);
    for (int d = 0; d < (has_derivs ? 3 : 1); ++d) {
        for (int c = firstcheck, e = c + nchecks; c < e; ++c) {
            int i = d * ncomps + c;
            mask.foreach ([=](ActiveLane lane) -> void {
                if (!OIIO::isfinite(vals[i * __OSL_WIDTH + lane])) {
                    ctx->errorfmt(
                        "Detected {} value in {}{} at {}:{} (op {}) batch lane:{}",
                        vals[i * __OSL_WIDTH + lane],
                        d > 0 ? "the derivatives of " : "", USTR(symbolname),
                        USTR(sourcefile), sourceline, USTR(opname),
                        lane.value());
                    // continue checking all data lanes, and all components
                    // for that matter, we want to find all issues, not just
                    // the 1st, right?
                    //return;
                }
            });
        }
    }
}



// Wide vals + mask + varying index
OSL_BATCHOP void
__OSL_MASKED_OP1(naninf_check_offset,
                 Wi)(int mask_value, int ncomps, const void* vals_,
                     int has_derivs, void* bsg_, ustring_pod sourcefile,
                     int sourceline, ustring_pod symbolname,
                     const void* wide_offsets_ptr, int nchecks,
                     ustring_pod opname)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;
    Wide<const int> wOffsets(wide_offsets_ptr);
    const Mask mask(mask_value);

    const float* vals = (const float*)vals_;
    for (int d = 0; d < (has_derivs ? 3 : 1); ++d) {
        mask.foreach ([=](ActiveLane lane) -> void {
            int firstcheck = wOffsets[lane];
            for (int c = firstcheck, e = c + nchecks; c < e; ++c) {
                int i = d * ncomps + c;
                if (!OIIO::isfinite(vals[i * __OSL_WIDTH + lane])) {
                    ctx->errorfmt(
                        "Detected {} value in {}{} at {}:{} (op {}) batch lane:{}",
                        vals[i * __OSL_WIDTH + lane],
                        d > 0 ? "the derivatives of " : "", USTR(symbolname),
                        USTR(sourcefile), sourceline, USTR(opname),
                        lane.value());
                    // continue checking all data lanes, and all components
                    // for that matter, we want to find all issues, not just
                    // the 1st, right?
                    //return;
                }
            }
        });
    }
}



// Many parameters, but the 2 parameter used in the function name
// correspond to:  "vals" and "firstcheck"
OSL_BATCHOP void
__OSL_OP2(uninit_check_values_offset, X,
          i)(long long typedesc_, void* vals_, void* bsg_,
             ustring_pod sourcefile, int sourceline, ustring_pod groupname_,
             int layer, ustring_pod layername_, ustring_pod shadername,
             int opnum, ustring_pod opname, int argnum, ustring_pod symbolname,
             int firstcheck, int nchecks)
{
    TypeDesc typedesc   = TYPEDESC(typedesc_);
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;
    bool uninit         = false;
    if (typedesc.basetype == TypeDesc::FLOAT) {
        float* vals = (float*)vals_;
        for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
            if (!OIIO::isfinite(vals[c])) {
                uninit  = true;
                vals[c] = 0;
            }
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int* vals = (int*)vals_;
        for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
            if (vals[c] == std::numeric_limits<int>::min()) {
                uninit  = true;
                vals[c] = 0;
            }
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring* vals = (ustring*)vals_;
        for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
            if (vals[c] == Strings::uninitialized_string) {
                uninit  = true;
                vals[c] = ustring();
            }
    }
    if (uninit) {
        ustringrep groupname = USTR(groupname_);
        ustringrep layername = USTR(layername_);
        ctx->errorfmt(
            "Detected possible use of uninitialized value in {} {} at {}:{} (group {}, layer {} {}, shader {}, op {} '{}', arg {})",
            typedesc, symbolname, sourcefile, sourceline,
            groupname.empty() ? "<unnamed group>" : groupname.c_str(), layer,
            layername.empty() ? "<unnamed layer>" : layername.c_str(),
            shadername, opnum, opname, argnum);
    }
}



// Many parameters, but the 2 parameter used in the function name
// correspond to:  "vals" and "firstcheck"
OSL_BATCHOP void
__OSL_MASKED_OP2(uninit_check_values_offset, WX,
                 i)(int mask_value, long long typedesc_, void* vals_,
                    void* bsg_, ustring_pod sourcefile, int sourceline,
                    ustring_pod groupname_, int layer, ustring_pod layername_,
                    ustring_pod shadername, int opnum, ustring_pod opname,
                    int argnum, ustring_pod symbolname, int firstcheck,
                    int nchecks)
{
    TypeDesc typedesc   = TYPEDESC(typedesc_);
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;
    const Mask mask(mask_value);

    Mask lanes_uninit(false);

    if (typedesc.basetype == TypeDesc::FLOAT) {
        float* vals = (float*)vals_;
        for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
            mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
                if (!OIIO::isfinite(vals[c * __OSL_WIDTH + lane])) {
                    lanes_uninit.set_on(lane);
                    vals[c * __OSL_WIDTH + lane] = 0;
                }
            });
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int* vals = (int*)vals_;
        for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
            mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
                if (vals[c * __OSL_WIDTH + lane]
                    == std::numeric_limits<int>::min()) {
                    lanes_uninit.set_on(lane);
                    vals[c * __OSL_WIDTH + lane] = 0;
                }
            });
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring* vals = (ustring*)vals_;
        for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
            mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
                if (vals[c * __OSL_WIDTH + lane]
                    == Strings::uninitialized_string) {
                    lanes_uninit.set_on(lane);
                    vals[c * __OSL_WIDTH + lane] = ustring();
                }
            });
    }
    if (lanes_uninit.any_on()) {
        ustringrep groupname = USTR(groupname_);
        ustringrep layername = USTR(layername_);
        ctx->errorfmt(
            "Detected possible use of uninitialized value in {} {} at {}:{} (group {}, layer {} {}, shader {}, op {} '{}', arg {}) for lanes({:x}) of batch",
            typedesc, symbolname, sourcefile, sourceline,
            groupname.empty() ? "<unnamed group>" : groupname.c_str(), layer,
            layername.empty() ? "<unnamed layer>" : layername.c_str(),
            shadername, opnum, opname, argnum, lanes_uninit.value());
    }
}



// Many parameters, but the 2 parameter used in the function name
// correspond to:  "vals" and "wide_offsets_ptr"
OSL_BATCHOP void
__OSL_MASKED_OP2(uninit_check_values_offset, X,
                 Wi)(int mask_value, long long typedesc_, void* vals_,
                     void* bsg_, ustring_pod sourcefile, int sourceline,
                     ustring_pod groupname_, int layer, ustring_pod layername_,
                     ustring_pod shadername, int opnum, ustring_pod opname,
                     int argnum, ustring_pod symbolname,
                     const void* wide_offsets_ptr, int nchecks)
{
    TypeDesc typedesc   = TYPEDESC(typedesc_);
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;
    Wide<const int> wOffsets(wide_offsets_ptr);
    const Mask mask(mask_value);
    //std::cout << "osl_uninit_check_u_values_w16_offset_masked="<< mask_value << std::endl;
    Mask lanes_uninit(false);
    if (typedesc.basetype == TypeDesc::FLOAT) {
        float* vals = (float*)vals_;
        mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
            int firstcheck = wOffsets[lane];
            for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
                if (!OIIO::isfinite(vals[c])) {
                    lanes_uninit.set_on(lane);
                    vals[c] = 0;
                }
        });
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int* vals = (int*)vals_;
        mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
            int firstcheck = wOffsets[lane];
            for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
                if (vals[c] == std::numeric_limits<int>::min()) {
                    lanes_uninit.set_on(lane);
                    vals[c] = 0;
                }
        });
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring* vals = (ustring*)vals_;
        mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
            int firstcheck = wOffsets[lane];
            for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
                if (vals[c] == Strings::uninitialized_string) {
                    lanes_uninit.set_on(lane);
                    vals[c] = ustring();
                }
        });
    }

    if (lanes_uninit.any_on()) {
        ustringrep groupname = USTR(groupname_);
        ustringrep layername = USTR(layername_);
        ctx->errorfmt(
            "Detected possible use of uninitialized value in {} {} at {}:{} (group {}, layer {} {}, shader {}, op {} '{}', arg {}) for lanes({:x}) of batch",
            typedesc, symbolname, sourcefile, sourceline,
            groupname.empty() ? "<unnamed group>" : groupname.c_str(), layer,
            layername.empty() ? "<unnamed layer>" : layername.c_str(),
            shadername, opnum, opname, argnum, lanes_uninit.value());
    }
}



// Many parameters, but the 2 parameter used in the function name
// correspond to:  "vals" and "wide_offsets_ptr"
OSL_BATCHOP void
__OSL_MASKED_OP2(uninit_check_values_offset, WX,
                 Wi)(int mask_value, long long typedesc_, void* vals_,
                     void* bsg_, ustring_pod sourcefile, int sourceline,
                     ustring_pod groupname_, int layer, ustring_pod layername_,
                     ustring_pod shadername, int opnum, ustring_pod opname,
                     int argnum, ustring_pod symbolname,
                     const void* wide_offsets_ptr, int nchecks)
{
    TypeDesc typedesc   = TYPEDESC(typedesc_);
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;
    Wide<const int> wOffsets(wide_offsets_ptr);
    const Mask mask(mask_value);
    //std::cout << "osl_uninit_check_w16_values_w16_offset_masked="<< mask_value << std::endl;
    Mask lanes_uninit(false);
    if (typedesc.basetype == TypeDesc::FLOAT) {
        float* vals = (float*)vals_;
        mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
            int firstcheck = wOffsets[lane];
            for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
                if (!OIIO::isfinite(vals[c * __OSL_WIDTH + lane])) {
                    lanes_uninit.set_on(lane);
                    vals[c * __OSL_WIDTH + lane] = 0;
                }
        });
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int* vals = (int*)vals_;
        mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
            int firstcheck = wOffsets[lane];
            for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
                if (vals[c * __OSL_WIDTH + lane]
                    == std::numeric_limits<int>::min()) {
                    lanes_uninit.set_on(lane);
                    vals[c * __OSL_WIDTH + lane] = 0;
                }
        });
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring* vals = (ustring*)vals_;
        mask.foreach ([=, &lanes_uninit](ActiveLane lane) -> void {
            int firstcheck = wOffsets[lane];
            for (int c = firstcheck, e = firstcheck + nchecks; c < e; ++c)
                if (vals[c * __OSL_WIDTH + lane]
                    == Strings::uninitialized_string) {
                    lanes_uninit.set_on(lane);
                    vals[c * __OSL_WIDTH + lane] = ustring();
                }
        });
    }

    if (lanes_uninit.any_on()) {
        ustringrep groupname = USTR(groupname_);
        ustringrep layername = USTR(layername_);
        ctx->errorfmt(
            "Detected possible use of uninitialized value in {} {} at {}:{} (group {}, layer {} {}, shader {}, op {} '{}', arg {}) for lanes({:x}) of batch",
            typedesc, symbolname, sourcefile, sourceline,
            groupname.empty() ? "<unnamed group>" : groupname.c_str(), layer,
            layername.empty() ? "<unnamed layer>" : layername.c_str(),
            shadername, opnum, opname, argnum, lanes_uninit.value());
    }
}



OSL_BATCHOP int
__OSL_OP(range_check)(int indexvalue, int length, ustring_pod symname,
                      void* bsg_, ustring_pod sourcefile, int sourceline,
                      ustring_pod groupname_, int layer, ustring_pod layername_,
                      ustring_pod shadername)
{
    if (indexvalue < 0 || indexvalue >= length) {
        ustringrep groupname = USTR(groupname_);
        ustringrep layername = USTR(layername_);
        auto* bsg            = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
        ShadingContext* ctx  = bsg->uniform.context;
        ctx->errorfmt("Index [{}] out of range {}[0..{}]: {}:{}"
                      " (group {}, layer {} {}, shader {})",
                      indexvalue, USTR(symname), length - 1, USTR(sourcefile),
                      sourceline,
                      groupname.empty() ? "<unnamed group>" : groupname.c_str(),
                      layer,
                      layername.empty() ? "<unnamed layer>" : layername.c_str(),
                      USTR(shadername));
        if (indexvalue >= length)
            indexvalue = length - 1;
        else
            indexvalue = 0;
    }
    return indexvalue;
}



OSL_BATCHOP void
__OSL_MASKED_OP(range_check)(void* wide_indexvalue, unsigned int mask_value,
                             int length, ustring_pod symname, void* bsg_,
                             ustring_pod sourcefile, int sourceline,
                             ustring_pod groupname_, int layer,
                             ustring_pod layername_, ustring_pod shadername)
{
    ustringrep groupname = USTR(groupname_);
    ustringrep layername = USTR(layername_);
    auto* bsg            = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    Masked<int> wIndexValue(wide_indexvalue, Mask(mask_value));
    wIndexValue.mask().foreach ([=](ActiveLane lane) -> void {
        int indexvalue = wIndexValue[lane];
        if (indexvalue < 0 || indexvalue >= length) {
            ShadingContext* ctx = bsg->uniform.context;
            ctx->errorfmt(
                "Index [{}] out of range {}[0..{}]: {}:{} (group {}, layer {} {}, shader {})",
                indexvalue, USTR(symname), length - 1, USTR(sourcefile),
                sourceline,
                groupname.empty() ? "<unnamed group>" : groupname.c_str(),
                layer,
                layername.empty() ? "<unnamed layer>" : layername.c_str(),
                USTR(shadername));
            if (indexvalue >= length)
                indexvalue = length - 1;
            else
                indexvalue = 0;
            // modify index value so it is not out of bounds
            wIndexValue[lane] = indexvalue;
        }
    });
}



OSL_BATCHOP int
__OSL_OP1(get_attribute, s)(void* bsg_, int dest_derivs, ustring_pod obj_name_,
                            ustring_pod attr_name_, int array_lookup, int index,
                            const void* attr_type, void* wide_attr_dest,
                            int mask_)
{
    Mask mask(mask_);
    ASSERT(mask.any_on());

    auto* bsg            = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ustringrep obj_name  = USTR(obj_name_);
    ustringrep attr_name = USTR(attr_name_);

    // Ignoring m_next_failed_attrib cache for now,
    // might be faster
    auto* renderer = bsg->uniform.context->batched<__OSL_WIDTH>().renderer();

    MaskedData dest(*(const TypeDesc*)attr_type, dest_derivs, mask,
                    wide_attr_dest);
    Mask success;
    if (array_lookup) {
        success = renderer->get_array_attribute(bsg, obj_name, attr_name, index,
                                                dest);
    } else {
        success = renderer->get_attribute(bsg, obj_name, attr_name, dest);
    }
    return success.value();
}



OSL_BATCHOP int
__OSL_MASKED_OP1(get_attribute,
                 Ws)(void* bsg_, int dest_derivs, ustring_pod obj_name_,
                     ustring_pod* wattr_name_, int array_lookup, int index,
                     const void* attr_type, void* wide_attr_dest, int mask_)
{
    Mask mask(mask_);
    ASSERT(mask.any_on());

    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ustringrep obj_name = USTR(obj_name_);
    Wide<const ustringrep> wAttrName(wattr_name_);
    auto* renderer = bsg->uniform.context->batched<__OSL_WIDTH>().renderer();

    Mask retVal(false);

    // We have a varying attribute name.
    // Lets find all the lanes with the same values and
    // make a call for each unique attr_name
    foreach_unique(
        wAttrName, mask,
        [=, &retVal](ustring attr_name, Mask matching_lanes) -> void {
            //        Mask lanesPopulated = bsg->uniform.context->batched<__OSL_WIDTH>().osl_get_attribute(bsg, bsg->uniform.objdata,
            //                                                           dest_derivs, obj_name, attr_name,
            //                                                           array_lookup, index,
            //                                                           *(const TypeDesc *)attr_type,
            //                                                           wide_attr_dest, matching_lanes);
            // Ignoring m_next_failed_attrib cache for now,
            // might be faster
            MaskedData dest(*(const TypeDesc*)attr_type, dest_derivs,
                            matching_lanes, wide_attr_dest);
            Mask lanesPopulated;
            if (array_lookup) {
                lanesPopulated = renderer->get_array_attribute(bsg, obj_name,
                                                               attr_name, index,
                                                               dest);
            } else {
                lanesPopulated = renderer->get_attribute(bsg, obj_name,
                                                         attr_name, dest);
            }
            retVal |= lanesPopulated;
        });

    return retVal.value();
}



OSL_BATCHOP bool
__OSL_OP(get_attribute_uniform)(void* bsg_, int dest_derivs,
                                ustring_pod obj_name_, ustring_pod attr_name_,
                                int array_lookup, int index,
                                const void* attr_type, void* attr_dest)
{
    auto* bsg            = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ustringrep obj_name  = USTR(obj_name_);
    ustringrep attr_name = USTR(attr_name_);

    auto* renderer = bsg->uniform.context->batched<__OSL_WIDTH>().renderer();

    RefData dest(*(const TypeDesc*)attr_type, dest_derivs, attr_dest);

    bool success;
    if (array_lookup) {
        success = renderer->get_array_attribute_uniform(bsg, obj_name,
                                                        attr_name, index, dest);
    } else {
        success = renderer->get_attribute_uniform(bsg, obj_name, attr_name,
                                                  dest);
    }

    return success;
}


OSL_BATCHOP int
__OSL_OP(bind_interpolated_param)(void* bsg_, ustring_pod name, long long type,
                                  int userdata_has_derivs, void* userdata_data,
                                  int symbol_has_derivs, void* symbol_data,
                                  int symbol_data_size,
                                  unsigned int* userdata_initialized,
                                  int userdata_index, unsigned int mask_value)
{
    // Top bit indicate if we have checked for user data yet or not
    // the bottom half is a mask of which lanes successfully retrieved
    // user data
    int status = (*userdata_initialized) >> 31;
    if (status == 0) {
        // First time retrieving this userdata
        auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
        MaskedData userDest(TYPEDESC(type), userdata_has_derivs,
                            Mask(mask_value), userdata_data);
        Mask foundUserData = bsg->uniform.renderer->batched(WidthTag())
                                 ->get_userdata(USTR(name), bsg, userDest);

        // print("Binding {} {} : index {}, ok = {}\n", name,
        //       TYPEDESC(type).c_str(),userdata_index, foundUserData.value());

        *userdata_initialized = (1 << 31) | foundUserData.value();
        bsg->uniform.context->incr_get_userdata_calls();
    }
    OSL_DASSERT((*userdata_initialized) >> 31 == 1);
    Mask foundUserData(*userdata_initialized & 0x7FFFFFFF);
    if (foundUserData.any_on()) {
        // If userdata was present, copy it to the shader variable
        // Don't bother masking as any lanes without user data
        // will be overwritten by init ops or by default value
        memcpy(symbol_data, userdata_data, symbol_data_size);
    }

    return foundUserData.value();
}


// Asked if the raytype includes a bit pattern.
OSL_BATCHOP int
__OSL_OP(raytype_bit)(void* bsg_, int bit)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    return (bsg->uniform.raytype & bit) != 0;
}



// Asked if the raytype is a name we can't know until mid-shader.
OSL_BATCHOP int
__OSL_OP(raytype_name)(void* bsg_, ustring_pod name)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    int bit   = bsg->uniform.context->shadingsys().raytype_bit(USTR(name));
    return (bsg->uniform.raytype & bit) != 0;
}



OSL_BATCHOP void
__OSL_MASKED_OP(raytype_name)(void* bsg_, void* r_, ustringrep* name_,
                              unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    Wide<const ustringrep> wname(name_);
    Mask mask(mask_value);

    foreach_unique(wname, mask, [=](ustring name, Mask matching_lanes) -> void {
        int bit = bsg->uniform.context->shadingsys().raytype_bit(name);
        int ray_is_named_type = ((bsg->uniform.raytype & bit) != 0);
        Masked<int> wr(r_, matching_lanes);
        assign_all(wr, ray_is_named_type);
    });
}

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
