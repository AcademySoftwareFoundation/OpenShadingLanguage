// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of string functions
/// such as format, concat, printf, etc.
///
/////////////////////////////////////////////////////////////////////////

//#include <OSL/rs_free_function.h>
#include <OSL/fmt_util.h>
#include <cstdarg>


#include <OpenImageIO/fmath.h>
#include <OpenImageIO/strutil.h>


#include "oslexec_pvt.h"
#include "shading_state_uniform.h"


OSL_NAMESPACE_ENTER
namespace pvt {



// Shims to convert llvm gen to rs free function C++ parameter types
// and forward on calls to re free functions.

OSL_RSOP OSL::ustringhash_pod
osl_gen_ustringhash_pod(const char* s)
{
    return USTR(s).hash();
}

OSL_RSOP void
osl_gen_errorfmt(OpaqueExecContextPtr exec_ctx,
                 OSL::ustringhash_pod fmt_specification, int32_t arg_count,
                 void* arg_types, uint32_t arg_values_size, uint8_t* arg_values)
{
    OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(
        fmt_specification);
    auto encoded_types = reinterpret_cast<const EncodedType*>(arg_types);

    rs_errorfmt(exec_ctx, rs_fmt_specification, arg_count, encoded_types,
                arg_values_size, arg_values);
}



OSL_RSOP void
osl_gen_warningfmt(OpaqueExecContextPtr exec_ctx,
                   OSL::ustringhash_pod fmt_specification, int32_t arg_count,
                   void* arg_types, uint32_t arg_values_size,
                   uint8_t* arg_values)
{
    OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(
        fmt_specification);
    auto encoded_types = reinterpret_cast<const EncodedType*>(arg_types);

    rs_warningfmt(exec_ctx, rs_fmt_specification, arg_count, encoded_types,
                  arg_values_size, arg_values);
}


OSL_RSOP void
osl_gen_printfmt(OpaqueExecContextPtr exec_ctx,
                 OSL::ustringhash_pod fmt_specification, int32_t arg_count,
                 void* arg_types, uint32_t arg_values_size, uint8_t* arg_values)
{
    OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(
        fmt_specification);
    auto encoded_types = reinterpret_cast<const EncodedType*>(arg_types);
    //auto argValues2 = reinterpret_cast<uint8_t *> (arg_values);


    rs_printfmt(exec_ctx, rs_fmt_specification, arg_count, encoded_types,
                arg_values_size,
                arg_values);  //not argValues2
}


OSL_RSOP void
osl_gen_filefmt(OpaqueExecContextPtr exec_ctx,
                OSL::ustringhash_pod filename_hash,
                OSL::ustringhash_pod fmt_specification, int32_t arg_count,
                void* arg_types, uint32_t arg_values_size, uint8_t* arg_values)
{
    OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(
        fmt_specification);
    OSL::ustringhash rs_filename = OSL::ustringhash_from(filename_hash);

    auto encoded_types = reinterpret_cast<const EncodedType*>(arg_types);
    rs_filefmt(exec_ctx, rs_filename, rs_fmt_specification, arg_count,
               encoded_types, arg_values_size, arg_values);
}

int
get_max_warnings_per_thread(OpaqueExecContextPtr oec)
{
    auto ec  = pvt::get_ec(oec);
    auto ssu = reinterpret_cast<const OSL::pvt::ShadingStateUniform*>(
        ec->shadingStateUniform);
    return ssu->m_max_warnings_per_thread;
}

}  //namespace pvt


OSL_NAMESPACE_EXIT