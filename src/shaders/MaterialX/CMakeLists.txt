# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

file (GLOB mx_shader_headers "*.h")
file (GLOB shader_source "*.mx")

# FIXME -- is this really necessary?
if (${CMAKE_GENERATOR} MATCHES "(Visual Studio.*)")
    # Work around visual studio outputting oslc.exe in a subfolder
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../../oslc/oslc"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/../../oslc/${CMAKE_BUILD_TYPE}/oslc.exe" "${CMAKE_CURRENT_BINARY_DIR}/../oslc/"
        DEPENDS oslc)
endif ()


# Macro to compile a shader with oslc, with MX type substitution. Syntax is:
#   mx_oslc_compile (MXTEMPLATE mx_template_file
#                    SHADERNAME shader_name_no_extension_or_type
#                    TYPE substitition_type
#                 [ OTHERTYPE substitition_type ]
#                 [ DEPENDS list_of_dependencies ]
#                 [ INCLUDE_DIRS list_of_include_dirs_for_oslc ]
#                 [ DEFINES list_of_extra_definitions_for_oslc ]
#                 [ OSL_FILE optional_osl_filename_override ]
#                 [ OSO_FILE optional_oso_filename_override ]
#                 [ OSO_LIST list_to_append_oso_filename ]
#                 [ GEN_OSL_LIST list_to_append_generated_osl_filename ]
#                 )
# This is based on the oslc_compile macro found in src/shaders, but we've
# modified it to do the type substitution that MaterialX needs.
macro (mx_oslc_compile)
    cmake_parse_arguments (_shader ""
                           "MXTEMPLATE;OSL_FILE;OSO_FILE;OSO_LIST;TYPE;OTHERTYPE;SHADERNAME;GEN_OSL_LIST"
                           "DEPENDS;INCLUDE_DIRS;DEFINES" ${ARGN})
    # ^^ syntax: prefix options one-arg-keywords multi-arg-keywords args
    # set (mxfile "${CMAKE_SOURCE_DIR}/src/shaders/MaterialX/${_shader_MXTEMPLATE}")
    set (mxfile "${_shader_MXTEMPLATE}")
    set (oslfile ${_shader_OSL_FILE})
    set (osofile ${_shader_OSO_FILE})
    set (stdosl_header "${CMAKE_SOURCE_DIR}/src/shaders/stdosl.h")
    set (oslc_args -q ${_shader_DEFINES} "-I${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_incdir ${_shader_INCLUDE_DIRS})
        list (APPEND oslc_args "-I${_incdir}")
    endforeach ()
    list (APPEND oslc_args "-I${CMAKE_SOURCE_DIR}/src/shaders")
    string (TOLOWER ${_shader_TYPE} osltype)
    if (NOT _shader_OTHERTYPE)
        set (_shader_OTHERTYPE "none")
    endif ()
    if (${_shader_OTHERTYPE} STREQUAL "same")
        set (_shader_OTHERTYPE ${_shader_TYPE})
    endif ()
    if (_shader_OSL_FILE)
        set (oslfile ${_shader_OSL_FILE})
    else ()
        if (NOT ${_shader_OTHERTYPE} STREQUAL "none")
            set (oslfile "${CMAKE_CURRENT_BINARY_DIR}/${_shader_SHADERNAME}_${osltype}_${_shader_OTHERTYPE}.osl")
        else ()
            set (oslfile "${CMAKE_CURRENT_BINARY_DIR}/${_shader_SHADERNAME}_${osltype}.osl")
        endif ()
        string (REGEX REPLACE "osl$" "oso" osofile ${oslfile})
    endif ()
    if (VERBOSE)
        get_filename_component(mxfile_justname ${mxfile} NAME)
        get_filename_component(oslfile_justname ${oslfile} NAME)
        get_filename_component(osofile_justname ${osofile} NAME)
        # message (STATUS "oslc will make ${mxfile_justname} -> ${oslfile_justname} -> ${osofile_justname}")
    endif ()
    add_custom_command (OUTPUT ${osofile}
        COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/build_materialX_osl.py"
                       -s "${_shader_SHADERNAME}"
                       -m "${mxfile}"
                       -o "${oslfile}"
                       -t "${osltype}"
                       --othertypes "${_shader_OTHERTYPE}"
        COMMAND oslc ${oslc_args} "${oslfile}" -o "${osofile}"
        MAIN_DEPENDENCY ${mxfile}
        BYPRODUCTS ${oslfile}
        DEPENDS ${_shader_DEPENDS} "${stdosl_header}" oslc "${CMAKE_CURRENT_SOURCE_DIR}/build_materialX_osl.py"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "oslc ${oslfile}")
    if (_shader_OSO_LIST)
        list (APPEND ${_shader_OSO_LIST} ${osofile})
    endif ()
    if (_shader_GEN_OSL_LIST)
        list (APPEND ${_shader_GEN_OSL_LIST} ${oslfile})
    endif ()
endmacro ()


# Macro to compile a bunch of mx templates, each in several type flavors.
# Syntax is:
#   make_mx_flavors (mx_template_source_list
#                    TYPES type_list
#                    [ OTHERTYPES other_type_list ] )
macro (make_mx_flavors)
    cmake_parse_arguments (mxflav "" "" "TYPES;OTHERTYPES" ${ARGN})
    # ^^ syntax: prefix options one-arg-keywords multi-arg-keywords args
    if (NOT mxflav_OTHERTYPES)
        set (mxflav_OTHERTYPES "none")
    endif ()
    # Find out if "same" is in the list. FIXME: When cmake minimum >= 3.3,
    # we can just use ("same" IN_LIST mxflav_OTHERTYPES) in the test before
    # the 'continue' statement.
    list (FIND mxflav_OTHERTYPES "same" same_index)
    foreach (oslsrc ${mxflav_UNPARSED_ARGUMENTS})
        get_filename_component ( oslsrc_we ${oslsrc} NAME_WE )
        foreach (osltype ${mxflav_TYPES})
            foreach (oslothertype ${mxflav_OTHERTYPES})
                if (${same_index} GREATER -1 AND ${oslothertype} STREQUAL ${osltype})
                    # Trick to make sure that we don't doubly-count 'same'
                    # and the first type. As an example of when this comes
                    # up, when OTHERTYPES is 'same float' and TYPES also
                    # contains 'float'.
                    continue()
                endif ()
                if (${oslothertype} STREQUAL "same")
                    set (oslothertype ${osltype})
                endif ()
                if (${oslothertype} STREQUAL "none"
                    #OR ${oslothertype} STREQUAL ${osltype}
                    )
                    set (_shadername "${oslsrc_we}_${osltype}")
                else ()
                    set (_shadername "${oslsrc_we}_${osltype}_${oslothertype}")
                endif ()
                set (oslfile "${CMAKE_CURRENT_BINARY_DIR}/${_shadername}.osl")
                set (osofile "${CMAKE_CURRENT_BINARY_DIR}/${_shadername}.oso")
                set (mx_template_file "${CMAKE_CURRENT_SOURCE_DIR}/${oslsrc}")
                mx_oslc_compile (MXTEMPLATE ${mx_template_file}
                                 SHADERNAME ${oslsrc_we}
                                 TYPE ${osltype}
                                 OTHERTYPE ${oslothertype}
                                 OSL_FILE ${oslfile}
                                 OSO_FILE ${osofile}
                                 DEPENDS ${mx_shader_headers}
                                 INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                                              ${CMAKE_SOURCE_DIR}/src/shaders
                                 OSO_LIST mx_shader_objs
                                 GEN_OSL_LIST mx_shader_osls
                                 )
            endforeach ()
        endforeach ()
    endforeach ()
endmacro()


# The MaterialX shader zoo is very complicated, many of our shader templates
# need to be compiled separately for each of many types. But not all the
# same types for all shaders. Try to organize this as cleanly as possible
# below.

# MX shaders that come in flavors for all color/vector types and float.
make_mx_flavors (
            mx_absval.mx mx_acos.mx mx_asin.mx
            mx_blur.mx mx_ceil.mx mx_compare.mx
            mx_constant.mx mx_cos.mx mx_dot.mx
            mx_exp.mx mx_floor.mx mx_fractal3d.mx mx_geomattrvalue.mx
            mx_image.mx mx_ln.mx mx_mix.mx
            mx_noise2d.mx mx_noise3d.mx
            mx_ramp4.mx mx_ramplr.mx mx_ramptb.mx
            mx_sign.mx mx_sin.mx mx_splitlr.mx
            mx_splittb.mx mx_sqrt.mx
            mx_switch.mx mx_swizzle_color.mx mx_swizzle_color2.mx
            mx_swizzle_color4.mx mx_swizzle_vector.mx mx_swizzle_vector2.mx
            mx_swizzle_vector4.mx mx_tan.mx mx_tiledimage.mx
            mx_triplanarprojection.mx
    TYPES   float color color2 color4 vector vector2 vector4)

# MX shaders that are templated on two types: colorN/vectorN/float,
# versus themselves and float.
make_mx_flavors (
            mx_atan2.mx mx_clamp.mx mx_contrast.mx mx_invert.mx
            mx_max.mx mx_min.mx mx_modulo.mx mx_power.mx mx_remap.mx
            mx_smoothstep.mx
    TYPES      float color color2 color4 vector vector2 vector4
    OTHERTYPES same float)

# MX basic math operate on two elements of the same type, or type and float
make_mx_flavors (
    mx_add.mx mx_divide.mx mx_multiply.mx mx_subtract.mx
    TYPES      float color color2 color4 vector vector2 vector4 matrix33 matrix44
    OTHERTYPES same float)
make_mx_flavors (mx_add.mx      TYPES surfaceshader OTHERTYPES same)
make_mx_flavors (mx_multiply.mx TYPES surfaceshader OTHERTYPES color float)

# MX shaders that come in flavors for all color/vector types.
make_mx_flavors (
            mx_extract.mx mx_fractal3d_fa.mx
            mx_noise2d_fa.mx
            mx_noise3d_fa.mx mx_separate.mx
            mx_swizzle_float.mx
    TYPES   color color2 color4 vector vector2 vector4)

# MX shaders that come in flavors for all color types, and float
make_mx_flavors (
            mx_burn.mx mx_dodge.mx mx_geomcolor.mx mx_inside.mx
            mx_outside.mx mx_overlay.mx mx_screen.mx
    TYPES float color color2 color4)

# All the rest of the odds and ends follow:
make_mx_flavors (mx_ambientocclusion.mx TYPES float)
make_mx_flavors (mx_bitangent.mx        TYPES vector)
make_mx_flavors (mx_cellnoise2d.mx      TYPES float)
make_mx_flavors (mx_cellnoise3d.mx      TYPES float)
make_mx_flavors (mx_constant.mx         TYPES matrix44 matrix33 string filename bool int)
make_mx_flavors (mx_crossproduct.mx     TYPES vector)
make_mx_flavors (mx_determinant.mx      TYPES matrix44 matrix33)
make_mx_flavors (mx_disjointover.mx     TYPES color2 color4)
make_mx_flavors (mx_dot.mx              TYPES matrix44 matrix33 string filename bool int surfaceshader)
make_mx_flavors (mx_dotproduct.mx       TYPES vector vector2 vector4)
make_mx_flavors (mx_frame.mx            TYPES float)
make_mx_flavors (mx_geomattrvalue.mx    TYPES int bool string)
make_mx_flavors (mx_heighttonormal.mx   TYPES vector)
make_mx_flavors (mx_hsvadjust.mx        TYPES color color4)
make_mx_flavors (mx_hsvtorgb.mx         TYPES color color4)
make_mx_flavors (mx_hueshift.mx         TYPES color color4)  # DEPRECATED in MX 1.36
make_mx_flavors (mx_in.mx               TYPES color2 color4)
make_mx_flavors (mx_luminance.mx        TYPES color color4)
make_mx_flavors (mx_magnitude.mx        TYPES vector vector2 vector4)
make_mx_flavors (mx_mask.mx             TYPES color2 color4)
make_mx_flavors (mx_matte.mx            TYPES color2 color4)
make_mx_flavors (mx_mix.mx              TYPES surfaceshader)
make_mx_flavors (mx_normal.mx           TYPES vector)
make_mx_flavors (mx_normalize.mx        TYPES vector vector2 vector4)
make_mx_flavors (mx_out.mx              TYPES color2 color4)
make_mx_flavors (mx_over.mx             TYPES color2 color4)
make_mx_flavors (mx_combine_cc.mx       TYPES color4)
make_mx_flavors (mx_combine_cf.mx       TYPES color4)
make_mx_flavors (mx_combine_vf.mx       TYPES vector4)
make_mx_flavors (mx_combine_vv.mx       TYPES vector4)
make_mx_flavors (mx_position.mx         TYPES vector)
make_mx_flavors (mx_premult.mx          TYPES color color2 color4)
make_mx_flavors (mx_rgbtohsv.mx         TYPES color color4)
make_mx_flavors (mx_rotate.mx           TYPES vector vector2)
make_mx_flavors (mx_rotate2d.mx         TYPES vector2)     # DEPRECATED in MX 1.36
make_mx_flavors (mx_saturate.mx         TYPES color color4)
make_mx_flavors (mx_scale.mx            TYPES vector vector2)
make_mx_flavors (mx_tangent.mx          TYPES vector)
make_mx_flavors (mx_texcoord.mx         TYPES vector vector2)
make_mx_flavors (mx_time.mx             TYPES float)
make_mx_flavors (mx_transformpoint.mx   TYPES vector vector4)
make_mx_flavors (mx_transformvector.mx  TYPES vector vector4)
make_mx_flavors (mx_transformnormal.mx  TYPES vector vector4)
make_mx_flavors (mx_transpose.mx        TYPES matrix44 matrix33)
make_mx_flavors (mx_unpremult.mx        TYPES color color2 color4)
make_mx_flavors (mx_matrix_invert.mx    TYPES matrix44 matrix33)

make_mx_flavors (mx_convert.mx          TYPES float
                                        OTHERTYPES color color2 color4
                                                   vector vector2 vector4)
make_mx_flavors (mx_convert.mx          TYPES color OTHERTYPES vector color4)
make_mx_flavors (mx_convert.mx          TYPES color2 OTHERTYPES vector2)
make_mx_flavors (mx_convert.mx          TYPES color4 OTHERTYPES color vector4)
make_mx_flavors (mx_convert.mx          TYPES vector OTHERTYPES color)
make_mx_flavors (mx_convert.mx          TYPES vector2 OTHERTYPES color2)
make_mx_flavors (mx_convert.mx          TYPES vector4 OTHERTYPES color4)

add_custom_target (mxshaders ALL
                   DEPENDS ${mx_shader_objs}
                   SOURCES ${shader_source} ${mx_shader_headers})

install (FILES ${mx_shader_headers} ${mx_shader_objs} ${mx_shader_osls}
         DESTINATION ${OSL_SHADER_INSTALL_DIR}/MaterialX)

