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
#   mx_oslc_compile (MX mx_template_file
#                    TYPE substitition_type
#                    SHADERNAME shader_name_no_extension_or_type
#                 [ DEPENDS list_of_dependencies ]
#                 [ INCLUDE_DIRS list_of_include_dirs_for_oslc ]
#                 [ DEFINES list_of_extra_definitions_for_oslc ]
#                 [ OSO_FILE optional_oso_filename_override ]
#                 [ OSO_LIST list_to_append_oso_filename ]
#                 [ GEN_OSL_LIST list_to_append_generated_osl_filename ]
#                 )
# This is based on the oslc_compile macro found in src/shaders, but we've
# modified it to do the type substitution that MaterialX needs.
macro (mx_oslc_compile)
    cmake_parse_arguments (_shader ""
                           "OSL;OSO_FILE;OSO_LIST;TYPE;SHADERNAME;GEN_OSL_LIST"
                           "DEPENDS;INCLUDE_DIRS;DEFINES" ${ARGN})
    # ^^ syntax: prefix options one-arg-keywords multi-arg-keywords args
    set (oslfile ${_shader_OSL})
    if (_shader_OSO_FILE)
        set (osofile ${_shader_OSO_FILE})
    else ()
        get_filename_component ( oslsrc_we ${_shader_OSL} NAME_WE )
        set (osofile "${CMAKE_CURRENT_BINARY_DIR}/${oslsrc_we}.oso")
    endif ()
    if (VERBOSE)
        message (STATUS "oslc will make '${oslfile}'  ->  '${osofile}'")
    endif ()
    set (stdosl_header "${CMAKE_SOURCE_DIR}/src/shaders/stdosl.h")
    set (oslc_args -q ${_shader_DEFINES} "-I${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_incdir ${_shader_INCLUDE_DIRS})
        list (APPEND oslc_args "-I${_incdir}")
    endforeach ()
    list (APPEND oslc_args "-I${CMAKE_SOURCE_DIR}/src/shaders")
    string (TOLOWER ${_shader_TYPE} osltype_lower)
    set (gen_osl_file "${CMAKE_CURRENT_BINARY_DIR}/${_shader_SHADERNAME}_${osltype_lower}.osl")
    add_custom_command (OUTPUT ${osofile}
        COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/build_materialX_osl.py" -s "${_shader_SHADERNAME}" -t "${osltype_lower}" -mx "${CMAKE_CURRENT_SOURCE_DIR}" -o "${CMAKE_CURRENT_BINARY_DIR}"
        COMMAND oslc ${oslc_args} "${gen_osl_file}" -o "${osofile}"
        MAIN_DEPENDENCY ${oslsrc}
        DEPENDS ${_shader_DEPENDS} "${stdosl_header}" oslc "${CMAKE_CURRENT_SOURCE_DIR}/build_materialX_osl.py"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "oslc ${gen_osl_file}")
    if (_shader_OSO_LIST)
        list (APPEND ${_shader_OSO_LIST} ${osofile})
    endif ()
    if (_shader_GEN_OSL_LIST)
        list (APPEND ${_shader_GEN_OSL_LIST} ${gen_osl_file})
    endif ()
endmacro ()


# Macro to compile a bunch of mx templates, each in several type flavors.
# Syntax is:
#   make_mx_flavors (mx_template_source_list
#                    TYPES type_list)
macro (make_mx_flavors)
    cmake_parse_arguments (mxflav "" "" "TYPES" ${ARGN})
    # ^^ syntax: prefix options one-arg-keywords multi-arg-keywords args
    foreach (oslsrc ${mxflav_UNPARSED_ARGUMENTS})
        get_filename_component ( oslsrc_we ${oslsrc} NAME_WE )
        foreach (osltype ${mxflav_TYPES})
            string (TOLOWER ${osltype} osltype_lower)
            set (_shadername "${oslsrc_we}_${osltype_lower}")
            set (osofile "${CMAKE_CURRENT_BINARY_DIR}/${_shadername}.oso")
            set (mx_template_file "${CMAKE_CURRENT_SOURCE_DIR}/${oslsrc}")
            mx_oslc_compile (OSL ${mx_template_file}
                          SHADERNAME ${oslsrc_we}
                          TYPE ${osltype_lower}
                          OSO_FILE ${osofile}
                          DEPENDS ${mx_shader_headers}
                          INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                                       ${CMAKE_SOURCE_DIR}/src/shaders
                          OSO_LIST mx_shader_objs
                          GEN_OSL_LIST mx_shader_osls
                          )
        endforeach ()
    endforeach ()
endmacro()


# The MaterialX shader zoo is very complicated, many of our shader templates
# need to be compiled separately for each of many types. But not all the
# same types for all shaders. Try to organize this as cleanly as possible
# below.

# MX shaders that come in flavors for all color/vector types, and float.
make_mx_flavors (
            mx_absval.mx mx_acos.mx mx_add.mx mx_asin.mx mx_atan2.mx
            mx_blur.mx mx_ceil.mx mx_clamp.mx mx_compare.mx
            mx_constant.mx mx_contrast.mx mx_cos.mx mx_dot.mx
            mx_exp.mx mx_power.mx mx_floor.mx mx_fractal3d.mx mx_geomattrvalue.mx
            mx_image.mx mx_invert.mx mx_ln.mx mx_max.mx mx_min.mx mx_mix.mx
            mx_modulo.mx mx_noise2d.mx mx_noise3d.mx
            mx_ramp4.mx mx_ramplr.mx mx_ramptb.mx mx_remap.mx
            mx_sin.mx mx_smoothstep.mx mx_splitlr.mx mx_splittb.mx mx_sqrt.mx mx_subtract.mx
            mx_switch.mx mx_swizzle_color.mx mx_swizzle_color2.mx
            mx_swizzle_color4.mx mx_swizzle_vector.mx mx_swizzle_vector2.mx
            mx_swizzle_vector4.mx mx_tan.mx mx_triplanarprojection.mx
    TYPES   float color color2 color4 vector vector2 vector4)

# MX shaders that come in flavors for all color/vector types.
make_mx_flavors (
            mx_add_float.mx mx_atan2_float.mx mx_clamp_float.mx mx_contrast_float.mx
            mx_divide_float.mx mx_power_float.mx mx_fractal3d_fa.mx
            mx_invert_float.mx mx_max_float.mx mx_min_float.mx
            mx_modulo_float.mx mx_multiply_float.mx mx_noise2d_fa.mx
            mx_noise3d_fa.mx mx_remap_float.mx
            mx_smoothstep_float.mx mx_subtract_float.mx mx_swizzle_float.mx
    TYPES   color color2 color4 vector vector2 vector4)

# MX shaders that come in flavors for all color types, and float
make_mx_flavors (
            mx_burn.mx mx_dodge.mx mx_geomcolor.mx mx_inside.mx
            mx_outside.mx mx_overlay.mx mx_screen.mx
    TYPES float color color2 color4)

# All the rest of the odds and ends follow:
make_mx_flavors (mx_add.mx              TYPES surfaceshader)
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
make_mx_flavors (mx_hueshift.mx         TYPES color color4)
make_mx_flavors (mx_in.mx               TYPES color2 color4)
make_mx_flavors (mx_luminance.mx        TYPES color color4)
make_mx_flavors (mx_magnitude.mx        TYPES vector vector2 vector4)
make_mx_flavors (mx_mask.mx             TYPES color2 color4)
make_mx_flavors (mx_matte.mx            TYPES color2 color4)
make_mx_flavors (mx_mix.mx              TYPES surfaceshader)
make_mx_flavors (mx_mult_surfaceshader.mx TYPES color float)
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
make_mx_flavors (mx_rotate2d.mx         TYPES vector2)
make_mx_flavors (mx_saturate.mx         TYPES color color4)
make_mx_flavors (mx_scale.mx            TYPES vector vector2)
make_mx_flavors (mx_tangent.mx          TYPES vector)
make_mx_flavors (mx_texcoord.mx         TYPES vector vector2)
make_mx_flavors (mx_time.mx             TYPES float)
make_mx_flavors (mx_transpose.mx        TYPES matrix44 matrix33)
make_mx_flavors (mx_unpremult.mx        TYPES color color2 color4)
make_mx_flavors (mx_matrix_invert.mx    TYPES matrix44 matrix33)
make_mx_flavors (mx_divide.mx           TYPES float color color2 color4 vector vector2 vector4 matrix44 matrix33)
make_mx_flavors (mx_multiply.mx         TYPES float color color2 color4 vector vector2 vector4 matrix44 matrix33)


add_custom_target (mxshaders ALL
                   DEPENDS ${mx_shader_objs}
                   SOURCES ${shader_source} ${mx_shader_headers})

install (FILES ${mx_shader_headers} ${mx_shader_objs} ${mx_shader_osls}
         DESTINATION shaders/MaterialX)

