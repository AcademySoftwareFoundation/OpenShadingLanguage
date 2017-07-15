file (GLOB mx_shader_headers "*.h")
file (GLOB shader_source "*.mx")

# FIXME -- is this really necessary?
if (${CMAKE_GENERATOR} MATCHES "(Visual Studio.*)")
    # Work around visual studio outputting oslc.exe in a subfolder
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../../oslc/oslc"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/../../oslc/${CMAKE_BUILD_TYPE}/oslc.exe" "${CMAKE_CURRENT_BINARY_DIR}/../oslc/"
        DEPENDS oslc)
endif ()


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
            set (oslname "${oslsrc_we}_${osltype}")
            string (TOLOWER ${oslname} osolower )
            set (osofile "${CMAKE_CURRENT_BINARY_DIR}/${osolower}.oso")
            # Rely on oslc_compile that was defined in src/shaders/CMakeLists.txt
            oslc_compile (OSL ${CMAKE_CURRENT_SOURCE_DIR}/${oslsrc}
                          OSO_FILE ${osofile}
                          DEFINES -D${osltype}
                          HEADERS ${mx_shader_headers}
                          INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                                       ${CMAKE_SOURCE_DIR}/src/shaders
                          OSO_LIST mx_shader_objs)
        endforeach ()
    endforeach ()
endmacro()


# The MaterialX shader zoo is very complicated, many of our shader templates
# need to be compiled separately for each of many types. But not all the
# same types for all shaders. Try to organize this as cleanly as possible
# below.

# MX shaders that come in flavors for all color/vector types, and float.
make_mx_flavors (
            mx_absval.mx mx_add.mx mx_blur.mx mx_clamp.mx mx_compare.mx
            mx_constant.mx mx_contrast.mx mx_divide.mx mx_dot.mx
            mx_exponent.mx mx_floor.mx mx_fractal3d.mx mx_geomattrvalue.mx
            mx_image.mx mx_invert.mx mx_max.mx mx_min.mx mx_mix.mx
            mx_modulo.mx mx_multiply.mx mx_noise2d.mx mx_noise3d.mx
            mx_ramp4.mx mx_ramplr.mx mx_ramptb.mx mx_remap.mx
            mx_smoothstep.mx mx_splitlr.mx mx_splittb.mx mx_subtract.mx
            mx_switch.mx mx_swizzle_color.mx mx_swizzle_color2.mx
            mx_swizzle_color4.mx mx_swizzle_vector.mx mx_swizzle_vector2.mx
            mx_swizzle_vector4.mx mx_triplanarprojection.mx
    TYPES   FLOAT COLOR COLOR2 COLOR4 VECTOR VECTOR2 VECTOR4)

# MX shaders that come in flavors for all color/vector types.
make_mx_flavors (
            mx_add_float.mx mx_clamp_float.mx mx_contrast_float.mx
            mx_divide_float.mx mx_exponent_float.mx mx_fractal3d_fa.mx
            mx_invert_float.mx mx_max_float.mx mx_min_float.mx
            mx_modulo_float.mx mx_multiply_float.mx mx_noise2d_fa.mx
            mx_noise3d_fa.mx mx_pack.mx mx_remap_float.mx
            mx_smoothstep_float.mx mx_subtract_float.mx mx_swizzle_float.mx
    TYPES   COLOR COLOR2 COLOR4 VECTOR VECTOR2 VECTOR4)

# MX shaders that come in flavors for all color types, and float
make_mx_flavors (
            mx_burn.mx mx_dodge.mx mx_geomcolor.mx mx_inside.mx
            mx_outside.mx mx_overlay.mx mx_screen.mx
    TYPES FLOAT COLOR COLOR2 COLOR4)

# All the rest of the odds and ends follow:
make_mx_flavors (mx_add.mx              TYPES SURFACESHADER)
make_mx_flavors (mx_ambientocclusion.mx TYPES FLOAT)
make_mx_flavors (mx_bitangent.mx        TYPES VECTOR)
make_mx_flavors (mx_cellnoise2d.mx      TYPES FLOAT)
make_mx_flavors (mx_cellnoise3d.mx      TYPES FLOAT)
make_mx_flavors (mx_constant.mx         TYPES MATRIX44 MATRIX33 STRING FILENAME BOOL INT)
make_mx_flavors (mx_crossproduct.mx     TYPES VECTOR)
make_mx_flavors (mx_disjointover.mx     TYPES COLOR2 COLOR4)
make_mx_flavors (mx_dot.mx              TYPES MATRIX44 MATRIX33 STRING FILENAME BOOL INT SURFACESHADER)
make_mx_flavors (mx_dotproduct.mx       TYPES VECTOR VECTOR2 VECTOR4)
make_mx_flavors (mx_frame.mx            TYPES FLOAT)
make_mx_flavors (mx_geomattrvalue.mx    TYPES INT BOOL STRING)
make_mx_flavors (mx_heighttonormal.mx   TYPES VECTOR)
make_mx_flavors (mx_hueshift.mx         TYPES COLOR COLOR4)
make_mx_flavors (mx_in.mx               TYPES COLOR2 COLOR4)
make_mx_flavors (mx_luminance.mx        TYPES COLOR COLOR4)
make_mx_flavors (mx_magnitude.mx        TYPES VECTOR VECTOR2 VECTOR4)
make_mx_flavors (mx_mask.mx             TYPES COLOR2 COLOR4)
make_mx_flavors (mx_matte.mx            TYPES COLOR2 COLOR4)
make_mx_flavors (mx_mix.mx              TYPES SURFACESHADER)
make_mx_flavors (mx_mult_surfaceshader.mx TYPES COLOR FLOAT)
make_mx_flavors (mx_normal.mx           TYPES VECTOR)
make_mx_flavors (mx_normalize.mx        TYPES VECTOR VECTOR2 VECTOR4)
make_mx_flavors (mx_out.mx              TYPES COLOR2 COLOR4)
make_mx_flavors (mx_over.mx             TYPES COLOR2 COLOR4)
make_mx_flavors (mx_pack_cc.mx          TYPES COLOR4)
make_mx_flavors (mx_pack_cf.mx          TYPES COLOR4)
make_mx_flavors (mx_pack_vf.mx          TYPES VECTOR4)
make_mx_flavors (mx_pack_vv.mx          TYPES VECTOR4)
make_mx_flavors (mx_position.mx         TYPES VECTOR)
make_mx_flavors (mx_premult.mx          TYPES COLOR COLOR2 COLOR4)
make_mx_flavors (mx_rotate2d.mx         TYPES VECTOR2)
make_mx_flavors (mx_saturate.mx         TYPES COLOR COLOR4)
make_mx_flavors (mx_scale.mx            TYPES VECTOR VECTOR2)
make_mx_flavors (mx_tangent.mx          TYPES VECTOR)
make_mx_flavors (mx_texcoord.mx         TYPES VECTOR VECTOR2)
make_mx_flavors (mx_time.mx             TYPES FLOAT)
make_mx_flavors (mx_unpremult.mx        TYPES COLOR COLOR2 COLOR4)


add_custom_target (mxshaders ALL
                   DEPENDS ${mx_shader_objs}
                   SOURCES ${shader_source} ${mx_shader_headers})

install (FILES ${mx_shader_headers} ${mx_shader_objs}
         DESTINATION shaders/MaterialX)

