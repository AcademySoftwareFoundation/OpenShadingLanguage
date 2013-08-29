add_custom_command (OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/oslutil.h"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/oslutil.h" "${CMAKE_CURRENT_BINARY_DIR}"
        MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/oslutil.h"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")


if (${CMAKE_GENERATOR} MATCHES "(Visual Studio.*)")
    # Work around visual studio outputting oslc.exe in a subfolder
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../../oslc/oslc"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/../../oslc/${CMAKE_BUILD_TYPE}/oslc.exe" "${CMAKE_CURRENT_BINARY_DIR}/../oslc/"
        DEPENDS oslc)
endif ()

macro (osl_compile oslsrc osltype objlist headers)
    # message (STATUS "OSL_COMPILE src=${oslsrc}")
    # message (STATUS "  src ${CMAKE_CURRENT_SOURCE_DIR}")
    # message (STATUS "  bin ${CMAKE_CURRENT_BINARY_DIR}")
    get_filename_component ( oslsrc_we ${oslsrc} NAME_WE )
    set ( oslname "${oslsrc_we}_${osltype}")
    string (TOLOWER ${oslname} osolower )
    set ( osofile "${CMAKE_CURRENT_BINARY_DIR}/${osolower}.oso" )
    
    # message (STATUS "  osl output ${osofile}")

    set ( ${objlist} ${${objlist}} ${osofile} )
    # message (STATUS "  obj list now ${${objlist}}")

#    include_directories ( ${CMAKE_CURRENT_BINARY_DIR} )
#    include_directories ( ${CMAKE_CURRENT_SOURCE_DIR} )
    #if (VERBOSE)
    #    message (STATUS "cmd: ${CMAKE_CURRENT_BINARY_DIR}/../../oslc/oslc  -D${osltype} ${CMAKE_CURRENT_SOURCE_DIR}/${oslsrc} -o ${osofile}")
    #endif ()
    add_custom_command (OUTPUT ${osofile}
        COMMAND "${CMAKE_CURRENT_BINARY_DIR}/../../oslc/oslc" -D${osltype} -q ${CMAKE_CURRENT_SOURCE_DIR}/${oslsrc} -o ${osofile}
        MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${oslsrc}
        DEPENDS ${${headers}} ${CMAKE_CURRENT_SOURCE_DIR}/${oslsrc} "${CMAKE_CURRENT_BINARY_DIR}/../../oslc/oslc"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endmacro ()

file (GLOB shader_headers "*.h")
file (GLOB shader_source "*.mx")
LIST(APPEND all_types FLOAT COLOR COLOR2 COLOR4 VECTOR VECTOR2 VECTOR4)
LIST(APPEND FA_op_types COLOR COLOR2 COLOR4 VECTOR VECTOR2 VECTOR4)

foreach (_shadertype ${all_types})
    osl_compile (mx_constant.mx ${_shadertype} shader_objs shader_headers)

    osl_compile (mx_absval.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_add.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_subtract.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_multiply.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_divide.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_modulo.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_invert.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_exponent.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_contrast.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_clamp.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_remap.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_floor.mx ${_shadertype} shader_objs shader_headers)

    osl_compile (mx_geomattrvalue.mx ${_shadertype} shader_objs shader_headers)
    
    osl_compile (mx_blur.mx ${_shadertype} shader_objs shader_headers)
    
    osl_compile (mx_compare.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_switch.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_smoothstep.mx ${_shadertype} shader_objs shader_headers)

    osl_compile (mx_splitlr.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_splittb.mx ${_shadertype} shader_objs shader_headers)

    osl_compile (mx_ramplr.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_ramptb.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_ramp4.mx ${_shadertype} shader_objs shader_headers)
    
    osl_compile (mx_noise2d.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_noise3d.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_fractal3d.mx ${_shadertype} shader_objs shader_headers)
        
    osl_compile (mx_image.mx ${_shadertype} shader_objs shader_headers)
    
    osl_compile (mx_max.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_min.mx ${_shadertype} shader_objs shader_headers)

    osl_compile (mx_mix.mx ${_shadertype} shader_objs shader_headers)

    osl_compile (mx_swizzle_color.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_swizzle_color2.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_swizzle_color4.mx ${_shadertype} shader_objs shader_headers)
    
    osl_compile (mx_swizzle_vector.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_swizzle_vector2.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_swizzle_vector4.mx ${_shadertype} shader_objs shader_headers)
    
    osl_compile (mx_triplanarprojection.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_dot.mx ${_shadertype} shader_objs shader_headers)

endforeach ()

foreach (_shadertype ${FA_op_types})
    osl_compile (mx_multiply_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_divide_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_invert_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_modulo_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_exponent_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_clamp_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_add_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_subtract_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_max_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_min_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_contrast_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_remap_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_smoothstep_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_swizzle_float.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_pack.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_noise2d_fa.mx ${_shadertype} shader_objs shader_headers)
    osl_compile (mx_noise3d_fa.mx ${_shadertype} shader_objs shader_headers)    
    osl_compile (mx_fractal3d_fa.mx ${_shadertype} shader_objs shader_headers)

endforeach ()

    osl_compile (mx_constant.mx MATRIX44 shader_objs shader_headers)
    osl_compile (mx_constant.mx MATRIX33 shader_objs shader_headers)
    osl_compile (mx_constant.mx STRING shader_objs shader_headers)
    osl_compile (mx_constant.mx FILENAME shader_objs shader_headers)
    osl_compile (mx_constant.mx BOOL shader_objs shader_headers)
    osl_compile (mx_constant.mx INT shader_objs shader_headers)
    
    osl_compile (mx_dot.mx MATRIX44 shader_objs shader_headers)
    osl_compile (mx_dot.mx MATRIX33 shader_objs shader_headers)
    osl_compile (mx_dot.mx STRING shader_objs shader_headers)
    osl_compile (mx_dot.mx FILENAME shader_objs shader_headers)
    osl_compile (mx_dot.mx BOOL shader_objs shader_headers)
    osl_compile (mx_dot.mx INT shader_objs shader_headers)
    
    osl_compile (mx_geomattrvalue.mx INT shader_objs shader_headers)
    osl_compile (mx_geomattrvalue.mx BOOL shader_objs shader_headers)
    osl_compile (mx_geomattrvalue.mx STRING shader_objs shader_headers)
        
    osl_compile (mx_ambientocclusion.mx FLOAT shader_objs shader_headers)

    osl_compile (mx_position.mx VECTOR shader_objs shader_headers)
    osl_compile (mx_normal.mx VECTOR shader_objs shader_headers)
    osl_compile (mx_tangent.mx VECTOR shader_objs shader_headers)
    osl_compile (mx_bitangent.mx VECTOR shader_objs shader_headers)

    osl_compile (mx_texcoord.mx VECTOR shader_objs shader_headers)
    osl_compile (mx_texcoord.mx VECTOR2 shader_objs shader_headers)
    
    osl_compile (mx_geomcolor.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_geomcolor.mx COLOR shader_objs shader_headers)
    osl_compile (mx_geomcolor.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_geomcolor.mx COLOR4 shader_objs shader_headers)

    osl_compile (mx_normalize.mx VECTOR shader_objs shader_headers)
    osl_compile (mx_normalize.mx VECTOR2 shader_objs shader_headers)
    osl_compile (mx_normalize.mx VECTOR4 shader_objs shader_headers)

    osl_compile (mx_magnitude.mx VECTOR shader_objs shader_headers)
    osl_compile (mx_magnitude.mx VECTOR2 shader_objs shader_headers)
    osl_compile (mx_magnitude.mx VECTOR4 shader_objs shader_headers)
    
    osl_compile (mx_rotate2d.mx VECTOR2 shader_objs shader_headers)

    osl_compile (mx_frame.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_time.mx FLOAT shader_objs shader_headers)

    osl_compile (mx_hueshift.mx COLOR shader_objs shader_headers)
    osl_compile (mx_hueshift.mx COLOR4 shader_objs shader_headers)
    
    osl_compile (mx_saturate.mx COLOR shader_objs shader_headers)
    osl_compile (mx_saturate.mx COLOR4 shader_objs shader_headers)
    
    osl_compile (mx_luminance.mx COLOR shader_objs shader_headers)
    osl_compile (mx_luminance.mx COLOR4 shader_objs shader_headers)

    osl_compile (mx_premult.mx COLOR shader_objs shader_headers)
    osl_compile (mx_premult.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_premult.mx COLOR4 shader_objs shader_headers)

    osl_compile (mx_unpremult.mx COLOR shader_objs shader_headers)
    osl_compile (mx_unpremult.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_unpremult.mx COLOR4 shader_objs shader_headers)
    
    osl_compile (mx_dotproduct.mx VECTOR shader_objs shader_headers)
    osl_compile (mx_dotproduct.mx VECTOR2 shader_objs shader_headers)
    osl_compile (mx_dotproduct.mx VECTOR4 shader_objs shader_headers)

    osl_compile (mx_burn.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_burn.mx COLOR shader_objs shader_headers)
    osl_compile (mx_burn.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_burn.mx COLOR4 shader_objs shader_headers)

    osl_compile (mx_dodge.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_dodge.mx COLOR shader_objs shader_headers)
    osl_compile (mx_dodge.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_dodge.mx COLOR4 shader_objs shader_headers)
    
    osl_compile (mx_screen.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_screen.mx COLOR shader_objs shader_headers)
    osl_compile (mx_screen.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_screen.mx COLOR4 shader_objs shader_headers)
    
    osl_compile (mx_overlay.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_overlay.mx COLOR shader_objs shader_headers)
    osl_compile (mx_overlay.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_overlay.mx COLOR4 shader_objs shader_headers)    
    
    osl_compile (mx_crossproduct.mx VECTOR shader_objs shader_headers)
    
    osl_compile (mx_disjointover.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_disjointover.mx COLOR4 shader_objs shader_headers)

    osl_compile (mx_in.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_in.mx COLOR4 shader_objs shader_headers)
    
    osl_compile (mx_mask.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_mask.mx COLOR4 shader_objs shader_headers)    
    
    osl_compile (mx_matte.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_matte.mx COLOR4 shader_objs shader_headers)    
    
    osl_compile (mx_out.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_out.mx COLOR4 shader_objs shader_headers)    
    
    osl_compile (mx_over.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_over.mx COLOR4 shader_objs shader_headers)
    
    osl_compile (mx_inside.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_inside.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_inside.mx COLOR shader_objs shader_headers)
    osl_compile (mx_inside.mx COLOR4 shader_objs shader_headers)
    
    osl_compile (mx_outside.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_outside.mx COLOR2 shader_objs shader_headers)
    osl_compile (mx_outside.mx COLOR shader_objs shader_headers)
    osl_compile (mx_outside.mx COLOR4 shader_objs shader_headers)
        
    osl_compile (mx_pack_cf.mx COLOR4 shader_objs shader_headers)
    osl_compile (mx_pack_cc.mx COLOR4 shader_objs shader_headers)

    osl_compile (mx_pack_vf.mx VECTOR4 shader_objs shader_headers)
    osl_compile (mx_pack_vv.mx VECTOR4 shader_objs shader_headers)

    osl_compile (mx_cellnoise2d.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_cellnoise3d.mx FLOAT shader_objs shader_headers)
    
    osl_compile (mx_heighttonormal.mx VECTOR shader_objs shader_headers)

    osl_compile (mx_scale.mx VECTOR shader_objs shader_headers)
    osl_compile (mx_scale.mx VECTOR2 shader_objs shader_headers)
    
    osl_compile (mx_add.mx SURFACESHADER shader_objs shader_headers)
    osl_compile (mx_mult_surfaceshader.mx COLOR shader_objs shader_headers)
    osl_compile (mx_mult_surfaceshader.mx FLOAT shader_objs shader_headers)
    osl_compile (mx_mix.mx SURFACESHADER shader_objs shader_headers)

add_custom_target (mxshaders ALL
                   DEPENDS ${shader_objs}
                   SOURCES ${shader_source} ${shader_headers})

install (FILES ${shader_headers} ${shader_objs}
         DESTINATION shaders/MaterialX)

