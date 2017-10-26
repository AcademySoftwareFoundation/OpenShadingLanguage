MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


if (OSL_BUILD_TESTS)


macro ( MXTESTSUITE )
    foreach (_testname ${ARGN})
        SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/${_testname})
            foreach (test ${SUBDIRS})
                set(_subdir "MaterialX/${_testname}/${_test}")
                set(_env MATERIALX_OSOS=$ENV{OSL_HOME}/shaders/MaterialX)
                TESTSUITE(${test})
            endforeach ()
    endforeach ()            

endmacro ()

MXTESTSUITE(
    mx_absval
    mx_add
    mx_add_float
    mx_subtract
    mx_subtract_float
    mx_multiply
    mx_multiply_float
    mx_divide
    mx_divide_float
    mx_modulo
    mx_modulo_float
    mx_min
    mx_min_float
    mx_max
    mx_max_float    
    mx_exponent
    mx_exponent_float
)

endif (OSL_BUILD_TESTS)