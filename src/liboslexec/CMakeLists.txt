SET ( liboslexec_srcs
          shadingsys.cpp closure.cpp
          dictionary.cpp
          context.cpp instance.cpp 
          loadshader.cpp master.cpp 
          opcolor.cpp 
          opmessage.cpp
          opnoise.cpp simplexnoise.cpp gabornoise.cpp
          opspline.cpp opstring.cpp
          oslexec.cpp osoreader.cpp
          pointcloud.cpp rendservices.cpp 
          constfold.cpp runtimeoptimize.cpp typespec.cpp
          lpexp.cpp lpeparse.cpp automata.cpp accum.cpp
          opclosure.cpp
          llvm_gen.cpp llvm_instance.cpp llvm_util.cpp
    )

# oslcomp symbols used in oslexec
if (NOT BUILDSTATIC)
    LIST(APPEND liboslexec_srcs
        ../liboslcomp/ast.cpp
        ../liboslcomp/codegen.cpp
        ../liboslcomp/oslcomp.cpp
        ../liboslcomp/symtab.cpp
        ../liboslcomp/typecheck.cpp
        )
endif ()

include_directories ( "${CMAKE_SOURCE_DIR}/src/liboslcomp" )

FILE ( GLOB exec_headers "*.h" )

FLEX_BISON ( osolex.l osogram.y oso liboslexec_srcs exec_headers )

SET ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS" )

MACRO ( LLVM_COMPILE llvm_src srclist )
    GET_FILENAME_COMPONENT ( llvmsrc_we ${llvm_src} NAME_WE )
    SET ( llvm_asm "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.s" )
    SET ( llvm_bc "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc" )
    SET ( llvm_bc_cpp "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc.cpp" )
    if (VERBOSE)
        MESSAGE (STATUS "LLVM_COMPILE in=${llvm_src}")
        MESSAGE (STATUS "LLVM_COMPILE asm=${llvm_asm}")
        MESSAGE (STATUS "LLVM_COMPILE bc=${llvm_bc}")
        MESSAGE (STATUS "LLVM_COMPILE cpp=${llvm_bc_cpp}")
    endif ()
    SET ( ${srclist} ${${srclist}} ${llvm_bc_cpp} )
    EXEC_PROGRAM ( "${LLVM_DIRECTORY}/bin/llvm-config" ARGS --cxxflags OUTPUT_VARIABLE LLVM_COMPILE_FLAGS )
    set (LLVM_COMPILE_FLAGS "${LLVM_COMPILE_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -O3 --combine")
    if (OSL_NAMESPACE)
        LIST (APPEND LLVM_COMPILE_FLAGS "-DOSL_NAMESPACE=${OSL_NAMESPACE}")
    endif ()
    if (OPENIMAGEIO_NAMESPACE)
        LIST (APPEND LLVM_COMPILE_FLAGS "-DOPENIMAGEIO_NAMESPACE=${OPENIMAGEIO_NAMESPACE}")
    endif ()
    get_property (CURRENT_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
    if (VERBOSE)
        message (STATUS "Current #defines are ${CURRENT_DEFINITIONS}")
    endif ()
    foreach (def ${CURRENT_DEFINITIONS})
      LIST (APPEND LLVM_COMPILE_FLAGS "-D${def}")
    endforeach()

    # First try looking in their build (clang++ first, then llvm-g++)
    FIND_PROGRAM(LLVM_BC_GENERATOR NAMES "clang++" "llvm-g++" PATHS "${LLVM_DIRECTORY}/bin" NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_PATH)

    if(NOT LLVM_BC_GENERATOR)
      # Wasn't in their build, look anywhere
      FIND_PROGRAM(LLVM_BC_GENERATOR NAMES clang++ llvm-g++)
    endif()

    if(NOT LLVM_BC_GENERATOR)
      message (FATAL_ERROR "You must have a valid llvm bitcode generator (either llvm-g++ or clang++) somewhere.")
    else()
      if (VERBOSE)
          message (STATUS "Using ${LLVM_BC_GENERATOR} to generate bitcode.")
      endif ()
    endif()

    # Command to turn the .cpp file into LLVM assembly language .s, into
    # LLVM bitcode .bc, then back into a C++ file with the bc embedded!
    ADD_CUSTOM_COMMAND ( OUTPUT ${llvm_bc_cpp}
      COMMAND ${LLVM_BC_GENERATOR}
      "-I${CMAKE_CURRENT_SOURCE_DIR}"
      "-I${CMAKE_SOURCE_DIR}/src/include"
      "-I${CMAKE_BINARY_DIR}/include"
      "-I${OPENIMAGEIO_INCLUDES}"
      "-I${ILMBASE_INCLUDE_DIR}"
      "-I${Boost_INCLUDE_DIRS}"
      ${LLVM_COMPILE_FLAGS}
      -O3 -S -emit-llvm -o ${llvm_asm} ${llvm_src}

      COMMAND "${LLVM_DIRECTORY}/bin/llvm-as" -f -o ${llvm_bc} ${llvm_asm}
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.bash" ${llvm_bc} ${llvm_bc_cpp}
      MAIN_DEPENDENCY ${llvm_src}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.bash"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
ENDMACRO ( )

if (USE_LLVM_BITCODE)
    LLVM_COMPILE ( llvm_ops.cpp liboslexec_srcs )
else ()
    # With MSVC/Mingw, we don't compile llvm_ops.cpp to LLVM bitcode, due
    # to clang being unable to compile MSVC C++ header files at this time.
    # Instead it is part of the regular build process.
    ADD_DEFINITIONS (-DOSL_LLVM_NO_BITCODE)
    SET (liboslexec_srcs ${liboslexec_srcs} llvm_ops.cpp)
endif ()

if (BUILDSTATIC)
    ADD_LIBRARY ( oslexec STATIC ${liboslexec_srcs} )
else ()
    ADD_LIBRARY ( oslexec SHARED ${liboslexec_srcs} )
endif ()

TARGET_LINK_LIBRARIES ( oslexec
                        ${VISIBILITY_COMMAND} ${VISIBILITY_MAP_COMMAND}
                        ${OPENIMAGEIO_LIBRARY} ${PUGIXML_LIBRARIES}
                        ${PARTIO_LIBRARIES} ${ZLIB_LIBRARIES}
                        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
                        ${LLVM_LIBRARY} ${EXTRA_OSLEXEC_LIBRARIES})
ADD_DEPENDENCIES (oslexec "${CMAKE_CURRENT_SOURCE_DIR}/liboslcexec.map")
LINK_ILMBASE ( oslexec )

if (BUILDSTATIC)
    INSTALL ( TARGETS oslexec LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
else ()
    INSTALL ( TARGETS oslexec LIBRARY DESTINATION lib )
endif ()

# Unit tests
add_executable (accum_test accum_test.cpp)
target_link_libraries ( accum_test oslexec oslcomp ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LIBRARY} ${EXTRA_OSLEXEC_LIBRARIES} )
link_ilmbase (accum_test)
add_test (unit_accum "${CMAKE_BINARY_DIR}/src/liboslexec/accum_test")
