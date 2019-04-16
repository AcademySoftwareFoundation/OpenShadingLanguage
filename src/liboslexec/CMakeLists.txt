SET ( liboslexec_srcs
          shadingsys.cpp closure.cpp
          dictionary.cpp
          context.cpp instance.cpp
          loadshader.cpp master.cpp
          opcolor.cpp opmatrix.cpp opmessage.cpp
          opnoise.cpp
          opspline.cpp opstring.cpp optexture.cpp
          oslexec.cpp
          pointcloud.cpp rendservices.cpp
          constfold.cpp runtimeoptimize.cpp typespec.cpp
          lpexp.cpp lpeparse.cpp automata.cpp accum.cpp
          opclosure.cpp
          shadeimage.cpp
          backendllvm.cpp
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

# oslnoise symbols used in oslexec
LIST(APPEND liboslexec_srcs
   ../liboslnoise/gabornoise.cpp
   ../liboslnoise/simplexnoise.cpp
   )

# oslquery symbols used in oslexec
if (NOT BUILDSTATIC)
    LIST(APPEND liboslexec_srcs
        ../liboslquery/oslquery.cpp
        )
endif ()

include_directories ( "${CMAKE_SOURCE_DIR}/src/liboslcomp" )

FILE ( GLOB exec_headers "*.h" )
FILE ( GLOB compiler_headers "../liboslcomp/*.h" )

FLEX_BISON ( osolex.l osogram.y oso liboslexec_srcs exec_headers )
FLEX_BISON ( ../liboslcomp/osllex.l ../liboslcomp/oslgram.y osl liboslexec_srcs compiler_headers )

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
    get_property (CURRENT_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
    if (VERBOSE)
        message (STATUS "Current #defines are ${CURRENT_DEFINITIONS}")
    endif ()
    foreach (def ${CURRENT_DEFINITIONS})
        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-D${def}")
    endforeach()
    set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} ${SIMD_COMPILE_FLAGS} ${CSTD_FLAGS} ${TOOLCHAIN_FLAGS})

    # Figure out what program we will use to make the bitcode.
    if (NOT LLVM_BC_GENERATOR)
        FIND_PROGRAM(LLVM_BC_GENERATOR NAMES "clang++" PATHS "${LLVM_DIRECTORY}/bin" NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_PATH)
    endif ()
    # If that didn't work, look anywhere
    if (NOT LLVM_BC_GENERATOR)
        # Wasn't in their build, look anywhere
        FIND_PROGRAM(LLVM_BC_GENERATOR NAMES clang++ llvm-g++)
    endif ()

    if (NOT LLVM_BC_GENERATOR)
        message (FATAL_ERROR "You must have a valid llvm bitcode generator (clang++) somewhere.")
    endif ()
    if (VERBOSE)
        message (STATUS "Using ${LLVM_BC_GENERATOR} to generate bitcode.")
    endif()

    # Fix specific problem I had on new Apple systems (e.g. Mavericks) with
    # LLVM/libc++ installed -- for some reason, LLVM 3.4 wasn't finding it,
    # so in that specific case, append another -I to point it in the right
    # direction.
    if (APPLE AND ${LLVM_BC_GENERATOR} MATCHES ".*clang.*")
        EXEC_PROGRAM ( "${LLVM_BC_GENERATOR}" ARGS --version OUTPUT_VARIABLE MY_CLANG_VERSION )
        string (REGEX REPLACE "clang version ([0-9][.][0-9]+).*" "\\1" MY_CLANG_VERSION "${MY_CLANG_VERSION}")
        if ((${MY_CLANG_VERSION} VERSION_GREATER "3.3")
              AND (EXISTS "/usr/lib/libc++.dylib")
              AND (EXISTS "/Library/Developer/CommandLineTools/usr/lib/c++/v1"))
            set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-I/Library/Developer/CommandLineTools/usr/lib/c++/v1")
        endif ()
    endif ()

    # Command to turn the .cpp file into LLVM assembly language .s, into
    # LLVM bitcode .bc, then back into a C++ file with the bc embedded!
    ADD_CUSTOM_COMMAND ( OUTPUT ${llvm_bc_cpp}
      COMMAND ${LLVM_BC_GENERATOR}
          ${LLVM_COMPILE_FLAGS}
          "-I${CMAKE_CURRENT_SOURCE_DIR}"
          "-I${CMAKE_SOURCE_DIR}/src/include"
          "-I${CMAKE_BINARY_DIR}/include"
          "-I${OPENIMAGEIO_INCLUDE_DIR}"
          "-I${ILMBASE_INCLUDE_DIR}"
          "-I${Boost_INCLUDE_DIRS}"
          -DOSL_COMPILING_TO_BITCODE=1
          -Wno-deprecated-register
          -O3 -fno-math-errno -S -emit-llvm -o ${llvm_asm} ${llvm_src}
      COMMAND "${LLVM_DIRECTORY}/bin/llvm-as" -f -o ${llvm_bc} ${llvm_asm}
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.bash" ${llvm_bc} ${llvm_bc_cpp} "osl_llvm_compiled_ops"
      MAIN_DEPENDENCY ${llvm_src}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.bash"
              ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

ENDMACRO ( )

MACRO ( CUDA_SHADEOPS_COMPILE srclist )
    # Add all of the "shadeops" sources that need to be compiled to LLVM bitcode for CUDA
    set ( shadeops_srcs
        llvm_ops.cpp
        opnoise.cpp
        opspline.cpp
        ../liboslnoise/gabornoise.cpp
        ../liboslnoise/simplexnoise.cpp
        )

    SET ( shadeops_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/shadeops_cuda.bc.cpp" )
    SET ( linked_shadeops_bc "${CMAKE_CURRENT_BINARY_DIR}/linked_shadeops.bc" )

    LIST ( APPEND ${srclist} ${shadeops_bc_cuda_cpp} )

    FOREACH ( shadeops_src ${shadeops_srcs} )
        MAKE_CUDA_BITCODE ( ${shadeops_src} "_cuda" shadeops_bc "" )
        LIST ( APPEND shadeops_bc_list ${shadeops_bc} )
    ENDFOREACH ()

    # Link all of the individual LLVM bitcode files
    ADD_CUSTOM_COMMAND ( OUTPUT ${linked_shadeops_bc}
        COMMAND "${LLVM_DIRECTORY}/bin/llvm-link" -internalize ${shadeops_bc_list} -o ${linked_shadeops_bc}
        DEPENDS ${shadeops_bc_list} ${exec_headers} ${PROJECT_PUBLIC_HEADERS} ${shadeops_srcs}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

    # Serialze the linked bitcode into a CPP file and add it to the list of liboslexec soures
    ADD_CUSTOM_COMMAND ( OUTPUT ${shadeops_bc_cuda_cpp}
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.bash"
            ${linked_shadeops_bc} ${shadeops_bc_cuda_cpp} "osl_llvm_compiled_ops_cuda"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.bash" ${linked_shadeops_bc}
        ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
ENDMACRO ( )

if (USE_LLVM_BITCODE)
    LLVM_COMPILE ( llvm_ops.cpp liboslexec_srcs )

    # Optionally repeat the bitcode compilation with CUDA-specific options
    if (CUDA_FOUND)
        ADD_DEFINITIONS (-DOSL_LLVM_CUDA_BITCODE)
        CUDA_SHADEOPS_COMPILE ( liboslexec_srcs )
    endif ()
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
set_target_properties (oslexec
                       PROPERTIES
                       VERSION ${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}
                       SOVERSION ${SOVERSION}
                       OUTPUT_NAME oslexec${OSL_LIBNAME_SUFFIX}
                      )

TARGET_LINK_LIBRARIES ( oslexec
                        ${VISIBILITY_COMMAND} ${VISIBILITY_MAP_COMMAND}
                        ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES}
                        ${PUGIXML_LIBRARIES}
                        ${PARTIO_LIBRARIES} ${ZLIB_LIBRARIES}
                        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
                        ${CLANG_LIBRARIES}
                        ${LLVM_LIBRARIES} ${LLVM_LDFLAGS}
                        ${LLVM_SYSTEM_LIBRARIES})
ADD_DEPENDENCIES (oslexec "${CMAKE_SOURCE_DIR}/src/build-scripts/hidesymbols.map")

install_targets (oslexec)

# Unit tests
if (OSL_BUILD_TESTS)
    add_executable (accum_test accum_test.cpp)
    target_link_libraries ( accum_test oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    add_test (unit_accum "${CMAKE_BINARY_DIR}/src/liboslexec/accum_test")

    add_executable (dual_test dual_test.cpp)
    target_link_libraries ( dual_test ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    add_test (unit_dual "${CMAKE_BINARY_DIR}/src/liboslexec/dual_test")

    add_executable (llvmutil_test llvmutil_test.cpp)
    target_link_libraries ( llvmutil_test oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    add_test (unit_llvmutil "${CMAKE_BINARY_DIR}/src/liboslexec/llvmutil_test")
endif ()
