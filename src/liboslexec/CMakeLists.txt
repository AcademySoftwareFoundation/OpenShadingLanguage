SET ( liboslexec_srcs
          shadingsys.cpp closure.cpp
          bsdf.cpp bsdf_diffuse.cpp bsdf_microfacet.cpp bsdf_phong.cpp
          bsdf_westin.cpp bsdf_ashikhmin_velvet.cpp
          bsdf_cloth_fncs.cpp bsdf_cloth.cpp bsdf_cloth_specular.cpp
          bsdf_reflection.cpp bsdf_refraction.cpp bsdf_transparent.cpp bsdf_ward.cpp
          bssrdf.cpp bsdf_hair.cpp bsdf_fakefur.cpp
          emissive.cpp background.cpp debug.cpp
          vol_subsurface.cpp
          context.cpp exec.cpp instance.cpp 
          loadshader.cpp master.cpp 
          oparray.cpp opassign.cpp opattribute.cpp
          opcolor.cpp opcompare.cpp opcontrol.cpp opderivs.cpp opinteger.cpp
          opmath.cpp opmathfunc.cpp opmatrix.cpp 
          opmessage.cpp opmisc.cpp opnoise.cpp 
          opspline.cpp opstring.cpp optexture.cpp opvector.cpp
          oslexec.cpp osoreader.cpp
          rendservices.cpp runtimeoptimize.cpp typespec.cpp
          lpexp.cpp lpeparse.cpp automata.cpp accum.cpp
          opclosure.cpp builtin_closures.cpp
          ../liboslcomp/ast.cpp ../liboslcomp/codegen.cpp
          ../liboslcomp/oslcomp.cpp ../liboslcomp/symtab.cpp
          ../liboslcomp/typecheck.cpp
    )

if (LLVM_FOUND)
    SET ( liboslexec_srcs ${liboslexec_srcs} llvm_instance.cpp )
endif ()

FILE ( GLOB compiler_headers "*.h" )

FLEX_BISON ( osolex.l osogram.y oso liboslexec_srcs compiler_headers )


MACRO ( LLVM_COMPILE llvm_src srclist )
    GET_FILENAME_COMPONENT ( llvmsrc_we ${llvm_src} NAME_WE )
    SET ( llvm_asm "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.s" )
    SET ( llvm_bc "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc" )
    SET ( llvm_bc_cpp "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc.cpp" )
    MESSAGE (STATUS "LLVM_COMPILE in=${llvm_src}")
    MESSAGE (STATUS "LLVM_COMPILE asm=${llvm_asm}")
    MESSAGE (STATUS "LLVM_COMPILE bc=${llvm_bc}")
    MESSAGE (STATUS "LLVM_COMPILE cpp=${llvm_bc_cpp}")
    SET ( ${srclist} ${${srclist}} ${llvm_bc_cpp} )
    EXEC_PROGRAM ( ${LLVM_DIRECTORY}/bin/llvm-config ARGS --cxxflags OUTPUT_VARIABLE LLVM_COMPILE_FLAGS )
    set (LLVM_COMPILE_FLAGS "${LLVM_COMPILE_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -O3 --combine")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS" )
    if (OSL_NAMESPACE)
        LIST (APPEND LLVM_COMPILE_FLAGS "-DOSL_NAMESPACE=${OSL_NAMESPACE}")
    endif ()
    if (OPENIMAGEIO_NAMESPACE)
        LIST (APPEND LLVM_COMPILE_FLAGS "-DOPENIMAGEIO_NAMESPACE=${OPENIMAGEIO_NAMESPACE}")
    endif ()

    # Command to turn the .cpp file into LLVM assembly language .s, into
    # LLVM bitcode .bc, then back into a C++ file with the bc embedded!
    ADD_CUSTOM_COMMAND ( OUTPUT ${llvm_bc_cpp}
      COMMAND ${LLVM_DIRECTORY}/bin/llvm-g++ ${LLVM_COMPILE_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_SOURCE_DIR}/include -I${OPENIMAGEIOHOME}/include -I${ILMBASE_INCLUDE_AREA} -I${Boost_INCLUDE_DIRS} -S -emit-llvm -o ${llvm_asm} ${llvm_src}
      COMMAND ${LLVM_DIRECTORY}/bin/llvm-as -f -o ${llvm_bc} ${llvm_asm}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.bash ${llvm_bc} ${llvm_bc_cpp}
      MAIN_DEPENDENCY ${llvm_src}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.bash
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
ENDMACRO ( )

if (LLVM_FOUND)
    LLVM_COMPILE ( llvm_ops.cpp liboslexec_srcs )
endif ()

ADD_LIBRARY ( oslexec SHARED ${liboslexec_srcs} )
TARGET_LINK_LIBRARIES ( oslexec oslcomp ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
if (LLVM_FOUND)
    TARGET_LINK_LIBRARIES ( oslexec ${LLVM_LIBRARY} )
endif()
LINK_ILMBASE ( oslexec )

INSTALL ( TARGETS oslexec LIBRARY DESTINATION lib )



# Unit tests
add_executable (closure_test closure_test.cpp)
add_executable (accum_test accum_test.cpp)
target_link_libraries ( closure_test oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
target_link_libraries ( accum_test oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
link_ilmbase (closure_test)
link_ilmbase (accum_test)
add_test (unit_closure ${CMAKE_BINARY_DIR}/liboslexec/closure_test)
add_test (unit_accum ${CMAKE_BINARY_DIR}/liboslexec/accum_test)
