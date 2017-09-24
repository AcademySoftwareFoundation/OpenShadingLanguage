SET ( liboslcomp_srcs ast.cpp codegen.cpp oslcomp.cpp symtab.cpp typecheck.cpp
    )

# oslexec symbols used in oslcomp
if (NOT BUILDSTATIC)
    LIST(APPEND liboslcomp_srcs
        ../liboslexec/oslexec.cpp
        ../liboslexec/typespec.cpp
        )
endif ()

FILE ( GLOB compiler_headers "*.h" )

FLEX_BISON ( osllex.l oslgram.y osl liboslcomp_srcs compiler_headers )

if (BUILDSTATIC)
    ADD_LIBRARY ( oslcomp STATIC ${liboslcomp_srcs} )
    ADD_DEFINITIONS ( -DBUILD_STATIC=1 )
else ()
    ADD_LIBRARY ( oslcomp SHARED ${liboslcomp_srcs} )
endif ()

TARGET_LINK_LIBRARIES ( oslcomp ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES}
                       ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
                       ${CLANG_LIBRARIES} ${LLVM_LIBRARIES} ${LLVM_LDFLAGS}
                       ${LLVM_SYSTEM_LIBRARIES})

install_targets (oslcomp)

