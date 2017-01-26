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

if (MSVC)
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	# Enable PDB generation for Release build
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
	set (CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
endif ()

TARGET_LINK_LIBRARIES ( oslcomp ${OPENIMAGEIO_LIBRARY} ${ILMBASE_LIBRARIES}
                       ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})

INSTALL ( TARGETS oslcomp RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib )

