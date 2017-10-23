file (GLOB public_headers OSL/*.h)

if (USE_CPP VERSION_GREATER 11)
    set (OSL_BUILD_CPP14 1)
endif ()
if (USE_CPP VERSION_GREATER 17)
    set (OSL_BUILD_CPP17 1)
endif ()

message (STATUS "Create oslversion.h from oslversion.h.in")
configure_file (OSL/oslversion.h.in "${CMAKE_BINARY_DIR}/include/OSL/oslversion.h" @ONLY)
list (APPEND public_headers "${CMAKE_BINARY_DIR}/include/OSL/oslversion.h")

INSTALL ( FILES ${public_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/OSL )

set (PROJECT_PUBLIC_HEADERS ${public_headers} PARENT_SCOPE)
