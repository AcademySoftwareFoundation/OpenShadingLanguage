file (GLOB public_headers OSL/*.h)

message (STATUS "Create oslversion.h from oslversion.h.in")
configure_file (OSL/oslversion.h.in "${CMAKE_BINARY_DIR}/include/OSL/oslversion.h" @ONLY)
list (APPEND public_headers "${CMAKE_BINARY_DIR}/include/OSL/oslversion.h")

INSTALL ( FILES ${public_headers} DESTINATION include/OSL )
