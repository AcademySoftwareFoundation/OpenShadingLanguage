file (GLOB public_headers "*.h")

message (STATUS "Create oslversion.h from oslversion.h.in")
configure_file (oslversion.h.in ${CMAKE_BINARY_DIR}/include/oslversion.h @ONLY)
list (APPEND public_headers ${CMAKE_BINARY_DIR}/include/oslversion.h)

INSTALL ( FILES ${public_headers} DESTINATION include/OSL )
