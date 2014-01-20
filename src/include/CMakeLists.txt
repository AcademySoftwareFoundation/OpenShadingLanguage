set (public_headers accum.h dual.h dual_vec.h
                    export.h genclosure.h Imathx.h llvm_util.h
                    matrix22.h optautomata.h oslclosure.h
                    oslcomp.h oslconfig.h oslexec.h oslquery.h
                    rendererservices.h shaderglobals.h)
message (STATUS "Create oslversion.h from oslversion.h.in")
configure_file (oslversion.h.in "${CMAKE_BINARY_DIR}/include/oslversion.h" @ONLY)
list (APPEND public_headers "${CMAKE_BINARY_DIR}/include/oslversion.h")

INSTALL ( FILES ${public_headers} DESTINATION include/OSL )
