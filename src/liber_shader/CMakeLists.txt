
# ER shader library
INCLUDE_DIRECTORIES (${ZLIB_INCLUDE_DIR})
INCLUDE_DIRECTORIES (${ELARA_SRC_DIR}/liber/include)

FILE (GLOB h_src ${ELARA_SRC_DIR}/liber_shader/*.h)
FILE (GLOB cpp_src ${ELARA_SRC_DIR}/liber_shader/*.cpp)

if (MSVC)
	# Enable PDB generation for Release build
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /DEBUG")
endif ()

ADD_LIBRARY (liber_shader SHARED ${h_src} ${cpp_src})
TARGET_LINK_LIBRARIES (liber_shader liber ${ZLIB_LIBRARY} ${OPENEXR_LIBRARIES})

if (MSVC)
	INSTALL (TARGETS liber_shader RUNTIME DESTINATION bin)
	INSTALL (TARGETS liber_shader LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
else ()
	INSTALL (TARGETS liber_shader RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif ()
