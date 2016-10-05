
# ER executable

INCLUDE_DIRECTORIES (${ELARA_SRC_DIR}/liber/include)

FILE (GLOB h_src ${ELARA_SRC_DIR}/er/*.h)
FILE (GLOB cpp_src ${ELARA_SRC_DIR}/er/*.cpp)

ADD_EXECUTABLE (er ${h_src} ${cpp_src})
TARGET_LINK_LIBRARIES (er liber)

if (MSVC)
	# Enable SEH to support minidump
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHa")
	# Enable PDB generation for Release build
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /DEBUG")
endif ()

INSTALL (TARGETS er RUNTIME DESTINATION bin)
