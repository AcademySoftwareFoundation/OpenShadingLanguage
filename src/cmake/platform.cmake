###########################################################################
# Figure out what platform we're on, and set some variables appropriately

# CMAKE_SYSTEM_PROCESSOR should not be used because it indicates the platform
# we are building on, but when cross compiling or using a chroot this is not
# what we want to use
if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set (SYSTEM_PROCESSOR "x86_64")
else()
    set (SYSTEM_PROCESSOR "i686")
endif()

message (STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message (STATUS "CMAKE_SYSTEM_VERSION = ${CMAKE_SYSTEM_VERSION}")
message (STATUS "SYSTEM_PROCESSOR = ${SYSTEM_PROCESSOR}")

if (UNIX)
    message (STATUS "Unix! ${CMAKE_SYSTEM_NAME}")
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        set (platform "linux")
        set (CXXFLAGS "${CXXFLAGS} -DLINUX")
        if (${SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set (platform "linux64")
            set (CXXFLAGS "${CXXFLAGS} -DLINUX64")
        endif ()
    endif ()
    if (APPLE)
        message (STATUS "Apple!")
        set (platform "macosx")
    endif ()
endif ()

if (WIN32)
    message (STATUS "Windows!")
    set (platform "windows")
endif ()

if (platform)
    message (STATUS "platform = ${platform}")
else ()
    message (FATAL_ERROR "'platform' not defined")
endif ()

unset(SYSTEM_PROCESSOR)
