###########################################################################
# OpenImageIO


# If 'OPENIMAGEHOME' not set, use the env variable of that name if available
if (NOT OPENIMAGEIOHOME)
    if (NOT $ENV{OPENIMAGEIOHOME} STREQUAL "")
        set (OPENIMAGEIOHOME $ENV{OPENIMAGEIOHOME})
    endif ()
endif ()


MESSAGE ( STATUS "OPENIMAGEIOHOME = ${OPENIMAGEIOHOME}" )

find_library ( OPENIMAGEIO_LIBRARY
               NAMES OpenImageIO
               HINTS ${OPENIMAGEIOHOME}
               PATH_SUFFIXES lib64 lib
               PATHS ${OPENIMAGEIOHOME}/lib )
find_path ( OPENIMAGEIO_INCLUDES
            NAMES OpenImageIO/imageio.h
            HINTS ${OPENIMAGEIOHOME}
            PATH_SUFFIXES include )
IF (OPENIMAGEIO_INCLUDES AND OPENIMAGEIO_LIBRARY )
    SET ( OPENIMAGEIO_FOUND TRUE )
    if (VERBOSE)
        MESSAGE ( STATUS "OpenImageIO includes = ${OPENIMAGEIO_INCLUDES}" )
        MESSAGE ( STATUS "OpenImageIO library = ${OPENIMAGEIO_LIBRARY}" )
    endif ()
ELSE ()
    MESSAGE ( STATUS "OpenImageIO not found" )
ENDIF ()

# end OpenImageIO setup
###########################################################################
