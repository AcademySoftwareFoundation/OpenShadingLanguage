###########################################################################
# OpenImageIO

find_library ( OPENIMAGEIO_LIBRARY
               NAMES OpenImageIO
               PATHS $ENV{IMAGEIOHOME}/lib/ )
find_path ( OPENIMAGEIO_INCLUDES OpenImageIO/imageio.h
            $ENV{IMAGEIOHOME}/include )
IF (OPENIMAGEIO_INCLUDES AND OPENIMAGEIO_LIBRARY )
    SET ( OPENIMAGEIO_FOUND TRUE )
    MESSAGE ( STATUS "OpenImageIO includes = ${OPENIMAGEIO_INCLUDES}" )
    MESSAGE ( STATUS "OpenImageIO library = ${OPENIMAGEIO_LIBRARY}" )
ELSE ()
    MESSAGE ( STATUS "OpenImageIO not found" )
ENDIF ()

# end OpenImageIO setup
###########################################################################
