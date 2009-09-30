###########################################################################
# OpenImageIO

MESSAGE ( STATUS "OPENIMAGEIOHOME = ${OPENIMAGEIOHOME}" )

find_library ( OPENIMAGEIO_LIBRARY
               NAMES OpenImageIO
               PATHS $ENV{IMAGEIOHOME}/lib/ ${OPENIMAGEIOHOME}/lib )
find_path ( OPENIMAGEIO_INCLUDES OpenImageIO/imageio.h
            $ENV{IMAGEIOHOME}/include ${OPENIMAGEIOHOME}/include )
IF (OPENIMAGEIO_INCLUDES AND OPENIMAGEIO_LIBRARY )
    SET ( OPENIMAGEIO_FOUND TRUE )
    MESSAGE ( STATUS "OpenImageIO includes = ${OPENIMAGEIO_INCLUDES}" )
    MESSAGE ( STATUS "OpenImageIO library = ${OPENIMAGEIO_LIBRARY}" )
    MESSAGE ( STATUS "OpenImageIO namespace = ${OPENIMAGEIO_NAMESPACE}" )
    if (OPENIMAGEIO_NAMESPACE)
        add_definitions ("-DOPENIMAGEIO_NAMESPACE=${OPENIMAGEIO_NAMESPACE}")
    endif ()
ELSE ()
    MESSAGE ( STATUS "OpenImageIO not found" )
ENDIF ()

# end OpenImageIO setup
###########################################################################
