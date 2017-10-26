###########################################################################
# Fonts are not available yet.
option (INSTALL_DOCS "Install documentation" ON)
option (INSTALL_FONTS "Install default fonts" OFF)

###########################################################################
# Rpath handling at the install step
set (MACOSX_RPATH ON)
if (CMAKE_SKIP_RPATH)
    # We need to disallow the user from truly setting CMAKE_SKIP_RPATH, since
    # we want to run the generated executables from the build tree in order to
    # generate the manual page documentation.  However, we make sure the
    # install rpath is unset so that the install tree is still free of rpaths
    # for linux packaging purposes.
    set (CMAKE_SKIP_RPATH FALSE)
    unset (CMAKE_INSTALL_RPATH)
else ()
    # the RPATH to be used when installing
    set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")

    # add the automatically determined parts of the RPATH that
    # point to directories outside the build tree to the install RPATH
    set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    if (VERBOSE)
        message (STATUS "CMAKE_INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}")
    endif ()
endif ()

# OSL considerations because we run oslc in the course of building:
# Use (i.e. don't skip) the full RPATH for the build tree
set (CMAKE_SKIP_BUILD_RPATH  FALSE)
# When building, don't use the install RPATH already (but later on when installing)
set (CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

