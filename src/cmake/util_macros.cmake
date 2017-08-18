# Macro to set a variable with our funny overrides:
# If the variable is already set (by -D on the command line), leave it alone.
# If an environment variable of the same name is set, use that value
# (making it super easy for sites to override external tool locations).
# If neither of those, then use the default passed.
macro (setup_string name defaultval explanation)
    # If the named variable already has a value (was set by -D...), leave
    # it alone.  But if it's not yet set...
    if ("${${name}}" STREQUAL "")
        # If there's an environment variable of the same name that's
        # nonempty, use the env variable.  Otherwise, use the default.
        if (NOT $ENV{${name}} STREQUAL "")
            set (${name} $ENV{${name}})
                  # CACHE STRING ${explanation})
        else ()
            set (${name} ${defaultval})
                  # CACHE STRING ${explanation})
        endif ()
    endif ()
    if (VERBOSE)
        message (STATUS "${name} = ${${name}}")
    endif ()
endmacro ()

macro (setup_path name defaultval explanation)
    # If the named variable already has a value (was set by -D...), leave
    # it alone.  But if it's not yet set...
    if ("${${name}}" STREQUAL "")
        # If there's an environment variable of the same name that's
        # nonempty, use the env variable.  Otherwise, use the default.
        if (NOT $ENV{${name}} STREQUAL "")
            set (${name} $ENV{${name}})
                  # CACHE PATH ${explanation})
        else ()
            set (${name} ${defaultval})
                  # CACHE PATH ${explanation})
        endif ()
    endif ()
    if (VERBOSE)
        message (STATUS "${name} = ${${name}}")
    endif ()
endmacro ()


# Macro to install targets to the appropriate locations.  Use this instead of
# the install(TARGETS ...) signature.
#
# Usage:
#
#    install_targets (target1 [target2 ...])
#
macro (install_targets)
    install (TARGETS ${ARGN}
             RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT user
             LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT user
             ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT developer)
endmacro()
