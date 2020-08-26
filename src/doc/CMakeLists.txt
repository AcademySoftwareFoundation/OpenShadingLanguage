# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

set (public_docs osl-languagespec.pdf
                 testshade.md.html
                 osltoy.md.html
                 markdeep.min.js
                 docs.css
                 )

install (FILES ${public_docs} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT documentation)

install ( FILES "${PROJECT_SOURCE_DIR}/LICENSE.md"
                "${PROJECT_SOURCE_DIR}/INSTALL.md"
                "${PROJECT_SOURCE_DIR}/CHANGES.md"
                "${PROJECT_SOURCE_DIR}/README.md"
          DESTINATION ${CMAKE_INSTALL_DOCDIR} )


file (GLOB testshade_figures "Figures/testshade/*.jpg")
install (FILES ${testshade_figures}
         DESTINATION ${CMAKE_INSTALL_DOCDIR}/Figures/testshade )

file (GLOB osltoy_figures "Figures/osltoy/*.jpg")
install (FILES ${osltoy_figures}
         DESTINATION ${CMAKE_INSTALL_DOCDIR}/Figures/osltoy )



# Macro to compile a shader with oslc. Syntax is:
#   docdeep_generate (SOURCE osl_source_file
#                 [ NAME doc_name ]
#                 [ DOC_LIST list_to_append_doc_filename ] )
#                 [ SRC_LIST list_to_append_src_filename ] )
macro (docdeep_generate)
    cmake_parse_arguments (_docdeep "" "NAME;DOC_LIST;SRC_LIST" "SOURCE" ${ARGN})
    # ^^ syntax: prefix options one-arg-keywords multi-arg-keywords args
    set (oslfile ${_docdeep_SOURCE})
    set (mdfile "${_docdeep_NAME}.md.html")
    if (VERBOSE)
        message (STATUS "docdeep will make '${mdfile}'")
    endif ()
    set (docdeep_program ${CMAKE_SOURCE_DIR}/src/build-scripts/docdeep.py)
    add_custom_command (OUTPUT ${mdfile}
        COMMAND python ${docdeep_program} -d ${_docdeep_NAME} -s docs.css
                ${_docdeep_SOURCE} > "${CMAKE_CURRENT_BINARY_DIR}/${mdfile}"
        MAIN_DEPENDENCY ${docdeep_program}
        DEPENDS ${_docdeep_SOURCE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "docdeep ${_docdeep_NAME}"
        )
    if (_docdeep_DOC_LIST)
        list (APPEND ${_docdeep_DOC_LIST} "${CMAKE_CURRENT_BINARY_DIR}/${mdfile}")
    endif ()
    if (_docdeep_SRC_LIST)
        list (APPEND ${_docdeep_SRC_LIST} ${_docdeep_SOURCE})
    endif ()
endmacro ()



set (all_docdeep_docs "")
set (all_docdeep_srcs "")
docdeep_generate (NAME OSLQuery
                  SOURCE ${PROJECT_SOURCE_DIR}/src/include/OSL/oslquery.h
                         ${PROJECT_SOURCE_DIR}/src/oslinfo/oslinfo.cpp
                  DOC_LIST all_docdeep_docs
                  SRC_LIST all_docdeep_srcs)
docdeep_generate (NAME docdeep
                  SOURCE ${PROJECT_SOURCE_DIR}/src/build-scripts/docdeep.py
                  DOC_LIST all_docdeep_docs
                  SRC_LIST all_docdeep_srcs)

add_custom_target (generated_docs ALL
                   DEPENDS ${all_docdeep_docs} ${all_docdeep_srcs}
                   SOURCES ${all_docdeep_srcs}
                   )

message (STATUS "All docdeep docs = ${all_docdeep_docs}")

install (FILES ${all_docdeep_docs}
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        COMPONENT documentation
        )
