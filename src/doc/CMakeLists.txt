# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

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



