set (public_docs osl-languagespec.pdf
                 testshade.md.html
                 markdeep.min.js
                 docs.css
                 )

install (FILES ${public_docs} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT documentation)

install ( FILES "${PROJECT_SOURCE_DIR}/LICENSE"
                "${PROJECT_SOURCE_DIR}/INSTALL.md"
                "${PROJECT_SOURCE_DIR}/CHANGES.md"
                "${PROJECT_SOURCE_DIR}/README.md"
          DESTINATION ${CMAKE_INSTALL_DOCDIR} )


file (GLOB testshade_figures "Figures/testshade/*.jpg")
install (FILES ${testshade_figures}
         DESTINATION ${CMAKE_INSTALL_DOCDIR}/Figures/testshade )
