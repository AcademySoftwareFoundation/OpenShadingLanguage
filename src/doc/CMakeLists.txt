set (public_docs osl-languagespec.pdf
                 testshade.md.html
                 markdeep.min.js
                 )

install (FILES ${public_docs} DESTINATION doc COMPONENT documentation)

install ( FILES "${PROJECT_SOURCE_DIR}/LICENSE"
                "${PROJECT_SOURCE_DIR}/INSTALL.md"
                "${PROJECT_SOURCE_DIR}/CHANGES.md"
                "${PROJECT_SOURCE_DIR}/README.md"
          DESTINATION doc )


file (GLOB testshade_figures "Figures/testshade/*.jpg")
install (FILES ${testshade_figures}
         DESTINATION doc/Figures/testshade )
