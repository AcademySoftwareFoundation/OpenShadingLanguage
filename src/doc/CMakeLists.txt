set (public_docs osl-languagespec.pdf)
# Some day: CLA-INDIVIDUAL CLA-CORPORATE

install (FILES ${public_docs} DESTINATION doc COMPONENT documentation)

install ( FILES "${PROJECT_SOURCE_DIR}/LICENSE"
                "${PROJECT_SOURCE_DIR}/INSTALL.md"
                "${PROJECT_SOURCE_DIR}/CHANGES.md"
                "${PROJECT_SOURCE_DIR}/README.md"
          DESTINATION doc )
