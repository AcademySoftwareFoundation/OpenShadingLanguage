MACRO ( RUNTEST testname )
    ADD_TEST ( ${testname} python ${PROJECT_SOURCE_DIR}/../testsuite/tests/${testname}/run.py ${PROJECT_SOURCE_DIR}/../testsuite/tests/${testname} ${CMAKE_BINARY_DIR} )
ENDMACRO ()

RUNTEST ( oslc-empty )

