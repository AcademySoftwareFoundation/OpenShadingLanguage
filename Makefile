#########################################################################
# The top-level Makefile is just a stub that merely includes
# src/make/master.mk
#########################################################################

include src/make/master.mk

$(info "dist_dir = ${dist_dir}")

OSL_MAKE_FLAGS ?=
OSL_CMAKE_FLAGS ?=

ifneq (${VERBOSE},)
OSL_MAKE_FLAGS += VERBOSE=${VERBOSE}
endif

ifdef DEBUG
OSL_CMAKE_FLAGS += -DCMAKE_BUILD_TYPE:STRING=Debug
endif

$(info OSL_CMAKE_FLAGS = ${OSL_CMAKE_FLAGS})
$(info OSL_MAKE_FLAGS = ${OSL_MAKE_FLAGS})



cmakesetup:
	- ${MKDIR} build/${platform}${variant}
	( cd build/${platform}${variant} ; \
	  cmake -DCMAKE_INSTALL_PREFIX=${working_dir}/dist/${platform}${variant} \
	        ${OSL_CMAKE_FLAGS} \
		-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
		-DBOOST_ROOT=${BOOST_HOME} \
		../../src )

cmake: cmakesetup
	( cd build/${platform}${variant} ; make ${OSL_MAKE_FLAGS} )

cmakeinstall: cmake
	( cd build/${platform}${variant} ; make ${OSL_MAKE_FLAGS} install )

cmaketest: cmake
	( cd build/${platform}${variant} ; make ${OSL_MAKE_FLAGS} test )

package: cmakeinstall
	( cd build/${platform}${variant} ; make ${OSL_MAKE_FLAGS} package )

package_source: cmakeinstall
	( cd build/${platform}${variant} ; make ${OSL_MAKE_FLAGS} package_source )
