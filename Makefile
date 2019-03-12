#########################################################################
#
# This is the master makefile.
# Here we put all the top-level make targets, platform-independent
# rules, etc.
#
# Run 'make help' to list helpful targets.
#
#########################################################################


.PHONY: all debug profile clean realclean nuke

working_dir	:= ${shell pwd}

# Figure out which architecture we're on
include ${working_dir}/src/make/detectplatform.mk

# Presence of make variables DEBUG and PROFILE cause us to make special
# builds, which we put in their own areas.
ifdef DEBUG
    variant +=.debug
endif
ifdef PROFILE
    variant +=.profile
endif

MY_MAKE_FLAGS ?=
MY_NINJA_FLAGS ?=
MY_CMAKE_FLAGS ?= 
#-g3 
#-DSELF_CONTAINED_INSTALL_TREE:BOOL=TRUE
BUILDSENTINEL ?= Makefile
NINJA ?= ninja
CMAKE ?= cmake

# Site-specific build instructions
OSL_SITE ?= ${shell uname -n}
ifneq (${shell echo ${OSL_SITE} | grep imageworks.com},)
include ${working_dir}/site/spi/Makefile-bits
endif

# Set up variables holding the names of platform-dependent directories --
# set these after evaluating site-specific instructions
top_build_dir := build
build_dir     := ${top_build_dir}/${platform}${variant}
top_dist_dir  := dist
dist_dir      := ${top_dist_dir}/${platform}${variant}
OSL_ROOT_DIR  ?= ${working_dir}/${dist_dir}

ifndef INSTALL_PREFIX
INSTALL_PREFIX := ${working_dir}/${dist_dir}
INSTALL_PREFIX_BRIEF := ${dist_dir}
else
INSTALL_PREFIX_BRIEF := ${INSTALL_PREFIX}
endif

VERBOSE ?= ${SHOWCOMMANDS}
ifneq (${VERBOSE},)
MY_MAKE_FLAGS += VERBOSE=${VERBOSE}
MY_CMAKE_FLAGS += -DVERBOSE:BOOL=${VERBOSE}
ifneq (${VERBOSE},0)
	MY_NINJA_FLAGS += -v
	TEST_FLAGS += -V
endif
$(info OSL_SITE = ${OSL_SITE})
$(info dist_dir = ${dist_dir})
$(info INSTALL_PREFIX = ${INSTALL_PREFIX})
endif

ifneq (${LLVM_DIRECTORY},)
MY_CMAKE_FLAGS += -DLLVM_DIRECTORY:STRING=${LLVM_DIRECTORY}
endif

ifneq (${LLVM_VERSION},)
MY_CMAKE_FLAGS += -DLLVM_VERSION:STRING=${LLVM_VERSION}
endif

ifneq (${LLVM_NAMESPACE},)
MY_CMAKE_FLAGS += -DLLVM_NAMESPACE:STRING=${LLVM_NAMESPACE}
endif

ifneq (${LLVM_STATIC},)
MY_CMAKE_FLAGS += -DLLVM_STATIC:BOOL=${LLVM_STATIC}
endif

ifneq (${USE_LLVM_BITCODE},)
MY_CMAKE_FLAGS += -DUSE_LLVM_BITCODE:BOOL=${USE_LLVM_BITCODE}
endif

ifneq (${USE_BOOST_WAVE},)
MY_CMAKE_FLAGS += -DUSE_BOOST_WAVE:BOOL=${USE_BOOST_WAVE}
endif

ifneq (${NAMESPACE},)
MY_CMAKE_FLAGS += -DOSL_NAMESPACE:STRING=${NAMESPACE}
endif

ifneq (${HIDE_SYMBOLS},)
MY_CMAKE_FLAGS += -DHIDE_SYMBOLS:BOOL=${HIDE_SYMBOLS}
endif

ifneq (${USE_FAST_MATH},)
MY_CMAKE_FLAGS += -DUSE_FAST_MATH:BOOL=${USE_FAST_MATH}
endif

# Old names -- DEPRECATED (1.9)
ifneq (${OPENEXR_HOME},)
MY_CMAKE_FLAGS += -DOPENEXR_ROOT_DIR:STRING=${OPENEXR_HOME}
endif
ifneq (${ILMBASE_HOME},)
MY_CMAKE_FLAGS += -DILMBASE_ROOT_DIR:STRING=${ILMBASE_HOME}
endif

ifneq (${OPENEXR_ROOT_DIR},)
MY_CMAKE_FLAGS += -DOPENEXR_ROOT_DIR:STRING=${OPENEXR_ROOT_DIR}
endif

ifneq (${ILMBASE_ROOT_DIR},)
MY_CMAKE_FLAGS += -DILMBASE_ROOT_DIR:STRING=${ILMBASE_ROOT_DIR}
endif

ifneq (${USE_PARTIO},)
MY_CMAKE_FLAGS += -DUSE_PARTIO:BOOL=${USE_PARTIO}
endif

ifneq (${PARTIO_HOME},)
MY_CMAKE_FLAGS += -DPARTIO_HOME:BOOL=${PARTIO_HOME} -DUSE_PARTIO:BOOL=1
endif

ifneq (${BOOST_HOME},)
MY_CMAKE_FLAGS += -DBOOST_ROOT:STRING=${BOOST_HOME}
endif

ifneq (${USE_QT},)
MY_CMAKE_FLAGS += -DUSE_QT:BOOL=${USE_QT}
endif

ifneq (${STOP_ON_WARNING},)
MY_CMAKE_FLAGS += -DSTOP_ON_WARNING:BOOL=${STOP_ON_WARNING}
endif

ifneq (${BUILDSTATIC},)
MY_CMAKE_FLAGS += -DBUILDSTATIC:BOOL=${BUILDSTATIC}
endif

ifneq (${SOVERSION},)
MY_CMAKE_FLAGS += -DSOVERSION:STRING=${SOVERSION}
endif

ifneq (${OSL_LIBNAME_SUFFIX},)
MY_CMAKE_FLAGS += -DOSL_LIBNAME_SUFFIX:STRING=${OSL_LIBNAME_SUFFIX}
endif

ifneq (${OIIO_LIBNAME_SUFFIX},)
MY_CMAKE_FLAGS += -DOIIO_LIBNAME_SUFFIX:STRING=${OIIO_LIBNAME_SUFFIX}
endif

ifneq (${LINKSTATIC},)
MY_CMAKE_FLAGS += -DLINKSTATIC:BOOL=${LINKSTATIC}
endif

ifneq (${OSL_BUILD_TESTS},)
MY_CMAKE_FLAGS += -DOSL_BUILD_TESTS:BOOL=${OSL_BUILD_TESTS}
endif

ifneq (${OSL_BUILD_SHADERS},)
MY_CMAKE_FLAGS += -DOSL_BUILD_SHADERS:BOOL=${OSL_BUILD_SHADERS}
endif

ifneq (${OSL_BUILD_MATERIALX},)
MY_CMAKE_FLAGS += -DOSL_BUILD_MATERIALX:BOOL=${OSL_BUILD_MATERIALX}
endif

ifdef DEBUG
MY_CMAKE_FLAGS += -DCMAKE_BUILD_TYPE:STRING=Debug
endif

ifdef PROFILE
MY_CMAKE_FLAGS += -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo
endif

ifneq (${MYCC},)
MY_CMAKE_FLAGS += -DCMAKE_C_COMPILER:STRING="${MYCC}"
endif
ifneq (${MYCXX},)
MY_CMAKE_FLAGS += -DCMAKE_CXX_COMPILER:STRING="${MYCXX}"
endif

ifneq (${USE_CPP},)
MY_CMAKE_FLAGS += -DUSE_CPP=${USE_CPP}
endif

ifneq (${USE_LIBCPLUSPLUS},)
MY_CMAKE_FLAGS += -DUSE_LIBCPLUSPLUS:BOOL=${USE_LIBCPLUSPLUS}
endif

ifneq (${EXTRA_CPP_ARGS},)
MY_CMAKE_FLAGS += -DEXTRA_CPP_ARGS:STRING="${EXTRA_CPP_ARGS}"
endif

ifneq (${USE_SIMD},)
MY_CMAKE_FLAGS += -DUSE_SIMD:STRING="${USE_SIMD}"
endif

ifneq (${TEST},)
TEST_FLAGS += -R ${TEST}
endif

ifneq (${USE_CCACHE},)
MY_CMAKE_FLAGS += -DUSE_CCACHE:BOOL=${USE_CCACHE}
endif

ifeq (${USE_NINJA},1)
MY_CMAKE_FLAGS += -G Ninja
BUILDSENTINEL := build.ninja
endif

ifneq (${CODECOV},)
MY_CMAKE_FLAGS += -DCMAKE_BUILD_TYPE:STRING=Debug -DCODECOV:BOOL=${CODECOV}
endif

ifneq (${SANITIZE},)
MY_CMAKE_FLAGS += -DSANITIZE=${SANITIZE}
endif

ifneq (${CLANG_TIDY},)
  MY_CMAKE_FLAGS += -DCLANG_TIDY:BOOL=1
endif
ifneq (${CLANG_TIDY_CHECKS},)
  MY_CMAKE_FLAGS += -DCLANG_TIDY_CHECKS:STRING=${CLANG_TIDY_CHECKS}
endif
ifneq (${CLANG_TIDY_ARGS},)
  MY_CMAKE_FLAGS += -DCLANG_TIDY_ARGS:STRING=${CLANG_TIDY_ARGS}
endif
ifneq (${CLANG_TIDY_FIX},)
  MY_CMAKE_FLAGS += -DCLANG_TIDY_FIX:BOOL=${CLANG_TIDY_FIX}
  MY_NINJA_FLAGS += -j 1
  # N.B. when fixing, you don't want parallel jobs!
endif

ifneq (${USE_OPTIX},)
MY_CMAKE_FLAGS += -DUSE_OPTIX:BOOL=${USE_OPTIX}
endif

#$(info MY_CMAKE_FLAGS = ${MY_CMAKE_FLAGS})
#$(info MY_MAKE_FLAGS = ${MY_MAKE_FLAGS})

#########################################################################




#########################################################################
# Top-level documented targets

all: dist

# 'make debug' is implemented via recursive make setting DEBUG
debug:
	${MAKE} DEBUG=1 --no-print-directory

# 'make profile' is implemented via recursive make setting PROFILE
profile:
	${MAKE} PROFILE=1 --no-print-directory

# 'make cmakesetup' constructs the build directory and runs 'cmake' there,
# generating makefiles to build the project.  For speed, it only does this when
# ${build_dir}/Makefile doesn't already exist, in which case we rely on the
# cmake generated makefiles to regenerate themselves when necessary.
cmakesetup:
	@ (if [ ! -e ${build_dir}/${BUILDSENTINEL} ] ; then \
		${CMAKE} -E make_directory ${build_dir} ; \
		cd ${build_dir} ; \
		${CMAKE} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
			${MY_CMAKE_FLAGS} -DBOOST_ROOT=${BOOST_HOME} \
			../.. ; \
	 fi)

ifeq (${USE_NINJA},1)

# 'make cmake' does a basic build (after first setting it up)
cmake: cmakesetup
	@ ( cd ${build_dir} ; ${NINJA} ${MY_NINJA_FLAGS} )

# 'make cmakeinstall' builds everthing and installs it in 'dist'.
# Suppress pointless output from docs installation.
cmakeinstall: cmake
	@ ( cd ${build_dir} ; ${NINJA} ${MY_NINJA_FLAGS} install | grep -v '^-- \(Installing\|Up-to-date\|Set runtime path\)' )

# 'make package' builds everything and then makes an installable package
# (platform dependent -- may be .tar.gz, .sh, .dmg, .rpm, .deb. .exe)
package: cmakeinstall
	@ ( cd ${build_dir} ; ${NINJA} ${MY_NINJA_FLAGS} package )

# 'make package_source' makes an installable source package
# (platform dependent -- may be .tar.gz, .sh, .dmg, .rpm, .deb. .exe)
package_source: cmakeinstall
	@ ( cd ${build_dir} ; ${NINJA} ${MY_NINJA_FLAGS} package_source )

else

# 'make cmake' does a basic build (after first setting it up)
cmake: cmakesetup
	@ ( cd ${build_dir} ; ${MAKE} ${MY_MAKE_FLAGS} )

# 'make cmakeinstall' builds everthing and installs it in 'dist'.
# Suppress pointless output from docs installation.
cmakeinstall: cmake
	@ ( cd ${build_dir} ; ${MAKE} ${MY_MAKE_FLAGS} install | grep -v '^-- \(Installing\|Up-to-date\|Set runtime path\)' )

# 'make package' builds everything and then makes an installable package
# (platform dependent -- may be .tar.gz, .sh, .dmg, .rpm, .deb. .exe)
package: cmakeinstall
	@ ( cd ${build_dir} ; ${MAKE} ${MY_MAKE_FLAGS} package )

# 'make package_source' makes an installable source package
# (platform dependent -- may be .tar.gz, .sh, .dmg, .rpm, .deb. .exe)
package_source: cmakeinstall
	@ ( cd ${build_dir} ; ${MAKE} ${MY_MAKE_FLAGS} package_source )

endif

# 'make dist' is just a synonym for 'make cmakeinstall'
dist : cmakeinstall

TEST_FLAGS += --force-new-ctest-process --output-on-failure

# 'make test' does a full build and then runs all tests
test: cmake
	@ ${CMAKE} -E cmake_echo_color --switch=$(COLOR) --cyan "Running tests ${TEST_FLAGS}..."
	@ # if [ "${CODECOV}" == "1" ] ; then lcov -b ${build_dir} -d ${build_dir} -z ; rm -rf ${build_dir}/cov ; fi
	@ ( cd ${build_dir} ; \
	    OSL_ROOT_DIR=${OSL_ROOT_DIR} \
	    LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH} \
	    DYLD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${DYLD_LIBRARY_PATH} \
	    OIIO_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${OIIO_LIBRARY_PATH} \
	    PYTHONPATH=${working_dir}/${build_dir}/src/python:${PYTHONPATH} \
	    ctest -E broken ${TEST_FLAGS} ;\
	  )
	@ ( if [ "${CODECOV}" == "1" ] ; then \
	      cd ${build_dir} ; \
	      lcov -b . -d . -c -o cov.info ; \
	      lcov --remove cov.info "/usr*" -o cov.info ; \
	      genhtml -o ./cov -t "OSL test coverage" --num-spaces 4 cov.info ; \
	  fi )

# 'make clean' clears out the build directory for this platform
clean:
	${CMAKE} -E remove_directory ${build_dir}

# 'make realclean' clears out both build and dist directories for this platform
realclean: clean
	${CMAKE} -E remove_directory ${dist_dir}

# 'make nuke' blows away the build and dist areas for all platforms
nuke:
	${CMAKE} -E remove_directory ${top_build_dir}
	${CMAKE} -E remove_directory ${top_dist_dir}

#########################################################################



# 'make help' prints important make targets
help:
	@echo "Targets:"
	@echo "  make              Build optimized binaries and libraries"
	@echo "  make debug        Build unoptimized with symbols"
	@echo "  make profile      Build for profiling"
	@echo "  make clean        Remove the temporary files in ${build_dir}"
	@echo "  make realclean    Remove both ${build_dir} AND ${dist_dir}"
	@echo "  make nuke         Remove ALL of build and dist (not just ${platform})"
	@echo "  make test         Run tests"
	@echo ""
	@echo "Helpful modifiers:"
	@echo "  C++ compiler and build process:"
	@echo "      VERBOSE=1                Show all compilation commands"
	@echo "      STOP_ON_WARNING=0        Do not stop building if compiler warns"
	@echo "      OSL_SITE=xx              Use custom site build mods"
	@echo "      MYCC=xx MYCXX=yy         Use custom compilers"
	@echo "      USE_CPP=14               Compile in C++14 mode (default is C++11)"
	@echo "      USE_LIBCPLUSPLUS=1       Use clang libc++"
	@echo "      EXTRA_CPP_ARGS=          Additional args to the C++ command"
	@echo "      USE_NINJA=1              Set up Ninja build (instead of make)"
	@echo "      USE_CCACHE=0             Disable ccache (even if available)"
	@echo "      CODECOV=1                Enable code coverage tests"
	@echo "      SANITIZE=name1,...       Enablie sanitizers (address, leak, thread)"
	@echo "      CLANG_TIDY=1             Run clang-tidy on all source (can be modified"
	@echo "                                  by CLANG_TIDY_ARGS=... and CLANG_TIDY_FIX=1"
	@echo "  Linking and libraries:"
	@echo "      HIDE_SYMBOLS=1           Hide symbols not in the public API"
	@echo "      BUILDSTATIC=1            Build static library instead of shared"
	@echo "      LINKSTATIC=1             Link with static external libs when possible"
	@echo "  Finding and Using Dependencies:"
	@echo "      BOOST_HOME=path          Custom Boost installation"
	@echo "      OPENEXR_ROOT_DIR=path    Custom OpenEXR installation"
	@echo "      ILMBASE_ROOT_DIR=path    Custom Ilmbase installation"
	@echo "      PARTIO_HOME=path         Use Partio from the given location"
	@echo "      USE_QT=0                 Skip anything that needs Qt"
	@echo "  LLVM-related options:"
	@echo "      LLVM_VERSION=6.0         Specify which LLVM version to use"
	@echo "      LLVM_DIRECTORY=xx        Specify where LLVM lives"
	@echo "      LLVM_NAMESPACE=xx        Specify custom LLVM namespace"
	@echo "      LLVM_STATIC=1            Use static LLVM libraries"
	@echo "      USE_LLVM_BITCODE=0       Don't generate embedded LLVM bitcode"
	@echo "      USE_BOOST_WAVE=1         Force boost wave rather than clang preprocessor"
	@echo "  OSL build-time options:"
	@echo "      INSTALL_PREFIX=path      Set installation prefix (default: ./${INSTALL_PREFIX_BRIEF})"
	@echo "      NAMESPACE=name           Override namespace base name (default: OSL)"
	@echo "      USE_FAST_MATH=1          Use faster, but less accurate math (set to 0 for libm defaults)"
	@echo "      OSL_BUILD_TESTS=0        Don't build unit tests, testshade, testrender"
	@echo "      OSL_BUILD_SHADERS=0      Don't build any shaders"
	@echo "      OSL_BUILD_MATERIALX=0    Don't build MaterialX shaders"
	@echo "      USE_SIMD=arch            Build with SIMD support (choices: 0, sse2, sse3,"
	@echo "                                  ssse3, sse4.1, sse4.2, f16c, avx, avx2"
	@echo "                                  comma-separated ok)"
	@echo "      USE_OPTIX=1              Build the OptiX test renderer"
	@echo "  make test, extra options:"
	@echo "      TEST=regex               Run only tests matching the regex"
	@echo ""

