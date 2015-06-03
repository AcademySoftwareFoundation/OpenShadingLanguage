#########################################################################
#
# This is the master makefile.
# Here we put all the top-level make targets, platform-independent
# rules, etc.
#
# Run 'make help' to list helpful targets.
#
#########################################################################


.PHONY: all debug profile clean realclean nuke doxygen

working_dir	:= ${shell pwd}
INSTALLDIR	=${working_dir}

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
MY_CMAKE_FLAGS ?= -g3 -DSELF_CONTAINED_INSTALL_TREE:BOOL=TRUE
BUILDSENTINEL ?= Makefile
NINJA ?= ninja
CMAKE ?= cmake

# Site-specific build instructions
ifndef OSL_SITE
    OSL_SITE := ${shell uname -n}
endif
$(info OSL_SITE = ${OSL_SITE})
ifneq (${shell echo ${OSL_SITE} | grep imageworks},)
include ${working_dir}/site/spi/Makefile-bits
endif

# Set up variables holding the names of platform-dependent directories --
# set these after evaluating site-specific instructions
top_build_dir := build
build_dir     := ${top_build_dir}/${platform}${variant}
top_dist_dir  := dist
dist_dir      := ${top_dist_dir}/${platform}${variant}

$(info dist_dir = ${dist_dir})
$(info INSTALLDIR = ${INSTALLDIR})


VERBOSE := ${SHOWCOMMANDS}
ifneq (${VERBOSE},)
MY_MAKE_FLAGS += VERBOSE=${VERBOSE}
MY_CMAKE_FLAGS += -DVERBOSE:BOOL=1
TEST_FLAGS += -V
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

ifneq (${NAMESPACE},)
MY_CMAKE_FLAGS += -DOSL_NAMESPACE:STRING=${NAMESPACE}
endif

ifneq (${HIDE_SYMBOLS},)
MY_CMAKE_FLAGS += -DHIDE_SYMBOLS:BOOL=${HIDE_SYMBOLS}
endif

ifneq (${USE_FAST_MATH},)
MY_CMAKE_FLAGS += -DUSE_FAST_MATH:BOOL=${USE_FAST_MATH}
endif

ifneq (${ILMBASE_HOME},)
MY_CMAKE_FLAGS += -DILMBASE_HOME:STRING=${ILMBASE_HOME}
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

ifneq (${STOP_ON_WARNING},)
MY_CMAKE_FLAGS += -DSTOP_ON_WARNING:BOOL=${STOP_ON_WARNING}
endif

ifneq (${BUILDSTATIC},)
MY_CMAKE_FLAGS += -DBUILDSTATIC:BOOL=${BUILDSTATIC}
endif

ifneq (${LINKSTATIC},)
MY_CMAKE_FLAGS += -DLINKSTATIC:BOOL=${LINKSTATIC}
endif

ifneq (${USE_EXTERNAL_PUGIXML},)
MY_CMAKE_FLAGS += -DUSE_EXTERNAL_PUGIXML:BOOL=${USE_EXTERNAL_PUGIXML} -DPUGIXML_HOME=${PUGIXML_HOME}
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

ifneq (${USE_CPP11},)
MY_CMAKE_FLAGS += -DOSL_BUILD_CPP11:BOOL=${USE_CPP11}
endif

ifneq (${USE_LIBCPLUSPLUS},)
MY_CMAKE_FLAGS += -DOSL_BUILD_LIBCPLUSPLUS:BOOL=${USE_LIBCPLUSPLUS}
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

ifeq (${USE_NINJA},1)
MY_CMAKE_FLAGS += -G Ninja
BUILDSENTINEL := build.ninja
RUN_BUILD := ${NINJA} ${MY_NINJA_FLAGS}
else
RUN_BUILD := ${MAKE} ${MY_MAKE_FLAGS}
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
		${CMAKE} -DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/${dist_dir} \
			${MY_CMAKE_FLAGS} -DBOOST_ROOT=${BOOST_HOME} \
			../.. ; \
	 fi)

# 'make cmake' does a basic build (after first setting it up)
cmake: cmakesetup
	( cd ${build_dir} ; ${RUN_BUILD} )

# 'make cmakeinstall' builds everthing and installs it in 'dist'.
# Suppress pointless output from docs installation.
cmakeinstall: cmake
	( cd ${build_dir} ; ${RUN_BUILD} install | grep -v '^-- \(Installing\|Up-to-date\).*doc/html' )

# 'make package' builds everything and then makes an installable package
# (platform dependent -- may be .tar.gz, .sh, .dmg, .rpm, .deb. .exe)
package: cmakeinstall
	( cd ${build_dir} ; ${RUN_BUILD} package )

# 'make package_source' makes an installable source package
# (platform dependent -- may be .tar.gz, .sh, .dmg, .rpm, .deb. .exe)
package_source: cmakeinstall
	( cd ${build_dir} ; ${RUN_BUILD} package_source )

# 'make dist' is just a synonym for 'make cmakeinstall'
dist : cmakeinstall

# 'make test' does a full build and then runs all tests
test: cmake
	${CMAKE} -E cmake_echo_color --switch=$(COLOR) --cyan "Running tests ${TEST_FLAGS}..."
	( cd ${build_dir} ; ctest --force-new-ctest-process ${TEST_FLAGS} -E broken )

# 'make testall' does a full build and then runs all tests (even the ones
# that are expected to fail on some platforms)
testall: cmake
	${CMAKE} -E cmake_echo_color --switch=$(COLOR) --cyan "Running all tests ${TEST_FLAGS}..."
	( cd ${build_dir} ; ctest --force-new-ctest-process ${TEST_FLAGS} )

#clean: testclean
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

doxygen:
	doxygen src/doc/Doxyfile

#########################################################################



# 'make help' prints important make targets
help:
	@echo "Targets:"
	@echo "  make              Build optimized binaries and libraries in ${dist_dir},"
	@echo "                        temporary build files in ${build_dir}"
	@echo "  make debug        Build unoptimized with symbols in ${dist_dir}.debug,"
	@echo "                        temporary build files in ${build_dir}.debug"
	@echo "  make profile      Build for profiling in ${dist_dir}.profile,"
	@echo "                        temporary build files in ${build_dir}.profile"
	@echo "  make clean        Remove the temporary files in ${build_dir}"
	@echo "  make realclean    Remove both ${build_dir} AND ${dist_dir}"
	@echo "  make nuke         Remove ALL of build and dist (not just ${platform})"
	@echo "  make test         Run tests"
	@echo "  make testall      Run all tests, even broken ones"
	@echo "  make doxygen      Build the Doxygen docs in ${top_build_dir}/doxygen"
	@echo ""
	@echo "Helpful modifiers:"
	@echo "  C++ compiler and build process:"
	@echo "      VERBOSE=1                Show all compilation commands"
	@echo "      STOP_ON_WARNING=0        Do not stop building if compiler warns"
	@echo "      OSL_SITE=xx              Use custom site build mods"
	@echo "      MYCC=xx MYCXX=yy         Use custom compilers"
	@echo "      USE_CPP11=1              Compile in C++11 mode"
	@echo "      USE_LIBCPLUSPLUS=1       Use clang libc++"
	@echo "      EXTRA_CPP_ARGS=          Additional args to the C++ command"
	@echo "      USE_NINJA=1              Set up Ninja build (instead of make)"
	@echo "  Linking and libraries:"
	@echo "      HIDE_SYMBOLS=1           Hide symbols not in the public API"
	@echo "      BUILDSTATIC=1            Build static library instead of shared"
	@echo "      LINKSTATIC=1             Link with static external libraries when possible"
	@echo "  Finding and Using Dependencies:"
	@echo "      BOOST_HOME=path          Custom Boost installation"
	@echo "      ILMBASE_HOME=path        Custom Ilmbase installation"
	@echo "      PARTIO_HOME=             Use Partio from the given location"
	@echo "      USE_EXTERNAL_PUGIXML=1   Use the system PugiXML, not the one in OIIO"
	@echo "  LLVM-related options:"
	@echo "      LLVM_VERSION=3.4         Specify which LLVM version to use"
	@echo "      LLVM_DIRECTORY=xx        Specify where LLVM lives"
	@echo "      LLVM_NAMESPACE=xx        Specify custom LLVM namespace"
	@echo "      LLVM_STATIC=1            Use static LLVM libraries"
	@echo "      USE_LLVM_BITCODE=0       Don't generate embedded LLVM bitcode"
	@echo "  OSL build-time options:"
	@echo "      NAMESPACE=name           Wrap OSL APIs in another namespace"
	@echo "      USE_FAST_MATH=1          Use faster, but less accurate math (set to 0 for libm defaults)"
	@echo "      USE_SIMD=arch            Build with SIMD support (choices: 0, sse2, sse3,"
	@echo "                                    ssse3, sse4.1, sse4.2, f16c, comma-separated ok)"
	@echo ""

