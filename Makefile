# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

#########################################################################
#
# This is the master makefile.
# Here we put all the top-level make targets, platform-independent
# rules, etc. This is just a fancy wrapper around cmake, but for many
# people, it's a lot simpler to just type "make" and have everything
# happen automatically.
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
MY_CMAKE_FLAGS += -g3
BUILDSENTINEL ?= Makefile
NINJA ?= ninja
CMAKE ?= cmake
CMAKE_BUILD_TYPE ?= Release

# Site-specific build instructions
OSL_SITE ?= ${shell uname -n}
ifneq (${shell echo ${OSL_SITE} | grep imageworks.com},)
include ${working_dir}/site/spi/Makefile-bits
endif

# Set up variables holding the names of platform-dependent directories --
# set these after evaluating site-specific instructions
top_build_dir ?= build
build_dir     ?= ${top_build_dir}/${platform}${variant}
top_dist_dir  ?= dist
dist_dir      ?= ${top_dist_dir}/${platform}${variant}

INSTALL_PREFIX ?= ${working_dir}/${dist_dir}

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

ifneq (${NAMESPACE},)
MY_CMAKE_FLAGS += -DOSL_NAMESPACE:STRING=${NAMESPACE}
endif

ifneq (${LLVM_DIRECTORY},)
MY_CMAKE_FLAGS += -DLLVM_DIRECTORY:STRING=${LLVM_DIRECTORY}
MY_CMAKE_FLAGS += -DLLVM_ROOT:STRING=${LLVM_DIRECTORY}
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

ifneq (${USE_FAST_MATH},)
MY_CMAKE_FLAGS += -DUSE_FAST_MATH:BOOL=${USE_FAST_MATH}
endif

ifneq (${STOP_ON_WARNING},)
MY_CMAKE_FLAGS += -DSTOP_ON_WARNING:BOOL=${STOP_ON_WARNING}
endif

ifneq (${BUILD_SHARED_LIBS},)
MY_CMAKE_FLAGS += -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
endif

ifneq (${LINKSTATIC},)
MY_CMAKE_FLAGS += -DLINKSTATIC:BOOL=${LINKSTATIC}
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
CMAKE_BUILD_TYPE=Debug
endif

ifdef PROFILE
CMAKE_BUILD_TYPE=RelWithDebInfo
endif

ifneq (${MYCC},)
MY_CMAKE_FLAGS += -DCMAKE_C_COMPILER:STRING="${MYCC}"
endif
ifneq (${MYCXX},)
MY_CMAKE_FLAGS += -DCMAKE_CXX_COMPILER:STRING="${MYCXX}"
endif

ifneq (${USE_CPP},)
MY_CMAKE_FLAGS += -DCMAKE_CXX_STANDARD=${USE_CPP}
endif

ifneq (${CMAKE_CXX_STANDARD},)
MY_CMAKE_FLAGS += -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
endif

ifneq (${USE_LIBCPLUSPLUS},)
MY_CMAKE_FLAGS += -DUSE_LIBCPLUSPLUS:BOOL=${USE_LIBCPLUSPLUS}
endif

ifneq (${GLIBCXX_USE_CXX11_ABI},)
MY_CMAKE_FLAGS += -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}
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

ifeq (${CODECOV},1)
  CMAKE_BUILD_TYPE=Debug
  MY_CMAKE_FLAGS += -DCODECOV:BOOL=${CODECOV}
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

ifneq (${CLANG_FORMAT_INCLUDES},)
  MY_CMAKE_FLAGS += -DCLANG_FORMAT_INCLUDES:STRING=${CLANG_FORMAT_INCLUDES}
endif
ifneq (${CLANG_FORMAT_EXCLUDES},)
  MY_CMAKE_FLAGS += -DCLANG_FORMAT_EXCLUDES:STRING=${CLANG_FORMAT_EXCLUDES}
endif

ifneq (${BUILD_MISSING_DEPS},)
  MY_CMAKE_FLAGS += -DBUILD_MISSING_DEPS:BOOL=${BUILD_MISSING_DEPS}
endif

ifneq (${USE_OPTIX},)
  MY_CMAKE_FLAGS += -DUSE_OPTIX:BOOL=${USE_OPTIX}
endif

ifneq (${USE_PYTHON},)
MY_CMAKE_FLAGS += -DUSE_PYTHON:BOOL=${USE_PYTHON}
endif

ifneq (${PYTHON_VERSION},)
MY_CMAKE_FLAGS += -DPYTHON_VERSION:STRING=${PYTHON_VERSION}
endif

ifneq (${PYLIB_LIB_PREFIX},)
MY_CMAKE_FLAGS += -DPYLIB_LIB_PREFIX:BOOL=${PYLIB_LIB_PREFIX}
endif

ifneq (${PYLIB_INCLUDE_SONAME},)
MY_CMAKE_FLAGS += -DPYLIB_INCLUDE_SONAME:BOOL=${PYLIB_INCLUDE_SONAME}
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

# 'make config' constructs the build directory and runs 'cmake' there,
# generating makefiles to build the project.  For speed, it only does this when
# ${build_dir}/Makefile doesn't already exist, in which case we rely on the
# cmake generated makefiles to regenerate themselves when necessary.
config:
	@ (if [ ! -e ${build_dir}/${BUILDSENTINEL} ] ; then \
		${CMAKE} -E make_directory ${build_dir} ; \
		cd ${build_dir} ; \
		${CMAKE} -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE} \
			 -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
			 ${MY_CMAKE_FLAGS} ../.. ; \
	 fi)


# 'make build' does a basic build (after first setting it up)
build: config
	@ ( cd ${build_dir} ; \
	    ${CMAKE} --build . --config ${CMAKE_BUILD_TYPE} \
	  )

# 'make install' builds everthing and installs it in 'dist'.
# Suppress pointless output from docs installation.
install: build
	@ ( cd ${build_dir} ; \
	    ${CMAKE} --build . --target install --config ${CMAKE_BUILD_TYPE} | grep -v '^-- \(Installing\|Up-to-date\|Set runtime path\)' \
	  )

# 'make package' builds everything and then makes an installable package
# (platform dependent -- may be .tar.gz, .sh, .dmg, .rpm, .deb. .exe)
package: install
	@ ( cd ${build_dir} ; \
	    ${CMAKE} --build . --target package --config ${CMAKE_BUILD_TYPE} \
	  )

# 'make package_source' makes an installable source package
# (platform dependent -- may be .tar.gz, .sh, .dmg, .rpm, .deb. .exe)
package_source: install
	@ ( cd ${build_dir} ; \
	    ${CMAKE} --build . --target package_source --config ${CMAKE_BUILD_TYPE} \
	  )

# 'make clang-format' runs clang-format on all source files (if it's installed)
clang-format: config
	@ ( cd ${build_dir} ; \
	    ${CMAKE} --build . --target clang-format --config ${CMAKE_BUILD_TYPE} \
	  )


# 'make dist' is just a synonym for 'make install'
dist : install

TEST_FLAGS += --force-new-ctest-process --output-on-failure

# 'make test' does a full build and then runs all tests
test: build
	@ ${CMAKE} -E cmake_echo_color --switch=$(COLOR) --cyan "Running tests ${TEST_FLAGS}..."
	@ ( cd ${build_dir} ; \
	    OSL_ROOT_DIR=${INSTALL_PREFIX} \
	    OSL_ROOT=${INSTALL_PREFIX} \
	    LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH} \
	    DYLD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${DYLD_LIBRARY_PATH} \
	    OIIO_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${OIIO_LIBRARY_PATH} \
	    PYTHONPATH=${working_dir}/${build_dir}/lib/python/site-packages:${PYTHONPATH} \
	    ctest -E broken ${TEST_FLAGS} \
	  )
	@ ( if [[ "${CODECOV}" == "1" ]] ; then \
	      cd ${build_dir} ; \
	      lcov -b . -d . -c -o cov.info ; \
	      lcov --remove cov.info "/usr*" -o cov.info ; \
	      genhtml -o ./cov -t "Test coverage" --num-spaces 4 cov.info ; \
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
	@echo "  make              Build and install optimized binaries and libraries"
	@echo "  make install      Build and install optimized binaries and libraries"
	@echo "  make build        Build only (no install) optimized binaries and libraries"
	@echo "  make config       Just configure cmake, don't build"
	@echo "  make debug        Build and install unoptimized with symbols"
	@echo "  make profile      Build and install for profiling"
	@echo "  make clean        Remove the temporary files in ${build_dir}"
	@echo "  make realclean    Remove both ${build_dir} AND ${dist_dir}"
	@echo "  make nuke         Remove ALL of build and dist (not just ${platform})"
	@echo "  make test         Run tests"
	@echo "  make testall      Run all tests, even broken ones"
	@echo "  make clang-format Run clang-format on all the source files"
	@echo ""
	@echo "Helpful modifiers:"
	@echo "  C++ compiler and build process:"
	@echo "      VERBOSE=1                Show all compilation commands"
	@echo "      STOP_ON_WARNING=0        Do not stop building if compiler warns"
	@echo "      OSL_SITE=xx              Use custom site build mods"
	@echo "      MYCC=xx MYCXX=yy         Use custom compilers"
	@echo "      CMAKE_CXX_STANDARD=14    Compile in C++14 mode (default is C++11)"
	@echo "      USE_LIBCPLUSPLUS=1       For clang, use libc++"
	@echo "      GLIBCXX_USE_CXX11_ABI=1  For gcc, use the new string ABI"
	@echo "      EXTRA_CPP_ARGS=          Additional args to the C++ command"
	@echo "      USE_NINJA=1              Set up Ninja build (instead of make)"
	@echo "      USE_CCACHE=0             Disable ccache (even if available)"
	@echo "      CODECOV=1                Enable code coverage tests"
	@echo "      SANITIZE=name1,...       Enable sanitizers (address, leak, thread)"
	@echo "      CLANG_TIDY=1             Run clang-tidy on all source (can be modified"
	@echo "                                  by CLANG_TIDY_ARGS=... and CLANG_TIDY_FIX=1"
	@echo "      CLANG_FORMAT_INCLUDES=... CLANG_FORMAT_EXCLUDES=..."
	@echo "                               Customize files for 'make clang-format'"
	@echo "  Linking and libraries:"
	@echo "      BUILD_SHARED_LIBS=0      Build static library instead of shared"
	@echo "      LINKSTATIC=1             Link with static external libs when possible"
	@echo "  Dependency hints:"
	@echo "      For each dependeny Foo, defining ENABLE_Foo=0 disables it, even"
	@echo "      if found. And you can hint where to find it with Foo_ROOT=path"
	@echo "      Note that it is case sensitive!"
	@echo "  Finding and Using Dependencies:"
	@echo "      BOOST_ROOT=path          Custom Boost installation"
	@echo "      USE_QT=0                 Skip anything that needs Qt"
	@echo "  LLVM-related options:"
	@echo "      LLVM_VERSION=7.0         Specify which LLVM version to use"
	@echo "      LLVM_DIRECTORY=xx        Specify where LLVM lives"
	@echo "      LLVM_NAMESPACE=xx        Specify custom LLVM namespace"
	@echo "      LLVM_STATIC=1            Use static LLVM libraries"
	@echo "      USE_LLVM_BITCODE=0       Don't generate embedded LLVM bitcode"
	@echo "  OSL build-time options:"
	@echo "      INSTALL_PREFIX=path      Set installation prefix (default: ${INSTALL_PREFIX})"
	@echo "      NAMESPACE=name           Override namespace base name (default: OSL)"
	@echo "      USE_FAST_MATH=1          Use faster, but less accurate math (set to 0 for libm defaults)"
	@echo "      OSL_BUILD_TESTS=0        Don't build unit tests, testshade, testrender"
	@echo "      OSL_BUILD_SHADERS=0      Don't build any shaders"
	@echo "      OSL_BUILD_MATERIALX=1    Build MaterialX shaders"
	@echo "      USE_SIMD=arch            Build with SIMD support (comma-separated choices:"
	@echo "                                  0, sse2, sse3, ssse3, sse4.1, sse4.2, f16c,"
	@echo "                                  avx, avx2, avx512f)"
	@echo "      USE_OPTIX=1              Build the OptiX test renderer"
	@echo "  make test, extra options:"
	@echo "      TEST=regex               Run only tests matching the regex"
	@echo ""

