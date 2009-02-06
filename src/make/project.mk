# License and copyright goes here

#########################################################################
# project.mk
#
# This is where we put all project-specific make commands, including 
# the list of files that becomes our public distribution
#########################################################################


all:

#########################################################################
# 'make help' prints important make targets
help:
	@echo "Targets:"
	@echo "  make            (default target is 'all')"
	@echo "  make all        Build optimized binaries and libraries in ${dist_dir},"
	@echo "                      temporary build files in ${build_dir}"
	@echo "  make debug      Build unoptimized with symbols in ${dist_dir}.debug,"
	@echo "                      temporary build files in ${build_dir}.debug"
	@echo "  make profile    Build for profiling in ${dist_dir}.profile,"
	@echo "                      temporary build files in ${build_dir}.profile"
	@echo "  make clean      Remove the temporary files in ${build_dir}"
	@echo "  make realclean  Remove both ${build_dir} AND ${dist_dir}"
	@echo "  make nuke       Remove ALL of build and dist (not just ${platform})"
	@echo "Helpful modifiers:"
	@echo "  make SHOWCOMMANDS=1 ...     Show all compilation commands"
	@echo ""



#########################################################################
# dist_files lists (relative to build) all files that end up in an
# external distribution
dist_bins    	:= oslc
dist_libs     	:= 

# include files that get included in the compiled distribution
dist_includes	:= 

# make the public distro have a subdirectory in include,
# to avoid name clashes
# dist_include_dir := include/ProjectName

# docs files that get copied to dist
dist_docs	:= 

# files at the root level that get copied to dist
dist_root	:= 



#########################################################################
# Project-specific make variables




#########################################################################
# Path for including things specific to this project

THIRD_PARTY_TOOLS_HOME ?= ../external/dist/${platform}

ILMBASE_VERSION ?= 1.0.1
ILMBASE_HOME ?= ${THIRD_PARTY_TOOLS_HOME}
ILMBASE_LIB_AREA ?= ${ILMBASE_HOME}/lib/ilmbase-${ILMBASE_VERSION}
ILMBASE_CXX ?= -I${ILMBASE_HOME}/include/ilmbase-${ILMBASE_VERSION}/OpenEXR
LINK_ILMBASE_HALF ?= ${ILMBASE_LIB_AREA}/half${OEXT}
LINK_ILMBASE ?= ${LD_LIBPATH}${ILMBASE_LIB_AREA} ${LINKWITH}Imath ${LINK_ILMBASE_HALF} ${LINKWITH}IlmThread ${LINKWITH}Iex

BOOST_VERSION ?= 1_35_0
BOOST_HOME ?= ${THIRD_PARTY_TOOLS_HOME}
BOOST_CXX ?= -I${BOOST_HOME}/include/boost_${BOOST_VERSION}
BOOST_LIB_AREA ?= ${BOOST_HOME}/lib/boost_${BOOST_VERSION}
ifeq (${platform},macosx)
  BOOST_SUFFIX ?= -mt-1_35
else
  GCCVERCODE ?= ${shell gcc --version | grep -o "[0-9]\.[0-9]" | head -1 | tr -d "."}
  BOOST_SUFFIX ?= -gcc${GCCVERCODE}-mt-1_35
endif

LINK_BOOST ?= ${LD_LIBPATH}${BOOST_LIB_AREA} \
              ${LINKWITH}boost_filesystem${BOOST_SUFFIX} \
              ${LINKWITH}boost_system${BOOST_SUFFIX} \
              ${LINKWITH}boost_thread${BOOST_SUFFIX}

# Here we put instructions to copy libraries from these external tools into
# this project's lib directory.
dist_extra_libs += $(wildcard ${BOOST_LIB_AREA}/libboost_filesystem${BOOST_SUFFIX}${SHLIBEXT}*) \
		   $(wildcard ${BOOST_LIB_AREA}/libboost_system${BOOST_SUFFIX}${SHLIBEXT}*) \
		   $(wildcard ${BOOST_LIB_AREA}/libboost_thread${BOOST_SUFFIX}${SHLIBEXT}*)

OPENIMAGEIO_HOME ?= ${IMAGEIOHOME}
OPENIMAGEIO_INCLUDE ?= -I${OPENIMAGEIO_HOME}/include -I${OPENIMAGEIO_HOME}/include/OpenImageIO
OPENIMAGEIO_LINK ?= ${LD_LIBPATH}${OPENIMAGEIO_HOME}/lib ${LINKWITH}OpenImageIO

# Here we put the ${FOO_CXX} commands for the packages we actually use
PROJECT_EXTRA_CXX ?= ${OPENIMAGEIO_INCLUDE} ${BOOST_CXX}
