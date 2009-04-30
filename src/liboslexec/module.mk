# Template for the makefile for an individual src/* directory.
# Fill in the blanks below.

# License and copyright goes here

# Name of the binary or library whose source is in this directory.
# Do NOT include .exe or any other suffix.
local_name := liboslexec

# Name of all source files in this directory
local_src := 
#sllex.l slgram.y

# Extra static libs needed to compile this binary (leave blank if this
# module is not for a binary executable)
local_libs := 

# Grammar and lexer object files
COMPILER_HEADERS := ${wildcard ${src_dir}/${local_name}/*.h}
OSOLEXL  := ${src_dir}/${local_name}/osolex.l
OSOLEXO  := ${build_obj_dir}/${local_name}/osolex${OEXT}
OSOLEXC  := ${build_obj_dir}/${local_name}/osolex.cpp
OSOGRAMY := ${src_dir}/${local_name}/osogram.y
OSOGRAMO := ${build_obj_dir}/${local_name}/osogram${OEXT}
OSOGRAMC := ${build_obj_dir}/${local_name}/osogram.cpp
OSOGRAMH := ${build_obj_dir}/${local_name}/osogram.hpp

# Extra objects from other libs we need to compile this library 
local_extra_objs := ${OSOGRAMO} ${OSOLEXO}

# Extra shared libs needed to compile this binary (leave blank if this
# module is not for a binary executable)
local_shlibs := 

# ld flags needed for this library
local_ldflags := ${OPENIMAGEIO_LINK} ${LINK_BOOST}


# 
${OSOLEXC}: ${OSOLEXL} ${OSOGRAMH} ${COMPILER_HEADERS}
	@ echo "  Compiling $@ ..."
	${FLEX} -+ -t ${OSOLEXL} > $@

${OSOLEXO}: ${OSOLEXC} ${OSOGRAMH}
	@ ${CXX} ${CFLAGS} ${CINCL}${src_dir}/liboslexec ${PROJECT_EXTRA_CXX} ${DASHC} ${OSOLEXC} ${DASHO}$@

# Action to build the object files
${OSOGRAMC}: ${OSOGRAMY} ${COMPILER_HEADERS}
	@ echo "  Compiling $@ from $< ..."
	${BISON} -dv -p oso -o $@ ${OSOGRAMY}

${OSOGRAMO}: ${OSOGRAMC} ${OSOGRAMH}
	@ echo "  Compiling $@ ..."
	${CXX} ${CFLAGS} ${CINCL}${src_dir}/liboslexec ${PROJECT_EXTRA_CXX} ${DASHC} ${OSOGRAMC} ${DASHO}$@

${OSOGRAMH}: ${OSOGRAMY}


## Include ONE of the includes below, depending on whether this module
## constructs a binary executable, a static library, or a shared library
## (DLL).

#include ${src_make_dir}/bin.mk
#include ${src_make_dir}/lib.mk
include ${src_make_dir}/shlib.mk
