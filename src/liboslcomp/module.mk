# Template for the makefile for an individual src/* directory.
# Fill in the blanks below.

# License and copyright goes here

# Name of the binary or library whose source is in this directory.
# Do NOT include .exe or any other suffix.
local_name := liboslcomp

# Name of all source files in this directory
local_src := ast.cpp oslcomp.cpp
#sllex.l slgram.y

# Extra static libs needed to compile this binary (leave blank if this
# module is not for a binary executable)
local_libs := 

# Grammar and lexer object files
COMPILER_HEADERS := ${wildcard ${src_dir}/${local_name}/*.h}
LEXL  := ${src_dir}/${local_name}/osllex.l
LEXO  := ${build_obj_dir}/${local_name}/osllex${OEXT}
LEXC  := ${build_obj_dir}/${local_name}/osllex.cpp
GRAMY := ${src_dir}/${local_name}/oslgram.y
GRAMO := ${build_obj_dir}/${local_name}/oslgram${OEXT}
GRAMC := ${build_obj_dir}/${local_name}/oslgram.cpp
GRAMH := ${build_obj_dir}/${local_name}/oslgram.hpp

# Extra objects from other libs we need to compile this library 
local_extra_objs := ${GRAMO} ${LEXO}

# Extra shared libs needed to compile this binary (leave blank if this
# module is not for a binary executable)
local_shlibs := 

# ld flags needed for this library
local_ldflags := ${OPENIMAGEIO_LINK}


# 
${LEXC}: ${LEXL} ${GRAMH} ${COMPILER_HEADERS}
	@ echo "  Compiling $@ ..."
	${FLEX} --c++ -o $@ ${LEXL}

${LEXO}: ${LEXC} ${GRAMH}
	@ ${CXX} ${CFLAGS} ${CINCL}${src_dir}/liboslcomp ${PROJECT_EXTRA_CXX} ${DASHC} ${LEXC} ${DASHO}$@

# Action to build the object files
${GRAMC}: ${GRAMY} ${COMPILER_HEADERS}
	@ echo "  Compiling $@ from $< ..."
	${BISON} -dv -p osl -o $@ ${GRAMY}

${GRAMO}: ${GRAMC} ${GRAMH}
	@ echo "  Compiling $@ ..."
	${CXX} ${CFLAGS} ${CINCL}${src_dir}/liboslcomp ${PROJECT_EXTRA_CXX} ${DASHC} ${GRAMC} ${DASHO}$@

${GRAMH}: ${GRAMY}


## Include ONE of the includes below, depending on whether this module
## constructs a binary executable, a static library, or a shared library
## (DLL).

#include ${src_make_dir}/bin.mk
#include ${src_make_dir}/lib.mk
include ${src_make_dir}/shlib.mk
