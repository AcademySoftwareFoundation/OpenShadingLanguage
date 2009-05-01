# Template for the makefile for an individual src/* directory.
# Fill in the blanks below.

# License and copyright goes here

# Name of the binary or library whose source is in this directory.
# Do NOT include .exe or any other suffix.
local_name := liboslquery

# Name of all source files in this directory
local_src := oslquery.cpp

# Extra static libs needed to compile this binary (leave blank if this
# module is not for a binary executable)
local_libs := 

# Grammar and lexer object files
OSOLEXO  := ${build_obj_dir}/liboslexec/osolex${OEXT}
OSOGRAMO := ${build_obj_dir}/liboslexec/osogram${OEXT}

# Extra objects from other libs we need to compile this library 
local_extra_objs := ${build_obj_dir}/liboslexec/osoreader${OEXT} \
	            ${OSOGRAMO} ${OSOLEXO} 

# Extra shared libs needed to compile this binary (leave blank if this
# module is not for a binary executable)
local_shlibs := liboslcomp

# ld flags needed for this library
local_ldflags := ${OPENIMAGEIO_LINK} ${LINK_BOOST}


## Include ONE of the includes below, depending on whether this module
## constructs a binary executable, a static library, or a shared library
## (DLL).

#include ${src_make_dir}/bin.mk
#include ${src_make_dir}/lib.mk
include ${src_make_dir}/shlib.mk
