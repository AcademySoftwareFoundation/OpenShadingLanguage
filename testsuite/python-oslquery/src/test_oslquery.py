#!/usr/bin/env python

# These lines make us compatible with both Python 2 and 3
from __future__ import print_function
from __future__ import absolute_import

# Import the Python bindings for OSLQuery.
# It may also be smart for many applications to
#     import OpenImageIO
# if you intend to do a lot of work with OIIO types such as TypeDesc
import oslquery


# printparam is called for each parameter or metadata item, it takes an
# argument that is an PyOSLQuery.Parameter
def printparam(p, indent="    ") :
    if p.isstruct :
        # If the parameter is a struct, then p.structname will be the name
        # of the struct, and p.fields will be a list of fields.
        # The master struct itself will have no data. The individual fields
        # are separate parameters the follow with names like "name.field",
        # for each field.
        print (indent, "struct {} {} with fields {} ...".format(p.structname, p.name, p.fields))
    elif p.isclosure :
        # If the parameter is a closure, it's special
        print (indent, "{}closure color {} = {}".format(
                    "output " if p.isoutput else "",
                    p.name, p.value))
    else :
        # All other parameter types. Note how we check for output-ness, the
        # type is an OpenImageIO::TypeDesc but it can print like a string,
        # if the type is a string we surround it with single quotes to make
        # it clear. Aggregate types will have their `value` print correctly
        # as tuples.
        print (indent, "{}{} {} = {}".format(
                    "output " if p.isoutput else "",
                    p.type, p.name,
                    "'{}'".format(p.value) if p.type == "string" else p.value))
    if p.spacename :
        print (indent, "    space:", p.spacename)
    # Metadata are themselves another tuple of Parameter objects hanging off
    # the parameter. We can iterate over them just like we iterated over the
    # shader params.
    for m in p.metadata :
        printparam (m, indent+"    meta: ")


######################################################################
# main test starts here

try:
    # Open a shader for query
    q = oslquery.OSLQuery('test')
    print ("Shader: ", q.shadertype(), q.shadername())

    # We can iterate over any shader-wide metadata as q.metadata
    for m in q.metadata :
        printparam (m, "  meta: ")

    # Iterating over the query object itself is iterating over the
    # parameters to the shader:
    print ("  Parameters:")
    for i in range(len(q)) :
        printparam(q[i])
    # FIXME(pybind11): The following way of looping over params should work.
    # But on Mac, with a combination of python 3.8/3.9 and pybind11 2.6, it
    # crashes. Works with older pybind11, so I think it's a pybind11 bug
    # that will get fixed at some point. Try it again later.
    #
    # for p in q :
    #     printparam(p)

    print ("Done.")
except Exception as detail:
    print ("Unknown exception:", detail)

