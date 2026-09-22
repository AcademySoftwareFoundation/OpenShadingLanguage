#!/usr/bin/env python

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
        # All other parameter types. Note how we check for output-ness.
        # p.type_name is the type as a plain string; if the type is a string
        # we surround the value with single quotes to make it clear.
        # Aggregate types will have their `value` print correctly as tuples.
        print (indent, "{}{} {} = {}".format(
                    "output " if p.isoutput else "",
                    p.type_name, p.name,
                    "'{}'".format(p.value) if p.type_name == "string" else p.value))
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
    for p in q :
        printparam(p)

    # Properties backed by a ustring that is empty or default-constructed.
    # These are worth exercising explicitly because printparam above only
    # reaches structname for parameters that are structs, so the empty case
    # went untested for years. It matters: ustring::c_str() is NULL for an
    # empty ustring, so any binding that converts via a raw char* instead of
    # ustring::string() crashes here rather than producing ''.
    print ("  Empty string properties:")
    print ("    structname of a non-struct param:", repr(q[0].structname))
    empty = oslquery.Parameter()
    print ("    default Parameter name:", repr(empty.name))
    print ("    default Parameter structname:", repr(empty.structname))
    print ("    default Parameter type_name:", repr(empty.type_name))
    print ("    default Parameter value:", repr(empty.value))
    print ("    default Parameter fields:", repr(empty.fields))
    print ("    default Parameter spacename:", repr(empty.spacename))

    print ("Done.")
except Exception as detail:
    print ("Unknown exception:", detail)

